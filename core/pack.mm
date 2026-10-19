// Safetensors pack loader: mmap every shard once, no-copy MTLBuffer windows per tensor. Owner: lab-core.
#import <Foundation/Foundation.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>

#include "core/engine.hpp"

namespace ne {

struct Mapping {
  std::string path;
  void *base = nullptr;
  uint64_t size = 0;     // file size
  uint64_t mapped = 0;   // page-rounded mapping length
  ~Mapping() {
    if (base) munmap(base, mapped);
  }
};

static std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  NE_CHECK(f.good(), "cannot read %s", path.c_str());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
static bool exists(const std::string &p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}
static std::string real_path(const std::string &p) {
  char buf[PATH_MAX];
  NE_CHECK(realpath(p.c_str(), buf) != nullptr, "realpath(%s): %s", p.c_str(), strerror(errno));
  return buf;
}
static bool ends_with(const std::string &s, const char *suf) {
  size_t n = strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

std::vector<SafetensorsEntry> read_safetensors_header(const std::string &path, uint64_t file_size, Json *metadata) {
  int fd = open(path.c_str(), O_RDONLY);
  NE_CHECK(fd >= 0, "open %s: %s", path.c_str(), strerror(errno));
  uint64_t hlen = 0;
  bool ok = pread(fd, &hlen, 8, 0) == 8;
  std::string h;
  if (ok) {
    NE_CHECK(hlen > 0 && hlen < (1ull << 30) && 8 + hlen <= file_size, "%s: bad safetensors header length %llu",
             path.c_str(), (unsigned long long)hlen);
    h.resize(hlen);
    ok = pread(fd, h.data(), hlen, 8) == (ssize_t)hlen;
  }
  close(fd);
  NE_CHECK(ok, "%s: short read of the header", path.c_str());
  Json j = Json::parse(h);
  NE_CHECK(j.is_object(), "%s: header is not an object", path.c_str());
  const uint64_t data0 = 8 + hlen;
  std::vector<SafetensorsEntry> out;
  for (auto &kv : j.obj()) {
    if (kv.first == "__metadata__") {
      if (metadata) *metadata = kv.second;
      continue;
    }
    const Json &v = kv.second;
    SafetensorsEntry e;
    e.name = kv.first;
    e.dtype = dtype_from_name(v["dtype"].str());
    NE_CHECK(e.dtype != DType::Invalid, "%s: %s has unsupported dtype %s", path.c_str(), e.name.c_str(),
             v["dtype"].str().c_str());
    int64_t numel = 1;
    for (auto &d : v["shape"].arr()) {
      e.shape.push_back(d.i64());
      numel *= d.i64();
    }
    NE_CHECK(e.shape.size() <= (size_t)kMaxDims, "%s: %s has too many dims", path.c_str(), e.name.c_str());
    const Json &off = v["data_offsets"];
    uint64_t b = (uint64_t)off[0].i64(), en = (uint64_t)off[1].i64();
    NE_CHECK(b <= en && en - b == (uint64_t)numel * dtype_size(e.dtype), "%s: %s: data_offsets [%llu, %llu] do not "
             "match %lld x %zu bytes", path.c_str(), e.name.c_str(), (unsigned long long)b, (unsigned long long)en,
             (long long)numel, dtype_size(e.dtype));
    e.begin = data0 + b;
    e.end = data0 + en;
    NE_CHECK(e.end <= file_size, "%s: %s extends past the end of the file", path.c_str(), e.name.c_str());
    out.push_back(std::move(e));
  }
  return out;
}

// A no-copy buffer over [begin, end) of the mapping, rounded out to pages. Returns the buffer and the byte offset of
// `begin` inside it. The buffer keeps the mapping alive.
static id<MTLBuffer> window(const std::shared_ptr<Mapping> &m, uint64_t begin, uint64_t end, size_t *offset) {
  Device &d = Device::get();
  const uint64_t pg = d.page_size;
  uint64_t a = begin / pg * pg, b = (end + pg - 1) / pg * pg;
  b = std::min(b, m->mapped);
  NE_CHECK(b - a <= d.max_buffer_length, "window of %llu bytes exceeds maxBufferLength %llu",
           (unsigned long long)(b - a), (unsigned long long)d.max_buffer_length);
  std::shared_ptr<Mapping> keep = m;
  id<MTLBuffer> buf = [d.dev newBufferWithBytesNoCopy:(char *)m->base + a
                                               length:b - a
                                              options:MTLResourceStorageModeShared
                                          deallocator:^(void *, NSUInteger) {
                                            (void)keep;  // released with the block: last one unmaps
                                          }];
  NE_CHECK(buf != nil, "newBufferWithBytesNoCopy failed (%s, %llu bytes at %llu)", m->path.c_str(),
           (unsigned long long)(b - a), (unsigned long long)a);
  *offset = begin - a;
  return buf;
}

// One mapped file and its tensors (names without a pack's file prefix; no-copy buffers over the mapping), shared by
// every open pack that lists the file. Keyed by (device, inode); the registry holds weak references: the entry dies
// with the last pack (and the buffers with the last tensor view) that uses it.
struct FileEntry {
  std::shared_ptr<Mapping> map;
  std::vector<TensorInfo> tensors;
};

static std::shared_ptr<FileEntry> shared_file(const std::string &rp) {
  static std::mutex mu;
  static std::map<std::pair<dev_t, ino_t>, std::weak_ptr<FileEntry>> reg;
  int fd = ::open(rp.c_str(), O_RDONLY);
  NE_CHECK(fd >= 0, "open %s: %s", rp.c_str(), strerror(errno));
  struct stat fs;
  if (fstat(fd, &fs) != 0) {
    ::close(fd);
    NE_CHECK(false, "fstat %s: %s", rp.c_str(), strerror(errno));
  }
  const std::pair<dev_t, ino_t> key{fs.st_dev, fs.st_ino};
  std::lock_guard<std::mutex> g(mu);
  if (auto it = reg.find(key); it != reg.end())
    if (auto fe = it->second.lock()) {
      ::close(fd);
      return fe;
    }
  Device &d = Device::get();
  auto fe = std::make_shared<FileEntry>();
  auto m = fe->map = std::make_shared<Mapping>();
  m->path = rp;
  m->size = (uint64_t)fs.st_size;
  m->mapped = (m->size + d.page_size - 1) / d.page_size * d.page_size;
  m->base = mmap(nullptr, m->mapped, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  NE_CHECK(m->base != MAP_FAILED, "mmap %s: %s", rp.c_str(), strerror(errno));
  for (auto &e : read_safetensors_header(rp, m->size, nullptr)) {
    TensorInfo ti;
    ti.name = e.name;
    ti.file = rp;
    ti.file_offset = e.begin;
    ti.nbytes = e.end - e.begin;
    ti.host = (const char *)m->base + e.begin;
    ti.map = m;
    ti.t.dtype = e.dtype;
    ti.t.ndim = (int)e.shape.size();
    for (size_t i = 0; i < e.shape.size(); i++) ti.t.shape[i] = e.shape[i];
    if (ti.nbytes > 0 && ti.nbytes + 2 * d.page_size <= d.max_buffer_length)
      ti.t.buf = window(m, e.begin, e.end, &ti.t.offset);
    fe->tensors.push_back(std::move(ti));
  }
  for (auto it = reg.begin(); it != reg.end();) it = it->second.expired() ? reg.erase(it) : std::next(it);
  reg[key] = fe;
  return fe;
}

std::unique_ptr<Pack> Pack::open(const std::string &dir_in) {
  @autoreleasepool {
    std::unique_ptr<Pack> p(new Pack());
    p->aligned_ = std::make_unique<AlignedCache>();
    std::string dir = dir_in;
    if (!dir.empty() && dir[0] == '~') dir = std::string(getenv("HOME") ? getenv("HOME") : "") + dir.substr(1);
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    p->dir_ = dir;
    struct stat st;
    NE_CHECK(stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode), "pack: %s is not a directory", dir.c_str());

    if (exists(dir + "/config.json")) {
      p->config_text_ = read_file(dir + "/config.json");
      p->config_ = Json::parse(p->config_text_);
    }

    // Which files.
    std::vector<std::string> files;
    std::multimap<std::string, std::string> expected;  // file -> tensor name (from the index)
    if (exists(dir + "/model.safetensors.index.json")) {
      Json idx = Json::parse(read_file(dir + "/model.safetensors.index.json"));
      std::set<std::string> seen;
      for (auto &kv : idx["weight_map"].obj()) {
        const std::string &f = kv.second.str();
        if (seen.insert(f).second) files.push_back(f);
        expected.emplace(f, kv.first);
      }
    } else {
      DIR *dd = opendir(dir.c_str());
      NE_CHECK(dd, "opendir %s", dir.c_str());
      while (dirent *e = readdir(dd)) {
        std::string n = e->d_name;
        if (ends_with(n, ".safetensors")) files.push_back(n);
      }
      closedir(dd);
    }
    std::sort(files.begin(), files.end());
    std::string ngram = "ngram_table.bin";
    if (const Json *nt = p->config_.find("ngram_table"); nt && nt->is_object()) ngram = nt->get_str("file", ngram);
    if (exists(dir + "/" + ngram) && std::find(files.begin(), files.end(), ngram) == files.end()) files.push_back(ngram);

    // Map each distinct file once -- per process: a file another open pack already maps (the same inode: a symlink,
    // a hard link, the same path) reuses that pack's mapping AND its tensors' no-copy MTLBuffers, so two packs that
    // share shards (base + fine-tune: Swift-1.5 shares the 75 GB of experts) are one set of buffers in RAM and in the
    // queue's residency sets (lab-srvperf).
    for (const std::string &f : files) {
      std::string path = dir + "/" + f;
      std::string rp = real_path(path);
      std::shared_ptr<FileEntry> fe = shared_file(rp);
      p->files_.push_back(fe);
      const bool st_file = ends_with(f, ".safetensors");
      std::string prefix = st_file ? "" : f.substr(0, f.rfind('.')) + ".";
      if (std::find(p->maps_.begin(), p->maps_.end(), fe->map) == p->maps_.end()) p->maps_.push_back(fe->map);
      std::set<std::string> here;
      for (const TensorInfo &e : fe->tensors) {
        TensorInfo ti = e;
        ti.name = prefix + e.name;
        here.insert(e.name);
        NE_CHECK(p->tensors_.find(ti.name) == p->tensors_.end(), "pack: tensor %s appears twice (%s)", ti.name.c_str(),
                 f.c_str());
        p->tensors_.emplace(ti.name, std::move(ti));
      }
      auto [lo, hi] = expected.equal_range(f);
      for (auto it = lo; it != hi; ++it)
        NE_CHECK(here.count(it->second), "pack: index lists %s in %s, but the file lacks it", it->second.c_str(),
                 f.c_str());
    }
    for (auto &kv : p->tensors_) p->names_.push_back(kv.first);
    return p;
  }
}

struct Pack::AlignedCache {
  std::mutex mu;
  std::map<std::string, Tensor> copies;  // key: name + "@" + align
  uint64_t bytes = 0;
};

const Tensor &Pack::aligned(std::string_view name, size_t align) const {
  const Tensor &t = at(name);
  if (t.gpu_address() % align == 0) return t;
  NE_CHECK(align <= 16384 && (align & (align - 1)) == 0, "aligned(): align must be a power of two <= 16384");
  std::lock_guard<std::mutex> g(aligned_->mu);
  std::string key = std::string(name) + "@" + std::to_string(align);
  auto it = aligned_->copies.find(key);
  if (it != aligned_->copies.end()) return it->second;
  Tensor c = t;
  c.buf = alloc_bytes(t.nbytes());  // newBuffer contents are page-aligned
  c.offset = 0;
  memcpy(c.buf.contents, t.host(), t.nbytes());
  aligned_->bytes += t.nbytes();
  return aligned_->copies.emplace(key, c).first->second;
}
uint64_t Pack::aligned_copy_bytes() const {
  std::lock_guard<std::mutex> g(aligned_->mu);
  return aligned_->bytes;
}

Pack::~Pack() {
  // Buffers hold the mappings; dropping our references unmaps whatever nobody else (another pack) still views.
  tensors_.clear();
  maps_.clear();
  files_.clear();
}

const TensorInfo *Pack::info(std::string_view name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}
const Tensor *Pack::get(std::string_view name) const {
  const TensorInfo *ti = info(name);
  return ti && ti->t.buf ? &ti->t : nullptr;
}
const Tensor &Pack::at(std::string_view name) const {
  const TensorInfo *ti = info(name);
  NE_CHECK(ti, "pack: no tensor %.*s", (int)name.size(), name.data());
  NE_CHECK(ti->t.buf, "pack: %.*s (%llu bytes) has no single GPU view; use view_rows()", (int)name.size(), name.data(),
           (unsigned long long)ti->nbytes);
  return ti->t;
}
const Json &Pack::text_config() const {
  const Json *tc = config_.find("text_config");
  return tc && tc->is_object() ? *tc : config_;
}
Tensor Pack::view_rows(std::string_view name, int64_t row0, int64_t nrows) const {
  const TensorInfo *ti = info(name);
  NE_CHECK(ti, "pack: no tensor %.*s", (int)name.size(), name.data());
  const Tensor &t = ti->t;
  NE_CHECK(t.ndim >= 1 && row0 >= 0 && nrows > 0 && row0 + nrows <= t.shape[0], "view_rows(%lld, %lld) out of range "
           "for %s", (long long)row0, (long long)nrows, t.desc().c_str());
  Tensor v = t;
  uint64_t rb = t.row_bytes();
  uint64_t b = ti->file_offset + (uint64_t)row0 * rb;
  v.buf = window(ti->map, b, b + (uint64_t)nrows * rb, &v.offset);
  v.shape[0] = nrows;
  return v;
}
uint64_t Pack::mapped_bytes() const {
  uint64_t n = 0;
  for (auto &m : maps_) n += m->mapped;
  return n;
}

}  // namespace ne
