// align-pack: write a copy of a pack in which every tensor's data starts at a multiple of `align` bytes in its file
// (so no-copy GPU views are vector-load aligned). Files that are already aligned (and every non-safetensors file,
// including the n-gram table) become symlinks to the original's real path; misaligned shards are rewritten with a
// space-padded header and padding between tensors. Owner: lab-core.
#import <Foundation/Foundation.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>

#include "core/engine.hpp"
#include "core/pack_align.h"

namespace ne {

namespace {
std::string realp(const std::string &p) {
  char b[PATH_MAX];
  NE_CHECK(realpath(p.c_str(), b), "realpath %s: %s", p.c_str(), strerror(errno));
  return b;
}
bool ends_with(const std::string &s, const char *suf) {
  size_t n = strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}
uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

void read_all(int fd, void *p, size_t n, uint64_t off) {
  char *c = (char *)p;
  while (n) {
    ssize_t k = pread(fd, c, n, (off_t)off);
    NE_CHECK(k > 0, "pread: %s", k < 0 ? strerror(errno) : "short file");
    c += k;
    n -= (size_t)k;
    off += (uint64_t)k;
  }
}

uint32_t crc32c(uint32_t c, const char *p, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t v;
    memcpy(&v, p + i, 8);
    c = __builtin_arm_crc32cd(c, v);
  }
  for (; i < n; i++) c = __builtin_arm_crc32cb(c, (uint8_t)p[i]);
  return c;
}

void write_all(int fd, const void *p, size_t n, uint64_t off) {
  const char *c = (const char *)p;
  while (n) {
    ssize_t k = pwrite(fd, c, std::min<size_t>(n, 64u << 20), (off_t)off);
    NE_CHECK(k > 0, "pwrite: %s", strerror(errno));
    c += k;
    n -= (size_t)k;
    off += (uint64_t)k;
  }
}

// Rewrites one shard. Returns bytes written.
uint64_t rewrite(const std::string &src, const std::string &dst, uint64_t align, bool verify) {
  struct stat st;
  NE_CHECK(stat(src.c_str(), &st) == 0, "stat %s", src.c_str());
  Json meta;
  auto entries = read_safetensors_header(src, (uint64_t)st.st_size, &meta);
  std::sort(entries.begin(), entries.end(), [](auto &a, auto &b) { return a.begin < b.begin; });
  // Header: original metadata (+ a note) and the entries with new relative offsets.
  std::vector<uint64_t> rel(entries.size());
  uint64_t cur = 0;
  for (size_t i = 0; i < entries.size(); i++) {
    cur = round_up(cur, align);
    rel[i] = cur;
    cur += entries[i].end - entries[i].begin;
  }
  const uint64_t data_bytes = cur;
  std::string h = "{";
  std::string m = "{";
  if (meta.is_object())
    for (auto &kv : meta.obj())
      if (kv.first != "aligned_by") m += json_quote(kv.first) + ":" + kv.second.dump() + ",";
  m += strfmt("\"aligned_by\":\"ijah align-pack, align=%llu\"}", (unsigned long long)align);
  h += "\"__metadata__\":" + m;
  for (size_t i = 0; i < entries.size(); i++) {
    const auto &e = entries[i];
    std::string shp = "[";
    for (size_t k = 0; k < e.shape.size(); k++) shp += (k ? "," : "") + std::to_string(e.shape[k]);
    shp += "]";
    h += strfmt(",%s:{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%llu,%llu]}", json_quote(e.name).c_str(),
                dtype_name(e.dtype), shp.c_str(), (unsigned long long)rel[i],
                (unsigned long long)(rel[i] + e.end - e.begin));
  }
  h += "}";
  // Pad with spaces (allowed by the format) so that 8 + len(header) is a multiple of align.
  h.append(round_up(8 + h.size(), align) - 8 - h.size(), ' ');
  const uint64_t data0 = 8 + h.size(), total = data0 + data_bytes;

  // Copy with pread/pwrite under F_NOCACHE (tens of GB must not flush everyone's page cache); CRC32C of every
  // tensor's source bytes, re-read from the written file (also uncached) and compared when verify is on.
  int sfd = ::open(src.c_str(), O_RDONLY);
  NE_CHECK(sfd >= 0, "open %s", src.c_str());
  fcntl(sfd, F_NOCACHE, 1);
  const std::string part = dst + ".part";
  int fd = ::open(part.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  NE_CHECK(fd >= 0, "create %s: %s", part.c_str(), strerror(errno));
  fcntl(fd, F_NOCACHE, 1);
  NE_CHECK(ftruncate(fd, (off_t)total) == 0, "ftruncate %s", part.c_str());
  uint64_t hl = h.size();
  write_all(fd, &hl, 8, 0);
  write_all(fd, h.data(), h.size(), 8);
  const size_t CH = 32u << 20;
  std::vector<char> buf(CH);
  std::vector<uint32_t> crc(entries.size());
  for (size_t i = 0; i < entries.size(); i++) {
    uint32_t c = 0xffffffffu;
    for (uint64_t o = 0, n = entries[i].end - entries[i].begin; o < n; o += CH) {
      size_t k = (size_t)std::min<uint64_t>(CH, n - o);
      read_all(sfd, buf.data(), k, entries[i].begin + o);
      c = crc32c(c, buf.data(), k);
      write_all(fd, buf.data(), k, data0 + rel[i] + o);
    }
    crc[i] = c;
  }
  NE_CHECK(fsync(fd) == 0, "fsync %s", part.c_str());
  ::close(sfd);
  if (verify) {
    auto ne2 = read_safetensors_header(part, total, nullptr);
    NE_CHECK(ne2.size() == entries.size(), "verify %s: entry count", part.c_str());
    std::sort(ne2.begin(), ne2.end(), [](auto &a, auto &b) { return a.begin < b.begin; });
    for (size_t i = 0; i < entries.size(); i++) {
      NE_CHECK(ne2[i].name == entries[i].name && ne2[i].dtype == entries[i].dtype && ne2[i].shape == entries[i].shape &&
                   ne2[i].begin % align == 0 && ne2[i].end - ne2[i].begin == entries[i].end - entries[i].begin,
               "verify %s: %s header", part.c_str(), entries[i].name.c_str());
      uint32_t c = 0xffffffffu;
      for (uint64_t o = 0, n = ne2[i].end - ne2[i].begin; o < n; o += CH) {
        size_t k = (size_t)std::min<uint64_t>(CH, n - o);
        read_all(fd, buf.data(), k, ne2[i].begin + o);
        c = crc32c(c, buf.data(), k);
      }
      NE_CHECK(c == crc[i], "verify %s: %s bytes differ (crc32c)", part.c_str(), entries[i].name.c_str());
    }
  }
  ::close(fd);
  NE_CHECK(rename(part.c_str(), dst.c_str()) == 0, "rename %s", part.c_str());
  return total;
}
}  // namespace

AlignStats align_pack(const std::string &src_in, const std::string &dst, const AlignOptions &opt) {
  NE_CHECK(opt.align >= 1 && (opt.align & (opt.align - 1)) == 0, "align must be a power of two");
  const std::string src = realp(src_in);
  AlignStats s;
  if (!opt.dry_run) {
    struct stat st;
    if (stat(dst.c_str(), &st) != 0) NE_CHECK(mkdir(dst.c_str(), 0755) == 0, "mkdir %s: %s", dst.c_str(), strerror(errno));
    NE_CHECK(realp(dst) != src, "align-pack: destination is the source");
  }
  DIR *dd = opendir(src.c_str());
  NE_CHECK(dd, "opendir %s", src.c_str());
  std::vector<std::string> names;
  while (dirent *e = readdir(dd))
    if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) names.push_back(e->d_name);
  closedir(dd);
  std::sort(names.begin(), names.end());
  for (const std::string &n : names) {
    const std::string sp = src + "/" + n, dp = dst + "/" + n;
    const std::string target = realp(sp);
    bool need = false;
    uint64_t fsize = 0;
    if (ends_with(n, ".safetensors")) {
      struct stat st;
      stat(target.c_str(), &st);
      fsize = (uint64_t)st.st_size;
      for (auto &e : read_safetensors_header(target, fsize, nullptr)) {
        if (e.begin % opt.align) {
          need = true;
          s.misaligned_tensors++;
          s.misaligned_bytes += e.end - e.begin;
        }
      }
      if (need && !opt.only.empty() && n.find(opt.only) == std::string::npos) need = false;
    }
    if (opt.verbose && need) printf("  rewrite %s (%.2f GB)\n", n.c_str(), fsize / 1e9);
    if (need) {
      s.files_rewritten++;
      s.bytes_to_write += fsize;
    } else {
      s.files_linked++;
    }
    if (opt.dry_run) continue;
    struct stat lst;
    if (lstat(dp.c_str(), &lst) == 0) {
      if (!need && S_ISLNK(lst.st_mode) && realp(dp) == target) continue;  // already linked
      if (need && S_ISREG(lst.st_mode)) {                                    // already rewritten (resume)
        bool ok = true;
        for (auto &e : read_safetensors_header(dp, (uint64_t)lst.st_size, nullptr)) ok = ok && e.begin % opt.align == 0;
        if (ok) continue;
      }
      NE_CHECK(unlink(dp.c_str()) == 0, "unlink %s", dp.c_str());
    }
    if (need) s.bytes_written += rewrite(target, dp, opt.align, opt.verify);
    else NE_CHECK(symlink(target.c_str(), dp.c_str()) == 0, "symlink %s: %s", dp.c_str(), strerror(errno));
  }
  return s;
}

}  // namespace ne
