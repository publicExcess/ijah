// Pack loader: a directory of safetensors shards (symlinks allowed) + config.json (+ ngram_table.bin, which is a
// safetensors-format file too). Every shard is mmapped once per process (read-only; packs that share a file -- same
// inode -- share its mapping and its tensors' buffers); every tensor is a no-copy MTLBuffer window over that mapping. Owner: lab-core.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.h"
#include "core/tensor.h"

namespace ne {

struct Mapping;  // one mmapped file (shared by the buffers that view it; unmapped when the last one dies)
struct FileEntry;  // a mapped file + its tensors' buffers, shared by every open pack listing that file (by inode)

struct TensorInfo {
  std::string name;
  Tensor t;                  // GPU view; t.buf == nil when nbytes > max buffer length (use Pack::view_rows) or 0
  std::string file;          // realpath of the shard
  uint64_t file_offset = 0;  // absolute offset of the data in the file
  uint64_t nbytes = 0;
  const void *host = nullptr;  // CPU pointer into the mapping (read-only pages)
  std::shared_ptr<Mapping> map;
};

// enable_shared_from_this: a pack opened through the C API (ne_pack_open) is shared-owned, so objects that keep a
// pointer to it (models/flash_next Model: lazy MTP head load, n-gram preads) hold it alive past ne_pack_close.
class Pack : public std::enable_shared_from_this<Pack> {
 public:
  // Opens `dir`. Files: the weight_map values of model.safetensors.index.json if present (else every *.safetensors),
  // plus config["ngram_table"]["file"] (default ngram_table.bin) if it exists; tensors of non-.safetensors files are
  // named "<file stem>.<name>", e.g. "ngram_table.weight". Throws ne::Error.
  static std::unique_ptr<Pack> open(const std::string &dir);
  ~Pack();

  const Tensor *get(std::string_view name) const;     // nullptr if missing or no single GPU view
  const Tensor &at(std::string_view name) const;      // throws if missing or no single GPU view
  const TensorInfo *info(std::string_view name) const;  // nullptr if missing
  bool has(std::string_view name) const { return info(name) != nullptr; }
  const std::vector<std::string> &names() const { return names_; }  // sorted

  const Json &config() const { return config_; }        // whole config.json (Null if absent)
  const Json &text_config() const;                      // config["text_config"] if present, else config()
  const std::string &config_text() const { return config_text_; }
  const std::string &dir() const { return dir_; }

  // The tensor with its GPU address a multiple of `align`: the no-copy view when it already is, else a copy into a
  // fresh shared buffer (made once, cached in the pack; costs RAM = the tensor's size). Thread-safe. Prefer an aligned
  // pack (engine-cli align-pack) for big tensors: the K5 pack has 56 GB of tensors that are not 16-byte aligned.
  const Tensor &aligned(std::string_view name, size_t align = 16) const;
  uint64_t aligned_copy_bytes() const;  // RAM used by aligned() copies so far

  // A fresh no-copy view of rows [row0, row0+nrows) along dim 0 (page-aligned window, own MTLBuffer).
  // Works for any tensor, including ones larger than max_buffer_length (the n-gram table).
  Tensor view_rows(std::string_view name, int64_t row0, int64_t nrows) const;

  size_t num_files() const { return maps_.size(); }
  uint64_t mapped_bytes() const;

 private:
  Pack() = default;
  std::string dir_;
  Json config_;
  std::string config_text_ = "{}";
  std::map<std::string, TensorInfo, std::less<>> tensors_;
  std::vector<std::string> names_;
  std::vector<std::shared_ptr<Mapping>> maps_;
  std::vector<std::shared_ptr<FileEntry>> files_;
  struct AlignedCache;
  std::unique_ptr<AlignedCache> aligned_;
};

// Safetensors header of one file (exposed for tools/tests).
struct SafetensorsEntry {
  std::string name;
  DType dtype;
  std::vector<int64_t> shape;
  uint64_t begin, end;  // absolute file offsets
};
std::vector<SafetensorsEntry> read_safetensors_header(const std::string &path, uint64_t file_size, Json *metadata);

}  // namespace ne
