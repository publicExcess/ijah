// align-pack: copy a pack so every tensor's data is `align`-byte aligned in its file. Owner: lab-core.
#pragma once
#include <cstdint>
#include <string>

namespace ne {

struct AlignOptions {
  uint64_t align = 64;
  bool dry_run = false;
  bool verify = true;   // re-read every rewritten file and memcmp each tensor against the source
  bool verbose = true;
  std::string only;     // rewrite only misaligned files whose name contains this (others are symlinked as they are)
};
struct AlignStats {
  uint64_t misaligned_tensors = 0, misaligned_bytes = 0;
  uint64_t files_rewritten = 0, files_linked = 0, bytes_to_write = 0, bytes_written = 0;
};
// dst is created if needed; re-running resumes (aligned rewritten files and correct symlinks are kept).
AlignStats align_pack(const std::string &src, const std::string &dst, const AlignOptions &opt);

}  // namespace ne
