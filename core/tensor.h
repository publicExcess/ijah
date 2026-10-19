// DType, Tensor views, shared-storage allocation, upload/download. Owner: lab-core.
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ne {

// Same numbering as ne_dtype in include/engine.h.
enum class DType : int { F32 = 0, F16, BF16, U8, U16, U32, I32, I64, I8, I16, U64, F64, BOOL, Invalid = -1 };

size_t dtype_size(DType t);                 // 0 for Invalid
const char *dtype_name(DType t);            // safetensors spelling: "F32", "BF16", ...
DType dtype_from_name(std::string_view s);  // Invalid if unknown

constexpr int kMaxDims = 8;

// A contiguous row-major view: `buf` + byte `offset`. Copyable (ARC retains the buffer).
struct Tensor {
  id<MTLBuffer> buf = nil;
  size_t offset = 0;
  DType dtype = DType::Invalid;
  int ndim = 0;
  int64_t shape[kMaxDims] = {};

  int64_t numel() const;
  size_t nbytes() const { return (size_t)numel() * dtype_size(dtype); }
  int64_t dim(int i) const { return shape[i < 0 ? i + ndim : i]; }
  size_t row_bytes() const;  // bytes of one index along dim 0
  bool valid() const { return buf != nil && dtype != DType::Invalid; }
  uint64_t gpu_address() const { return buf.gpuAddress + offset; }
  void *host() const { return (char *)buf.contents + offset; }  // shared storage: CPU view of the same bytes
  std::string desc() const;  // "BF16[512,2560]"

  // Sub-views (no copy). rows(): dim-0 slice. view(): reinterpret bytes at a relative offset.
  Tensor rows(int64_t row0, int64_t nrows) const;
  Tensor view(size_t byte_offset, DType dt, std::initializer_list<int64_t> shape) const;
};

Tensor make_tensor(id<MTLBuffer> buf, size_t offset, DType dt, std::initializer_list<int64_t> shape);
// Shared-storage allocation (zero-filled), at least 16 bytes, rounded up to 16.
Tensor alloc(DType dt, std::initializer_list<int64_t> shape);
Tensor alloc(DType dt, const int64_t *shape, int ndim);
id<MTLBuffer> alloc_bytes(size_t nbytes);
// Unified memory: these are memcpys. Call them only when no GPU work on the tensor is in flight.
void upload(const Tensor &t, const void *src, size_t nbytes);    // nbytes <= t.nbytes()
void download(const Tensor &t, void *dst, size_t nbytes);

}  // namespace ne
