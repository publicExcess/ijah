// Device singleton, dtype table, Tensor helpers, allocation. Owner: lab-core.
#import <Foundation/Foundation.h>

#include <mach/mach_time.h>
#include <unistd.h>

#include "core/engine.hpp"

namespace ne {

double now_us() {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e3;
}

Device::Device() {
  @autoreleasepool {
    dev = MTLCreateSystemDefaultDevice();
    NE_CHECK(dev != nil, "no Metal device");
    name = dev.name.UTF8String;
    queue = [dev newCommandQueueWithMaxCommandBufferCount:64];
    NE_CHECK(queue != nil, "newCommandQueue failed");
    max_buffer_length = dev.maxBufferLength;
    page_size = (uint64_t)getpagesize();
    has_nax = [dev supportsFamily:MTLGPUFamilyApple10];
    if (@available(macOS 26.0, *)) {
      if ([dev supportsFamily:MTLGPUFamilyMetal4]) {
        queue4 = [dev newMTL4CommandQueue];
        has_metal4 = queue4 != nil;
      }
    }
  }
}

Device &Device::get() {
  static Device *d = new Device();  // never destroyed: buffers may outlive static destructors
  return *d;
}

// ---- dtypes ----
static const struct { DType t; const char *name; size_t size; } kDTypes[] = {
    {DType::F32, "F32", 4}, {DType::F16, "F16", 2}, {DType::BF16, "BF16", 2}, {DType::U8, "U8", 1},
    {DType::U16, "U16", 2}, {DType::U32, "U32", 4}, {DType::I32, "I32", 4}, {DType::I64, "I64", 8},
    {DType::I8, "I8", 1},   {DType::I16, "I16", 2}, {DType::U64, "U64", 8}, {DType::F64, "F64", 8},
    {DType::BOOL, "BOOL", 1},
};
size_t dtype_size(DType t) {
  for (auto &d : kDTypes) if (d.t == t) return d.size;
  return 0;
}
const char *dtype_name(DType t) {
  for (auto &d : kDTypes) if (d.t == t) return d.name;
  return "?";
}
DType dtype_from_name(std::string_view s) {
  for (auto &d : kDTypes) if (s == d.name) return d.t;
  return DType::Invalid;
}

// ---- Tensor ----
int64_t Tensor::numel() const {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) n *= shape[i];
  return n;
}
size_t Tensor::row_bytes() const {
  if (ndim == 0) return dtype_size(dtype);
  int64_t n = 1;
  for (int i = 1; i < ndim; i++) n *= shape[i];
  return (size_t)n * dtype_size(dtype);
}
std::string Tensor::desc() const {
  std::string s = dtype_name(dtype);
  s += "[";
  for (int i = 0; i < ndim; i++) s += (i ? "," : "") + std::to_string(shape[i]);
  return s + "]";
}
Tensor Tensor::rows(int64_t row0, int64_t nrows) const {
  NE_CHECK(ndim >= 1 && row0 >= 0 && nrows >= 0 && row0 + nrows <= shape[0], "rows(%lld, %lld) out of range for %s",
           (long long)row0, (long long)nrows, desc().c_str());
  Tensor t = *this;
  t.offset += (size_t)row0 * row_bytes();
  t.shape[0] = nrows;
  return t;
}
Tensor Tensor::view(size_t byte_offset, DType dt, std::initializer_list<int64_t> shp) const {
  Tensor t = make_tensor(buf, offset + byte_offset, dt, shp);
  NE_CHECK(byte_offset + t.nbytes() <= nbytes() || (buf && t.offset + t.nbytes() <= buf.length),
           "view out of range");
  return t;
}

Tensor make_tensor(id<MTLBuffer> buf, size_t offset, DType dt, std::initializer_list<int64_t> shp) {
  NE_CHECK(shp.size() <= (size_t)kMaxDims, "too many dims");
  Tensor t;
  t.buf = buf;
  t.offset = offset;
  t.dtype = dt;
  t.ndim = (int)shp.size();
  int i = 0;
  for (int64_t s : shp) t.shape[i++] = s;
  return t;
}

id<MTLBuffer> alloc_bytes(size_t nbytes) {
  size_t n = (std::max<size_t>(nbytes, 16) + 15) & ~size_t(15);
  id<MTLBuffer> b = [Device::get().dev newBufferWithLength:n options:MTLResourceStorageModeShared];
  NE_CHECK(b != nil, "alloc of %zu bytes failed", n);
  memset(b.contents, 0, n);  // newBufferWithLength zero-fills too; be explicit
  return b;
}
Tensor alloc(DType dt, const int64_t *shape, int ndim) {
  NE_CHECK(ndim >= 0 && ndim <= kMaxDims, "bad ndim %d", ndim);
  NE_CHECK(dtype_size(dt) > 0, "bad dtype");
  Tensor t;
  t.dtype = dt;
  t.ndim = ndim;
  for (int i = 0; i < ndim; i++) t.shape[i] = shape[i];
  t.buf = alloc_bytes(t.nbytes());
  return t;
}
Tensor alloc(DType dt, std::initializer_list<int64_t> shape) {
  return alloc(dt, shape.begin(), (int)shape.size());
}
void upload(const Tensor &t, const void *src, size_t nbytes) {
  NE_CHECK(t.buf && t.offset <= t.buf.length && nbytes <= t.buf.length - t.offset && nbytes <= t.nbytes(),
           "upload out of range (%zu bytes into a %lld-byte tensor)", nbytes, (long long)t.nbytes());
  memcpy(t.host(), src, nbytes);
}
void download(const Tensor &t, void *dst, size_t nbytes) {
  NE_CHECK(t.buf && t.offset <= t.buf.length && nbytes <= t.buf.length - t.offset && nbytes <= t.nbytes(),
           "download out of range (%zu bytes into a %lld-byte tensor)", nbytes, (long long)t.nbytes());
  memcpy(dst, t.host(), nbytes);
}

}  // namespace ne
