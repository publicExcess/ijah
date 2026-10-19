// Umbrella header for the C++ core (Objective-C++). Include this from kernels/ and models/ .mm files. Owner: lab-core.
#pragma once
#include "include/engine.h"

#include "core/device.h"
#include "core/error.h"
#include "core/json.h"
#include "core/pack.h"
#include "core/pipelines.h"
#include "core/recorder.h"
#include "core/tensor.h"

// The C handle behind ne_buf*. Pack-owned handles have owned == false (ne_free ignores them); views of pack files are
// readonly (ne_upload refuses them: the mapping is PROT_READ).
struct ne_buf {
  id<MTLBuffer> buf = nil;
  bool owned = true;
  bool readonly = false;
};

namespace ne {

// --- bridging helpers for C entry points written in other directories ---
// Wrap a buffer as a new caller-owned C handle (the caller ne_free()s it).
ne_buf *to_c(id<MTLBuffer> b);
// The MTLBuffer behind a handle (nil for NULL).
inline id<MTLBuffer> from_c(const ne_buf *b) { return b ? b->buf : nil; }
// A Tensor over a C handle (+ byte offset); shape/dtype as given.
Tensor tensor_from_c(const ne_buf *b, uint64_t offset, DType dt, std::initializer_list<int64_t> shape);
class Pack;
// The C++ Pack behind an ne_pack* handle.
const Pack *pack_from_c(const ne_pack *p);

// Run `fn` and translate exceptions into the C convention: returns 0, or -1 with ne_last_error() set.
void set_last_error(const std::string &msg);
template <class F> int c_call(F &&fn) {
  try {
    fn();
    return 0;
  } catch (const std::exception &e) {
    set_last_error(e.what());
    return -1;
  }
}

}  // namespace ne
