// Recorder: record a sequence of compute dispatches + barriers once, replay it many times. Owner: lab-core.
//
//   ne::Recorder r;                         // backend: $NE_BACKEND or auto (mtl4 > icb > classic)
//   r.dispatch(pso, {ne::Arg(x), ne::Arg(w), ne::Arg::value(n)}, MTLSizeMake(tgs,1,1), MTLSizeMake(256,1,1));
//   r.barrier();                            // the next dispatch waits for every earlier one
//   r.dispatch(...);
//   r.end();                                // finalise (builds the ICB / residency set / constant arena)
//   r.run();  r.gpu_ms();                   // commit + wait; replay as often as you like
//
// Arguments bind at buffer index = position in the list. Inline bytes (Arg::bytes / Arg::value) are copied into a
// recorder-owned constant arena at record time; patch them between replays through bytes_at(cmd, slot).
// Without barrier(), consecutive dispatches may overlap (all backends except Serial).
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace ne {

enum class Backend { Auto, MTL4, ICB, Classic, Serial };
const char *backend_name(Backend b);
Backend backend_from_name(const std::string &s);  // "mtl4" "icb" "classic" "serial" "auto"/""; throws on others
Backend resolve_backend(Backend b);               // Auto -> $NE_BACKEND or the best available

struct Arg {
  id<MTLBuffer> buf = nil;
  size_t offset = 0;
  std::vector<uint8_t> data;  // inline bytes when buf == nil
  Arg(const Tensor &t) : buf(t.buf), offset(t.offset) {}
  Arg(id<MTLBuffer> b, size_t off = 0) : buf(b), offset(off) {}
  static Arg bytes(const void *p, size_t n) {
    Arg a(nil);
    a.data.assign((const uint8_t *)p, (const uint8_t *)p + n);
    return a;
  }
  template <class T> static Arg value(const T &v) { return bytes(&v, sizeof(T)); }
};

class Recorder {
 public:
  explicit Recorder(Backend b = Backend::Auto);
  ~Recorder();
  Recorder(const Recorder &) = delete;
  Recorder &operator=(const Recorder &) = delete;

  void begin();  // clear the recording (keeps the backend)
  // grid = threadgroups per grid, tg = threads per threadgroup, tg_mem = threadgroup memory bytes at index 0.
  // Returns the command index.
  size_t dispatch(id<MTLComputePipelineState> pso, const std::vector<Arg> &args, MTLSize grid, MTLSize tg,
                  uint32_t tg_mem = 0);
  // Non-uniform grid: `threads` = total threads per grid (edge threadgroups are partial; Apple GPUs support this).
  size_t dispatch_threads(id<MTLComputePipelineState> pso, const std::vector<Arg> &args, MTLSize threads, MTLSize tg,
                          uint32_t tg_mem = 0);
  // Threadgroup count read by the GPU from `ind` (+offset): 3 x uint32 (MTLDispatchThreadgroupsIndirectArguments).
  // Not supported by the ICB backend (throws at end()).
  size_t dispatch_indirect(id<MTLComputePipelineState> pso, const std::vector<Arg> &args, id<MTLBuffer> ind,
                           size_t ind_offset, MTLSize tg, uint32_t tg_mem = 0);
  void barrier();
  // Make a buffer resident for every replay although no dispatch binds it directly: buffers that kernels reach
  // through GPU addresses stored in other buffers (pointer tables, argument buffers). Bound buffers need no use().
  void use(id<MTLBuffer> b);
  void use(const Tensor &t) { use(t.buf); }
  void end();

  // Replays: up to kMaxInFlight may be committed before waiting (they run in commit order; each waits for all
  // earlier work on the queue). Patching bytes_at() while a replay is in flight races with it.
  static constexpr int kMaxInFlight = 3;
  void run();          // commit() + wait()
  void commit();       // submit one replay (after end())
  void wait();         // wait for every replay in flight; throws on GPU error
  void wait_oldest();  // wait for the oldest replay in flight only
  size_t in_flight() const;
  double gpu_ms() const { return gpu_ms_; }        // last waited replay: GPUEndTime - GPUStartTime (command buffer)
  double encode_us() const { return encode_us_; }  // last commit: CPU time to encode + commit
  double wall_us() const { return wall_us_; }      // last waited replay: its commit() start -> wait return

  // lab-pfpath: requestResidency on this recording's residency set (MTL4, after end()): the engine's idle keep-warm.
  // After ~1 s without GPU work the next command buffer of a recording waits ~0.1 s for its residency (and the first
  // of any ~0.6 s), docs/prefill.md "server path"
  void touch();
  uint8_t *bytes_at(size_t cmd, int slot);  // inline-bytes argument `slot` of command `cmd` (valid after end())
  Backend backend() const { return backend_; }
  size_t size() const;
  bool ended() const { return ended_; }

  struct Impl;

 private:
  Backend backend_;
  std::unique_ptr<Impl> p_;
  bool ended_ = false;
  double gpu_ms_ = 0, encode_us_ = 0, wall_us_ = 0;
};

}  // namespace ne
