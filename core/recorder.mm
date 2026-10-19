// Recorder backends: Metal 4 encoder (MTL4), indirect command buffer (ICB), classic concurrent encoder with memory
// barriers (Classic), classic serial encoder (Serial). Encodings follow lab lead 4's measurements
// (the lab tree's engine/fused/disp.mm). Owner: lab-core.
#import <Foundation/Foundation.h>

#include <algorithm>
#include <set>

#include "core/engine.hpp"

namespace ne {

const char *backend_name(Backend b) {
  switch (b) {
    case Backend::Auto: return "auto";
    case Backend::MTL4: return "mtl4";
    case Backend::ICB: return "icb";
    case Backend::Classic: return "classic";
    case Backend::Serial: return "serial";
  }
  return "?";
}
Backend backend_from_name(const std::string &s) {
  if (s.empty() || s == "auto") return Backend::Auto;
  if (s == "mtl4") return Backend::MTL4;
  if (s == "icb") return Backend::ICB;
  if (s == "classic") return Backend::Classic;
  if (s == "serial") return Backend::Serial;
  throw Error("unknown backend '" + s + "' (mtl4 | icb | classic | serial | auto)");
}
Backend resolve_backend(Backend b) {
  if (b == Backend::Auto) {
    const char *e = getenv("NE_BACKEND");
    b = backend_from_name(e ? e : "");
  }
  if (b == Backend::Auto) b = Device::get().has_metal4 ? Backend::MTL4 : Backend::ICB;
  if (b == Backend::MTL4 && !Device::get().has_metal4) throw Error("backend mtl4 requested but Metal 4 is unavailable");
  return b;
}

namespace {
struct Slot {
  id<MTLBuffer> buf = nil;
  size_t offset = 0;
  int64_t arena_off = -1;  // inline bytes: offset into the arena
  uint32_t nbytes = 0;
};
struct Cmd {
  id<MTLComputePipelineState> pso = nil;
  std::vector<Slot> slots;
  MTLSize grid{1, 1, 1}, tg{1, 1, 1};
  uint32_t tg_mem = 0;
  bool barrier_before = false;
  id<MTLBuffer> ind = nil;
  size_t ind_off = 0;
  bool threads = false;  // grid counts threads (dispatchThreads), not threadgroups
};
constexpr size_t kArenaAlign = 16;
}  // namespace

struct Recorder::Impl {
  std::vector<Cmd> cmds;
  std::vector<uint8_t> staging;
  id<MTLBuffer> arena = nil;
  bool pending_barrier = false;
  size_t max_slots = 0;
  std::vector<id<MTLBuffer>> resources;
  std::vector<id<MTLBuffer>> extra;  // use()d: reached through GPU addresses only
  id<MTLIndirectCommandBuffer> icb = nil;
  id<MTL4ArgumentTable> table4 = nil;
  id<MTLResidencySet> rs4 = nil;
  // Replays in flight: a ring of up to kMaxInFlight, oldest at `head`.
  struct Flight {
    id<MTLCommandBuffer> cb = nil;          // classic / ICB / serial
    id<MTL4CommandAllocator> alloc4 = nil;  // MTL4: one allocator per slot (reset only when its replay is done)
    dispatch_semaphore_t sem = nil;
    double g0 = 0, g1 = 0;
    NSError *err = nil;
    double t_commit = 0;
  };
  Flight fl[Recorder::kMaxInFlight];
  int head = 0, count = 0;
};

Recorder::Recorder(Backend b) : backend_(resolve_backend(b)), p_(new Impl()) {}
Recorder::~Recorder() {
  while (p_ && p_->count) {
    try { wait_oldest(); } catch (...) {}
  }
}
size_t Recorder::in_flight() const { return (size_t)p_->count; }

void Recorder::begin() {
  NE_CHECK(p_->count == 0, "Recorder::begin while a replay is in flight");
  Backend b = backend_;
  p_.reset(new Impl());
  backend_ = b;
  ended_ = false;
}

size_t Recorder::size() const { return p_->cmds.size(); }

static Cmd make_cmd(Recorder::Impl &p, id<MTLComputePipelineState> pso, const std::vector<Arg> &args, uint32_t tg_mem) {
  NE_CHECK(pso != nil, "dispatch: null pipeline");
  NE_CHECK(args.size() <= 31, "dispatch: %zu arguments (max 31)", args.size());
  Cmd c;
  c.pso = pso;
  c.tg_mem = tg_mem;
  c.barrier_before = p.pending_barrier;
  p.pending_barrier = false;
  for (const Arg &a : args) {
    Slot s;
    if (a.buf) {
      s.buf = a.buf;
      s.offset = a.offset;
    } else {
      NE_CHECK(!a.data.empty(), "dispatch: argument with neither buffer nor bytes");
      size_t off = (p.staging.size() + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
      p.staging.resize(off + a.data.size());
      memcpy(p.staging.data() + off, a.data.data(), a.data.size());
      s.arena_off = (int64_t)off;
      s.nbytes = (uint32_t)a.data.size();
    }
    c.slots.push_back(s);
  }
  p.max_slots = std::max(p.max_slots, c.slots.size());
  return c;
}

size_t Recorder::dispatch(id<MTLComputePipelineState> pso, const std::vector<Arg> &args, MTLSize grid, MTLSize tg,
                          uint32_t tg_mem) {
  NE_CHECK(!ended_, "Recorder::dispatch after end(); call begin() to re-record");
  Cmd c = make_cmd(*p_, pso, args, tg_mem);
  c.grid = grid;
  c.tg = tg;
  NE_CHECK(tg.width * tg.height * tg.depth <= pso.maxTotalThreadsPerThreadgroup, "dispatch %s: %lu threads per "
           "threadgroup > pipeline max %lu", pso.label.UTF8String, (unsigned long)(tg.width * tg.height * tg.depth),
           (unsigned long)pso.maxTotalThreadsPerThreadgroup);
  p_->cmds.push_back(std::move(c));
  return p_->cmds.size() - 1;
}

size_t Recorder::dispatch_threads(id<MTLComputePipelineState> pso, const std::vector<Arg> &args, MTLSize threads,
                                  MTLSize tg, uint32_t tg_mem) {
  size_t i = dispatch(pso, args, threads, tg, tg_mem);
  p_->cmds[i].threads = true;
  return i;
}

size_t Recorder::dispatch_indirect(id<MTLComputePipelineState> pso, const std::vector<Arg> &args, id<MTLBuffer> ind,
                                   size_t ind_offset, MTLSize tg, uint32_t tg_mem) {
  NE_CHECK(!ended_, "Recorder::dispatch_indirect after end()");
  NE_CHECK(ind != nil && ind_offset % 4 == 0, "dispatch_indirect: bad indirect buffer");
  Cmd c = make_cmd(*p_, pso, args, tg_mem);
  c.tg = tg;
  c.ind = ind;
  c.ind_off = ind_offset;
  p_->cmds.push_back(std::move(c));
  return p_->cmds.size() - 1;
}

void Recorder::barrier() { p_->pending_barrier = true; }

void Recorder::use(id<MTLBuffer> b) {
  NE_CHECK(!ended_, "Recorder::use after end()");
  NE_CHECK(b != nil, "Recorder::use(nil)");
  p_->extra.push_back(b);
}

uint8_t *Recorder::bytes_at(size_t cmd, int slot) {
  NE_CHECK(ended_, "bytes_at before end()");
  NE_CHECK(cmd < p_->cmds.size() && slot >= 0 && (size_t)slot < p_->cmds[cmd].slots.size(), "bytes_at out of range");
  const Slot &s = p_->cmds[cmd].slots[slot];
  NE_CHECK(s.arena_off >= 0, "bytes_at: command %zu slot %d is a buffer, not inline bytes", cmd, slot);
  return (uint8_t *)p_->arena.contents + s.arena_off;
}

void Recorder::end() {
  NE_CHECK(!ended_, "Recorder::end twice");
  Impl &p = *p_;
  Device &d = Device::get();
  @autoreleasepool {
    // Constant arena.
    if (!p.staging.empty()) {
      p.arena = alloc_bytes(p.staging.size());
      memcpy(p.arena.contents, p.staging.data(), p.staging.size());
      for (Cmd &c : p.cmds)
        for (Slot &s : c.slots)
          if (s.arena_off >= 0) {
            s.buf = p.arena;
            s.offset = (size_t)s.arena_off;
          }
    }
    // Unique resources.
    std::set<void *> seen;
    for (Cmd &c : p.cmds) {
      for (Slot &s : c.slots)
        if (seen.insert((__bridge void *)s.buf).second) p.resources.push_back(s.buf);
      if (c.ind && seen.insert((__bridge void *)c.ind).second) p.resources.push_back(c.ind);
    }
    std::vector<id<MTLBuffer>> extra;
    for (id<MTLBuffer> b : p.extra)
      if (seen.insert((__bridge void *)b).second) {
        p.resources.push_back(b);
        extra.push_back(b);
      }
    p.extra = extra;  // classic/serial: only these need an explicit useResources

    if (backend_ == Backend::ICB && !p.cmds.empty()) {
      for (Cmd &c : p.cmds)
        NE_CHECK(c.ind == nil, "the ICB backend does not support dispatch_indirect (use mtl4 or classic)");
      MTLIndirectCommandBufferDescriptor *icd = [MTLIndirectCommandBufferDescriptor new];
      icd.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
      icd.inheritBuffers = NO;
      icd.inheritPipelineState = NO;
      icd.maxKernelBufferBindCount = std::max<size_t>(p.max_slots, 1);
      icd.maxKernelThreadgroupMemoryBindCount = 1;
      p.icb = [d.dev newIndirectCommandBufferWithDescriptor:icd maxCommandCount:p.cmds.size() options:0];
      NE_CHECK(p.icb != nil, "newIndirectCommandBuffer(%zu) failed", p.cmds.size());
      for (size_t i = 0; i < p.cmds.size(); i++) {
        const Cmd &c = p.cmds[i];
        id<MTLIndirectComputeCommand> ic = [p.icb indirectComputeCommandAtIndex:i];
        [ic setComputePipelineState:c.pso];
        for (size_t k = 0; k < c.slots.size(); k++) [ic setKernelBuffer:c.slots[k].buf offset:c.slots[k].offset atIndex:k];
        if (c.tg_mem) [ic setThreadgroupMemoryLength:(c.tg_mem + 15) & ~15u atIndex:0];
        if (c.barrier_before) [ic setBarrier];
        if (c.threads) [ic concurrentDispatchThreads:c.grid threadsPerThreadgroup:c.tg];
        else [ic concurrentDispatchThreadgroups:c.grid threadsPerThreadgroup:c.tg];
      }
    }
    if (backend_ == Backend::MTL4) {
      NSError *err = nil;
      for (auto &f : p.fl) {
        f.alloc4 = [d.dev newCommandAllocator];
        NE_CHECK(f.alloc4 != nil, "newCommandAllocator failed");
        f.sem = dispatch_semaphore_create(0);
      }
      MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
      atd.maxBufferBindCount = std::max<size_t>(p.max_slots, 1);
      p.table4 = [d.dev newArgumentTableWithDescriptor:atd error:&err];
      NE_CHECK(p.table4 != nil, "newArgumentTable: %s", err ? err.localizedDescription.UTF8String : "?");
      MTLResidencySetDescriptor *rsd = [MTLResidencySetDescriptor new];
      rsd.initialCapacity = std::max<size_t>(p.resources.size(), 1);
      p.rs4 = [d.dev newResidencySetWithDescriptor:rsd error:&err];
      NE_CHECK(p.rs4 != nil, "newResidencySet: %s", err ? err.localizedDescription.UTF8String : "?");
      for (id<MTLBuffer> b : p.resources) [p.rs4 addAllocation:b];
      [p.rs4 commit];
    }
  }
  ended_ = true;
}

void Recorder::commit() {
  NE_CHECK(ended_, "Recorder::commit before end()");
  Impl &p = *p_;
  NE_CHECK(p.count < kMaxInFlight, "Recorder::commit with %d replays in flight (max %d; call wait())", p.count,
           kMaxInFlight);
  Device &d = Device::get();
  double t0 = now_us();
  Impl::Flight &f = p.fl[(p.head + p.count) % kMaxInFlight];
  f.t_commit = t0;
  f.g0 = f.g1 = 0;
  f.err = nil;
  @autoreleasepool {
    if (backend_ == Backend::MTL4) {
      [f.alloc4 reset];
      id<MTL4CommandBuffer> cb = [d.dev newCommandBuffer];
      [cb beginCommandBufferWithAllocator:f.alloc4];
      [cb useResidencySet:p.rs4];
      id<MTL4ComputeCommandEncoder> e = [cb computeCommandEncoder];
      // MTL4 does not order command buffers on a queue by itself: wait for the dispatches of everything committed
      // before (other recorders' replays included), so back-to-back commits behave like the classic queue.
      [e barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch
               visibilityOptions:MTL4VisibilityOptionDevice];
      [e setArgumentTable:p.table4];
      id<MTLComputePipelineState> last = nil;
      for (const Cmd &c : p.cmds) {
        if (c.barrier_before)
          [e barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
        if (c.pso != last) { [e setComputePipelineState:c.pso]; last = c.pso; }
        for (size_t k = 0; k < c.slots.size(); k++)
          [p.table4 setAddress:c.slots[k].buf.gpuAddress + c.slots[k].offset atIndex:k];
        if (c.tg_mem) [e setThreadgroupMemoryLength:(c.tg_mem + 15) & ~15u atIndex:0];
        if (c.ind) [e dispatchThreadgroupsWithIndirectBuffer:c.ind.gpuAddress + c.ind_off threadsPerThreadgroup:c.tg];
        else if (c.threads) [e dispatchThreads:c.grid threadsPerThreadgroup:c.tg];
        else [e dispatchThreadgroups:c.grid threadsPerThreadgroup:c.tg];
      }
      [e endEncoding];
      [cb endCommandBuffer];
      MTL4CommitOptions *opt = [MTL4CommitOptions new];
      Impl::Flight *pf = &f;
      [opt addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        pf->g0 = fb.GPUStartTime;
        pf->g1 = fb.GPUEndTime;
        pf->err = fb.error;
        dispatch_semaphore_signal(pf->sem);
      }];
      id<MTL4CommandBuffer> cbs[1] = {cb};
      [d.queue4 commit:cbs count:1 options:opt];
    } else {
      id<MTLCommandBuffer> cb = [d.queue commandBuffer];
      const bool serial = backend_ == Backend::Serial;
      id<MTLComputeCommandEncoder> e =
          [cb computeCommandEncoderWithDispatchType:serial ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
      if (backend_ == Backend::ICB) {
        if (!p.resources.empty())
          [e useResources:p.resources.data() count:p.resources.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        if (p.icb) [e executeCommandsInBuffer:p.icb withRange:NSMakeRange(0, p.cmds.size())];
      } else {
        if (!p.extra.empty())
          [e useResources:p.extra.data() count:p.extra.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        id<MTLComputePipelineState> last = nil;
        for (const Cmd &c : p.cmds) {
          if (c.barrier_before && !serial) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          if (c.pso != last) { [e setComputePipelineState:c.pso]; last = c.pso; }
          for (size_t k = 0; k < c.slots.size(); k++) [e setBuffer:c.slots[k].buf offset:c.slots[k].offset atIndex:k];
          if (c.tg_mem) [e setThreadgroupMemoryLength:(c.tg_mem + 15) & ~15u atIndex:0];
          if (c.ind) [e dispatchThreadgroupsWithIndirectBuffer:c.ind indirectBufferOffset:c.ind_off threadsPerThreadgroup:c.tg];
          else if (c.threads) [e dispatchThreads:c.grid threadsPerThreadgroup:c.tg];
          else [e dispatchThreadgroups:c.grid threadsPerThreadgroup:c.tg];
        }
      }
      [e endEncoding];
      [cb commit];
      f.cb = cb;
    }
  }
  p.count++;
  encode_us_ = now_us() - t0;
}

void Recorder::touch() {
  Impl &p = *p_;
  if (backend_ == Backend::MTL4 && ended_ && p.rs4 != nil) [p.rs4 requestResidency];
}

void Recorder::wait_oldest() {
  Impl &p = *p_;
  if (!p.count) return;
  Impl::Flight &f = p.fl[p.head];
  p.head = (p.head + 1) % kMaxInFlight;
  p.count--;
  NSError *err = nil;
  if (backend_ == Backend::MTL4) {
    dispatch_semaphore_wait(f.sem, DISPATCH_TIME_FOREVER);
    wall_us_ = now_us() - f.t_commit;
    gpu_ms_ = (f.g1 - f.g0) * 1e3;
    err = f.err;
    f.err = nil;
  } else {
    [f.cb waitUntilCompleted];
    wall_us_ = now_us() - f.t_commit;
    gpu_ms_ = (f.cb.GPUEndTime - f.cb.GPUStartTime) * 1e3;
    err = f.cb.error;
    f.cb = nil;
  }
  NE_CHECK(err == nil, "GPU error (%s): %s", backend_name(backend_), err.localizedDescription.UTF8String);
}

void Recorder::wait() {
  while (p_->count) wait_oldest();
}

void Recorder::run() {
  commit();
  wait();
}

}  // namespace ne
