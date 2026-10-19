// Core self-test kernels (compiled into build/engine.metallib with everything else). Owner: lab-core.
// Names start with ne_st_ so they never collide with model kernels.
#include <metal_stdlib>
using namespace metal;

// Byte copy (any alignment): dst[i] = src[i], i < n. Grid-stride.
kernel void ne_st_copy_bytes(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                             constant ulong &n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                             uint nth [[threads_per_grid]]) {
  for (ulong i = gid; i < n; i += nth) dst[i] = src[i];
}

// One step of a dependent chain: dst[i] = lcg^32(src[n - 1 - i] * MUL + add). The first threadgroups of a step read
// what the last threadgroups of the previous step wrote, and the LCG loop lengthens each thread, so a missing barrier
// between steps shows up as wrong values. MUL is a function constant (default 3).
constant uint fc_mul [[function_constant(0)]];
constant uint MUL = is_function_constant_defined(fc_mul) ? fc_mul : 3u;
kernel void ne_st_step(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],
                       constant uint &n [[buffer(2)]], constant uint &add [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  uint v = src[n - 1u - gid] * MUL + add;
  for (uint k = 0; k < 32u; k++) v = v * 1664525u + 1013904223u;
  dst[gid] = v;
}

// Lead 4's dispatch-cost probes (the lab tree's engine/fused/disp.metal).
// One threadgroup bumps a counter: a dependent chain of these measures pure dispatch + barrier cost.
kernel void ne_st_tiny(device float *buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  if (tid == 0u) buf[0] = buf[0] + 1.0f;
}
// 256 KiB read (64 B per thread per step); `dep` makes it depend on the previous dispatch's output.
kernel void ne_st_read256(device const uint4 *src [[buffer(0)]], device uint *out [[buffer(1)]],
                          device const uint *dep [[buffer(2)]], constant uint &n4 [[buffer(3)]],
                          uint gid [[thread_position_in_grid]], uint nth [[threads_per_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  uint4 a = uint4(0u);
  for (uint b = gid; b < n4; b += nth) a ^= src[b];
  uint z = a.x ^ a.y ^ a.z ^ a.w;
  z = simd_xor(z);
  if (lane == 0u) out[gid / 32u] = z ^ (dep[0] == 1234u ? 1u : 0u);
}

// Threadgroup-memory check: reverse 256 values through threadgroup memory bound at index 0 by the recorder.
kernel void ne_st_tgmem(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                        threadgroup float *tmp [[threadgroup(0)]], uint lid [[thread_position_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]]) {
  tmp[lid] = src[tgid * 256u + lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  dst[tgid * 256u + lid] = tmp[255u - lid];
}

// Reads through GPU addresses stored in a buffer: out[i] = tab.p[i % 4][i]. Tests Recorder::use().
struct NePtrTable { device const uint *p[4]; };
kernel void ne_st_ptr(device const NePtrTable &tab [[buffer(0)]], device uint *out [[buffer(1)]],
                      constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) out[gid] = tab.p[gid % 4u][gid];
}
