// Utility kernels (timing aids). Owner: lab-layer.
#include <metal_stdlib>
using namespace metal;

// Streaming read of n4 uint4 (64 B contiguous per thread per step; lab lead 1's best read pattern), used to evict
// the SLC between timed steps. out gets one word per simdgroup so the reads cannot be elided.
kernel void ne_stream_read(device const uint4 *src [[buffer(0)]], device uint *out [[buffer(1)]],
                           constant uint &n4 [[buffer(2)]], uint gid [[thread_position_in_grid]],
                           uint nth [[threads_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint4 a = uint4(0u);
  for (uint b = gid * 4u; b + 3u < n4; b += nth * 4u) {
    const uint4 v0 = src[b], v1 = src[b + 1u], v2 = src[b + 2u], v3 = src[b + 3u];
    a ^= (v0 ^ v1) ^ (v2 ^ v3);
  }
  const uint z = simd_xor(a.x ^ a.y ^ a.z ^ a.w);
  if (lane == 0u) out[gid / 32u] = z;
}

// Empty dispatch (kernels/bench/fn_prof.mm: the cost of one dispatch + barrier boundary).
kernel void ne_empty(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (gid == 0xFFFFFFFFu) out[0] = 0u;
}
