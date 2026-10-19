// lab-ttft: split-K GEMM over the dense-decoded W' for chunks of few rows (kernels/trunk6/trunk6.mm record_prefill).
// At M <= ~128 rows lab-prefill's NAX GEMM (pf_h16_hf, one 64 x 64 tile per threadgroup over the whole K) runs a few
// threadgroups per Linear (HC down 10240 -> 320: 5; GDN a / b: 1; out_proj / o_proj: 40) and is latency-bound: 73% of
// the T6 trunk's time in a 16-row chunk (docs/prefill.md 4c). Here the K range is split over S threadgroup slices
// (grid z), partials f32 [S][Mp][NP] reduced in a fixed order (deterministic, run to run bit-identical), then the
// unchanged t6_pf_post. Self-contained (metal_stdlib only): the host compiles this file at run time when the
// metallib lacks it (a build without the Metal toolchain).
#include <metal_stdlib>
using namespace metal;

struct T6SkP {
  uint K;   // reduction length = row stride of x and w (halves)
  uint NP;  // output columns (multiple of 32), row stride of part / y
  uint M;   // rows
  uint Mp;  // rows rounded up to 16 (8 MT: part rows per slice)
  uint ks;  // slice length (multiple of 32)
  uint S;   // slices
};

// part[s][m][n] = sum_{k in slice s} x[m][k] * w[n][k]. x half [>= Mp][K] (rows >= M: finite garbage, their outputs
// are never read), w half [NP][K] (W' dense, HF orientation). Grid (NP / 32, 1, S), 128 threads: simdgroup sg owns
// columns n0 + 8 sg .. + 8 for all Mp = 8 MT rows (MT accumulators), so every W' element is read once. S == 1: part =
// the y buffer itself.
template <int MT>
kernel void t6_pf_sk(device const half *w [[buffer(0)]], device const half *x [[buffer(1)]],
                     device float *part [[buffer(2)]], constant T6SkP &p [[buffer(3)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint KB = 32u, LD = KB + 8u;  // x staged per 32-wide k block (row pad: fewer bank conflicts)
  threadgroup half xs[8 * MT * LD];
  const uint n0 = tg.x * 32u + sg * 8u, s = tg.z;
  const uint k0 = s * p.ks, k1 = min(p.K, k0 + p.ks);
  simdgroup_float8x8 acc[MT];
  for (short i = 0; i < MT; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const device half *wp = w + (size_t)n0 * p.K;
  for (uint kb = k0; kb < k1; kb += KB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lid; j < 8u * MT * (KB / 8u); j += 128u) {  // (row, 8-half chunk) per thread, coalesced along k
      const uint r = j / (KB / 8u), c = (j - r * (KB / 8u)) * 8u;
      *(threadgroup half4 *)(xs + r * LD + c) = *(const device half4 *)(x + (size_t)r * p.K + kb + c);
      *(threadgroup half4 *)(xs + r * LD + c + 4u) = *(const device half4 *)(x + (size_t)r * p.K + kb + c + 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0u; kk < KB; kk += 8u) {
      simdgroup_half8x8 b;
      simdgroup_load(b, wp + kb + kk, p.K, ulong2(0, 0), true);  // b[kk][nn] = w[n0 + nn][k + kk]
      for (short i = 0; i < MT; i++) {
        simdgroup_half8x8 a;
        simdgroup_load(a, xs + (8 * i) * LD + kk, LD);  // a[mm][kk] = x[8 i + mm][k + kk]
        simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
      }
    }
  }
  device float *o = part + (size_t)s * p.Mp * p.NP + n0;
  for (short i = 0; i < MT; i++) simdgroup_store(acc[i], o + (size_t)(8 * i) * p.NP, p.NP);
}
template [[host_name("t6_pf_sk2")]] [[kernel]] decltype(t6_pf_sk<2>) t6_pf_sk<2>;
template [[host_name("t6_pf_sk4")]] [[kernel]] decltype(t6_pf_sk<4>) t6_pf_sk<4>;
template [[host_name("t6_pf_sk8")]] [[kernel]] decltype(t6_pf_sk<8>) t6_pf_sk<8>;
template [[host_name("t6_pf_sk16")]] [[kernel]] decltype(t6_pf_sk<16>) t6_pf_sk<16>;

// y[m][n] = sum_s part[s][m][n] in slice order, m < M, n < NP (y = yraw [M][NP]). Grid ceil(M NP / 256) x 256.
kernel void t6_pf_skr(device const float *part [[buffer(0)]], device float *y [[buffer(1)]],
                      constant T6SkP &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.M * p.NP) return;
  const uint m = i / p.NP, c = i - m * p.NP;
  float v = 0.0f;
  for (uint s = 0; s < p.S; s++) v += part[((size_t)s * p.Mp + m) * p.NP + c];
  y[i] = v;
}
