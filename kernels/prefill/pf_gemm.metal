// Prefill GEMMs on the M5 neural accelerators (NAX, MPP matmul2d through MLX's steel NAXTile helpers). Owner:
// lab-prefill. Design and numbers: docs/prefill.md.
//
//   pf_q8_*   : y[M][N] = x[M][K] @ W^T for the pack's MLX 8-bit affine trunk weights (W U32 [N][K/4], bf16 scales /
//               biases per group G), read in place. The weight tile is dequantised to bf16 in threadgroup memory
//               (one 64 x 64 tile per k step), x is read straight from device memory into NAX fragments (bf16).
//               f32 accumulation; y stored as bf16 (= the reference's bf16 Linear outputs) or f32. Strided x / y
//               (ldx, ldy) so projections can land in column slices of a wider buffer.
//   pf_bf16_* : the same over a dense bf16 weight [N][K] (router), B fragments loaded from device memory.
//
// The GEMM body is MLX's qmm_t_nax_tgp_impl (mlx/backend/metal/kernels/quantized_nax.h, MLX 0.32.3) with its input /
// output types split and the leading dimensions made explicit; NAXTile / QuantizedBlockLoader are vendored unchanged
// in kernels/prefill/mlx/ (MIT License, Copyright (c) 2023 Apple Inc.; full text in kernels/prefill/mlx/LICENSE.mlx).
#include "kernels/prefill/mlx/gemm_nax.h"
#include "kernels/prefill/mlx/quantized_nax.h"
#ifndef PF_WT
#define PF_WT half  // GEMM operand type of the q8 kernels (weights and x tiles)
#endif

constant bool PF_REFR_FC [[function_constant(0)]];  // = ne_common.h NE_REFR
constant bool PF_REFR = is_function_constant_defined(PF_REFR_FC) ? PF_REFR_FC : true;
static inline float pf_bf16r(float x) {  // = ne_common.h bf16r
  const uint u = as_type<uint>(x);
  return as_type<float>((u + 0x7FFFu + ((u >> 16u) & 1u)) & 0xFFFF0000u);
}
static inline float pf_rr(float x) { return PF_REFR ? pf_bf16r(x) : x; }

struct PfGemm {  // = prefill.mm PfGemm
  int K, N, M, ldx, ldy;
};

// Weight-tile loader (8-bit, BK = BN = 64, 128 threads: thread = 32 consecutive bytes of one row; the group (>= 64)
// covers the whole 64-wide k step): w = scale * q + bias in f32, stored as T (half by default: 11-bit mantissa, so the
// dequantised weights stay within 2^-12 of the decode path's f32 weights; MLX's own loader stores bf16).
template <typename T, int G>
struct PfQ8Loader {
  const device uint8_t *src;
  const device bfloat *sc, *bi;
  threadgroup T *dst;
  int step = 0;
  short row;
  PfQ8Loader(const device uint8_t *w, const device bfloat *s, const device bfloat *b, int K, threadgroup T *ws, int ldd,
             uint tid) {
    row = short(tid / 2u);
    const short col = short((tid % 2u) * 32u);
    src = w + (size_t)row * K + col;
    sc = s + (size_t)row * (K / G);
    bi = b + (size_t)row * (K / G);
    dst = ws + row * ldd + col;
  }
  void load(short nrows) {
    if (row >= nrows) {
      for (int i = 0; i < 32; i++) dst[i] = T(0);
      return;
    }
    const float s = float(*sc), b = float(*bi);
    const device uchar4 *w4 = (const device uchar4 *)src;
    for (int i = 0; i < 8; i++) {
      const float4 q = float4(w4[i]);
      for (int j = 0; j < 4; j++) dst[4 * i + j] = T(fma(s, q[j], b));
    }
  }
  void next() {
    src += 64;
    if (++step == G / 64) { step = 0; sc++; bi++; }
  }
};

// EPI = 1: the HC read's activation in the epilogue (pf_hc_act: y = rr(silu(bf16(acc) / 4)), bf16 out; lab-pflevers)
template <typename XT, typename YT, int G, bool AN, typename T = half, int BM = 64, int BK = 64, int BN = 64, int WM = 2,
          int WN = 2, int EPI = 0>
METAL_FUNC void pf_qmm_t(const device uint32_t *w, const device bfloat *scales, const device bfloat *biases,
                         const device XT *x, device YT *y, threadgroup T *Ws, const constant PfGemm &p,
                         uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  const int K = p.K, N = p.N, M = p.M;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  if (y_row >= M) return;
  auto wl = (const device uint8_t *)w;
  x += y_row * (int64_t)p.ldx;
  wl += (size_t)y_col * K;
  scales += y_col * (K / G);
  biases += y_col * (K / G);
  y += y_row * (int64_t)p.ldy + y_col;
  PfQ8Loader<T, G> loader_w(wl, scales, biases, K, Ws, BK_padded, simd_gid * 32u + simd_lid);
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);
  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);
  const short sgp_sn = AN ? SN : min(int(SN), N - (y_col + tn));
  const short tgp_bn = AN ? BN : min(BN, int(N - (y_col)));
  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  x += tm * (int64_t)p.ldx;
  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_w.load(tgp_bn);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;
          volatile int compiler_barrier;
          if (sgp_sm > 0) {
            if constexpr (kAlignedM.value) Atile.load(x + kk1, p.ldx);
            else Atile.load_safe(x + kk1, p.ldx, short2(SK, sgp_sm));
          }
          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
          tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
          (void)compiler_barrier;
        }
        x += BK;
        loader_w.next();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sgp_sm <= 0 || sgp_sn <= 0) return;
      if constexpr (EPI == 1) {
        for (short i = 0; i < decltype(Dtile)::kNumFrags; i++)
          for (short e = 0; e < decltype(Dtile)::kElemsPerFrag; e++) {
            const float b = pf_bf16r(Dtile.val_frags[i][e]);
            Dtile.val_frags[i][e] = pf_bf16r(pf_rr(b * 0.25f / (1.0f + exp(-b * 0.25f))));
          }
      }
      if (kAlignedM.value && sgp_sn == SN) Dtile.store(y + tm * (int64_t)p.ldy + tn, p.ldy);
      else Dtile.store_safe(y + tm * (int64_t)p.ldy + tn, p.ldy, short2(sgp_sn, sgp_sm));
  });
}

// grid (ceil(N / 64), ceil(M / 64)), 128 threads
template <typename XT, typename YT, int G, bool AN>
kernel void pf_q8(const device uint32_t *w [[buffer(0)]], const device bfloat *s [[buffer(1)]],
                  const device bfloat *b [[buffer(2)]], const device XT *x [[buffer(3)]], device YT *y [[buffer(4)]],
                  const constant PfGemm &p [[buffer(5)]], uint3 tid [[threadgroup_position_in_grid]],
                  uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup PF_WT Ws[64 * (64 + 8)];
  pf_qmm_t<XT, YT, G, AN, PF_WT>(w, s, b, x, y, Ws, p, tid, simd_gid, simd_lid);
}
// pf_q8 bf16 -> bf16 with the HC activation epilogue (the HC down GEMM, N = 320)
template <int G>
kernel void pf_q8_act(const device uint32_t *w [[buffer(0)]], const device bfloat *s [[buffer(1)]],
                      const device bfloat *b [[buffer(2)]], const device bfloat *x [[buffer(3)]], device bfloat *y [[buffer(4)]],
                      const constant PfGemm &p [[buffer(5)]], uint3 tid [[threadgroup_position_in_grid]],
                      uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup PF_WT Ws[64 * (64 + 8)];
  pf_qmm_t<bfloat, bfloat, G, true, PF_WT, 64, 64, 64, 2, 2, 1>(w, s, b, x, y, Ws, p, tid, simd_gid, simd_lid);
}

// Dense bf16 weight [N][K] (row-major): y = x W^T. Same tiling, B fragments straight from device memory.
// grid (ceil(N / 64), ceil(M / 64)), 128 threads; N % 64 == 0, K % 32 == 0.
// (T = half, XT = half: lab-t6wire's EXL3 prefill GEMM on the dense-decoded W' and the prepared x, "pf_h16_hf".)
template <typename XT, typename YT, typename T = bfloat>
kernel void pf_bf16(const device T *w [[buffer(0)]], const device XT *x [[buffer(1)]], device YT *y [[buffer(2)]],
                    const constant PfGemm &p [[buffer(3)]], uint3 tid [[threadgroup_position_in_grid]],
                    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 64, BN = 64, WM = 2, WN = 2;
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32, TM = SM / 16, TN = SN / 16, TK = SK / 16;
  const int K = p.K, M = p.M;
  const int y_row = tid.y * BM, y_col = tid.x * BN;
  if (y_row >= M) return;
  const short tm = SM * (simd_gid / WN), tn = SN * (simd_gid % WN);
  const short sgp_sm = min(int(SM), M - (y_row + tm));
  if (sgp_sm <= 0) return;
  x += (y_row + tm) * (int64_t)p.ldx;
  w += (y_col + tn) * (int64_t)K;
  y += (y_row + tm) * (int64_t)p.ldy + y_col + tn;
  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  for (int k = 0; k < K; k += SK) {
    NAXTile<T, TM, TK> Atile;
    NAXTile<T, TN, TK> Btile;
    if (sgp_sm == SM) Atile.load(x + k, p.ldx);
    else Atile.load_safe(x + k, p.ldx, short2(SK, sgp_sm));
    Btile.load(w + k, K);
    tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
  }
  if (sgp_sm == SM) Dtile.store(y, p.ldy);
  else Dtile.store_safe(y, p.ldy, short2(SN, sgp_sm));
}

#define PF_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define PF_Q8(xs, xt, ys, yt)                                                \
  PF_INST("pf_q8_" xs ys "_g64", pf_q8, xt, yt, 64, true)                    \
  PF_INST("pf_q8_" xs ys "_g128", pf_q8, xt, yt, 128, true)                  \
  PF_INST("pf_q8_" xs ys "_g64_u", pf_q8, xt, yt, 64, false)                 \
  PF_INST("pf_q8_" xs ys "_g128_u", pf_q8, xt, yt, 128, false)
PF_Q8("b", bfloat, "b", bfloat)
PF_Q8("b", bfloat, "f", float)
PF_Q8("f", float, "b", bfloat)
PF_Q8("f", float, "f", float)
PF_INST("pf_q8_bb_g64_act", pf_q8_act, 64)
PF_INST("pf_q8_bb_g128_act", pf_q8_act, 128)
PF_INST("pf_bf16_bb", pf_bf16, bfloat, bfloat)
PF_INST("pf_bf16_bf", pf_bf16, bfloat, float)
PF_INST("pf_h16_hf", pf_bf16, half, float, half)
