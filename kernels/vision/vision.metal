// Flash-Next vision tower (27-layer ViT, hidden 1152, 16 heads x 72, MLP 4304, patch merger -> 2560). Owner:
// lab-vision. Design, rounding points and numbers: docs/vision.md.
//
// Activations are bf16 between ops with the reference's (mlx-vlm qwen3_vl VisionModel, bf16) rounding points, found
// empirically against MLX 0.32.3 (tests/vision): Linear = one rounding of acc + bias (addmm), the Conv3d patch embed
// rounds the conv and the bias add separately, LayerNorm = bf16(w * bf16(xhat) + b) with f32 statistics, GELU runs
// op by op in bf16 (MLX's compiled elementwise graph), the vision rope in f32 with one rounding.
//
//   vit_gemm_<epi>  y[M][N] = epi(x[M][K] @ W^T) on NAX, bf16 x / W (dense [N][K], N % 64 == 0, K % 32 == 0)
//   vit_layernorm   bf16 [M][1152] -> bf16 (eps 1e-6)
//   vit_pos_interp  bilinear interpolation of the 48 x 48 learned position table (bf16 ops as the reference)
//   vit_rope        2-D vision rope in place on the q / k slots of the padded qkv rows
//   vit_attn        MLX's NAX flash attention (head dim 72 zero-padded to 80 / 96), no mask
//
// The GEMM body follows MLX's NAX steel GEMM through lab-prefill's pf_bf16 (kernels/prefill/pf_gemm.metal);
// NAXTile / attention_nax are vendored from MLX 0.32.3 (kernels/prefill/mlx/, MIT License, Copyright (c) 2023-26
// Apple Inc.; LICENSE.mlx). erf / expm1f below are MLX's (mlx/backend/metal/kernels/erf.h, expm1f.h: MIT, Apple;
// expm1f after Norbert Juffa, BSD-2-Clause, Copyright (c) 2015-2023 Norbert Juffa: redistribution with or without
// modification permitted provided the copyright notice is retained; THIS SOFTWARE IS PROVIDED "AS IS").
#include "kernels/prefill/mlx/attention_nax.h"

using namespace mlx::steel;

namespace vit {
constant constexpr int HID = 1152;
static inline float bfr(float x) {  // round to the nearest bf16 (ties to even)
  const uint u = as_type<uint>(x);
  return as_type<float>((u + 0x7FFFu + ((u >> 16u) & 1u)) & 0xFFFF0000u);
}

static inline float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;
  j = fma(1.442695f, a, 12582912.f);
  j = j - 12582912.0f;
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);
  s = f * f;
  if (a == 0.0f) s = a;
  r = 1.97350979e-4f;
  r = fma(r, f, 1.39309070e-3f);
  r = fma(r, f, 8.33343994e-3f);
  r = fma(r, f, 4.16668020e-2f);
  r = fma(r, f, 1.66666716e-1f);
  r = fma(r, f, 4.99999970e-1f);
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s;
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0) r = v;
  if (j == 1) r = v + v;
  return r;
}
static inline float mlx_expm1f(float a) {
  float r = expm1f_scaled_unchecked(a, 1.0f);
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}
static inline float mlx_erf(float a) {
  float r, s, t, u;
  t = abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    r = fma(-1.72853470e-5f, t, 3.83197126e-4f);
    u = fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = fma(r, s, u);
    r = fma(r, t, -1.06777877e-1f);
    r = fma(r, t, -6.34846687e-1f);
    r = fma(r, t, -1.28717512e-1f);
    r = fma(r, t, -t);
    r = -mlx_expm1f(r);
    r = copysign(r, a);
  } else {
    r = -5.96761703e-4f;
    r = fma(r, s, 4.99119423e-3f);
    r = fma(r, s, -2.67681349e-2f);
    r = fma(r, s, 1.12819925e-1f);
    r = fma(r, s, -3.76125336e-1f);
    r = fma(r, s, 1.28379166e-1f);
    r = fma(r, a, a);
  }
  return r;
}
// nn.gelu_approx in bf16, op by op: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static inline float gelu_tanh_bf(float x) {
  const float c1 = bfr(0.044715f), c2 = bfr(0.7978845608028654f);
  const float t1 = bfr(x * x * x), t2 = bfr(c1 * t1), t3 = bfr(x + t2), t4 = bfr(c2 * t3);
  const float t5 = bfr(precise::tanh(t4)), t6 = bfr(1.0f + t5), t7 = bfr(0.5f * x);
  return bfr(t7 * t6);
}
// nn.gelu in bf16, op by op: x * (1 + erf(x / sqrt(2))) / 2
static inline float gelu_erf_bf(float x) {
  const float t1 = bfr(precise::divide(x, bfr(1.4142135623730951f))), t2 = bfr(mlx_erf(t1)), t3 = bfr(1.0f + t2);
  return bfr(bfr(x * t3) * 0.5f);
}
}  // namespace vit
using namespace vit;

struct VitGemm {  // = vision.mm VitGemm
  int K, N, M, ldx, ldy, ldr;
};
enum : int { EPI_BIAS = 0, EPI_RES = 1, EPI_GELU_T = 2, EPI_GELU_E = 3, EPI_CONV = 4 };

// y[r][c] = epi(sum_k x[r][k] w[c][k]); grid (N / 64, ceil(M / 64)), 128 threads (2 x 2 simdgroups of 32 x 32)
//   EPI_BIAS   bf16(acc + b[c])                       (nn.Linear = addmm)
//   EPI_RES    bf16(res[r][c] + bf16(acc + b[c]))     (Linear + residual add)
//   EPI_GELU_T gelu_tanh_bf(bf16(acc + b[c]))         (MLP fc1)
//   EPI_GELU_E gelu_erf_bf(bf16(acc + b[c]))          (merger fc1)
//   EPI_CONV   bf16(bf16(bf16(acc) + b[c]) + res[r][c])  (Conv3d, + bias, + position embedding)
template <int EPI>
kernel void vit_gemm(const device bfloat *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                     const device bfloat *bias [[buffer(2)]], const device bfloat *res [[buffer(3)]],
                     device bfloat *y [[buffer(4)]], const constant VitGemm &p [[buffer(5)]],
                     uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 64, BN = 64, WN = 2;
  constexpr short SM = 32, SN = 32, SK = 32, TM = SM / 16, TN = SN / 16, TK = SK / 16;
  const int K = p.K, M = p.M;
  const int y_row = tid.y * BM, y_col = tid.x * BN;
  if (y_row >= M) return;
  const short tm = SM * (simd_gid / WN), tn = SN * (simd_gid % WN);
  const short sgp_sm = min(int(SM), M - (y_row + tm));
  if (sgp_sm <= 0) return;
  x += (y_row + tm) * (int64_t)p.ldx;
  w += (y_col + tn) * (int64_t)K;
  NAXTile<float, TM, TN> D;
  D.clear();
  for (int k = 0; k < K; k += SK) {
    NAXTile<bfloat, TM, TK> A;
    NAXTile<bfloat, TN, TK> B;
    if (sgp_sm == SM) A.load(x + k, p.ldx);
    else A.load_safe(x + k, p.ldx, short2(SK, sgp_sm));
    B.load(w + k, K);
    tile_matmad_nax(D, A, metal::bool_constant<false>{}, B, metal::bool_constant<true>{});
  }
  const short2 sc = decltype(D)::NAXFrag_t::get_coord();
  for (short i = 0; i < TM; i++)
    for (short j = 0; j < TN; j++) {
      thread auto &f = D.frag_at(i, j);
      for (short e = 0; e < 8; e++) {
        const int r = y_row + tm + i * 16 + sc.y + (e >> 2) * 8;
        const int c = y_col + tn + j * 16 + sc.x + (e & 3);
        if (r >= M) continue;
        const float acc = f[e];
        const float b = float(bias[c]);
        float v;
        if (EPI == EPI_BIAS) v = bfr(acc + b);
        else if (EPI == EPI_RES) v = bfr(float(res[(int64_t)r * p.ldr + c]) + bfr(acc + b));
        else if (EPI == EPI_GELU_T) v = gelu_tanh_bf(bfr(acc + b));
        else if (EPI == EPI_GELU_E) v = gelu_erf_bf(bfr(acc + b));
        else v = bfr(bfr(bfr(acc) + b) + float(res[(int64_t)r * p.ldr + c]));
        y[(int64_t)r * p.ldy + c] = bfloat(v);
      }
    }
}
template [[host_name("vit_gemm_bias")]] [[kernel]] decltype(vit_gemm<EPI_BIAS>) vit_gemm<EPI_BIAS>;
template [[host_name("vit_gemm_res")]] [[kernel]] decltype(vit_gemm<EPI_RES>) vit_gemm<EPI_RES>;
template [[host_name("vit_gemm_gelu_t")]] [[kernel]] decltype(vit_gemm<EPI_GELU_T>) vit_gemm<EPI_GELU_T>;
template [[host_name("vit_gemm_gelu_e")]] [[kernel]] decltype(vit_gemm<EPI_GELU_E>) vit_gemm<EPI_GELU_E>;
template [[host_name("vit_gemm_conv")]] [[kernel]] decltype(vit_gemm<EPI_CONV>) vit_gemm<EPI_CONV>;

// LayerNorm over 1152 (one row per threadgroup of 256): f32 mean / variance (two passes), y = bf16(w * bf16(xhat) + b)
kernel void vit_layernorm(const device bfloat *x [[buffer(0)]], const device bfloat *w [[buffer(1)]],
                          const device bfloat *b [[buffer(2)]], device bfloat *y [[buffer(3)]],
                          constant int &M [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  if ((int)r >= M) return;
  const device bfloat *xr = x + (size_t)r * HID;
  float v[5];
  float s = 0.0f;
  for (int j = 0; j < 5; j++) {
    const int c = lid + 256 * j;
    v[j] = c < HID ? float(xr[c]) : 0.0f;
    s += v[j];
  }
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int i = 0; i < 8; i++) tot += part[i];
  const float mean = tot / float(HID);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (int j = 0; j < 5; j++) {
    const int c = lid + 256 * j;
    const float d = c < HID ? v[j] - mean : 0.0f;
    q += d * d;
  }
  q = simd_sum(q);
  if (lane == 0) part[sg] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (int i = 0; i < 8; i++) var += part[i];
  const float nrm = precise::rsqrt(var / float(HID) + 1e-6f);
  for (int j = 0; j < 5; j++) {
    const int c = lid + 256 * j;
    if (c < HID) y[(size_t)r * HID + c] = bfloat(bfr(float(w[c]) * bfr((v[j] - mean) * nrm) + float(b[c])));
  }
}

// Position embedding of each patch (merge-block order): sum over the 4 bilinear corners k of bf16(table[idx[k][n]] *
// wt[k][n]), added left to right in bf16 (= fast_pos_embed_interpolate in bf16). idx / wt [4][N] from the host.
kernel void vit_pos_interp(const device bfloat *table [[buffer(0)]], const device int *idx [[buffer(1)]],
                           const device float *wt [[buffer(2)]], device bfloat *out [[buffer(3)]],
                           constant int &N [[buffer(4)]], uint n [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
  if ((int)n >= N) return;
  for (int c = lid; c < HID; c += 256) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) {
      const float p = bfr(float(table[(size_t)idx[k * N + n] * HID + c]) * wt[k * N + n]);
      s = k == 0 ? p : bfr(s + p);
    }
    out[(size_t)n * HID + c] = bfloat(s);
  }
}

// Vision rope in place: qkv rows [N][3][16][96] (q | k | v slots, 72 used dims per head). For q and k, pairs
// (d, d + 36), d < 36: angle = pos[n][d < 18 ? 0 : 1] * invf[d % 18] (pos = (row, col) of the patch);
// out = bf16(x cos + rotate_half(x) sin) in f32. grid (N, 2 * 16), 64 threads (36 used).
kernel void vit_rope(device bfloat *qkv [[buffer(0)]], const device int2 *pos [[buffer(1)]],
                     constant float *invf [[buffer(2)]], constant int &N [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const int n = tg.x;
  if (n >= N || d >= 36u) return;
  const int which = tg.y / 16, h = tg.y % 16;
  device bfloat *e = qkv + (size_t)n * 4608 + which * 1536 + h * 96;
  const int2 p = pos[n];
  const float a = float(d < 18u ? p.x : p.y) * invf[d % 18u];
  const float c = precise::cos(a), s = precise::sin(a);
  const float x1 = float(e[d]), x2 = float(e[d + 36]);
  const float p1 = x1 * c, p2 = (-x2) * s, p3 = x2 * c, p4 = x1 * s;
  e[d] = bfloat(bfr(p1 + p2));
  e[d + 36] = bfloat(bfr(p3 + p4));
}

// MLX NAX flash attention, q / k head dim 80 (72 + zero pad), v 96 (72 + pad): BQ 64, BK 32, 4 simdgroups.
template [[host_name("vit_attn")]] [[kernel]] decltype(attention_nax<bfloat, 64, 32, 80, 96, 4, 1>)
    attention_nax<bfloat, 64, 32, 80, 96, 4, 1>;
