// Flash-Next full attention with QSA sparse selection (layers 3, 7, ..., 47), R = 1..8 rows (decode / MTP verify).
// Owner: lab-attn. Math: docs/reference.md 3.4; layouts, tolerances, timings: docs/attn.md.
//
//   qsa_prep    q / k rmsnorm + partial rope (q * 1/16 -> qr), new k / v -> KV cache (f32 | bf16 | kv8 g64), indexer
//               queries (q_layernorm + rope -> iq), raw indexer keys -> ikeys (bf16), block keys of the blocks this
//               step completes (k_layernorm(bf16(mean f32 of 4 raw keys)) + rope at 4b -> bkeys, bf16)
//   qsa_score   scores[t][b] = sum_h relu(iq[t,h] . bkeys[b]) / sqrt(128) (f32), b < (T + R) / 4
//   qsa_select  per row: comp = (T + t + 1) / 4 complete blocks; comp <= 512 -> blocks 0..comp-1 (= causal), else
//               the top 512 by score (exact radix select, ties -> lower block id), emitted ascending
//   qsa_attn    split-K over each row's block list (8 blocks = 32 keys per threadgroup) + its tail
//               [4 comp, T + t]: TG = (kv group, row, chunk), simdgroup = q head (12 per group), online softmax
//   qsa_merge   chunk partials in fixed order (deterministic), * sigmoid(gate) -> [R][6144]
//
// Attribution: the radix select (MSD digits over the f32 ordinal, run-length histogram adds, early stop when a bin
// holds exactly what is still needed, index-ordered compaction with the lower-index tie rule) follows Sushi's
// sushi_qsa_select (src/qwen4_qsa.zig, github.com/beamivalice/sushi, MIT, (c) 2026 Theinruj Toranavikrai, David
// Dalcu); written anew here for one row per threadgroup. The kv8 quantizer reproduces MLX's affine_quantize
// (bits 8, group 64; MIT, (c) 2023 Apple Inc.). MIT notice: permission is hereby granted, free of charge, to any
// person obtaining a copy of this software ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "kernels/ne_common.h"

namespace qsa {
constant constexpr uint NKV = 2u, HD = 256u, NQH = 24u, QPG = 12u, QROW = 512u, QD = NQH * QROW, KVD = NKV * HD;
constant constexpr uint IH = 4u, IHD = 128u, IQD = 5u * IHD, RATIO = 4u, KB = 512u;
// attention chunks: BCH blocks (4 BCH keys) per threadgroup, KB / BCH block chunks + 1 tail chunk per (kv group,
// row); BCH is a template parameter of qsa_attn / qsa_merge (default 8; the host passes the same to both)
constant constexpr uint PW = HD + 2u;        // partial record: acc[256], m, l
constant constexpr uint HSL = 16u;           // head slots per (row, chunk) in the partials (12 used; MMA tiles are 16)
constant constexpr uint GQ = 64u;            // kv8 group size
constant constexpr float ISQ128 = 11.313708498984761f;
}  // namespace qsa
using namespace qsa;

// KV cache formats: 0 = f32 [NKV][tmax][256], 1 = bf16 (ushort), 2 = kv8: codes u8 [NKV][tmax][256] + per (token,
// group of 64) ushort2 (bf16 scale, bf16 bias) [NKV][tmax][4]; w = scale * code + bias (MLX affine, bits 8).
enum : uint { KV_F32 = 0u, KV_BF16 = 1u, KV_Q8 = 2u };

static inline ushort bfbits(float x) { return ushort(as_type<uint>(bf16r(x)) >> 16u); }
static inline uint pack2bf(float a, float b) { return uint(bfbits(a)) | (uint(bfbits(b)) << 16u); }

// rmsnorm of a head held as N per lane (dims lane*N ..), * w (pack weight as stored), f32 -> bf16
template <uint N, uint DIM>
static inline void head_norm(thread float *e, const device ushort *w, uint lane) {
  float ss = 0.0f;
  for (uint j = 0u; j < N; j++) ss += e[j] * e[j];
  const float inv = 1.0f / precise::sqrt(simd_sum(ss) / float(DIM) + 1e-6f);
  for (uint j = 0u; j < N; j++) e[j] = bf16r(e[j] * inv * bf(w[lane * N + j]));
}
// partial rope (64 rotated dims, half-split pairs (d, d + 32)) at cache row `row`; N dims per lane, so dims < 32 live
// in lanes [0, 32/N) and their partners in the next 32/N lanes (shuffle xor 32/N). invf[d] = 1e7^(-2d/64) (host
// table). Multimodal positions (lab-vision, interleaved MRoPE, mrope_section [11, 11, 10]): invf[32] = mlen,
// invf[33] = delta (int bits); rows < mlen take (t, h, w) from the table at invf + 36 (int4 per row), later rows
// row + delta on all three axes; frequency d uses axis h if d % 3 == 1, w if d % 3 == 2 and d < 30, else t
// (mlx_vlm rope_utils _interleaved_position_selector). mlen = delta = 0 (text) = plain rope at the row index.
static inline float3 mrope_pos(constant float *invf, uint row) {
  const int mlen = as_type<int>(invf[32]), delta = as_type<int>(invf[33]);
  if ((int)row < mlen) {
    constant int *t = (constant int *)(invf + 36) + 4u * row;
    return float3(float(t[0]), float(t[1]), float(t[2]));
  }
  return float3(float(int(row) + delta));
}
template <uint N>
static inline void head_rope(thread float *e, constant float *invf, uint row, uint lane) {
  constexpr uint X = 32u / N;
  const float3 p = mrope_pos(invf, row);
  float o[N];
  for (uint j = 0u; j < N; j++) o[j] = simd_shuffle_xor(e[j], ushort(X));
  if (lane < 2u * X) {
    for (uint j = 0u; j < N; j++) {
      const uint d = (lane % X) * N + j;
      const float pd = d % 3u == 1u ? p.y : (d % 3u == 2u && d < 30u ? p.z : p.x);
      const float a = pd * invf[d];
      const float c = precise::cos(a), s = precise::sin(a);
      e[j] = lane < X ? e[j] * c - o[j] * s : e[j] * c + o[j] * s;
    }
  }
}

// One cache row slice (8 dims of lane, group = lane / 8) in format F.
template <uint F>
static inline void kv_store(thread const float *e, device uchar *base, device ushort2 *sb, size_t row, uint lane) {
  if (F == KV_F32) {
    device float4 *p = (device float4 *)((device float *)base + row * HD + lane * 8u);
    p[0] = float4(e[0], e[1], e[2], e[3]);
    p[1] = float4(e[4], e[5], e[6], e[7]);
  } else if (F == KV_BF16) {
    device uint4 *p = (device uint4 *)((device ushort *)base + row * HD + lane * 8u);
    p[0] = uint4(pack2bf(e[0], e[1]), pack2bf(e[2], e[3]), pack2bf(e[4], e[5]), pack2bf(e[6], e[7]));
  } else {
    // MLX affine_quantize (bits 8, group 64 = 8 lanes): w_min starts at +max, w_max at 0 (as MLX)
    float mn = e[0], mx = 0.0f;
    for (uint j = 0u; j < 8u; j++) { mn = min(mn, e[j]); mx = max(mx, e[j]); }
    for (ushort x = 1u; x < 8u; x <<= 1u) {
      mn = min(mn, simd_shuffle_xor(mn, x));
      mx = max(mx, simd_shuffle_xor(mx, x));
    }
    float scale = max(precise::divide(mx - mn, 255.0f), 1e-7f);
    const bool side = fabs(mn) > fabs(mx);
    scale = side ? scale : -scale;
    const float edge = side ? mn : mx;
    const float q0 = round(precise::divide(edge, scale));
    const bool at_zero = q0 == 0.0f;
    scale = at_zero ? scale : precise::divide(edge, q0);
    const float bias = at_zero ? 0.0f : edge;
    uint c[8];
    const float inv = precise::divide(1.0f, scale);
    for (uint j = 0u; j < 8u; j++) c[j] = uint(clamp(round((e[j] - bias) * inv), 0.0f, 255.0f));
    device uint2 *p = (device uint2 *)(base + row * HD + lane * 8u);
    p[0] = uint2(c[0] | (c[1] << 8u) | (c[2] << 16u) | (c[3] << 24u), c[4] | (c[5] << 8u) | (c[6] << 16u) | (c[7] << 24u));
    if ((lane & 7u) == 0u) sb[row * (HD / GQ) + lane / 8u] = ushort2(bfbits(scale), bfbits(bias));
  }
}
// 8 dims of a cached row (lane's slice) as floats
template <uint F>
static inline void kv_load(const device uchar *base, const device ushort2 *sb, size_t row, uint lane, thread float4 &a,
                           thread float4 &b) {
  if (F == KV_F32) {
    const device float4 *p = (const device float4 *)((const device float *)base + row * HD + lane * 8u);
    a = p[0];
    b = p[1];
  } else if (F == KV_BF16) {
    const uint4 u = *(const device uint4 *)((const device ushort *)base + row * HD + lane * 8u);
    a = bf4(u.xy);
    b = bf4(u.zw);
  } else {
    const uint2 u = *(const device uint2 *)(base + row * HD + lane * 8u);
    const ushort2 s = sb[row * (HD / GQ) + lane / 8u];
    const float sc = bf(s.x), bi = bf(s.y);
    a = float4(as_type<uchar4>(u.x)) * sc + bi;
    b = float4(as_type<uchar4>(u.y)) * sc + bi;
  }
}

// Block key b from 4 raw keys (bf16, 4 dims per lane): tokens < T from the cache, the others from this step's iqk.
static inline void block_key(uint b, uint T, const device float *iqk, const device ushort *ikeys,
                             const device ushort *iknw, constant float *invf, device ushort *bkeys, uint lane) {
  float s[4] = {0, 0, 0, 0};
  for (uint k = 0u; k < RATIO; k++) {
    const uint tok = b * RATIO + k;
    float4 v;
    if (tok < T) v = bf4(*(const device uint2 *)(ikeys + (size_t)tok * IHD + lane * 4u));
    else {
      const float4 x = *(const device float4 *)(iqk + (tok - T) * IQD + IH * IHD + lane * 4u);
      v = float4(bf16r(x.x), bf16r(x.y), bf16r(x.z), bf16r(x.w));
    }
    s[0] += v.x; s[1] += v.y; s[2] += v.z; s[3] += v.w;
  }
  float e[4];
  for (uint j = 0u; j < 4u; j++) e[j] = bf16r(s[j] * 0.25f);
  head_norm<4, IHD>(e, iknw, lane);
  head_rope<4>(e, invf, b * RATIO, lane);
  *(device uint2 *)(bkeys + (size_t)b * IHD + lane * 4u) = uint2(pack2bf(e[0], e[1]), pack2bf(e[2], e[3]));
}

// grid R*24 + R*2 + R*4 + (R+3)/4 threadgroups x 32 threads (one simdgroup per task):
//   [0, 24R)      q head (t, h): norm + rope + bf16 + 1/16 -> qr [R][24][256]
//   [24R, 26R)    kv head (t, g): k norm + rope + bf16, v bf16 -> cache row T + t
//   [26R, 30R)    indexer query (t, h): q_layernorm + rope -> iq [R][4][128]; h == 0 also stores the raw key
//   [30R, ...)    block keys of the blocks completed by tokens T .. T+R-1: b in [T/4, (T+R)/4)
// The tasks of qsa_prep for R rows at T (R may be a runtime count: the chunk prep). blocks = false: no block keys
// (qsa_chunk_blocks writes them after a barrier).
template <uint F>
static inline void prep_body(uint tg, uint R, uint T, bool blocks, device const float *q, device const float *kv,
                             device const float *iqk, device const ushort *qnw, device const ushort *knw,
                             device const ushort *iqnw, device const ushort *iknw, constant float *invf,
                             device uchar *kc, device uchar *vc, device ushort2 *ksb, device ushort2 *vsb,
                             device ushort *ikeys, device ushort *bkeys, uint tmax, device float *qr, device float *iq,
                             uint lane) {
  if (tg < R * NQH) {
    const uint t = tg / NQH, h = tg % NQH;
    float e[8];
    for (uint j = 0u; j < 8u; j++) e[j] = rr(q[t * QD + h * QROW + lane * 8u + j]);
    head_norm<8, HD>(e, qnw, lane);
    head_rope<8>(e, invf, T + t, lane);
    for (uint j = 0u; j < 8u; j++) qr[(t * NQH + h) * HD + lane * 8u + j] = bf16r(e[j]) * 0.0625f;
    return;
  }
  uint k = tg - R * NQH;
  if (k < R * NKV) {
    const uint t = k / NKV, g = k % NKV;
    float e[8], v[8];
    for (uint j = 0u; j < 8u; j++) {
      e[j] = rr(kv[t * 2u * KVD + g * HD + lane * 8u + j]);
      v[j] = bf16r(kv[t * 2u * KVD + KVD + g * HD + lane * 8u + j]);
    }
    head_norm<8, HD>(e, knw, lane);
    head_rope<8>(e, invf, T + t, lane);
    for (uint j = 0u; j < 8u; j++) e[j] = bf16r(e[j]);
    const size_t row = (size_t)g * tmax + T + t;
    kv_store<F>(e, kc, ksb, row, lane);
    kv_store<F>(v, vc, vsb, row, lane);
    return;
  }
  k -= R * NKV;
  if (k < R * IH) {
    const uint t = k / IH, h = k % IH;
    float e[4];
    for (uint j = 0u; j < 4u; j++) e[j] = bf16r(iqk[t * IQD + h * IHD + lane * 4u + j]);
    head_norm<4, IHD>(e, iqnw, lane);
    head_rope<4>(e, invf, T + t, lane);
    for (uint j = 0u; j < 4u; j++) iq[(t * IH + h) * IHD + lane * 4u + j] = bf16r(e[j]);
    if (h == 0u) {
      const float4 x = *(const device float4 *)(iqk + t * IQD + IH * IHD + lane * 4u);
      *(device uint2 *)(ikeys + (size_t)(T + t) * IHD + lane * 4u) = uint2(pack2bf(x.x, x.y), pack2bf(x.z, x.w));
    }
    return;
  }
  k -= R * IH;
  if (!blocks) return;
  const uint b = T / RATIO + k;
  if (b >= (T + R) / RATIO) return;
  block_key(b, T, iqk, ikeys, iknw, invf, bkeys, lane);
}

template <uint R, uint F>
kernel void qsa_prep(device const float *q [[buffer(0)]], device const float *kv [[buffer(1)]],
                     device const float *iqk [[buffer(2)]], device const ushort *qnw [[buffer(3)]],
                     device const ushort *knw [[buffer(4)]], device const ushort *iqnw [[buffer(5)]],
                     device const ushort *iknw [[buffer(6)]], constant float *invf [[buffer(7)]],
                     device uchar *kc [[buffer(8)]], device uchar *vc [[buffer(9)]], device ushort2 *ksb [[buffer(10)]],
                     device ushort2 *vsb [[buffer(11)]], device ushort *ikeys [[buffer(12)]],
                     device ushort *bkeys [[buffer(13)]], device const uint *pos [[buffer(14)]],
                     constant uint &tmax [[buffer(15)]], device float *qr [[buffer(16)]], device float *iq [[buffer(17)]],
                     uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  prep_body<F>(tg, R, pos[0], true, q, kv, iqk, qnw, knw, iqnw, iknw, invf, kc, vc, ksb, vsb, ikeys, bkeys, tmax, qr, iq,
               lane);
}

// The chunk's prep: qsa_prep's tasks for n rows at T = posbase[0] (grid 30 n x 32), no block keys.
template <uint F>
kernel void qsa_chunk_prep(device const float *q [[buffer(0)]], device const float *kv [[buffer(1)]],
                           device const float *iqk [[buffer(2)]], device const ushort *qnw [[buffer(3)]],
                           device const ushort *knw [[buffer(4)]], device const ushort *iqnw [[buffer(5)]],
                           device const ushort *iknw [[buffer(6)]], constant float *invf [[buffer(7)]],
                           device uchar *kc [[buffer(8)]], device uchar *vc [[buffer(9)]],
                           device ushort2 *ksb [[buffer(10)]], device ushort2 *vsb [[buffer(11)]],
                           device ushort *ikeys [[buffer(12)]], device ushort *bkeys [[buffer(13)]],
                           device const uint *posbase [[buffer(14)]], constant uint &tmax [[buffer(15)]],
                           device float *qr [[buffer(16)]], device float *iq [[buffer(17)]],
                           constant uint &n [[buffer(18)]], uint tg [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  prep_body<F>(tg, n, posbase[0], false, q, kv, iqk, qnw, knw, iqnw, iknw, invf, kc, vc, ksb, vsb, ikeys, bkeys, tmax,
               qr, iq, lane);
}

// Block keys of the blocks completed by the chunk's n rows, b in [T/4, (T+n)/4), from the cached raw keys (after
// qsa_chunk_prep): grid n/4 + 1 x 32. Same values as qsa_prep's (its staged keys are the bf16 of the same f32).
kernel void qsa_chunk_blocks(device const ushort *ikeys [[buffer(0)]], device const ushort *iknw [[buffer(1)]],
                             constant float *invf [[buffer(2)]], device ushort *bkeys [[buffer(3)]],
                             device const uint *posbase [[buffer(4)]], constant uint &n [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint T = posbase[0], b = T / RATIO + tg;
  if (b >= (T + n) / RATIO) return;
  block_key(b, 0xFFFFFFFFu, nullptr, ikeys, iknw, invf, bkeys, lane);
}

// Block keys b0 .. b1-1 from the cached raw keys (cache import / prefill hand-over): grid (b1 - b0) x 32.
kernel void qsa_build_blocks(device const ushort *ikeys [[buffer(0)]], device const ushort *iknw [[buffer(1)]],
                             constant float *invf [[buffer(2)]], device ushort *bkeys [[buffer(3)]],
                             constant uint &b0 [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  block_key(b0 + tg, 0xFFFFFFFFu, nullptr, ikeys, iknw, invf, bkeys, lane);
}

// Cache rows p0 .. p0+n-1 from f32 staging k, v [NKV][n][256] (already roped / bf16 values): grid NKV*n x 32.
template <uint F>
kernel void qsa_import(device const float *sk [[buffer(0)]], device const float *sv [[buffer(1)]],
                       device uchar *kc [[buffer(2)]], device uchar *vc [[buffer(3)]], device ushort2 *ksb [[buffer(4)]],
                       device ushort2 *vsb [[buffer(5)]], constant uint &p0 [[buffer(6)]], constant uint &n [[buffer(7)]],
                       constant uint &tmax [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg / n, i = tg % n;
  float e[8], v[8];
  for (uint j = 0u; j < 8u; j++) {
    e[j] = bf16r(sk[((size_t)g * n + i) * HD + lane * 8u + j]);
    v[j] = bf16r(sv[((size_t)g * n + i) * HD + lane * 8u + j]);
  }
  const size_t row = (size_t)g * tmax + p0 + i;
  kv_store<F>(e, kc, ksb, row, lane);
  kv_store<F>(v, vc, vsb, row, lane);
}

// Dequantized cache rows (tests): out [NKV][n][256] f32 for k and v. grid NKV*n x 32.
template <uint F>
kernel void qsa_export(device const uchar *kc [[buffer(0)]], device const uchar *vc [[buffer(1)]],
                       device const ushort2 *ksb [[buffer(2)]], device const ushort2 *vsb [[buffer(3)]],
                       device float *ok [[buffer(4)]], device float *ov [[buffer(5)]], constant uint &n [[buffer(6)]],
                       constant uint &tmax [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg / n, i = tg % n;
  const size_t row = (size_t)g * tmax + i;
  float4 a, b;
  kv_load<F>(kc, ksb, row, lane, a, b);
  device float4 *pk = (device float4 *)(ok + ((size_t)g * n + i) * HD + lane * 8u);
  pk[0] = a;
  pk[1] = b;
  kv_load<F>(vc, vsb, row, lane, a, b);
  device float4 *pv = (device float4 *)(ov + ((size_t)g * n + i) * HD + lane * 8u);
  pv[0] = a;
  pv[1] = b;
}

// grid G threadgroups x 256 threads, grid-stride over groups of 32 blocks; 8 lanes per block (16 dims each).
// scores [R][tmax/4] f32; only b < (T + R) / 4 written.
template <uint R>
kernel void qsa_score(device const float *iq [[buffer(0)]], device const ushort *bkeys [[buffer(1)]],
                      device const uint *pos [[buffer(2)]], device float *scores [[buffer(3)]],
                      constant uint &tmax [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                      uint ntg [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float4 qs[R * IH * IHD / 4];
  for (uint i = tid; i < R * IH * IHD / 4; i += 256u) qs[i] = ((const device float4 *)iq)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint T = pos[0], NB = (T + R) / RATIO, sl = tid & 7u, bl = tid >> 3u, nbs = tmax / RATIO;
  for (uint base = tg * 32u; base < NB; base += ntg * 32u) {
    const uint b = base + bl;
    float4 k[4] = {float4(0), float4(0), float4(0), float4(0)};
    if (b < NB) {
      const device uint4 *p = (const device uint4 *)(bkeys + (size_t)b * IHD + sl * 16u);
      const uint4 u0 = p[0], u1 = p[1];
      k[0] = bf4(u0.xy); k[1] = bf4(u0.zw); k[2] = bf4(u1.xy); k[3] = bf4(u1.zw);
    }
    for (uint t = 0u; t < R; t++) {
      float s = 0.0f;
      for (uint h = 0u; h < IH; h++) {
        const threadgroup float4 *qq = qs + (t * IH + h) * (IHD / 4u) + sl * 4u;
        float d = dot(k[0], qq[0]) + dot(k[1], qq[1]) + dot(k[2], qq[2]) + dot(k[3], qq[3]);
        d += simd_shuffle_xor(d, ushort(1));
        d += simd_shuffle_xor(d, ushort(2));
        d += simd_shuffle_xor(d, ushort(4));
        s += max(d, 0.0f);
      }
      if (sl == 0u && b < NB) scores[t * nbs + b] = precise::divide(s, ISQ128);
    }
  }
}

// The same scores on the simdgroup matrix units: S = IQ [4R x 128] . K^T [128 x nb] with bfloat operands (both are
// bf16 values: exact) and f32 accumulation, then relu, sum over the 4 heads of each row, / sqrt(128).
// grid G threadgroups x 256 threads (8 simdgroups x 8 blocks), grid-stride over groups of 64 blocks.
template <uint R>
static inline void score_mma_body(device const float *iq, device const bfloat *bkeys, uint T, device float *scores,
                                  uint tmax, uint tg, uint ntg, uint tid, uint sg, uint lane, threadgroup bfloat *qs,
                                  threadgroup float *cs, uint rows = R) {
  constexpr uint M = IH * R, MP = (M + 7u) / 8u * 8u, MT = MP / 8u;
  for (uint i = tid; i < MP * IHD; i += 256u) qs[i] = bfloat(i < M * IHD ? iq[i] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // blocks visible to the slice's real rows (a chunk's last slice can have rows < R); never past the cache, so the
  // 8-block simdgroup loads stay inside bkeys (tmax / 4 is a multiple of 16) and no score lands past column nbs
  const uint nbs = tmax / RATIO, NB = min((T + rows) / RATIO, nbs);
  for (uint base = tg * 64u; base < NB; base += ntg * 64u) {
    const uint b0 = base + sg * 8u;
    if (b0 >= NB) continue;  // uniform per simdgroup
    simdgroup_float8x8 acc[MT];
    for (uint m = 0u; m < MT; m++) acc[m] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < IHD / 8u; k++) {
      simdgroup_bfloat8x8 kb;
      simdgroup_load(kb, bkeys + (size_t)b0 * IHD + k * 8u, IHD, ulong2(0, 0), true);  // [dim][block]
      for (uint m = 0u; m < MT; m++) {
        simdgroup_bfloat8x8 qa;
        simdgroup_load(qa, qs + m * 8u * IHD + k * 8u, IHD);
        simdgroup_multiply_accumulate(acc[m], qa, kb, acc[m]);
      }
    }
    for (uint m = 0u; m < MT; m++) simdgroup_store(acc[m], cs + sg * MP * 8u + m * 64u, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint j = lane & 7u;
    for (uint t = lane >> 3u; t < R; t += 4u) {
      float s = 0.0f;
      for (uint h = 0u; h < IH; h++) s += max(cs[sg * MP * 8u + (t * IH + h) * 8u + j], 0.0f);
      if (b0 + j < NB) scores[t * nbs + b0 + j] = precise::divide(s, ISQ128);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}


template <uint R>
kernel void qsa_score_mma(device const float *iq [[buffer(0)]], device const bfloat *bkeys [[buffer(1)]],
                          device const uint *pos [[buffer(2)]], device float *scores [[buffer(3)]],
                          constant uint &tmax [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                          uint ntg [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint MP = (IH * R + 7u) / 8u * 8u;
  threadgroup bfloat qs[MP * IHD];
  threadgroup float cs[8 * MP * 8u];
  score_mma_body<R>(iq, bkeys, pos[0], scores, tmax, tg, ntg, tid, sg, lane, qs, cs);
}
// The chunk's block scores: 8-row slices s0 + y (grid G x nslices x 256), slice j at T = posbase[0] + 8 j with its
// rows' iq [8][4][128] at iq + 8 j and scores rows at 8 j: per element exactly qsa_score_mma<8>'s. n = the chunk's
// rows: the last slice sees only the blocks of its real rows (min(8, n - 8 j)).
kernel void qsa_chunk_score(device const float *iq [[buffer(0)]], device const bfloat *bkeys [[buffer(1)]],
                            device const uint *posbase [[buffer(2)]], device float *scores [[buffer(3)]],
                            constant uint &tmax [[buffer(4)]], constant uint &s0 [[buffer(5)]],
                            constant uint &n [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]], uint2 ntgp [[threadgroups_per_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat qs[32 * IHD];
  threadgroup float cs[8 * 32 * 8u];
  const uint j = s0 + tgp.y;
  score_mma_body<8>(iq + (size_t)j * 8u * IH * IHD, bkeys, posbase[0] + 8u * j, scores + (size_t)j * 8u * (tmax / RATIO),
                    tmax, tgp.x, ntgp.x, tid, sg, lane, qs, cs, min(8u, n - 8u * j));
}

// qsa_chunk_score on the MPP tensor op (pfkern, 18 Oct; NE_QSA_TSCORE=0: the simdgroup-matrix kernel): per 8-row
// slice the 32 (row, head) queries as a threadgroup tensor (bf16, as qsa_chunk_score stages them), one matmul2d
// 32 x 64 x 128 per 64 blocks against the block keys (device tensor), the 4 heads' ReLU sum in the epilogue through
// threadgroup memory. Same formula and staging precision; f32 summation order inside the op differs. grid G x
// nslices x 128.
kernel void qsa_chunk_score_t(device const float *iq [[buffer(0)]], device const bfloat *bkeys [[buffer(1)]],
                              device const uint *posbase [[buffer(2)]], device float *scores [[buffer(3)]],
                              constant uint &tmax [[buffer(4)]], constant uint &s0 [[buffer(5)]],
                              constant uint &n [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
                              uint2 ntgp [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  threadgroup bfloat qs[32 * IHD];
  threadgroup float cs[32 * 65];
  const uint j = s0 + tgp.y, rows = min(8u, n - 8u * j);
  const device float *q = iq + (size_t)j * 8u * IH * IHD;
  for (uint i = tid; i < 32u * IHD; i += 128u) qs[i] = bfloat(i < IH * rows * IHD ? q[i] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nbs = tmax / RATIO, NB = min((posbase[0] + 8u * j + rows) / RATIO, nbs);
  device float *sc = scores + (size_t)j * 8u * nbs;
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> A(qs, dextents<int32_t, 2>(IHD, 32));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B((device bfloat *)bkeys, dextents<int32_t, 2>(IHD, int(nbs)));
  constexpr auto desc = matmul2d_descriptor(32, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mB0 = B.slice(0, 0);
  auto ct = op.template get_destination_cooperative_tensor<decltype(A), decltype(mB0), float>();
  for (uint base = tgp.x * 64u; base < NB; base += ntgp.x * 64u) {
    auto mB = B.slice(0, int(base));
    op.run(A, mB, ct);
    for (int i = 0; i < 16; i++) {
      const auto ix = ct.get_multidimensional_index(i);
      cs[uint(ix[1]) * 65u + uint(ix[0])] = ct[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 8u * 64u; e += 128u) {
      const uint t = e / 64u, b = e % 64u;
      if (t >= rows || base + b >= NB) continue;
      float s2 = 0.0f;
      for (uint h = 0u; h < IH; h++) s2 += max(cs[(t * IH + h) * 65u + b], 0.0f);
      sc[t * nbs + base + b] = precise::divide(s2, ISQ128);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Monotone f32 -> u32 ordinal (ascending order kept; -0 and +0 equal; NaN above all), as Sushi's sushi_qsa_ord.
static inline uint ford(float v) {
  if (isnan(v)) return 0xFFFFFFFFu;
  if (v == 0.0f) return 0x80000000u;
  const uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Top-K selection over a candidate sequence (element i -> (ordinal, block id) or invalid), one threadgroup of STG
// threads: MSD radix select over the ordinal in 8-bit digits (run-length histogram adds; stops when a bin holds
// exactly what is still needed), then compaction in sequence order: selected iff ord > thr, or ord == thr among the
// first need_eq such. The sequence is ascending in block id, so the output is ascending and ties go to the lower id.
// Up to SCACHE * STG elements are kept in registers across the passes; longer sequences are re-read per pass.
constant constexpr uint STG = 1024u, SNS = STG / 32u, SCACHE = 8u, SENT = 0xFFFFFFFFu;
constant constexpr uint SLICE = 2048u;  // blocks per local-select slice (qsa_select_local)
struct ScoreSrc {  // scores[lo .. hi)
  const device float *sc;
  uint lo, hi;
  uint n() const { return hi - lo; }
  bool get(uint i, thread uint &u, thread uint &id) const {
    if (lo + i >= hi) return false;
    id = lo + i;
    u = ford(sc[id]);
    return true;
  }
};
struct CandSrc {  // candidate ids (SENT = empty slot), ascending among the valid ones
  const device float *sc;
  const device uint *cand;
  uint cnt;
  uint n() const { return cnt; }
  bool get(uint i, thread uint &u, thread uint &id) const {
    if (i >= cnt) return false;
    id = cand[i];
    if (id == SENT) return false;
    u = ford(sc[id]);
    return true;
  }
};
struct SelShared {
  threadgroup atomic_uint *hist;  // [256]
  threadgroup uint *sh;           // [4]
  threadgroup uint *sgs;          // [2 * SNS]
};

template <bool cached, class Src>
static inline void radix_topk_impl(thread const Src &src, uint nvalid, device uint *out, SelShared S, uint tid,
                                   uint lane, uint sg) {
  const uint n = src.n();
  uint cu[SCACHE], ci[SCACHE];
  if (cached) {
#pragma unroll
    for (uint k = 0u; k < SCACHE; k++) {
      uint u = 0u, id = SENT;
      if (!src.get(k * STG + tid, u, id)) id = SENT;
      cu[k] = u;
      ci[k] = id;
    }
  }
  uint thr = 0u, need_eq = 0xFFFFFFFFu;  // nvalid <= KB: every valid element is selected
  if (nvalid > KB) {
    uint pref = 0u, fixed = 0u, need = KB;
    for (uint lv = 0u; lv < 4u; lv++) {
      const uint shift = 24u - 8u * lv;
      for (uint b = tid; b < 256u; b += STG) atomic_store_explicit(&S.hist[b], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint last = 0xFFFFFFFFu, run = 0u;
      const uint steps = cached ? SCACHE : (n + STG - 1u) / STG;
#pragma unroll
      for (uint k = 0u; k < steps; k++) {
        uint u = 0u, id = SENT;
        if (cached) { u = cu[k]; id = ci[k]; }
        else if (!src.get(k * STG + tid, u, id)) id = SENT;
        if (id == SENT) continue;
        if (fixed != 0u && (u >> (32u - fixed)) != pref) continue;
        const uint d = (u >> shift) & 255u;
        if (d == last) run++;
        else {
          if (run) atomic_fetch_add_explicit(&S.hist[last], run, memory_order_relaxed);
          last = d;
          run = 1u;
        }
      }
      if (run) atomic_fetch_add_explicit(&S.hist[last], run, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {  // lane l owns bins [248 - 8l, 256 - 8l), scanned from the top
        const uint base = 256u - (lane + 1u) * 8u;
        uint tot = 0u;
        for (uint j = 0u; j < 8u; j++) tot += atomic_load_explicit(&S.hist[base + j], memory_order_relaxed);
        const uint pre = simd_prefix_exclusive_sum(tot);
        if (pre < need && need <= pre + tot) {
          uint acc = pre;
          for (uint jj = 8u; jj > 0u; jj--) {
            const uint bi = base + jj - 1u;
            const uint c = atomic_load_explicit(&S.hist[bi], memory_order_relaxed);
            if (acc + c >= need) { S.sh[0] = bi; S.sh[1] = acc; S.sh[2] = c; break; }
            acc += c;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint dsel = S.sh[0], above = S.sh[1], cnt = S.sh[2];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      need -= above;
      pref = (pref << 8u) | dsel;
      fixed += 8u;
      if (cnt == need || fixed == 32u) {
        thr = fixed == 32u ? pref : (pref << (32u - fixed));
        need_eq = need;
        break;
      }
    }
  }
  uint run_gt = 0u, run_eq = 0u, total = 0u;
  const uint steps = cached ? min(SCACHE, (n + STG - 1u) / STG) : (n + STG - 1u) / STG;
#pragma unroll
  for (uint k = 0u; k < (cached ? SCACHE : steps); k++) {
    if (k >= steps) break;
    uint u = 0u, id = SENT;
    if (cached) { u = cu[k]; id = ci[k]; }
    else if (!src.get(k * STG + tid, u, id)) id = SENT;
    const uint ok = id != SENT ? 1u : 0u;
    const uint gt = ok && (u > thr || nvalid <= KB) ? 1u : 0u;
    const uint eq = ok && !gt && u == thr ? 1u : 0u;
    const uint pg = simd_prefix_exclusive_sum(gt), pe = simd_prefix_exclusive_sum(eq);
    const uint sgg = simd_sum(gt), sge = simd_sum(eq);
    if (lane == 0u) { S.sgs[sg] = sgg; S.sgs[SNS + sg] = sge; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint a = lane < SNS ? S.sgs[lane] : 0u, bq = lane < SNS ? S.sgs[SNS + lane] : 0u;
    const uint pa = simd_prefix_exclusive_sum(a), pb = simd_prefix_exclusive_sum(bq);
    const uint off_g = simd_shuffle(pa, ushort(sg)), off_e = simd_shuffle(pb, ushort(sg));
    const uint tot_g = simd_sum(a), tot_e = simd_sum(bq);
    const uint gb = run_gt + off_g + pg, eb = run_eq + off_e + pe;
    if (gt) {
      const uint p = gb + min(eb, need_eq);
      if (p < KB) out[p] = id;
    } else if (eq && eb < need_eq) {
      const uint p = gb + eb;
      if (p < KB) out[p] = id;
    }
    run_gt += tot_g;
    run_eq += tot_e;
    total = run_gt + min(run_eq, need_eq);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (total >= KB) break;
  }
  for (uint i = min(total, KB) + tid; i < KB; i += STG) out[i] = SENT;
}
template <class Src>
static inline void radix_topk(thread const Src &src, uint nvalid, device uint *out, SelShared S, uint tid, uint lane,
                              uint sg) {
  if (src.n() <= SCACHE * STG) radix_topk_impl<true>(src, nvalid, out, S, tid, lane, sg);
  else radix_topk_impl<false>(src, nvalid, out, S, tid, lane, sg);
}

// grid R x NSL threadgroups x 1024 (NSL = slices of SLICE blocks in tmax/4): rows with comp > SLICE complete blocks
// keep the top 512 of each slice [s SLICE, (s+1) SLICE) in cand [R][NSL][512] (SENT-padded, ascending).
// DIR (pfkern, the chunk path's "_d" variants): rows with comp <= SCACHE * STG (8192 blocks, 32K tokens) select
// straight from their scores in qsa_select (the cached radix holds all of them in registers) and skip the per-slice
// pass. Same result: the top 512 of the slices' top 512s is the top 512, and both sequences ascend in block id
// (ties to the lower id).
template <uint R, bool DIR = false>
kernel void qsa_select_local(device const float *scores [[buffer(0)]], device const uint *pos [[buffer(1)]],
                             device uint *cand [[buffer(2)]], constant uint &tmax [[buffer(3)]],
                             uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint sh[4], sgs[2u * SNS];
  const uint t = tgp.x, s = tgp.y, nsl = (tmax / RATIO + SLICE - 1u) / SLICE;
  const uint T = pos[0], comp = (T + t + 1u) / RATIO, lo = s * SLICE;
  if (comp <= SLICE || lo >= comp || (DIR && comp <= SCACHE * STG)) return;
  const uint hi = min(lo + SLICE, comp);
  const ScoreSrc src{scores + (size_t)t * (tmax / RATIO), lo, hi};
  radix_topk(src, hi - lo, cand + ((size_t)t * nsl + s) * KB, SelShared{hist, sh, sgs}, tid, lane, sg);
}

// grid R threadgroups x 1024 threads: the block list of row t -> sel[t][0..n), nsel[t] = n. comp <= 512: blocks
// 0..comp-1 (causal); comp <= SLICE: top 512 of the scores; else top 512 of qsa_select_local's candidates.
template <uint R, bool DIR = false>
kernel void qsa_select(device const float *scores [[buffer(0)]], device const uint *pos [[buffer(1)]],
                       device uint *sel [[buffer(2)]], device uint *nsel [[buffer(3)]], constant uint &tmax [[buffer(4)]],
                       device const uint *cand [[buffer(5)]], uint t [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint sh[4], sgs[2u * SNS];
  const uint T = pos[0], comp = (T + t + 1u) / RATIO, nsl = (tmax / RATIO + SLICE - 1u) / SLICE;
  device uint *out = sel + t * KB;
  if (comp <= KB) {
    for (uint i = tid; i < comp; i += STG) out[i] = i;
    if (tid == 0u) nsel[t] = comp;
    return;
  }
  const device float *sc = scores + (size_t)t * (tmax / RATIO);
  if (comp <= SLICE || (DIR && comp <= SCACHE * STG)) {
    const ScoreSrc src{sc, 0u, comp};
    radix_topk(src, comp, out, SelShared{hist, sh, sgs}, tid, lane, sg);
  } else {
    const uint ns = (comp + SLICE - 1u) / SLICE;
    const uint last = comp - (ns - 1u) * SLICE;  // blocks in the last slice
    const CandSrc src{sc, cand + (size_t)t * nsl * KB, ns * KB};
    radix_topk(src, (ns - 1u) * KB + min(last, KB), out, SelShared{hist, sh, sgs}, tid, lane, sg);
  }
  if (tid == 0u) nsel[t] = KB;
}

// One cached K / V row pair as loaded (lane's 8 dims), per format; decoded at use so that the next block's rows can
// be in flight while the current one is computed.
template <uint F> struct KVRaw;
template <> struct KVRaw<KV_F32> { float4 k0, k1, v0, v1; };
template <> struct KVRaw<KV_BF16> { uint4 k, v; };
template <> struct KVRaw<KV_Q8> { uint2 k, v; ushort2 ks, vs; };
template <uint F>
static inline KVRaw<F> kv_raw(const device uchar *kc, const device uchar *vc, const device ushort2 *ksb,
                              const device ushort2 *vsb, size_t row, uint lane) {
  KVRaw<F> r;
  if constexpr (F == KV_F32) {
    const device float4 *pk = (const device float4 *)((const device float *)kc + row * HD + lane * 8u);
    const device float4 *pv = (const device float4 *)((const device float *)vc + row * HD + lane * 8u);
    r.k0 = pk[0]; r.k1 = pk[1]; r.v0 = pv[0]; r.v1 = pv[1];
  } else if constexpr (F == KV_BF16) {
    r.k = *(const device uint4 *)((const device ushort *)kc + row * HD + lane * 8u);
    r.v = *(const device uint4 *)((const device ushort *)vc + row * HD + lane * 8u);
  } else {
    r.k = *(const device uint2 *)(kc + row * HD + lane * 8u);
    r.v = *(const device uint2 *)(vc + row * HD + lane * 8u);
    r.ks = ksb[row * (HD / GQ) + lane / 8u];
    r.vs = vsb[row * (HD / GQ) + lane / 8u];
  }
  return r;
}
// lane's part of q . k (qs = sum of the lane's 8 q values, for the kv8 bias term)
template <uint F>
static inline float kv_dot(thread const KVRaw<F> &r, float4 q0, float4 q1, float qs) {
  if constexpr (F == KV_F32) return dot(q0, r.k0) + dot(q1, r.k1);
  else if constexpr (F == KV_BF16) return dot8bf(r.k, q0, q1);
  else return bf(r.ks.x) * (dot(float4(as_type<uchar4>(r.k.x)), q0) + dot(float4(as_type<uchar4>(r.k.y)), q1)) +
              bf(r.ks.y) * qs;
}
// a += p * v
template <uint F>
static inline void kv_acc(thread const KVRaw<F> &r, float p, thread float4 &a0, thread float4 &a1) {
  if constexpr (F == KV_F32) { a0 += p * r.v0; a1 += p * r.v1; }
  else if constexpr (F == KV_BF16) { a0 += p * bf4(r.v.xy); a1 += p * bf4(r.v.zw); }
  else {
    const float sp = p * bf(r.vs.x), pb = p * bf(r.vs.y);
    a0 += sp * float4(as_type<uchar4>(r.v.x)) + pb;
    a1 += sp * float4(as_type<uchar4>(r.v.y)) + pb;
  }
}

// Four simd_sums at once (keys j = 0..3 of a block): a butterfly that halves the values per lane on the first two
// levels (xor 16: lanes keep keys {0,1} or {2,3}; xor 8: one key per lane), 3 more levels, then 4 broadcasts:
// 10 shuffles instead of 20. Summation order differs from simd_sum (f32 rounding only).
static inline void sum4(thread const float *x, thread float *s, uint lane) {
  const bool h16 = (lane & 16u) != 0u, h8 = (lane & 8u) != 0u;
  const float y0 = (h16 ? x[2] : x[0]) + simd_shuffle_xor(h16 ? x[0] : x[2], ushort(16));
  const float y1 = (h16 ? x[3] : x[1]) + simd_shuffle_xor(h16 ? x[1] : x[3], ushort(16));
  float z = (h8 ? y1 : y0) + simd_shuffle_xor(h8 ? y0 : y1, ushort(8));
  z += simd_shuffle_xor(z, ushort(4));
  z += simd_shuffle_xor(z, ushort(2));
  z += simd_shuffle_xor(z, ushort(1));
  // key j = 2 * bit4 + bit3 of the lane
  s[0] = simd_shuffle(z, ushort(0));
  s[1] = simd_shuffle(z, ushort(8));
  s[2] = simd_shuffle(z, ushort(16));
  s[3] = simd_shuffle(z, ushort(24));
}

// grid NKV * R * NCH threadgroups x (QPG / HPS) * 32 threads: simdgroup j = q heads j*HPS .. j*HPS+HPS-1 of kv group g.
// Each K / V row is loaded and decoded once per simdgroup and used for its HPS heads (with one head per simdgroup the
// 12 re-reads of every row through L1 were the bound: 12 x 4 KB per block). TG (g, t, c): c < 64 = blocks
// sel[t][8c .. 8c+8) (exits if 8c >= nsel[t]); c = 64 = the tail [4 comp, T + t] (exits if empty). Per block of 4
// keys: 4 x HPS dot products, a butterfly reduction per head, one online-softmax update per head; the next block's
// rows are loaded before the current one is used. part [NKV][R][NCH][16][PW] (head slots 12..15 unused).
template <uint R, uint F, uint HPS, uint BCH>
kernel void qsa_attn(device const float *qr [[buffer(0)]], device const uchar *kc [[buffer(1)]],
                     device const uchar *vc [[buffer(2)]], device const ushort2 *ksb [[buffer(3)]],
                     device const ushort2 *vsb [[buffer(4)]], device const uint *sel [[buffer(5)]],
                     device const uint *nsel [[buffer(6)]], device const uint *pos [[buffer(7)]],
                     constant uint &tmax [[buffer(8)]], device float *part [[buffer(9)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NCHB = KB / BCH, NCH = NCHB + 1u;
  const uint c = tg % NCH, t = (tg / NCH) % R, g = tg / (NCH * R);
  const uint T = pos[0], end = T + t + 1u, comp = end / RATIO, n = nsel[t];
  uint b0 = 0u, b1 = 0u;
  if (c < NCHB) {
    b0 = c * BCH;
    if (b0 >= n) return;
    b1 = min(b0 + BCH, n);
  } else if (comp * RATIO >= end) return;
  float4 q0[HPS], q1[HPS], a0[HPS], a1[HPS];
  float qs[HPS], m[HPS], l[HPS];
  for (uint k = 0u; k < HPS; k++) {
    const uint h = g * QPG + sg * HPS + k;
    const device float4 *qp = (const device float4 *)(qr + (t * NQH + h) * HD + lane * 8u);
    q0[k] = qp[0];
    q1[k] = qp[1];
    qs[k] = dot(q0[k] + q1[k], float4(1.0f));
    m[k] = -INFINITY;
    l[k] = 0.0f;
    a0[k] = float4(0.0f);
    a1[k] = float4(0.0f);
  }
  const size_t rbase = (size_t)g * tmax;
  if (c < NCHB) {
    const device uint *sl = sel + t * KB;
    uint blk[BCH];  // all block ids up front: one load latency, not one per block
    for (uint i = 0u; i < BCH; i++) blk[i] = b0 + i < b1 ? sl[b0 + i] : 0u;
    KVRaw<F> cur[4], nxt[4];
    size_t r0 = rbase + (size_t)blk[0] * RATIO;
    for (uint j = 0u; j < 4u; j++) cur[j] = kv_raw<F>(kc, vc, ksb, vsb, r0 + j, lane);
    for (uint i = b0; i < b1; i++) {
      if (i + 1u < b1) {
        r0 = rbase + (size_t)blk[i + 1u - b0] * RATIO;
        for (uint j = 0u; j < 4u; j++) nxt[j] = kv_raw<F>(kc, vc, ksb, vsb, r0 + j, lane);
      }
      for (uint k = 0u; k < HPS; k++) {
        float x[4], s[4];
        for (uint j = 0u; j < 4u; j++) x[j] = kv_dot<F>(cur[j], q0[k], q1[k], qs[k]);
        sum4(x, s, lane);
        const float mn = max(max(m[k], max(s[0], s[1])), max(s[2], s[3]));
        const float cs = exp(m[k] - mn);
        float p[4];
        for (uint j = 0u; j < 4u; j++) p[j] = exp(s[j] - mn);
        l[k] = l[k] * cs + ((p[0] + p[1]) + (p[2] + p[3]));
        a0[k] *= cs;
        a1[k] *= cs;
        for (uint j = 0u; j < 4u; j++) kv_acc<F>(cur[j], p[j], a0[k], a1[k]);
        m[k] = mn;
      }
      for (uint j = 0u; j < 4u; j++) cur[j] = nxt[j];
    }
  } else {
    for (uint j = comp * RATIO; j < end; j++) {
      const KVRaw<F> r = kv_raw<F>(kc, vc, ksb, vsb, rbase + j, lane);
      for (uint k = 0u; k < HPS; k++) {
        const float s = simd_sum(kv_dot<F>(r, q0[k], q1[k], qs[k]));
        const float mn = max(m[k], s);
        const float cs = exp(m[k] - mn), p = exp(s - mn);
        l[k] = l[k] * cs + p;
        a0[k] *= cs;
        a1[k] *= cs;
        kv_acc<F>(r, p, a0[k], a1[k]);
        m[k] = mn;
      }
    }
  }
  for (uint k = 0u; k < HPS; k++) {
    device float *pr = part + ((((size_t)g * R + t) * NCH + c) * HSL + sg * HPS + k) * PW;
    ((device float4 *)(pr + lane * 8u))[0] = a0[k];
    ((device float4 *)(pr + lane * 8u))[1] = a1[k];
    if (lane == 0u) { pr[HD] = m[k]; pr[HD + 1u] = l[k]; }
  }
}

// The same chunk partials on the simdgroup matrix units:
//   S [16 x 32] = Q [16 x 256] . K^T   (Q = the kv group's 12 q heads + 4 zero rows; K = the chunk's 32 keys)
//   P = exp(S - rowmax), l = rowsum P;  O [16 x 256] = P . V        (f32 accumulation)
// Staged in threadgroup memory: Q and K as KT = bfloat for the bf16 cache (bit copies: exact products) or half for
// kv8 (dequantized s * c + b rounded to 11 bits, finer than the 8-bit codes); P and V as half (P rel 2^-11; bf16 V
// values are exact in half within its normal range). TG = (g, t, c) as qsa_attn, 4 simdgroups: S key tile =
// simdgroup, softmax rows sg*4 .. sg*4+3 (lane = key), O dims sg*64 .. sg*64+63.
template <uint F> struct StageT { typedef half T; };
template <> struct StageT<KV_BF16> { typedef bfloat T; };
// stage the lane's 64 values (key j, dims qd*64 ..) of cache row `row` (or zeros) into dst, converted to DT
template <uint F, class DT>
static inline void stage64(threadgroup DT *dst, const device uchar *src, const device ushort2 *sb, size_t row, bool ok,
                           uint qd) {
  if (!ok) {
    for (uint e = 0u; e < 64u; e++) dst[e] = DT(0.0f);
  } else if (F == KV_BF16 && is_same<DT, bfloat>::value) {
    const device uint4 *p = (const device uint4 *)((const device ushort *)src + row * HD + qd * 64u);
    threadgroup uint4 *o = (threadgroup uint4 *)dst;
    for (uint e = 0u; e < 8u; e++) o[e] = p[e];
  } else if (F == KV_BF16) {
    const device uint4 *p = (const device uint4 *)((const device ushort *)src + row * HD + qd * 64u);
    for (uint e = 0u; e < 8u; e++) {
      const uint4 u = p[e];
      const float4 x = bf4(u.xy), y = bf4(u.zw);
      dst[8u * e] = DT(x.x); dst[8u * e + 1u] = DT(x.y); dst[8u * e + 2u] = DT(x.z); dst[8u * e + 3u] = DT(x.w);
      dst[8u * e + 4u] = DT(y.x); dst[8u * e + 5u] = DT(y.y); dst[8u * e + 6u] = DT(y.z); dst[8u * e + 7u] = DT(y.w);
    }
  } else {
    const device uchar4 *p = (const device uchar4 *)(src + row * HD + qd * 64u);
    const ushort2 s2 = sb[row * (HD / GQ) + qd];
    const float sc = bf(s2.x), bi = bf(s2.y);
    for (uint e = 0u; e < 16u; e++) {
      const float4 x = float4(p[e]) * sc + bi;
      dst[4u * e] = DT(x.x); dst[4u * e + 1u] = DT(x.y); dst[4u * e + 2u] = DT(x.z); dst[4u * e + 3u] = DT(x.w);
    }
  }
}
template <uint R, uint F, uint BCH>
kernel void qsa_attn_mma(device const float *qr [[buffer(0)]], device const uchar *kc [[buffer(1)]],
                         device const uchar *vc [[buffer(2)]], device const ushort2 *ksb [[buffer(3)]],
                         device const ushort2 *vsb [[buffer(4)]], device const uint *sel [[buffer(5)]],
                         device const uint *nsel [[buffer(6)]], device const uint *pos [[buffer(7)]],
                         constant uint &tmax [[buffer(8)]], device float *part [[buffer(9)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  typedef typename StageT<F>::T KT;
  constexpr uint NCHB = KB / BCH, NCH = NCHB + 1u, NK = BCH * RATIO, NKT = NK / 8u;  // keys, key tiles per chunk
  threadgroup KT qs[16 * HD];       // 8 KB
  threadgroup ushort kvraw[NK * HD];  // 16 KB: K (KT), then V (half)
  threadgroup float ss[16 * NK];
  threadgroup half ps[16 * NK];
  threadgroup float sm[16], sl[16];
  threadgroup KT *ks = (threadgroup KT *)kvraw;
  threadgroup half *vs = (threadgroup half *)kvraw;
  const uint c = tg % NCH, t = (tg / NCH) % R, g = tg / (NCH * R);
  const uint T = pos[0], end = T + t + 1u, comp = end / RATIO, n = nsel[t];
  uint b0 = 0u, nk = 0u;
  if (c < NCHB) {
    b0 = c * BCH;
    if (b0 >= n) return;
    nk = (min(b0 + BCH, n) - b0) * RATIO;
  } else {
    if (comp * RATIO >= end) return;
    nk = end - comp * RATIO;
  }
  for (uint i = tid; i < 16u * HD; i += 128u) {  // Q (12 heads of group g, row t); rows 12..15 zero
    const uint hh = i / HD, d = i % HD;
    qs[i] = KT(hh < QPG ? qr[(t * NQH + g * QPG + hh) * HD + d] : 0.0f);
  }
  // staging: item i = (key j = i / 4, dims qd*64 .. qd*64+63 = kv8 group qd), NK * 4 items over 128 threads
  constexpr uint NIT = (NK * 4u + 127u) / 128u;
  size_t rows[NIT];
  for (uint u = 0u; u < NIT; u++) {
    const uint i = tid + 128u * u, j = i / 4u;
    rows[u] = 0;
    if (j < nk) rows[u] = (size_t)g * tmax + (c < NCHB ? (size_t)sel[t * KB + b0 + j / RATIO] * RATIO + j % RATIO : comp * RATIO + j);
  }
  for (uint u = 0u; u < NIT; u++) {
    const uint i = tid + 128u * u, j = i / 4u, qd = i % 4u;
    if (j < NK) stage64<F, KT>(ks + j * HD + qd * 64u, kc, ksb, rows[u], j < nk, qd);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tile = sg; tile < 2u * NKT; tile += 4u) {  // S tiles (row tile rm, key tile kt)
    const uint rm = tile / NKT, kt = tile % NKT;
    simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<KT, 8, 8> a0, kb;
    for (uint k = 0u; k < HD / 8u; k++) {
      simdgroup_load(kb, ks + kt * 8u * HD + k * 8u, HD, ulong2(0, 0), true);  // [dim][key]
      simdgroup_load(a0, qs + rm * 8u * HD + k * 8u, HD);
      simdgroup_multiply_accumulate(s0, a0, kb, s0);
    }
    simdgroup_store(s0, ss + rm * 8u * NK + kt * 8u, NK);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint u = 0u; u < NIT; u++) {  // V (K no longer needed)
    const uint i = tid + 128u * u, j = i / 4u, qd = i % 4u;
    if (j < NK) stage64<F, half>(vs + j * HD + qd * 64u, vc, vsb, rows[u], j < nk, qd);
  }
  for (uint r = sg * 4u; r < sg * 4u + 4u; r++) {  // softmax rows, lane = key
    const bool v = lane < nk && r < QPG;
    const float x = v ? ss[r * NK + min(lane, NK - 1u)] : -INFINITY;
    const float m = simd_max(x);
    const float p = v ? exp(x - m) : 0.0f;
    const half ph = half(p);
    const float l = simd_sum(float(ph));  // the weights the MMA uses
    if (lane < NK) ps[r * NK + lane] = ph;
    if (lane == 0u) { sm[r] = m; sl[r] = l; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 o[2][8];
  for (uint a = 0u; a < 2u; a++)
    for (uint d = 0u; d < 8u; d++) o[a][d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k = 0u; k < NK / 8u; k++) {
    simdgroup_half8x8 p0, p1, vb;
    simdgroup_load(p0, ps + k * 8u, NK);
    simdgroup_load(p1, ps + 8u * NK + k * 8u, NK);
    for (uint d = 0u; d < 8u; d++) {
      simdgroup_load(vb, vs + k * 8u * HD + sg * 64u + d * 8u, HD);  // [key][dim]
      simdgroup_multiply_accumulate(o[0][d], p0, vb, o[0][d]);
      simdgroup_multiply_accumulate(o[1][d], p1, vb, o[1][d]);
    }
  }
  device float *pb = part + (((size_t)g * R + t) * NCH + c) * HSL * PW;
  for (uint d = 0u; d < 8u; d++) {
    simdgroup_store(o[0][d], pb + sg * 64u + d * 8u, PW);
    simdgroup_store(o[1][d], pb + 8u * PW + sg * 64u + d * 8u, PW);
  }
  if (tid < QPG) { pb[tid * PW + HD] = sm[tid]; pb[tid * PW + HD + 1u] = sl[tid]; }
}

// grid R * 24 threadgroups x 256 threads: merge row t / head h's chunk partials, * sigmoid(gate). Simdgroup s takes
// chunks s, s + 8, ... (lane = 8 dims) and merges them in order; then thread = dim merges the 8 simdgroups in order
// (fixed order: deterministic). The tail partial (chunk 64) goes to simdgroup 0 last.
template <uint R, uint BCH>
kernel void qsa_merge(device const float *q [[buffer(0)]], device const float *part [[buffer(1)]],
                      device const uint *nsel [[buffer(2)]], device const uint *pos [[buffer(3)]],
                      device float *out [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NCHB = KB / BCH, NCH = NCHB + 1u;
  threadgroup float so[8][HD];
  threadgroup float sm[8], sl[8];
  const uint t = tg / NQH, h = tg % NQH, g = h / QPG, hh = h % QPG;
  const uint T = pos[0], end = T + t + 1u, comp = end / RATIO;
  const uint nc = (nsel[t] + BCH - 1u) / BCH;
  const bool tail = comp * RATIO < end;
  const device float *pb = part + (((size_t)g * R + t) * NCH) * HSL * PW + hh * PW;
  float M = -INFINITY, L = 0.0f;
  float4 o0 = float4(0.0f), o1 = float4(0.0f);
  const uint nk = nc > sg ? (nc - sg + 7u) / 8u : 0u;  // chunks sg, sg + 8, ... < nc; simdgroup 0 also the tail
  const uint nall = nk + (tail && sg == 0u ? 1u : 0u);
  for (uint k = 0u; k < nall; k++) {
    const uint c = k < nk ? sg + 8u * k : NCHB;
    const device float *pr = pb + c * HSL * PW;
    const float mc = pr[HD], lc = pr[HD + 1u];
    const float4 x0 = ((const device float4 *)(pr + lane * 8u))[0], x1 = ((const device float4 *)(pr + lane * 8u))[1];
    const float mn = max(M, mc);
    const float w0 = exp(M - mn), w1 = exp(mc - mn);
    o0 = o0 * w0 + x0 * w1;
    o1 = o1 * w0 + x1 * w1;
    L = L * w0 + lc * w1;
    M = mn;
  }
  ((threadgroup float4 *)(&so[sg][lane * 8u]))[0] = o0;
  ((threadgroup float4 *)(&so[sg][lane * 8u]))[1] = o1;
  if (lane == 0u) { sm[sg] = M; sl[sg] = L; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float MM = -INFINITY;
  for (uint s2 = 0u; s2 < 8u; s2++) MM = max(MM, sm[s2]);
  float o = 0.0f, LL = 0.0f;
  for (uint s2 = 0u; s2 < 8u; s2++) {
    if (sm[s2] == -INFINITY) continue;
    const float w = exp(sm[s2] - MM);
    o += so[s2][lid] * w;
    LL += sl[s2] * w;
  }
  const float gate = q[t * QD + h * QROW + HD + lid];
  // the reference: bf16 core * sigmoid(bf16 gate) as an uncompiled bf16 MLX op chain, 1 / (1 + exp(-x)) with every
  // step rounded (rel 1.2e-3 vs 3.4e-3 for one rounding of an f32 sigmoid, measured on the L03 goldens)
  const float gq = rr(gate);
  const float sg2 = NE_REFR ? bf16r(1.0f / bf16r(1.0f + bf16r(precise::exp(-gq)))) : sigm(gq);
  out[t * NQH * HD + h * HD + lid] = rr(rr(o / LL) * sg2);
}

// pos[0] = pos[0] - rows + accepted[0] (MTP: after a verify step that advanced pos by rows); grid 1 x 1.
kernel void qsa_pos_rollback(device uint *pos [[buffer(0)]], device const uint *accepted [[buffer(1)]],
                             constant uint &rows [[buffer(2)]]) {
  pos[0] = pos[0] - rows + accepted[0];
}
// pos[0] += n; grid 1 x 1.
kernel void qsa_pos_add(device uint *pos [[buffer(0)]], constant uint &n [[buffer(1)]]) { pos[0] += n; }

#define QSA_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define QSA_INST_RF(R, F, FN)                                \
  QSA_INST("qsa_prep_" FN "_r" #R, qsa_prep, R, F)           \
  QSA_INST("qsa_attn_" FN "_r" #R, qsa_attn, R, F, (F == KV_Q8 ? 2 : 1), 8)
#define QSA_INST_R(R)                                        \
  QSA_INST("qsa_attn_mma_bf16_r" #R, qsa_attn_mma, R, KV_BF16, 8) \
  QSA_INST("qsa_attn_mma_kv8_r" #R, qsa_attn_mma, R, KV_Q8, 8)    \
  QSA_INST_RF(R, KV_F32, "f32")                              \
  QSA_INST_RF(R, KV_BF16, "bf16")                            \
  QSA_INST_RF(R, KV_Q8, "kv8")                               \
  QSA_INST("qsa_score_r" #R, qsa_score, R)                   \
  QSA_INST("qsa_score_mma_r" #R, qsa_score_mma, R)           \
  QSA_INST("qsa_select_r" #R, qsa_select, R)                 \
  QSA_INST("qsa_select_local_r" #R, qsa_select_local, R)     \
  QSA_INST("qsa_merge_r" #R, qsa_merge, R, 8)
QSA_INST("qsa_select_r8_d", qsa_select, 8, true)
QSA_INST("qsa_select_local_r8_d", qsa_select_local, 8, true)
QSA_INST_R(1)
QSA_INST_R(2)
QSA_INST_R(3)
QSA_INST_R(4)
QSA_INST_R(5)
QSA_INST_R(6)
QSA_INST_R(7)
QSA_INST_R(8)
QSA_INST("qsa_import_f32", qsa_import, KV_F32)
QSA_INST("qsa_import_bf16", qsa_import, KV_BF16)
QSA_INST("qsa_import_kv8", qsa_import, KV_Q8)
QSA_INST("qsa_export_f32", qsa_export, KV_F32)
QSA_INST("qsa_export_bf16", qsa_export, KV_BF16)
QSA_INST("qsa_export_kv8", qsa_export, KV_Q8)
QSA_INST("qsa_chunk_prep_f32", qsa_chunk_prep, KV_F32)
QSA_INST("qsa_chunk_prep_bf16", qsa_chunk_prep, KV_BF16)
QSA_INST("qsa_chunk_prep_kv8", qsa_chunk_prep, KV_Q8)



