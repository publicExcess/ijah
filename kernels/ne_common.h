// Ijah kernels: shared Metal helpers (8-bit affine dots, bf16, H128, EXL3 trellis lane reader).
//
// Attribution: the trellis lane reader (lane_load / lane_funnel / cw_at) and the q8 dot are lab lead 4's
// (the lab tree's engine/fused/fused.metal, from lead 1's engine/roofline/kernels.py); the MCG
// decode rule, the tensor-core slot mapping and the simd-shuffle FWHT follow sushi (github.com/beamivalice/sushi,
// MIT, (c) 2026 Theinruj Toranavikrai, David Dalcu) and exllamav3 (turboderp, MIT). MIT notice: permission is
// hereby granted, free of charge, to any person obtaining a copy of this software ... THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND (full text in each upstream LICENSE).
#pragma once
#include <metal_stdlib>
using namespace metal;

#define NE_HADS 0.08838834764831845f  // 1/sqrt(128)

static inline float bf(ushort b) { return as_type<float>(uint(b) << 16u); }

// Reference rounding points (function constant 0, default ON when not supplied): the MLX reference keeps its
// activations in bf16; where that rounding visibly moves a result (HC inject logits ~ -20 -> 2 sigmoid(p/4); the GDN
// gate chain a -> a + dt_bias -> softplus; b; the HC down logits) the kernels round to bf16 too, so goldens validate
// tightly. OFF = the plain f32 path (to be judged against the bf16 teacher by KLD).
constant bool NE_REFR_FC [[function_constant(0)]];
constant bool NE_REFR = is_function_constant_defined(NE_REFR_FC) ? NE_REFR_FC : true;
static inline float bf16r(float x) {  // round to the nearest bf16 (ties to even), back to f32
  const uint u = as_type<uint>(x);
  return as_type<float>((u + 0x7FFFu + ((u >> 16u) & 1u)) & 0xFFFF0000u);
}
static inline float rr(float x) { return NE_REFR ? bf16r(x) : x; }
static inline float silu(float v) { return v / (1.0f + exp(-v)); }
static inline float sigm(float v) { return 1.0f / (1.0f + exp(-v)); }

// ---------------------------------------------------------------- 8-bit affine (MLX layout)
// weight bytes row-major [rows][IN] (MLX packs 4 bytes per u32 little-endian = byte order), scales / biases bf16
// [rows][IN/G]; w = scale * q + bias.
// 16 q8 bytes . 16 floats -> (sum q*x, sum x)
static inline float2 dot16q(uint4 u, float4 x0, float4 x1, float4 x2, float4 x3) {
  const float qx = dot(float4(as_type<uchar4>(u.x)), x0) + dot(float4(as_type<uchar4>(u.y)), x1) +
                   dot(float4(as_type<uchar4>(u.z)), x2) + dot(float4(as_type<uchar4>(u.w)), x3);
  return float2(qx, dot(x0 + x1 + x2 + x3, float4(1.0f)));
}
static inline float2 dot8q(uint2 u, float4 x0, float4 x1) {
  const float qx = dot(float4(as_type<uchar4>(u.x)), x0) + dot(float4(as_type<uchar4>(u.y)), x1);
  return float2(qx, dot(x0 + x1, float4(1.0f)));
}
// 8 bf16 . 8 floats
static inline float dot8bf(uint4 u, float4 x0, float4 x1) {
  const float4 a = float4(as_type<float>(u.x << 16u), as_type<float>(u.x & 0xFFFF0000u), as_type<float>(u.y << 16u),
                          as_type<float>(u.y & 0xFFFF0000u));
  const float4 b = float4(as_type<float>(u.z << 16u), as_type<float>(u.z & 0xFFFF0000u), as_type<float>(u.w << 16u),
                          as_type<float>(u.w & 0xFFFF0000u));
  return dot(a, x0) + dot(b, x1);
}
static inline float4 bf4(uint2 u) {
  return float4(as_type<float>(u.x << 16u), as_type<float>(u.x & 0xFFFF0000u), as_type<float>(u.y << 16u),
                as_type<float>(u.y & 0xFFFF0000u));
}

// One q8 row of IN inputs (IN % 512 == 0 not required: IN % 16 == 0), G = group size (64 or 128), x f32 in device
// memory (all simdgroups read the same x: it stays in cache). One simdgroup; returns the full dot on every lane.
template <uint IN, uint G>
static inline float q8row_dev(const device uchar *w, const device ushort *sc, const device ushort *bi,
                              const device float *x, uint lane) {
  const device uint4 *w4 = (const device uint4 *)w;
  const device float4 *x4 = (const device float4 *)x;
  float acc = 0.0f;
  for (uint c = lane; c < IN / 16u; c += 32u) {
    const uint g = (16u * c) / G;
    const float2 d = dot16q(w4[c], x4[4u * c], x4[4u * c + 1u], x4[4u * c + 2u], x4[4u * c + 3u]);
    acc += bf(sc[g]) * d.x + bf(bi[g]) * d.y;
  }
  return simd_sum(acc);
}

// R rows at once: x [R][IN] (row stride IN), each weight chunk loaded once; out[r] = full dot on every lane.
template <uint IN, uint G, uint R>
static inline void q8rows_dev(const device uchar *w, const device ushort *sc, const device ushort *bi,
                              const device float *x, uint lane, thread float *out) {
  const device uint4 *w4 = (const device uint4 *)w;
  float acc[R];
  for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
  for (uint c = lane; c < IN / 16u; c += 32u) {
    const uint g = (16u * c) / G;
    const float s = bf(sc[g]), b = bf(bi[g]);
    const uint4 u = w4[c];
    for (uint r = 0u; r < R; r++) {
      const device float4 *x4 = (const device float4 *)(x + r * IN);
      const float2 d = dot16q(u, x4[4u * c], x4[4u * c + 1u], x4[4u * c + 2u], x4[4u * c + 3u]);
      acc[r] += s * d.x + b * d.y;
    }
  }
  for (uint r = 0u; r < R; r++) out[r] = simd_sum(acc[r]);
}

// ---------------------------------------------------------------- H128
// normalised H128 of a 128-block held as (lane, lane+32, lane+64, lane+96); returns the same element order
static inline float4 fwht128(float4 v, uint lane) {
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float4 p = float4(simd_shuffle_xor(v.x, bit), simd_shuffle_xor(v.y, bit), simd_shuffle_xor(v.z, bit),
                            simd_shuffle_xor(v.w, bit));
    v = ((lane & bit) == 0u) ? v + p : p - v;
  }
  const float s0 = v.x + v.y, s1 = v.x - v.y, s2 = v.z + v.w, s3 = v.z - v.w;
  return float4(s0 + s2, s1 + s3, s0 - s2, s1 - s3) * NE_HADS;
}

// ---------------------------------------------------------------- EXL3 trellis (MCG, window 15), 1 row
// Tile = 256 weights in 16K bits (8K u32 words; the pack's u16 halfwords read as little-endian u32 = Sushi's
// wordU32). A lane decodes 8 weights of a tile: rows (2q, 2q+1, 2q+8, 2q+9) x col c (w0) and the same rows x col
// c+8 (w1), q = lane & 3, c = lane >> 2 (the tensor-core slot order).
template <int K>
static inline uint3 lane_load(const device uint *wp, uint lane) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint TOT = 256u * uint(K);
  constexpr uint S0 = TOT + uint(K) - 16u;
  uint s = 8u * uint(K) * lane + S0;
  s = (s >= TOT) ? s - TOT : s;
  const uint i0 = s >> 5u;
  const uint i1 = (i0 + 1u >= NW) ? i0 + 1u - NW : i0 + 1u;
  if ((8u * uint(K)) % 32u == 0u) return uint3(wp[i0], wp[i1], 0u);
  const uint i2 = (i1 + 1u >= NW) ? i1 + 1u - NW : i1 + 1u;
  return uint3(wp[i0], wp[i1], wp[i2]);
}
template <int K>
static inline uint2 lane_funnel(uint3 w, uint lane) {
  if ((8u * uint(K)) % 32u == 0u) return w.xy;
  constexpr uint TOT = 256u * uint(K);
  const uint a = (8u * uint(K) * lane + TOT + uint(K) - 16u) & 31u;
  return uint2((w.x << a) | ((w.y >> 1u) >> (31u - a)), (w.y << a) | ((w.z >> 1u) >> (31u - a)));
}
template <int K>
static inline constexpr uint win_off() {
  return ((8u * uint(K)) % 32u == 0u) ? ((256u * uint(K) + uint(K) - 16u) & 31u) : 0u;
}
static inline uint cw_at(uint2 v, uint E) {
  const uint c = 64u - E;
  if (c >= 32u) return (v.x >> (c - 32u)) & 0xFFFFu;
  if (c <= 16u) return (v.y >> c) & 0xFFFFu;
  return ((v.x << (32u - c)) | (v.y >> c)) & 0xFFFFu;
}
static inline float2 dec_mcg(uint2 cw) {  // Sushi MCG, window 15
  cw &= uint2(0x7FFFu);
  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
  const half4 h = as_type<half4>(r);
  return float2(half2(h.x + h.y, h.z + h.w));
}
template <int K>
static inline void dec8(uint3 ww, uint lane, thread float4 &w0, thread float4 &w1) {
  const uint2 v = lane_funnel<K>(ww, lane);
  constexpr uint OFF = win_off<K>();
  float wv[8];
  for (uint p = 0u; p < 4u; p++) {
    const float2 d = dec_mcg(uint2(cw_at(v, OFF + 16u + (2u * p) * uint(K)), cw_at(v, OFF + 16u + (2u * p + 1u) * uint(K))));
    wv[2u * p] = d.x;
    wv[2u * p + 1u] = d.y;
  }
  w0 = float4(wv[0], wv[1], wv[2], wv[3]);
  w1 = float4(wv[4], wv[5], wv[6], wv[7]);
}
// position of input k inside the per-tile lane order (2q, 2q+1, 2q+8, 2q+9): a lane reads one half4 per tile
static inline uint kp(uint p) {
  return (p & ~15u) | (((p & 7u) >> 1u) << 2u) | (p & 1u) | (((p >> 3u) & 1u) << 1u);
}
