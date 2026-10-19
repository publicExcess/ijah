// Flash-Next full-attention (QSA layers 3, 7, ...) decode kernels at R rows, dense path. Owner: lab-layer.
//   fn_attn_proj : q_proj (24 heads x [256 query | 256 output gate]), k_proj, v_proj (2 kv heads x 256) and the QSA
//                  indexer's index_qk_proj (4 query heads + 1 raw key, x 128) in one 8-bit affine GEMV dispatch; the
//                  raw indexer key of each row is appended to the index-key cache (f32 [Tmax][128]).
//   fn_attn_core : one threadgroup per query head: q / k rmsnorm per head (q_norm / k_norm, ones-centred pack weights:
//                  multiplied directly), partial rope (first 64 of 256 dims, half-split pairs (d, d+32), inv_freq
//                  1e7^(-2d/64), native: no YaRN), the new k / v appended to the KV cache (f32 [2][Tmax][256], keys
//                  stored roped; written by the first q head of each kv group), causal softmax(q k^T / 16) v over
//                  [0, T + t] for row t, times sigmoid(gate).
// Then out_proj + HC write is fn_gdn_outproj (same 6144 -> 2560 shape). pos[0] = T = tokens in the cache before this
// step (a GPU buffer, so a recording replays; the host advances it). The QSA block keys / sparse selection
// (> 512 complete blocks) are not implemented: see docs/kernels.md.
#include "ne_common.h"

constant constexpr uint HID = 2560u;
constant constexpr uint NQH = 24u, NKV = 2u, HD = 256u, QROW = 2u * HD;  // q_proj row block per head: q | gate
constant constexpr uint QD = NQH * QROW, KVD = NKV * HD, IQD = 5u * 128u;   // 12288, 512, 640
constant constexpr uint ROT = 64u;                                          // rotated dims per head

// grid ceil((QD + 2 KVD + IQD) / 8) threadgroups x 256 threads; one weight row per simdgroup, R input rows.
template <uint G, uint R>
kernel void fn_attn_proj(device const float *x [[buffer(0)]],
                         device const uchar *qw [[buffer(1)]], device const ushort *qs [[buffer(2)]], device const ushort *qb [[buffer(3)]],
                         device const uchar *kw [[buffer(4)]], device const ushort *ks [[buffer(5)]], device const ushort *kb [[buffer(6)]],
                         device const uchar *vw [[buffer(7)]], device const ushort *vs [[buffer(8)]], device const ushort *vb [[buffer(9)]],
                         device const uchar *iw [[buffer(10)]], device const ushort *is [[buffer(11)]], device const ushort *ib [[buffer(12)]],
                         device float *q [[buffer(13)]], device float *kv [[buffer(14)]], device float *iqk [[buffer(15)]],
                         device float *ikeys [[buffer(16)]], device const uint *pos [[buffer(17)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 8u + sg;
  constexpr uint R1 = QD, R2 = R1 + KVD, R3 = R2 + KVD, R4 = R3 + IQD;
  if (row >= R4) return;
  constexpr uint NG = HID / G;
  uint r;
  device const uchar *w;
  device const ushort *s, *b;
  if (row < R1) { r = row; w = qw; s = qs; b = qb; }
  else if (row < R2) { r = row - R1; w = kw; s = ks; b = kb; }
  else if (row < R3) { r = row - R2; w = vw; s = vs; b = vb; }
  else { r = row - R3; w = iw; s = is; b = ib; }
  float v[R];
  q8rows_dev<HID, G, R>(w + (size_t)r * HID, s + (size_t)r * NG, b + (size_t)r * NG, x, lane, v);
  if (lane != 0u) return;
  const uint T = pos[0];
  for (uint t = 0u; t < R; t++) {
    if (row < R1) q[t * QD + r] = v[t];
    else if (row < R2) kv[t * 2u * KVD + r] = v[t];
    else if (row < R3) kv[t * 2u * KVD + KVD + r] = v[t];
    else {
      iqk[t * IQD + r] = v[t];
      if (r >= 4u * 128u) ikeys[(size_t)(T + t) * 128u + (r - 512u)] = v[t];
    }
  }
}

// rmsnorm over the 256 dims of a head held as 8 per lane (lane*8 .. lane*8+7), * w, then partial rope at position p:
// dims d < 32 pair with d + 32 (both inside lanes 0..7).
static inline void norm_rope(thread float *e, const device ushort *w, float p, uint lane) {
  float ss = 0.0f;
  for (uint j = 0u; j < 8u; j++) ss += e[j] * e[j];
  const float inv = rsqrt(simd_sum(ss) / float(HD) + 1e-6f);
  for (uint j = 0u; j < 8u; j++) e[j] *= inv * bf(w[lane * 8u + j]);
  // rope: lane l < 4 holds d = 8l..8l+7 (< 32), its partner lane l + 4 holds d + 32
  float o[8];
  for (uint j = 0u; j < 8u; j++) o[j] = simd_shuffle_xor(e[j], ushort(4u));
  if (lane < 8u) {
    for (uint j = 0u; j < 8u; j++) {
      const uint d = (lane & 3u) * 8u + j;  // 0..31
      const float f = pow(10000000.0f, -2.0f * float(d) / float(ROT));
      const float a = p * f;
      const float c = cos(a), sn = sin(a);
      e[j] = lane < 4u ? e[j] * c - o[j] * sn : e[j] * c + o[j] * sn;
    }
  }
}

// Flash-decoding over position chunks with grouped queries, two dispatches:
// fn_attn_core: grid NKV x (tmax / ACH) threadgroups x 384 threads (12 simdgroups = the 12 q heads of kv group g).
//   Threadgroup (g, chunk c) covers positions [c*ACH, (c+1)*ACH); chunks past T + R exit at once. The new k / v rows of
//   the R tokens are recomputed by every threadgroup into threadgroup memory (no cross-threadgroup dependency) and
//   written to the cache by chunk 0. Per row t, simdgroup hh: q head norm + rope + 1/16, then an online softmax over
//   the chunk's positions <= T + t (K / V rows read by all 12 simdgroups: L1 hits) -> partial (acc[256], max, sum).
// fn_attn_merge: grid R x NQH threadgroups x 256 threads (thread = dim): merges the chunks in order (deterministic)
//   and writes attention * sigmoid(gate).
// kcache / vcache f32 [NKV][tmax][HD]; part [NKV][tmax / ACH][R][12][HD + 2] (plain stores: the dispatch boundary
// makes them visible).
constant constexpr uint ACH = 32u, QPG = NQH / NKV;  // positions per chunk, q heads per kv group
constant constexpr uint PW = HD + 2u;                // partial record: acc[HD], m, l
template <uint R>
kernel void fn_attn_core(device const float *q [[buffer(0)]], device const float *kv [[buffer(1)]],
                         device const ushort *qnw [[buffer(2)]], device const ushort *knw [[buffer(3)]],
                         device float *kcache [[buffer(4)]], device float *vcache [[buffer(5)]],
                         device const uint *pos [[buffer(6)]], constant uint &tmax [[buffer(7)]],
                         device float *part [[buffer(8)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float kn[R][HD], vn[R][HD];
  const uint g = tg % NKV, c = tg / NKV;
  const uint T = pos[0];
  if (c * ACH >= T + R) return;
  const uint p0 = c * ACH;
  if (sg < R) {  // new keys / values (simdgroup t does row t)
    const uint t = sg;
    float e[8];
    for (uint j = 0u; j < 8u; j++) e[j] = kv[t * 2u * KVD + g * HD + lane * 8u + j];
    norm_rope(e, knw, float(T + t), lane);
    for (uint j = 0u; j < 8u; j++) {
      const float kk = rr(e[j]), vv = kv[t * 2u * KVD + KVD + g * HD + lane * 8u + j];
      kn[t][lane * 8u + j] = kk;
      vn[t][lane * 8u + j] = vv;
      if (c == 0u) {
        kcache[((size_t)g * tmax + T + t) * HD + lane * 8u + j] = kk;
        vcache[((size_t)g * tmax + T + t) * HD + lane * 8u + j] = vv;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint hh = sg, h = g * QPG + hh;
  for (uint t = 0u; t < R; t++) {
    float qv[8];
    for (uint j = 0u; j < 8u; j++) qv[j] = q[t * QD + h * QROW + lane * 8u + j];
    norm_rope(qv, qnw, float(T + t), lane);
    for (uint j = 0u; j < 8u; j++) qv[j] = rr(qv[j]) * 0.0625f;  // 1/sqrt(256)
    const uint n = min(p0 + ACH, T + t + 1u);  // causal
    float m = -INFINITY, l = 0.0f, a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint j = p0; j < n; j++) {
      const bool cached = j < T;
      const device float4 *kr = (const device float4 *)(kcache + ((size_t)g * tmax + j) * HD + lane * 8u);
      const device float4 *vr = (const device float4 *)(vcache + ((size_t)g * tmax + j) * HD + lane * 8u);
      float4 k0, k1, v0, v1;
      if (cached) { k0 = kr[0]; k1 = kr[1]; v0 = vr[0]; v1 = vr[1]; }
      else {
        const threadgroup float4 *kt = (const threadgroup float4 *)(&kn[j - T][lane * 8u]);
        const threadgroup float4 *vt = (const threadgroup float4 *)(&vn[j - T][lane * 8u]);
        k0 = kt[0]; k1 = kt[1]; v0 = vt[0]; v1 = vt[1];
      }
      const float sd = simd_sum(dot(float4(qv[0], qv[1], qv[2], qv[3]), k0) + dot(float4(qv[4], qv[5], qv[6], qv[7]), k1));
      const float mn = max(m, sd);
      const float cs = exp(m - mn), pj = exp(sd - mn);
      l = l * cs + pj;
      a[0] = a[0] * cs + pj * v0.x; a[1] = a[1] * cs + pj * v0.y; a[2] = a[2] * cs + pj * v0.z; a[3] = a[3] * cs + pj * v0.w;
      a[4] = a[4] * cs + pj * v1.x; a[5] = a[5] * cs + pj * v1.y; a[6] = a[6] * cs + pj * v1.z; a[7] = a[7] * cs + pj * v1.w;
      m = mn;
    }
    device float *pr = part + ((((size_t)g * (tmax / ACH) + c) * R + t) * QPG + hh) * PW;
    for (uint j = 0u; j < 8u; j++) pr[lane * 8u + j] = a[j];
    if (lane == 0u) { pr[HD] = m; pr[HD + 1u] = l; }
  }
}

template <uint R>
kernel void fn_attn_merge(device const float *q [[buffer(0)]], device const float *part [[buffer(1)]],
                          device const uint *pos [[buffer(2)]], constant uint &tmax [[buffer(3)]],
                          device float *out [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
  const uint t = tg / NQH, h = tg % NQH, g = h / QPG, hh = h % QPG;
  const uint T = pos[0];
  const uint nct = (T + t + 1u + ACH - 1u) / ACH;  // chunks holding positions of row t
  float M = -INFINITY;
  for (uint cc = 0u; cc < nct; cc++) M = max(M, part[((((size_t)g * (tmax / ACH) + cc) * R + t) * QPG + hh) * PW + HD]);
  float o = 0.0f, L = 0.0f;
  for (uint cc = 0u; cc < nct; cc++) {
    const device float *pr = part + ((((size_t)g * (tmax / ACH) + cc) * R + t) * QPG + hh) * PW;
    const float mc = pr[HD];
    if (mc == -INFINITY) continue;
    const float w = exp(mc - M);
    o += pr[lid] * w;
    L += pr[HD + 1u] * w;
  }
  const float gate = q[t * QD + h * QROW + HD + lid];
  out[t * NQH * HD + h * HD + lid] = rr(rr(o / L) * rr(sigm(gate)));
}

#define NE_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define NE_INST_ATTN(R)                                     \
  NE_INST("fn_attn_proj_g64_r" #R, fn_attn_proj, 64, R)     \
  NE_INST("fn_attn_proj_g128_r" #R, fn_attn_proj, 128, R)   \
  NE_INST("fn_attn_core_r" #R, fn_attn_core, R)            \
  NE_INST("fn_attn_merge_r" #R, fn_attn_merge, R)
NE_INST_ATTN(1)
NE_INST_ATTN(2)
NE_INST_ATTN(4)
NE_INST_ATTN(6)
NE_INST_ATTN(8)
