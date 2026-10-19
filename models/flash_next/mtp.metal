// Flash-Next MTP head: the small kernels around the head's layer (lab-mtp). Host side: models/flash_next/mtp.mm.
#include "kernels/ne_common.h"

constant constexpr uint MH = 2560u, MHC = 4u, MHCD = MH * MHC;

// Per row r (one threadgroup of 256): e = 8-bit embedding row tok[r] (bf16 values, MLX QuantizedEmbedding g64),
// en = rr(e * rsqrt(mean e^2 + eps) * w_e); hn = rr(hid * rsqrt(mean over all 10240 + eps) * w_h) (one global norm).
// The ones-centred pack norm weights multiply as stored. Reference: Qwen4ExpRMSNorm (f32 math, bf16 out).
kernel void mtp_fuse_pre(device const uint *tok [[buffer(0)]], device const uint *ew [[buffer(1)]],
                         device const ushort *es [[buffer(2)]], device const ushort *eb [[buffer(3)]],
                         device const ushort *we [[buffer(4)]], device const float *hid [[buffer(5)]],
                         device const ushort *wh [[buffer(6)]], device float *en [[buffer(7)]],
                         device float *hn [[buffer(8)]], constant uint &vocab [[buffer(9)]],
                         uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[2][8];
  const uint t = min(tok[r], vocab - 1u);
  const device uchar *q = (const device uchar *)(ew + (size_t)t * (MH / 4u));
  const device ushort *sc = es + (size_t)t * (MH / 64u), *bi = eb + (size_t)t * (MH / 64u);
  float e[10];
  float s1 = 0.0f;
  for (uint j = 0u; j < 10u; j++) {
    const uint c = lid + 256u * j;
    e[j] = bf16r(bf(sc[c / 64u]) * float(q[c]) + bf(bi[c / 64u]));
    s1 += e[j] * e[j];
  }
  const device float *h = hid + (size_t)r * MHCD;
  float s2 = 0.0f;
  for (uint c = lid; c < MHCD; c += 256u) s2 += h[c] * h[c];
  s1 = simd_sum(s1);
  s2 = simd_sum(s2);
  if (lane == 0u) { red[0][sg] = s1; red[1][sg] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t1 = 0.0f, t2 = 0.0f;
  for (uint g = 0u; g < 8u; g++) { t1 += red[0][g]; t2 += red[1][g]; }
  const float i1 = rsqrt(t1 / float(MH) + 1e-6f), i2 = rsqrt(t2 / float(MHCD) + 1e-6f);
  for (uint j = 0u; j < 10u; j++) {
    const uint c = lid + 256u * j;
    en[(size_t)r * MH + c] = rr(e[j] * i1 * bf(we[c]));
  }
  for (uint c = lid; c < MHCD; c += 256u) hn[(size_t)r * MHCD + c] = rr(h[c] * i2 * bf(wh[c]));
}

// argmax per row over n logits (lowest index on ties, = mx.argmax); one threadgroup of 1024 per row.
kernel void mtp_argmax(device const float *logits [[buffer(0)]], device uint *out [[buffer(1)]],
                       constant uint &n [[buffer(2)]], constant uint &at [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  const device float *l = logits + (size_t)r * n;
  float m = -INFINITY;
  uint mi = 0xFFFFFFFFu;
  for (uint i = lid; i < n; i += 1024u) {
    const float v = l[i];
    if (v > m) { m = v; mi = i; }  // i increases per thread: the first max wins
  }
  const float sm = simd_max(m);
  const uint si = simd_min(m == sm ? mi : 0xFFFFFFFFu);
  if (lane == 0u) { bv[sg] = sm; bi[sg] = si; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  const float v = bv[lane];
  const float gm = simd_max(v);
  const uint gi = simd_min(v == gm ? bi[lane] : 0xFFFFFFFFu);
  if (lane == 0u) out[at + r] = gi;
}

// lab-sampler: the sampler's logit transform (kernels/fn_model.metal fn_sample_p1: logit_bias, then presence / frequency /
// repetition penalties; the token mask is not supported by the MTP loop) applied IN PLACE to the verify rows' logits
// lg [R][nv] before the top-K / argmax readout, so the target distributions the acceptance compares are the step path's.
// Verify row r predicts the token after [t1, d_1 .. d_r] (vtok[0 .. r]): its counts are the generated-token counts cnt
// (t1 and everything before it counted) plus the drafts d_1 .. d_r. Grid (ceil(nv / 1024), R) x 1024, dispatched
// indirectly (0 threadgroups when neither bias nor penalties are on). SampleParamsX = SampleParams of fn_model.metal /
// model.h (the same buffer, Model::sp).
struct SampleParamsX {
  float temp, top_p;
  uint top_k, seed, step, nv, mask_on;
  float min_p, presence, frequency, repetition;
  uint bias_on, pen_on, cand_k, lp_on;
};
kernel void mtp_logit_xform(device float *lg [[buffer(0)]], device const uint *vtok [[buffer(1)]],
                            device const SampleParamsX *P [[buffer(2)]], device const float *bias [[buffer(3)]],
                            device const uint *cnt [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]]) {
  const uint r = tgp.y, i = tgp.x * 1024u + lid, nv = P->nv;
  if (i >= nv) return;
  device float *l = lg + (size_t)r * nv;
  float v = l[i];
  if (P->bias_on != 0u) {
    const float b = bias[i];
    v = b <= -100.0f ? -INFINITY : v + b;
  }
  if (P->pen_on != 0u) {
    const uint c = cnt[i];
    uint extra = 0u;
    for (uint k = 1u; k <= r; k++) extra += vtok[k] == i ? 1u : 0u;
    if (c != 0u || extra != 0u) {
      if (P->repetition != 1.0f) v = v < 0.0f ? v * P->repetition : v / P->repetition;
      const uint g = (c & 0x0FFFFFFFu) + extra;
      v -= P->frequency * float(g) + (g != 0u ? P->presence : 0.0f);
    }
  }
  l[i] = v;
}

// pos[0] += rows (the head's KV length), so chained steps replay without the host
kernel void mtp_pos_add(device uint *pos [[buffer(0)]], constant uint &rows [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid == 0u) pos[0] += rows;
}

// ------------------------------------------------------------------------------------------------ GDN rollback v1
// MTP verify GDN step with rollback points: lab-forward's fn_gdn_step (kernels/fn_trunk.metal, same math, same
// rounding points) that also stores the recurrent state after each row t < R - 1 into snap[t] ([R-1][48][128][128]
// f32); the state after the last row goes to `state` as usual. Cost: (R - 1) x 3 MB of extra writes per GDN layer.
constant constexpr uint G_NKH = 16u, G_NVH = 48u, G_DK = 128u, G_DV = 128u;
constant constexpr uint G_KEYD = G_NKH * G_DK, G_VALD = G_NVH * G_DV, G_CONVD = 2u * G_KEYD + G_VALD;
template <uint R>
kernel void mtp_gdn_step_rows(device const float *qkv [[buffer(0)]], device const float *gb [[buffer(1)]],
                              device const float *z [[buffer(2)]], device const ushort *normw [[buffer(3)]],
                              device float *state [[buffer(4)]], device float *o [[buffer(5)]],
                              device float *snap [[buffer(6)]],
                              uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float ys[R][G_DV];
  const uint hv = tg, hk = hv / (G_NVH / G_NKH);
  device float4 *S4 = (device float4 *)(state + (size_t)hv * G_DV * G_DK);
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(sg * 4u + r) * (G_DK / 4u) + lane];
  for (uint t = 0u; t < R; t++) {
    const device float *qt = qkv + t * G_CONVD;
    const float4 q = ((const device float4 *)(qt + hk * G_DK))[lane];
    const float4 k = ((const device float4 *)(qt + G_KEYD + hk * G_DK))[lane];
    float4 qq, kk;
    if (NE_REFR) {
      const float qn = bf16r(rsqrt(bf16r(simd_sum(dot(q, q))) + 1e-6f));
      const float kn = bf16r(rsqrt(bf16r(simd_sum(dot(k, k))) + 1e-6f));
      const float4 q1 = q * qn, k1 = k * kn;
      qq = float4(bf16r(bf16r(q1.x) * 0.08838834764831845f), bf16r(bf16r(q1.y) * 0.08838834764831845f),
                  bf16r(bf16r(q1.z) * 0.08838834764831845f), bf16r(bf16r(q1.w) * 0.08838834764831845f));
      kk = float4(bf16r(k1.x), bf16r(k1.y), bf16r(k1.z), bf16r(k1.w));
    } else {
      const float qn = rsqrt(simd_sum(dot(q, q)) + 1e-6f) * rsqrt(float(G_DK));
      const float kn = rsqrt(simd_sum(dot(k, k)) + 1e-6f);
      qq = q * qn;
      kk = k * kn;
    }
    const float g = gb[t * 2u * G_NVH + hv], beta = gb[t * 2u * G_NVH + G_NVH + hv];
    const device float *v = qt + 2u * G_KEYD + hv * G_DV;
    for (uint r = 0u; r < 4u; r++) {
      const uint i = sg * 4u + r;
      float4 s = S[r] * g;
      const float kv = simd_sum(dot(s, kk));
      const float delta = (v[i] - kv) * beta;
      s += kk * delta;
      const float y = simd_sum(dot(s, qq));
      S[r] = s;
      if (lane == 0u) ys[t][i] = rr(y);
    }
    if (t + 1u < R) {
      device float4 *P4 = (device float4 *)(snap + ((size_t)t * G_NVH + hv) * G_DV * G_DK);
      for (uint r = 0u; r < 4u; r++) P4[(sg * 4u + r) * (G_DK / 4u) + lane] = S[r];
    }
  }
  for (uint r = 0u; r < 4u; r++) S4[(sg * 4u + r) * (G_DK / 4u) + lane] = S[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= R) return;
  const uint t = sg;
  float y[4];
  float ss = 0.0f;
  for (uint m = 0u; m < 4u; m++) { y[m] = ys[t][lane + 32u * m]; ss += y[m] * y[m]; }
  const float inv = rsqrt(simd_sum(ss) / float(G_DV) + 1e-6f);
  for (uint m = 0u; m < 4u; m++) {
    const uint i = lane + 32u * m;
    o[t * G_VALD + hv * G_DV + i] = rr(rr(y[m] * inv * bf(normw[i])) * sigm(z[t * G_VALD + hv * G_DV + i]));
  }
}
#define MTP_INST(R) template [[host_name("mtp_gdn_step_rows_r" #R)]] [[kernel]] decltype(mtp_gdn_step_rows<R>) mtp_gdn_step_rows<R>;
MTP_INST(2) MTP_INST(3) MTP_INST(4) MTP_INST(5) MTP_INST(6) MTP_INST(7) MTP_INST(8)

// Roll a GDN layer back to "after verify row a": keep[0] = rows kept (1 .. R; = accepted drafts + 1, the same u32
// GPU buffer lab-attn's attn::record_rollback reads), a = keep[0] - 1: ssm <- snap[a] (a < R - 1; a = R - 1 is the
// state already in place), conv state <- rows a+1 .. a+3 of hist = [conv_in (3 rows); raw qkv of the R rows].
// Grid: 48 * 128 * 128 / 4 / 256 + 120 threadgroups of 256 (state float4s, then 3 x 10240 conv floats).
kernel void mtp_gdn_rollback(device const float *snap [[buffer(0)]], device float *ssm [[buffer(1)]],
                             device const float *hist [[buffer(2)]], device float *conv [[buffer(3)]],
                             device const uint *keep [[buffer(4)]], constant uint &R [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
  const uint a = clamp(keep[0], 1u, R) - 1u;
  constexpr uint NS4 = G_NVH * G_DV * G_DK / 4u;
  if (tid < NS4) {
    if (a + 1u < R) ((device float4 *)ssm)[tid] = ((const device float4 *)(snap + (size_t)a * G_NVH * G_DV * G_DK))[tid];
    return;
  }
  const uint i = tid - NS4;
  if (i < 3u * G_CONVD) conv[i] = hist[(size_t)(a + 1u) * G_CONVD + i];
}

// ------------------------------------------------------------------------------------------------ draft shortlist
// Draft-only coarse head (a 4 / 2-bit requantised copy of lm_head) -> per row the K best coarse logits, re-scored
// EXACTLY through the 8-bit g128 lm_head rows (same rounding as the full head), argmax among them (lowest id on ties)
// -> amax[at + r]; the K (id, exact logit) pairs -> sl [R][K][2] (a proposal distribution for sampled drafting).
// Verification always reads the full head: only acceptance can change, never the output distribution.
// Stage 1 (mtp_topk_local): grid (SL_TG, R) x 256: each threadgroup takes a slice of the row and keeps its top K
// (K rounds of a threadgroup argmax over values held in registers) -> cand [R][SL_TG][K] (value, id).
// Stage 2 (mtp_shortlist): one threadgroup of 1024 per row: top K of the SL_TG * K candidates, exact re-score, argmax.
constant constexpr uint SL_TG = 128u, SL_PT = 8u;  // slices per row; values per thread (256 x 8 = 2048 per slice)
kernel void mtp_topk_local(device const float *coarse [[buffer(0)]], device float2 *cand [[buffer(1)]],
                           constant uint &n [[buffer(2)]], constant uint &K [[buffer(3)]],
                           uint2 tgp [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[8];
  threadgroup uint bi[8];
  const uint r = tgp.y, sl = tgp.x;
  const uint per = (n + SL_TG - 1u) / SL_TG, i0 = sl * per, i1 = min(n, i0 + per);
  const device float *l = coarse + (size_t)r * n;
  float v[SL_PT];
  uint ix[SL_PT];
  for (uint j = 0u; j < SL_PT; j++) {
    const uint i = i0 + lid + 256u * j;
    ix[j] = i;
    v[j] = i < i1 ? l[i] : -INFINITY;
  }
  for (uint k = 0u; k < K; k++) {
    float m = v[0];
    uint mi = ix[0];
    for (uint j = 1u; j < SL_PT; j++) if (v[j] > m || (v[j] == m && ix[j] < mi)) { m = v[j]; mi = ix[j]; }
    const float sm = simd_max(m);
    const uint si = simd_min(m == sm ? mi : 0xFFFFFFFFu);
    if (lane == 0u) { bv[sg] = sm; bi[sg] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = bv[0];
    uint gi = bi[0];
    for (uint q = 1u; q < 8u; q++) if (bv[q] > gm || (bv[q] == gm && bi[q] < gi)) { gm = bv[q]; gi = bi[q]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < SL_PT; j++) if (ix[j] == gi) v[j] = -INFINITY;
    if (lid == 0u) cand[((size_t)r * SL_TG + sl) * K + k] = float2(gm, as_type<float>(gi));
  }
}
kernel void mtp_shortlist(device const float2 *cand [[buffer(0)]], device const float *x [[buffer(1)]],
                          device const uchar *hw [[buffer(2)]], device const ushort *hs [[buffer(3)]],
                          device const ushort *hb [[buffer(4)]], device uint *amax [[buffer(5)]],
                          device float *sl [[buffer(6)]], constant uint &n [[buffer(7)]], constant uint &at [[buffer(8)]],
                          constant uint &K [[buffer(9)]], device const uint *ids [[buffer(10)]],
                          constant uint &mapped [[buffer(11)]],
                          uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  threadgroup uint found[32];
  threadgroup uint fcomp[32];  // the candidates' rows in the draft head (= found unless mapped)
  threadgroup float ex[32];
  const uint NC = SL_TG * K;  // <= 4096: 4 per thread
  float v[4];
  uint ix[4];
  for (uint j = 0u; j < 4u; j++) {
    const uint c = lid + 1024u * j;
    const float2 e = c < NC ? cand[(size_t)r * NC + c] : float2(-INFINITY, as_type<float>(0xFFFFFFFFu));
    v[j] = e.x;
    ix[j] = as_type<uint>(e.y);
  }
  for (uint k = 0u; k < K; k++) {
    float m = v[0];
    uint mi = ix[0];
    for (uint j = 1u; j < 4u; j++) if (v[j] > m || (v[j] == m && ix[j] < mi)) { m = v[j]; mi = ix[j]; }
    const float sm = simd_max(m);
    const uint si = simd_min(m == sm ? mi : 0xFFFFFFFFu);
    if (lane == 0u) { bv[sg] = sm; bi[sg] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      const float a = bv[lane];
      const float gm = simd_max(a);
      const uint gi = simd_min(a == gm ? bi[lane] : 0xFFFFFFFFu);
      // lab-draft: a pruned draft head's rows are compact (ascending ids): map to the token id (n = its rows)
      if (lane == 0u) { fcomp[k] = gi; found[k] = mapped ? ids[min(gi, n - 1u)] : min(gi, n - 1u); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < 4u; j++) if (ix[j] == fcomp[k]) v[j] = -INFINITY;
  }
  if (sg < K) {  // simdgroup k re-scores candidate k through the 8-bit g128 head
    const uint id = found[sg];
    const float e = q8row_dev<2560u, 128u>(hw + (size_t)id * 2560u, hs + (size_t)id * 20u, hb + (size_t)id * 20u,
                                           x + (size_t)r * 2560u, lane);
    if (lane == 0u) ex[sg] = rr(e);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid != 0u) return;
  uint best = found[0];
  float bvv = ex[0];
  for (uint k = 0u; k < K; k++) {
    sl[((size_t)r * K + k) * 2u] = as_type<float>(found[k]);
    sl[((size_t)r * K + k) * 2u + 1u] = ex[k];
    if (ex[k] > bvv || (ex[k] == bvv && found[k] < best)) { bvv = ex[k]; best = found[k]; }
  }
  amax[at + r] = best;
}

// ------------------------------------------------------------------------------------------------ GDN rollback v2
// Lazy commit: the layer's recurrent state is "committed" (after every accepted row of earlier rounds except the
// previous round's) and the previous round's accepted rows are pending (pq: conv'd qkv f32 [8][10240], pg: g / beta
// [8][96]; P = keep[0] = rows kept by the previous accept). One step: S = committed; apply the P pending rows (no
// outputs); store S (the new committed state: the one state write a step makes anyway); then the R new rows from
// registers -> outputs (their state is never stored). Rollback is then free: the next accept just sets keep.
// Same math and rounding points as fn_gdn_step.
template <uint R>
kernel void mtp_gdn_step_lazy(device const float *qkv [[buffer(0)]], device const float *gb [[buffer(1)]],
                              device const float *z [[buffer(2)]], device const ushort *normw [[buffer(3)]],
                              device float *state [[buffer(4)]], device float *o [[buffer(5)]],
                              device const float *pq [[buffer(6)]], device const float *pg [[buffer(7)]],
                              device const uint *keep [[buffer(8)]],
                              uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float ys[R][G_DV];
  const uint hv = tg, hk = hv / (G_NVH / G_NKH);
  device float4 *S4 = (device float4 *)(state + (size_t)hv * G_DV * G_DK);
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(sg * 4u + r) * (G_DK / 4u) + lane];
  const uint P = min(keep[0], 8u);
  for (uint t = 0u; t < P + R; t++) {
    const bool pend = t < P;
    const device float *qt = pend ? pq + t * G_CONVD : qkv + (t - P) * G_CONVD;
    const device float *gt = pend ? pg + t * 2u * G_NVH : gb + (t - P) * 2u * G_NVH;
    const float4 q = ((const device float4 *)(qt + hk * G_DK))[lane];
    const float4 k = ((const device float4 *)(qt + G_KEYD + hk * G_DK))[lane];
    float4 qq, kk;
    if (NE_REFR) {
      const float qn = bf16r(rsqrt(bf16r(simd_sum(dot(q, q))) + 1e-6f));
      const float kn = bf16r(rsqrt(bf16r(simd_sum(dot(k, k))) + 1e-6f));
      const float4 q1 = q * qn, k1 = k * kn;
      qq = float4(bf16r(bf16r(q1.x) * 0.08838834764831845f), bf16r(bf16r(q1.y) * 0.08838834764831845f),
                  bf16r(bf16r(q1.z) * 0.08838834764831845f), bf16r(bf16r(q1.w) * 0.08838834764831845f));
      kk = float4(bf16r(k1.x), bf16r(k1.y), bf16r(k1.z), bf16r(k1.w));
    } else {
      const float qn = rsqrt(simd_sum(dot(q, q)) + 1e-6f) * rsqrt(float(G_DK));
      const float kn = rsqrt(simd_sum(dot(k, k)) + 1e-6f);
      qq = q * qn;
      kk = k * kn;
    }
    const float g = gt[hv], beta = gt[G_NVH + hv];
    const device float *v = qt + 2u * G_KEYD + hv * G_DV;
    for (uint r = 0u; r < 4u; r++) {
      const uint i = sg * 4u + r;
      float4 s = S[r] * g;
      const float kv = simd_sum(dot(s, kk));
      const float delta = (v[i] - kv) * beta;
      s += kk * delta;
      S[r] = s;
      if (!pend) {
        const float y = simd_sum(dot(s, qq));
        if (lane == 0u) ys[t - P][i] = rr(y);
      }
    }
    if (t + 1u == P)
      for (uint r = 0u; r < 4u; r++) S4[(sg * 4u + r) * (G_DK / 4u) + lane] = S[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= R) return;
  const uint t = sg;
  float y[4];
  float ss = 0.0f;
  for (uint m = 0u; m < 4u; m++) { y[m] = ys[t][lane + 32u * m]; ss += y[m] * y[m]; }
  const float inv = rsqrt(simd_sum(ss) / float(G_DV) + 1e-6f);
  for (uint m = 0u; m < 4u; m++) {
    const uint i = lane + 32u * m;
    o[t * G_VALD + hv * G_DV + i] = rr(rr(y[m] * inv * bf(normw[i])) * sigm(z[t * G_VALD + hv * G_DV + i]));
  }
}
#define MTP_INST_L(R) template [[host_name("mtp_gdn_step_lazy_r" #R)]] [[kernel]] decltype(mtp_gdn_step_lazy<R>) mtp_gdn_step_lazy<R>;
MTP_INST_L(1) MTP_INST_L(2) MTP_INST_L(3) MTP_INST_L(4) MTP_INST_L(5) MTP_INST_L(6) MTP_INST_L(7) MTP_INST_L(8)

// After the lazy step (behind a barrier: every read of the old pending rows is done): this round's R new rows become
// the pending set (pq <- qkv rows, pg <- gb rows); the accept's keep[0] says how many of them count. Grid: R x 41 x 256.
kernel void mtp_gdn_pending_copy(device const float *qkv [[buffer(0)]], device const float *gb [[buffer(1)]],
                                 device float *pq [[buffer(2)]], device float *pg [[buffer(3)]],
                                 constant uint &R [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
  const uint per = G_CONVD + 2u * G_NVH;
  if (tid >= R * per) return;
  const uint t = tid / per, i = tid % per;
  if (i < G_CONVD) pq[t * G_CONVD + i] = qkv[t * G_CONVD + i];
  else pg[t * 2u * G_NVH + (i - G_CONVD)] = gb[t * 2u * G_NVH + (i - G_CONVD)];
}

// ------------------------------------------------------------------------------------------------ MTP token loop
// (lab-loop) PLE mix of the verify step with lazy commit (layer 1's dilated conv state, the same scheme as the GDN
// rollback v2): `state` [9][10240] holds the committed conv rows (oldest first), `pend` [8][10240] the previous verify
// step's normed rows, of which keep[0] were accepted. At the start the committed state becomes the last 9 rows of
// [state; pend[0, keep)] (stored), then the R new rows run exactly as fn_ple_mix (kernels/fn_model.metal, same math
// and rounding points) and their normed rows become the new pending set; the state after them is never stored.
// Projections: kb [R][10240] (key_proj), vb [R][2560] (value_proj), from the NAX GEMM. Grid 4 x 1024.
static inline float ml_tg_sum1024(float v, threadgroup float *red, uint sg, uint lane) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0u; i < 32u; i++) t += red[i];
  return t;
}
static inline float ml_sigm_bf(float x) { return NE_REFR ? bf16r(1.0f / bf16r(1.0f + bf16r(exp(-x)))) : sigm(x); }
static inline float ml_silu_bf(float x) { return NE_REFR ? bf16r(x * bf16r(sigm(x))) : silu(x); }
constant constexpr uint ML_ST = 9u, ML_PMAX = 8u;
template <uint R>
kernel void mtp_ple_mix_lazy(device const float *h [[buffer(0)]], device const float *kb [[buffer(1)]],
                             device const float *vb [[buffer(2)]], device const ushort *nk [[buffer(3)]],
                             device const ushort *nq [[buffer(4)]], device const ushort *nc [[buffer(5)]],
                             device const ushort *cw [[buffer(6)]], device float *state [[buffer(7)]],
                             device float *hp [[buffer(8)]], device float *pend [[buffer(9)]],
                             device const uint *keep [[buffer(10)]],
                             uint s [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  constexpr uint NI = (MH + 1023u) / 1024u;
  float seq[NI][ML_ST + R];
  float4 w4[NI];
  const uint P = min(keep[0], ML_PMAX);
  for (uint j = 0u; j < NI; j++) {
    const uint i = lid + 1024u * j;
    if (i >= MH) break;
    const uint c = s * MH + i;
    float all[ML_ST + ML_PMAX];
    for (uint k = 0u; k < ML_ST; k++) all[k] = state[k * MHCD + c];
    for (uint k = 0u; k < P; k++) all[ML_ST + k] = pend[k * MHCD + c];
    for (uint k = 0u; k < ML_ST; k++) {
      seq[j][k] = all[P + k];
      if (P) state[k * MHCD + c] = all[P + k];
    }
    w4[j] = float4(bf(cw[c * 4u]), bf(cw[c * 4u + 1u]), bf(cw[c * 4u + 2u]), bf(cw[c * 4u + 3u]));
  }
  for (uint t = 0u; t < R; t++) {
    float kp[NI], hv[NI], vv[NI];
    float ssk = 0.0f, ssh = 0.0f;
    for (uint j = 0u; j < NI; j++) {
      const uint i = lid + 1024u * j;
      kp[j] = hv[j] = vv[j] = 0.0f;
      if (i >= MH) continue;
      kp[j] = kb[t * MHCD + s * MH + i];
      vv[j] = vb[t * MH + i];
      hv[j] = h[t * MHCD + s * MH + i];
      ssk += kp[j] * kp[j];
      ssh += hv[j] * hv[j];
    }
    const float ik = 1.0f / sqrt(ml_tg_sum1024(ssk, red, sg, lane) / float(MH) + 1e-6f);
    const float ih = 1.0f / sqrt(ml_tg_sum1024(ssh, red, sg, lane) / float(MH) + 1e-6f);
    float d = 0.0f;
    for (uint j = 0u; j < NI; j++) {
      const uint i = lid + 1024u * j;
      if (i >= MH) continue;
      d += rr(kp[j] * ik * bf(nk[s * MH + i])) * rr(hv[j] * ih * bf(nq[s * MH + i]));
    }
    float g = rr(ml_tg_sum1024(d, red, sg, lane));
    g = rr(g / sqrt(float(MH)));
    const float gate = rr(sign(g) * sqrt(max(fabs(g), 1e-6f)));
    const float sgt = ml_sigm_bf(gate);
    float gv[NI];
    float ssg = 0.0f;
    for (uint j = 0u; j < NI; j++) {
      gv[j] = rr(sgt * vv[j]);
      ssg += gv[j] * gv[j];
    }
    const float ig = 1.0f / sqrt(ml_tg_sum1024(ssg, red, sg, lane) / float(MH) + 1e-6f);
    for (uint j = 0u; j < NI; j++) {
      const uint i = lid + 1024u * j;
      if (i >= MH) continue;
      const float nrm = rr(gv[j] * ig * bf(nc[s * MH + i]));
      seq[j][ML_ST + t] = nrm;
      pend[t * MHCD + s * MH + i] = nrm;
      const float co = rr(seq[j][t] * w4[j].x + seq[j][t + 3u] * w4[j].y + seq[j][t + 6u] * w4[j].z +
                          seq[j][t + 9u] * w4[j].w);
      const float o = rr(gv[j] + ml_silu_bf(co));
      hp[t * MHCD + s * MH + i] = rr(hv[j] + o);
    }
  }
}
#define MTP_INST_PLE(R) template [[host_name("mtp_ple_mix_lazy_r" #R)]] [[kernel]] decltype(mtp_ple_mix_lazy<R>) mtp_ple_mix_lazy<R>;
MTP_INST_PLE(1) MTP_INST_PLE(2) MTP_INST_PLE(3) MTP_INST_PLE(4) MTP_INST_PLE(5) MTP_INST_PLE(6) MTP_INST_PLE(7) MTP_INST_PLE(8)

// Stage 2 of a verify row's top K (after mtp_topk_local on the target's own logits; no re-score): the K best
// (id, logit) of each row, descending, lowest id on ties -> sl [R][K][2] (id bits, logit), and the row's argmax ->
// amax[r]. One threadgroup of 1024 per row; K <= 32.
kernel void mtp_topk_merge(device const float2 *cand [[buffer(0)]], device float *sl [[buffer(1)]],
                           device uint *amax [[buffer(2)]], constant uint &K [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  threadgroup uint found[32];
  threadgroup float fv[32];
  const uint NC = SL_TG * K;
  float v[4];
  uint ix[4];
  for (uint j = 0u; j < 4u; j++) {
    const uint c = lid + 1024u * j;
    const float2 e = c < NC ? cand[(size_t)r * NC + c] : float2(-INFINITY, as_type<float>(0xFFFFFFFFu));
    v[j] = e.x;
    ix[j] = as_type<uint>(e.y);
  }
  for (uint k = 0u; k < K; k++) {
    float m = v[0];
    uint mi = ix[0];
    for (uint j = 1u; j < 4u; j++) if (v[j] > m || (v[j] == m && ix[j] < mi)) { m = v[j]; mi = ix[j]; }
    const float sm = simd_max(m);
    const uint si = simd_min(m == sm ? mi : 0xFFFFFFFFu);
    if (lane == 0u) { bv[sg] = sm; bi[sg] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      const float a = bv[lane];
      const float gm = simd_max(a);
      const uint gi = simd_min(a == gm ? bi[lane] : 0xFFFFFFFFu);
      if (lane == 0u) { found[k] = gi; fv[k] = gm; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < 4u; j++) if (ix[j] == found[k]) v[j] = -INFINITY;
  }
  if (lid < K) {
    sl[((size_t)r * K + lid) * 2u] = as_type<float>(found[lid]);
    sl[((size_t)r * K + lid) * 2u + 1u] = fv[lid];
  }
  if (lid == 0u) amax[r] = found[0];
}

// dst[d0 + i] = src[s0 + i], i < n (u32; one threadgroup): the drafts of a chain into the verify token buffer.
kernel void mtp_copy_u32(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],
                         constant uint4 &a [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  if (tid < a.z) dst[a.y + tid] = src[a.x + tid];
}

// Sampled drafting (lab-loop): after a head step's draft readout, draw each row's draft from its shortlist (K exact
// (id, logit) pairs in sl) under the target's sampler (temperature -> top-k -> top-p over the K), u from a PCG hash
// of (seed, round, slot); amax[at + r] = the draw (the next chain step's token) and qb[at + r] = the K (id, prob)
// pairs it was drawn from (ids 0xFFFFFFFF past the kept ones): the host's exact accept uses these very probabilities.
struct MtpDraftSP { float temp, top_p; uint top_k, seed, round, K, pad0, pad1; };
static inline uint ml_pcg(uint v) {
  const uint s = v * 747796405u + 2891336453u;
  const uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
kernel void mtp_draft_sample(device const float *sl [[buffer(0)]], device uint *amax [[buffer(1)]],
                             device float *qb [[buffer(2)]], device const MtpDraftSP *P [[buffer(3)]],
                             constant uint &at [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
  if (lid != 0u) return;
  const uint K = min(P->K, 32u);
  float lg[32];
  uint id[32];
  for (uint k = 0u; k < K; k++) {
    id[k] = as_type<uint>(sl[((size_t)r * K + k) * 2u]);
    lg[k] = sl[((size_t)r * K + k) * 2u + 1u];
  }
  for (uint i = 1u; i < K; i++) {  // descending logit, lowest id on ties
    const float v = lg[i];
    const uint d = id[i];
    uint j = i;
    while (j > 0u && (lg[j - 1u] < v || (lg[j - 1u] == v && id[j - 1u] > d))) { lg[j] = lg[j - 1u]; id[j] = id[j - 1u]; j--; }
    lg[j] = v;
    id[j] = d;
  }
  const uint kk = P->top_k == 0u ? K : min(K, P->top_k);
  float w[32];
  float z = 0.0f;
  for (uint i = 0u; i < kk; i++) { w[i] = exp((lg[i] - lg[0]) / P->temp); z += w[i]; }
  uint n = kk;
  if (P->top_p < 1.0f) {
    float c = 0.0f;
    for (uint i = 0u; i < kk; i++) {
      c += w[i] / z;
      if (c >= P->top_p) { n = i + 1u; break; }
    }
  }
  float zn = 0.0f;
  for (uint i = 0u; i < n; i++) zn += w[i];
  const uint slot = at + r;
  // the proposal first (what the host's accept uses), then the draw from exactly these probabilities
  device float *q = qb + (size_t)slot * 64u;
  for (uint i = 0u; i < 32u; i++) {
    const bool keep = i < n;
    q[2u * i] = as_type<float>(keep ? id[i] : 0xFFFFFFFFu);
    q[2u * i + 1u] = keep ? w[i] / zn : 0.0f;
  }
  const float u = float(ml_pcg(P->seed ^ ml_pcg(P->round * 64u + slot)) >> 8u) * (1.0f / 16777216.0f);
  uint pick = id[n - 1u];
  float cum = 0.0f;
  for (uint i = 0u; i + 1u < n; i++) {
    cum += w[i] / zn;
    if (u < cum) { pick = id[i]; break; }
  }
  amax[slot] = pick;
}
