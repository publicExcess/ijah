// Flash-Next model-level kernels (lab-forward): token embedding (8-bit, x4 tile), PLE (layer 1: n-gram key/value
// projections + the signed-sqrt gate, grouped norms and the dilated conv), the 8-bit lm_head and the GPU sampler
// (two passes: per-block top-K + logsumexp partials, then the global top-K, temperature, top-p and the draw).
//
// Reference math: docs/reference.md sections 3.1 (embedding), 3.6 (PLE), HEAD (mixer + lm_head). The rounding
// points follow tools/opref.py (NE_REFR = round to bf16 where the MLX reference stores bf16).
#include "ne_common.h"

constant constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID;
constant constexpr uint PLE_KV = HCD + HID;  // key_proj rows (10240) | value_proj rows (2560)

// ------------------------------------------------------------------------------------------------ embedding
// h[r][s][c] = dequant(embed[tok[r]])[c] for the 4 streams (layer 0's input = tile(embed(ids), 4)); bf16 values.
// tok[r] >= VOCAB (lab-vision): image row, h = feat[tok[r] - VOCAB] (bf16 [rows][2560], the vision tower's output
// scattered at the image-token positions, = the reference's masked_scatter of the image features).
// Grid R threadgroups x 256 threads.
constant constexpr uint EMB_VOCAB = 248320u;
template <uint G>
kernel void fn_embed(device const uint *tok [[buffer(0)]], device const uchar *w [[buffer(1)]],
                     device const ushort *sc [[buffer(2)]], device const ushort *bi [[buffer(3)]],
                     device float *h [[buffer(4)]], device const ushort *feat [[buffer(5)]],
                     uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  const size_t t = tok[r];
  if (t >= EMB_VOCAB) {
    const device ushort *fr = feat + (t - EMB_VOCAB) * HID;
    for (uint c = lid; c < HID; c += 256u) {
      const float v = bf(fr[c]);
      for (uint s = 0u; s < HC; s++) h[r * HCD + s * HID + c] = v;
    }
    return;
  }
  const device uchar *wr = w + t * HID;
  for (uint c = lid; c < HID; c += 256u) {
    const uint g = c / G;
    const float v = bf16r(bf(sc[t * (HID / G) + g]) * float(wr[c]) + bf(bi[t * (HID / G) + g]));
    for (uint s = 0u; s < HC; s++) h[r * HCD + s * HID + c] = v;
  }
}

// ------------------------------------------------------------------------------------------------ PLE projections
// kv[r][0:10240) = key_proj(emb_r), kv[r][10240:12800) = value_proj(emb_r) (8-bit, group G, 2560 inputs, bf16 out);
// emb = the 16 gathered n-gram rows (f32 [R][2560], gathered on the CPU). 8 rows per threadgroup.
template <uint G, uint R>
kernel void fn_ple_proj(device const float *emb [[buffer(0)]], device const uchar *kw [[buffer(1)]],
                        device const ushort *ks [[buffer(2)]], device const ushort *kb [[buffer(3)]],
                        device const uchar *vw [[buffer(4)]], device const ushort *vs [[buffer(5)]],
                        device const ushort *vb [[buffer(6)]], device float *kv [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 8u + sg;
  if (row >= PLE_KV) return;
  constexpr uint NG = HID / G;
  const bool key = row < HCD;
  const uint r = key ? row : row - HCD;
  const device uchar *w = key ? kw : vw;
  const device ushort *s = key ? ks : vs, *b = key ? kb : vb;
  float v[R];
  q8rows_dev<HID, G, R>(w + (size_t)r * HID, s + (size_t)r * NG, b + (size_t)r * NG, emb, lane, v);
  if (lane == 0u)
    for (uint t = 0u; t < R; t++) kv[t * PLE_KV + row] = rr(v[t]);
}

// threadgroup-wide sum over 1024 threads (32 simdgroups); every thread gets the result
static inline float tg_sum1024(float v, threadgroup float *red, uint sg, uint lane) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0u; i < 32u; i++) t += red[i];  // fixed order: deterministic, same on every thread
  return t;
}
static inline float sigm_bf(float x) {  // MLX's uncompiled bf16 sigmoid: every step rounded (opref.sigmoid_bf)
  return NE_REFR ? bf16r(1.0f / bf16r(1.0f + bf16r(exp(-x)))) : sigm(x);
}
static inline float silu_bf(float x) { return NE_REFR ? bf16r(x * bf16r(sigm(x))) : silu(x); }

// ------------------------------------------------------------------------------------------------ PLE mix
// Per stream s (one threadgroup of 1024 threads; the R rows in order): keys = rmsnorm(kv_key_s) * norm_key_s,
// queries = rmsnorm(h_s) * norm_query_s, gate = signed sqrt((keys . queries) / sqrt(2560)), gv = sigmoid(gate) *
// values, normed = rmsnorm(gv) * norm_conv_s, conv = silu(dilated (3) depthwise conv K=4 over [state (9 rows);
// normed]), hp = h + (gv + conv). state f32 [9][10240] (bf16 values, oldest first), updated in place.
constant constexpr uint PLE_ST = 9u;  // (K - 1) * dilation
template <uint R>
kernel void fn_ple_mix(device const float *h [[buffer(0)]], device const float *kv [[buffer(1)]],
                       device const ushort *nk [[buffer(2)]], device const ushort *nq [[buffer(3)]],
                       device const ushort *nc [[buffer(4)]], device const ushort *cw [[buffer(5)]],
                       device float *state [[buffer(6)]], device float *hp [[buffer(7)]],
                       uint s [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  constexpr uint NI = (HID + 1023u) / 1024u;  // columns per thread (3; the last only for lid < 512)
  float seq[NI][PLE_ST + R];
  float4 w4[NI];
  for (uint j = 0u; j < NI; j++) {
    const uint i = lid + 1024u * j;
    if (i >= HID) break;
    const uint c = s * HID + i;
    for (uint k = 0u; k < PLE_ST; k++) seq[j][k] = state[k * HCD + c];
    w4[j] = float4(bf(cw[c * 4u]), bf(cw[c * 4u + 1u]), bf(cw[c * 4u + 2u]), bf(cw[c * 4u + 3u]));
  }
  for (uint t = 0u; t < R; t++) {
    float kp[NI], hv[NI], vv[NI];
    float ssk = 0.0f, ssh = 0.0f;
    for (uint j = 0u; j < NI; j++) {
      const uint i = lid + 1024u * j;
      kp[j] = hv[j] = vv[j] = 0.0f;
      if (i >= HID) continue;
      kp[j] = kv[t * PLE_KV + s * HID + i];
      vv[j] = kv[t * PLE_KV + HCD + i];
      hv[j] = h[t * HCD + s * HID + i];
      ssk += kp[j] * kp[j];
      ssh += hv[j] * hv[j];
    }
    const float ik = 1.0f / sqrt(tg_sum1024(ssk, red, sg, lane) / float(HID) + 1e-6f);
    const float ih = 1.0f / sqrt(tg_sum1024(ssh, red, sg, lane) / float(HID) + 1e-6f);
    float d = 0.0f;
    for (uint j = 0u; j < NI; j++) {
      const uint i = lid + 1024u * j;
      if (i >= HID) continue;
      d += rr(kp[j] * ik * bf(nk[s * HID + i])) * rr(hv[j] * ih * bf(nq[s * HID + i]));
    }
    float g = rr(tg_sum1024(d, red, sg, lane));
    g = rr(g / sqrt(float(HID)));
    const float gate = rr(sign(g) * sqrt(max(fabs(g), 1e-6f)));
    const float sgt = sigm_bf(gate);
    float gv[NI];
    float ssg = 0.0f;
    for (uint j = 0u; j < NI; j++) {
      gv[j] = rr(sgt * vv[j]);
      ssg += gv[j] * gv[j];
    }
    const float ig = 1.0f / sqrt(tg_sum1024(ssg, red, sg, lane) / float(HID) + 1e-6f);
    for (uint j = 0u; j < NI; j++) {
      const uint i = lid + 1024u * j;
      if (i >= HID) continue;
      const float nrm = rr(gv[j] * ig * bf(nc[s * HID + i]));
      seq[j][PLE_ST + t] = nrm;
      const float co = rr(seq[j][t] * w4[j].x + seq[j][t + 3u] * w4[j].y + seq[j][t + 6u] * w4[j].z +
                          seq[j][t + 9u] * w4[j].w);
      const float o = rr(gv[j] + silu_bf(co));
      hp[t * HCD + s * HID + i] = rr(hv[j] + o);
    }
  }
  for (uint j = 0u; j < NI; j++) {
    const uint i = lid + 1024u * j;
    if (i >= HID) break;
    for (uint k = 0u; k < PLE_ST; k++) state[k * HCD + s * HID + i] = seq[j][R + k];
  }
}

// ------------------------------------------------------------------------------------------------ lm_head
// logits[r][v] = lm_head(x_r) (8-bit, group G, 2560 inputs, bf16 out), 8 vocabulary rows per threadgroup.
template <uint G, uint R>
kernel void fn_lm_head(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],
                       device const ushort *sc [[buffer(2)]], device const ushort *bi [[buffer(3)]],
                       device float *logits [[buffer(4)]], constant uint &nv [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 8u + sg;
  if (row >= nv) return;
  constexpr uint NG = HID / G;
  float v[R];
  q8rows_dev<HID, G, R>(w + (size_t)row * HID, sc + (size_t)row * NG, bi + (size_t)row * NG, x, lane, v);
  if (lane == 0u)
    for (uint t = 0u; t < R; t++) logits[(size_t)t * nv + row] = bf16r(v[t]);  // the reference's logits are bf16
}

// ------------------------------------------------------------------------------------------------ sampler
// params (host-written): temp (<= 0: greedy), top_p, top_k (1..SK_MAX; 1 = greedy), seed, step (the RNG counter),
// nv. Pass 1, grid ceil(nv / 1024) threadgroups x 1024: block b's top-K (value desc, lowest id first on ties) ->
// cand[b][K], and (max, sum exp(v - max)) -> lse[b]. Pass 2, one threadgroup x 1024: global top-K from the
// candidates, logsumexp, temperature softmax over the K, top-p cut (smallest prefix with mass >= top_p), draw.
// out: [0] token, [1] its log-probability (temperature 1, full vocabulary) as f32 bits, [2] the argmax token,
// [3] K used; tok_next[0] = token (the next step's embedding input: the loop can stay on the GPU).
// mask_on != 0: pass 1 treats every token whose bit is clear in `mask` (u32 words, bit i % 32 of word i / 32 = token i
// allowed; ne_fn_model_set_mask) as logit -inf, so the top-K, the logsumexp (= the log-probability) and the draw are
// over the allowed tokens only (grammar-constrained decoding, lab-api). mask_on == 0: the buffer is not read.
// lab-sampler (16 Oct): logits are transformed in pass 1 in this order: logit_bias (bias_on: += bias[i]; bias <= -100 =
// the token is banned, -inf), penalties (pen_on: cnt[i] = prompt flag (bit 31, set by the host) | generated count of
// this request (low 28 bits, incremented by pass 2 for every token it samples): repetition (seen anywhere: v < 0 ?
// v * rep : v / rep), then v -= frequency * count + presence * (count > 0)), then the mask. Pass 2: `cand_k` = the
// candidates kept per block / globally (>= top_k; the sampler draws from the first top_k; more only for logprobs),
// min_p (> 0: after the temperature softmax, tokens under min_p * p_max are dropped, before top-p), and lp_on != 0
// writes the candidates' (id, log-probability under the transformed logits) pairs to `top` (out[3] = their count).
struct SampleParams {
  float temp, top_p;
  uint top_k, seed, step, nv, mask_on;
  float min_p, presence, frequency, repetition;
  uint bias_on, pen_on, cand_k, lp_on;
};
constant constexpr uint SK_MAX = 64u;

// argmax over 1024 threads: (value desc, index asc); every thread gets (value, index)
static inline float2 tg_argmax1024(float v, uint idx, threadgroup float *rv, threadgroup uint *ri, uint sg,
                                   uint lane, thread uint &out_idx) {
  float m = simd_max(v);
  uint mi = simd_min(v == m ? idx : 0xFFFFFFFFu);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) { rv[sg] = m; ri[sg] = mi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float a = rv[lane];
  const uint ai = ri[lane];
  m = simd_max(a);
  mi = simd_min(a == m ? ai : 0xFFFFFFFFu);
  out_idx = mi;
  return float2(m, 0.0f);
}

kernel void fn_sample_p1(device const float *logits [[buffer(0)]], device const SampleParams *P [[buffer(1)]],
                         device float2 *cand [[buffer(2)]], device float2 *lse [[buffer(3)]],
                         device const uint *mask [[buffer(4)]], device const float *bias [[buffer(5)]],
                         device const uint *cnt [[buffer(6)]],
                         uint b [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float rv[32];
  threadgroup uint ri[32];
  const uint nv = P->nv, K = clamp(max(P->top_k, P->cand_k), 1u, SK_MAX);
  const uint i = b * 1024u + lid;
  float v = i < nv ? logits[i] : -INFINITY;
  if (i < nv) {
    if (P->bias_on != 0u) {
      const float bi = bias[i];
      v = bi <= -100.0f ? -INFINITY : v + bi;
    }
    if (P->pen_on != 0u) {
      const uint c = cnt[i];
      if (c != 0u) {  // seen in the prompt (flag) or generated
        if (P->repetition != 1.0f) v = v < 0.0f ? v * P->repetition : v / P->repetition;
        const uint g = c & 0x0FFFFFFFu;
        v -= P->frequency * float(g) + (g != 0u ? P->presence : 0.0f);
      }
    }
    if (P->mask_on != 0u && ((mask[i >> 5] >> (i & 31u)) & 1u) == 0u) v = -INFINITY;
  }
  // logsumexp partial (temperature 1)
  uint dummy;
  const float m = tg_argmax1024(v, i, rv, ri, sg, lane, dummy).x;
  const float e = (i < nv && v > -INFINITY) ? exp(v - m) : 0.0f;  // v = m = -inf (a block fully masked out): 0, not NaN
  threadgroup float red[32];
  float se = simd_sum(e);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) red[sg] = se;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    float t = 0.0f;
    for (uint k = 0u; k < 32u; k++) t += red[k];
    lse[b] = float2(m, t);
  }
  for (uint k = 0u; k < K; k++) {
    uint wi;
    const float wv = tg_argmax1024(v, i, rv, ri, sg, lane, wi).x;
    if (lid == 0u) cand[b * SK_MAX + k] = float2(wv, as_type<float>(wi));
    if (i == wi) v = -INFINITY;
  }
}

static inline uint pcg_hash(uint x) {
  uint s = x * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

kernel void fn_sample_p2(device const float2 *cand [[buffer(0)]], device const float2 *lse [[buffer(1)]],
                         device const SampleParams *P [[buffer(2)]], device uint *out [[buffer(3)]],
                         device uint *tok_next [[buffer(4)]], device uint2 *top [[buffer(5)]],
                         device uint *cnt [[buffer(6)]], uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float rv[32];
  threadgroup uint ri[32];
  threadgroup float topv[SK_MAX];
  threadgroup uint topi[SK_MAX];
  threadgroup float red[32];
  const uint nv = P->nv, K = clamp(max(P->top_k, P->cand_k), 1u, SK_MAX);  // candidates
  const uint Ks = clamp(P->top_k, 1u, SK_MAX);                              // the draw's top-k
  const uint nb = (nv + 1023u) / 1024u;
  constexpr uint CPT = 16u;  // candidates per thread (nb * K <= 243 * 64 < 16384)
  float cv[CPT];
  uint ci[CPT];
  for (uint j = 0u; j < CPT; j++) {
    const uint q = lid + 1024u * j;  // candidate q = (block q / K, rank q % K)
    const uint bb = q / K, kk = q % K;
    if (bb < nb) {
      const float2 c = cand[bb * SK_MAX + kk];
      cv[j] = c.x;
      ci[j] = as_type<uint>(c.y);
    } else {
      cv[j] = -INFINITY;
      ci[j] = 0xFFFFFFFFu;
    }
  }
  // logsumexp over the blocks
  float M = -INFINITY;
  for (uint bb = lid; bb < nb; bb += 1024u) M = max(M, lse[bb].x);
  uint dummy;
  M = tg_argmax1024(M, 0u, rv, ri, sg, lane, dummy).x;
  float S = 0.0f;
  for (uint bb = lid; bb < nb; bb += 1024u) S += lse[bb].y * exp(lse[bb].x - M);
  S = simd_sum(S);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) red[sg] = S;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float St = 0.0f;
  for (uint k = 0u; k < 32u; k++) St += red[k];
  const float lZ = M + log(St);
  // global top-K: K rounds of (local best, then the threadgroup argmax)
  for (uint k = 0u; k < K; k++) {
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint j = 0u; j < CPT; j++)
      if (cv[j] > bv || (cv[j] == bv && ci[j] < bi)) { bv = cv[j]; bi = ci[j]; }
    uint wi;
    const float wv = tg_argmax1024(bv, bi, rv, ri, sg, lane, wi).x;
    if (lid == 0u) { topv[k] = wv; topi[k] = wi; }
    for (uint j = 0u; j < CPT; j++)
      if (ci[j] == wi) cv[j] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid != 0u) return;
  if (!(topv[0] > -INFINITY)) {
    // nothing can be sampled (every token -inf: the mask allows only tokens logit_bias bans, or the logits are
    // -inf / NaN): flag it (out[0] = ~0u, out[3] = 0; the host raises) and feed token 0 to a step already queued
    // behind this one, so its embedding gather stays in range; no counts, no logprobs
    out[0] = 0xFFFFFFFFu;
    out[1] = as_type<uint>(NAN);
    out[2] = 0xFFFFFFFFu;
    out[3] = 0u;
    tok_next[0] = 0u;
    return;
  }
  uint pick = 0u;
  const float temp = P->temp;
  if (temp > 0.0f && Ks > 1u) {
    float p[SK_MAX];
    float tot = 0.0f;
    for (uint k = 0u; k < Ks; k++) { p[k] = exp((topv[k] - topv[0]) / temp); tot += p[k]; }
    // min_p: p[k] / p[0] = exp((v_k - v_0) / temp) >= min_p (the list is sorted, so a prefix)
    uint nk = Ks;
    if (P->min_p > 0.0f) {
      nk = 1u;
      while (nk < Ks && p[nk] >= P->min_p) nk++;
      tot = 0.0f;
      for (uint k = 0u; k < nk; k++) tot += p[k];
    }
    // top-p on the (top-k, min-p) distribution: smallest prefix whose mass >= top_p
    uint keep = nk;
    float cum = 0.0f;
    for (uint k = 0u; k < nk; k++) {
      cum += p[k];
      if (cum >= P->top_p * tot) { keep = k + 1u; break; }
    }
    float kept = 0.0f;
    for (uint k = 0u; k < keep; k++) kept += p[k];
    const uint rnd = pcg_hash(pcg_hash(P->seed) ^ (P->step * 0x9E3779B9u));
    const float u = float(rnd >> 8u) * (1.0f / 16777216.0f) * kept;
    float acc = 0.0f;
    pick = keep - 1u;
    for (uint k = 0u; k < keep; k++) {
      acc += p[k];
      if (u < acc) { pick = k; break; }
    }
  }
  const uint t = topi[pick];
  out[0] = t;
  out[1] = as_type<uint>(topv[pick] - lZ);
  out[2] = topi[0];
  out[3] = K;
  tok_next[0] = t;
  if (P->pen_on != 0u) cnt[t] += 1u;  // this request's generated-token counts (presence / frequency / repetition)
  if (P->lp_on != 0u)
    for (uint k = 0u; k < K; k++) top[k] = uint2(topi[k], as_type<uint>(topv[k] - lZ));
}

#define NE_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
NE_INST("fn_embed_g64", fn_embed, 64)
NE_INST("fn_embed_g128", fn_embed, 128)
NE_INST("fn_ple_proj_g64_r1", fn_ple_proj, 64, 1)
NE_INST("fn_ple_proj_g64_r2", fn_ple_proj, 64, 2)
NE_INST("fn_ple_proj_g64_r4", fn_ple_proj, 64, 4)
NE_INST("fn_ple_mix_r1", fn_ple_mix, 1)
NE_INST("fn_ple_mix_r2", fn_ple_mix, 2)
NE_INST("fn_ple_mix_r3", fn_ple_mix, 3)  // lab-verify: fn-prof at 3 / 5 rows
NE_INST("fn_ple_mix_r4", fn_ple_mix, 4)
NE_INST("fn_ple_mix_r5", fn_ple_mix, 5)
NE_INST("fn_lm_head_g64_r1", fn_lm_head, 64, 1)
NE_INST("fn_lm_head_g128_r1", fn_lm_head, 128, 1)
NE_INST("fn_lm_head_g128_r4", fn_lm_head, 128, 4)
