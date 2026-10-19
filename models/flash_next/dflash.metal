// DFlash2 block drafter for Flash-Next (lab-dflash; docs/mtp.md section 13). Host side: models/flash_next/dflash.mm.
// Drafter: Inco's Qwen3.8-Flash-Next-DFlash2 (Apache-2.0), design after z-lab/dflash (DFlash, Chen / Liang / Liu,
// ICML 2026; dflash/model.py DFlash2DraftModel) and Splash's DFlash2 port for the dense 27B (splash-m5
// runtime/metal/kernels/decode/draft.metal: the grouped two-tap convolution, the candidate selector of sampling.metal
// draft_select_*). Written for this engine's layouts; no code copied.
//
// Activations f32 row-major; drafter weights 8-bit MLX affine (dflash.mm quantises the bf16 checkpoint at load),
// small vectors bf16; the context ring bf16. H = 2560, 32 query heads / 8 KV heads of 128, block rows R <= 8.
#include "kernels/ne_common.h"

constant constexpr uint DH = 2560u, DFEAT = 5u * DH, DHD = 128u, DNQ = 32u, DNKV = 8u, DW = 2048u, DG = DH / 16u;

// Target features: layer l's completed output (the 4-stream hyper-connection residual, pre-mixer) averaged over the
// 4 streams (the capture patch's hidden.unflatten(-1, (4, 2560)).mean(-2)), into slot `a.y` of the [R][5 * 2560] row.
// h f32 [..][4][2560] rows a.z .. a.z + a.x -> f f32 [a.x][12800]. Grid: ceil(a.x * 2560 / 256) threadgroups of 256.
kernel void dfl_hc_mean(device const float *h [[buffer(0)]], device float *f [[buffer(1)]],
                        constant uint4 &a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  const uint n = a.x * DH;
  if (gid >= n) return;
  const uint r = gid / DH, i = gid % DH;
  device const float *p = h + (ulong)(r + a.z) * (4u * DH) + i;
  const float s = (p[0] + p[DH]) + (p[2u * DH] + p[3u * DH]);
  f[(ulong)r * DFEAT + a.y * DH + i] = s * 0.25f;
}

// The same for a prefill chunk (lab-dflash): chunk rows a.z .. a.z + a.x -> the feature ring row (posbase + row) % 2048.
kernel void dfl_hc_mean_ring(device const float *h [[buffer(0)]], device float *f [[buffer(1)]],
                             device const uint *posbase [[buffer(2)]], constant uint4 &a [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
  const uint n = a.x * DH;
  if (gid >= n) return;
  const uint r = gid / DH, i = gid % DH, row = r + a.z;
  device const float *p = h + (ulong)row * (4u * DH) + i;
  const float s = (p[0] + p[DH]) + (p[2u * DH] + p[3u * DH]);
  f[(ulong)((posbase[0] + row) % DW) * DFEAT + a.y * DH + i] = s * 0.25f;
}

// 16 rows of the feature ring by position (P[2] + r0 + j) % 2048 -> a contiguous staging buffer (the context drain).
kernel void dfl_fs_gather(device const float4 *fs [[buffer(0)]], device float4 *st [[buffer(1)]],
                          device const uint *P [[buffer(2)]], constant uint &r0 [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint per = DFEAT / 4u;
  if (gid >= 16u * per) return;
  const uint j = gid / per, i = gid % per;
  st[gid] = fs[(ulong)((P[2] + r0 + j) % DW) * per + i];
}

// 16 c-ring rows by position (P[2] + r0 + j) % 2048 -> s.cc f32 (the rebuild of the key / value rings)
kernel void dfl_cr_gather(device const bfloat *cr [[buffer(0)]], device float *cc [[buffer(1)]],
                          device const uint *P [[buffer(2)]], constant uint &r0 [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid >= 16u * DH) return;
  const uint j = gid / DH, i = gid % DH;
  cc[gid] = float(cr[(ulong)((P[2] + r0 + j) % DW) * DH + i]);
}

// Block input rows: 8-bit embedding (the target's, group G) of tok[r] -> h f32 [R][2560] (bf16 values, as fn_embed).
kernel void dfl_embed(device const uint *tok [[buffer(0)]], device const uchar *w [[buffer(1)]],
                      device const ushort *sc [[buffer(2)]], device const ushort *bi [[buffer(3)]],
                      device float *h [[buffer(4)]], constant uint &G [[buffer(5)]],
                      uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  const ulong t = tok[r];
  const uint ng = DH / G;
  for (uint c = lid; c < DH; c += 256u)
    h[(ulong)r * DH + c] = bf16r(bf(sc[t * ng + c / G]) * float(w[t * DH + c]) + bf(bi[t * ng + c / G]));
}

// RMSNorm (Qwen3RMSNorm): y = w * x / sqrt(mean x^2 + eps), one threadgroup of 256 per row. round: bf16-round y
// (the context rows c_j, so a rebuild from the bf16 ring gives the same keys).
kernel void dfl_rms(device const float *x [[buffer(0)]], device const ushort *w [[buffer(1)]], device float *y [[buffer(2)]],
                    constant uint2 &a [[buffer(3)]],  // a.x = width, a.y = round
                    uint r [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const uint n = a.x;
  device const float *xr = x + (ulong)r * n;
  float s = 0.0f;
  for (uint c = lid; c < n; c += 256u) s += xr[c] * xr[c];
  s = simd_sum(s);
  if (lane == 0u) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0u; i < 8u; i++) t += part[i];
  const float inv = rsqrt(t / float(n) + 1e-6f);
  for (uint c = lid; c < n; c += 256u) {
    const float v = xr[c] * inv * bf(w[c]);
    y[(ulong)r * n + c] = a.y ? bf16r(v) : v;
  }
}

// Grouped dynamic two-tap convolution over the block rows (z-lab GroupedDynamicCausalConv, kernel 2, group 16):
// out[r][c] = sum_t x[r - t][c] * (base[s][t][c] + dyn[r][s][t][c / 16]) (+ res[r][c] when finishing; x[-1] = 0).
// dyn f32 [R][2 stages][2 taps][160] (the kernel_projection output), base f32 [2][2][2560]. a = (R, stage, finish).
// out may alias res (each element reads its own res entry first).
kernel void dfl_conv(device const float *x [[buffer(0)]], device const float *dyn [[buffer(1)]],
                     device const float *base [[buffer(2)]], device const float *res [[buffer(3)]],
                     device float *out [[buffer(4)]], constant uint4 &a [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= a.x * DH) return;
  const uint r = gid / DH, c = gid % DH, s = a.y, g = c / 16u;
  device const float *d = dyn + (ulong)r * (4u * DG);
  float v = x[gid] * (base[(s * 2u) * DH + c] + d[(s * 2u) * DG + g]);
  if (r > 0u) v += x[gid - DH] * (base[(s * 2u + 1u) * DH + c] + d[(s * 2u + 1u) * DG + g]);
  if (a.z) v += res[gid];
  out[gid] = v;
}

kernel void dfl_silu_mul(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *o [[buffer(2)]],
                         constant uint &n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) o[gid] = silu(g[gid]) * u[gid];
}

// One head vector: per-head RMSNorm (w) + rope (rotate-half, positions pos, inverse frequencies invf[64]) in place on
// 128 threads; t = thread. Returns the rotated value of this thread's element.
static inline float head_norm_rope(float v, device const ushort *w, device const float *invf, uint pos, uint t,
                                   threadgroup float *red, threadgroup float *hv) {
  const float ss = simd_sum(v * v);
  if ((t & 31u) == 0u) red[t / 32u] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt((red[0] + red[1] + red[2] + red[3]) / float(DHD) + 1e-6f);
  const float nv = v * inv * bf(w[t]);
  hv[t] = nv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint i = t & 63u;
  const float ang = float(pos) * invf[i];
  const float cs = cos(ang), sn = sin(ang);
  const float o = t < 64u ? nv * cs - hv[t + 64u] * sn : nv * cs + hv[t - 64u] * sn;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return o;
}

// Block rows after the q / k / v projections (q f32 [R][4096], k / v f32 [R][1024]): q <- rope(q_norm(q)) in place;
// block keys kb <- rope(k_norm(k)), values vb <- v, both f32 [8][Rmax][128]. Positions P[0] + r. Grid R * 40 x 128.
kernel void dfl_qk_prep(device float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                        device const ushort *qn [[buffer(3)]], device const ushort *kn [[buffer(4)]],
                        device const float *invf [[buffer(5)]], device float *kb [[buffer(6)]], device float *vb [[buffer(7)]],
                        device const uint *P [[buffer(8)]], constant uint2 &a [[buffer(9)]],  // a = (R, Rmax)
                        uint task [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float red[4];
  threadgroup float hv[128];
  const uint R = a.x, Rm = a.y;
  const uint r = task / 40u, hh = task % 40u;
  if (r >= R) return;
  const uint pos = P[0] + r;
  if (hh < DNQ) {
    device float *x = q + (ulong)r * (DNQ * DHD) + hh * DHD;
    x[t] = head_norm_rope(x[t], qn, invf, pos, t, red, hv);
  } else {
    const uint g = hh - DNQ;
    const float o = head_norm_rope(k[(ulong)r * (DNKV * DHD) + g * DHD + t], kn, invf, pos, t, red, hv);
    kb[((ulong)g * Rm + r) * DHD + t] = o;
    vb[((ulong)g * Rm + r) * DHD + t] = v[(ulong)r * (DNKV * DHD) + g * DHD + t];
  }
}

// Context rows -> the ring: row r (r < count = P[1]) at position P[2] + r, slot (pos % 2048): keys rope(k_norm(k)),
// values v, bf16 [8][2048][128]. Grid nrows * 8 x 128.
kernel void dfl_ctx_kv(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                       device const ushort *kn [[buffer(2)]], device const float *invf [[buffer(3)]],
                       device bfloat *kr [[buffer(4)]], device bfloat *vr [[buffer(5)]], device const uint *P [[buffer(6)]],
                       constant uint &r0 [[buffer(7)]],  // this dispatch's first row among the P[1] rows
                       uint task [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float red[4];
  threadgroup float hv[128];
  const uint r = task / DNKV, g = task % DNKV;
  if (r0 + r >= P[1]) return;
  const uint pos = P[2] + r0 + r, slot = pos % DW;
  const float o = head_norm_rope(k[(ulong)r * (DNKV * DHD) + g * DHD + t], kn, invf, pos, t, red, hv);
  kr[((ulong)g * DW + slot) * DHD + t] = bfloat(o);
  vr[((ulong)g * DW + slot) * DHD + t] = bfloat(v[(ulong)r * (DNKV * DHD) + g * DHD + t]);
}

// The context rows c (f32 [n][2560], already bf16 values) -> the bf16 c ring (persistent state: snapshots rebuild the
// key / value ring from it). Grid ceil(n * 2560 / 256) x 256.
kernel void dfl_ctx_c(device const float *c [[buffer(0)]], device bfloat *cr [[buffer(1)]], device const uint *P [[buffer(2)]],
                      constant uint2 &a [[buffer(3)]],  // a = (rows of this dispatch, r0)
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= a.x * DH) return;
  const uint r = gid / DH, i = gid % DH;
  if (a.y + r >= P[1]) return;
  const uint slot = (P[2] + a.y + r) % DW;
  cr[(ulong)slot * DH + i] = bfloat(c[gid]);
}

// Attention of the block's R query rows (non-causal; z-lab's sliding window: |q - k| < 2048) over the context ring
// (positions [P[3], P[0]): key kp is visible to row r iff P[0] + r - kp < 2048) and the R block keys (always visible).
// Grid (8 KV heads, S splits) x 256: simdgroup j takes the 4 queries qi = 4 j .. 4 j + 3 (qi = hh * 8 + r, query head
// 4 g + hh, block row r); lanes hold 4 dims each. Split s streams its share of the context keys with an online softmax;
// the last split adds the block keys. Partials f32 [8][S][32][130] (acc 128, max, sum) for dfl_attn_reduce.
kernel void dfl_attn(device const float *q [[buffer(0)]], device const bfloat *kr [[buffer(1)]],
                     device const bfloat *vr [[buffer(2)]], device const float *kb [[buffer(3)]],
                     device const float *vb [[buffer(4)]], device float *part [[buffer(5)]], device const uint *P [[buffer(6)]],
                     constant uint2 &a [[buffer(7)]],  // a = (R, Rmax)
                     uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg.x, s = tg.y, S = ntg.y, R = a.x, Rm = a.y;
  const uint T = P[0], lo = P[3];
  const float scale = NE_HADS;
  float4 qv[4], acc[4];
  float m[4], l[4];
  bool on[4];
  uint rr4[4];
  for (uint j = 0u; j < 4u; j++) {
    const uint qi = sg * 4u + j, hh = qi / 8u, r = qi % 8u;
    rr4[j] = r;
    on[j] = r < R;
    const uint h = g * 4u + hh;
    qv[j] = on[j] ? *(device const float4 *)(q + (ulong)r * (DNQ * DHD) + h * DHD + lane * 4u) * scale : float4(0.0f);
    acc[j] = float4(0.0f);
    m[j] = -INFINITY;
    l[j] = 0.0f;
  }
  const uint nctx = T - lo, per = (nctx + S - 1u) / S;
  const uint k0 = lo + min(nctx, s * per), k1 = lo + min(nctx, (s + 1u) * per);
  for (uint kp = k0; kp < k1; kp++) {
    const uint slot = kp % DW;
    const float4 kv = float4(*(device const bfloat4 *)(kr + ((ulong)g * DW + slot) * DHD + lane * 4u));
    const float4 vv = float4(*(device const bfloat4 *)(vr + ((ulong)g * DW + slot) * DHD + lane * 4u));
    for (uint j = 0u; j < 4u; j++) {
      const float sc = simd_sum(dot(qv[j], kv));
      if (!on[j] || T + rr4[j] - kp >= DW) continue;
      const float mn = max(m[j], sc), c = exp(m[j] - mn), p = exp(sc - mn);
      acc[j] = acc[j] * c + p * vv;
      l[j] = l[j] * c + p;
      m[j] = mn;
    }
  }
  if (s + 1u == S)
    for (uint b = 0u; b < R; b++) {
      const float4 kv = *(device const float4 *)(kb + ((ulong)g * Rm + b) * DHD + lane * 4u);
      const float4 vv = *(device const float4 *)(vb + ((ulong)g * Rm + b) * DHD + lane * 4u);
      for (uint j = 0u; j < 4u; j++) {
        const float sc = simd_sum(dot(qv[j], kv));
        if (!on[j]) continue;
        const float mn = max(m[j], sc), c = exp(m[j] - mn), p = exp(sc - mn);
        acc[j] = acc[j] * c + p * vv;
        l[j] = l[j] * c + p;
        m[j] = mn;
      }
    }
  for (uint j = 0u; j < 4u; j++) {
    device float *o = part + (((ulong)g * S + s) * 32u + sg * 4u + j) * 130u;
    *(device float4 *)(o + lane * 4u) = acc[j];
    if (lane == 0u) { o[128] = m[j]; o[129] = l[j]; }
  }
}
// Fixed-order combine of the S partials of (KV head g, query qi) -> o f32 [R][4096]. Grid (8 * 32) x 128.
kernel void dfl_attn_reduce(device const float *part [[buffer(0)]], device float *o [[buffer(1)]],
                            constant uint2 &a [[buffer(2)]],  // a = (R, S)
                            uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint g = tg / 32u, qi = tg % 32u, hh = qi / 8u, r = qi % 8u, S = a.y;
  if (r >= a.x) return;
  float M = -INFINITY;
  for (uint s = 0u; s < S; s++) M = max(M, part[(((ulong)g * S + s) * 32u + qi) * 130u + 128u]);
  float L = 0.0f, A = 0.0f;
  for (uint s = 0u; s < S; s++) {
    device const float *p = part + (((ulong)g * S + s) * 32u + qi) * 130u;
    if (p[128] == -INFINITY) continue;
    const float c = exp(p[128] - M);
    L += c * p[129];
    A += c * p[t];
  }
  o[(ulong)r * (DNQ * DHD) + (g * 4u + hh) * DHD + t] = A / L;
}

// Candidate selector (z-lab CandidateSelector.select): positions p = 0 .. P - 1 (block rows 1 .. P), one
// threadgroup of 256 (8 simdgroups x 2 candidates). unary = the 16 exact logits of the readout's shortlist sl
// [P][16][2] (id bits, logit); score_i = unary_i + sum_d pcb[prev][d] * sh[p][d] * scb[cand_i][d] (rank 256, bf16
// codebooks, sh = hidden_projection(hn) f32 [P][256]); prev = the anchor, then the chosen candidate. Greedy: argmax
// (lowest slot on ties). Sampled: q = softmax(score / temp), u from a PCG hash of (seed, round, p); the proposal q
// (16 (id bits, prob)) -> qo [P][16][2], the draft -> tok[1 + p] (the verify token buffer) and dr[p].
struct DflSelP { uint anchor, P, sampled, seed, round, pad0, pad1, pad2; float temp, pad3, pad4, pad5; };
static inline uint dfl_pcg(uint v) {
  const uint s = v * 747796405u + 2891336453u;
  const uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
kernel void dfl_select(device const float *sl [[buffer(0)]], device const float *sh [[buffer(1)]],
                       device const bfloat *pcb [[buffer(2)]], device const bfloat *scb [[buffer(3)]],
                       device const DflSelP *SP [[buffer(4)]], device uint *tok [[buffer(5)]], device float *qo [[buffer(6)]],
                       device uint *dr [[buffer(7)]], uint lid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float score[16];
  threadgroup uint prev_t;
  const uint P = min(SP->P, 7u);
  if (lid == 0u) prev_t = SP->anchor;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = 0u; p < P; p++) {
    const uint prev = prev_t;
    float ctx[8];
    for (uint i = 0u; i < 8u; i++) {
      const uint d = lane + 32u * i;
      ctx[i] = float(pcb[(ulong)prev * 256u + d]) * sh[(ulong)p * 256u + d];
    }
    for (uint j = 0u; j < 2u; j++) {
      const uint ci = sg * 2u + j;
      const uint id = as_type<uint>(sl[((ulong)p * 16u + ci) * 2u]);
      float e = 0.0f;
      for (uint i = 0u; i < 8u; i++) e += ctx[i] * float(scb[(ulong)id * 256u + lane + 32u * i]);
      e = simd_sum(e);
      if (lane == 0u) score[ci] = sl[((ulong)p * 16u + ci) * 2u + 1u] + e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      uint pick = 0u;
      device float *q = qo + (ulong)p * 32u;
      if (SP->sampled) {
        float mx = score[0];
        for (uint i = 1u; i < 16u; i++) mx = max(mx, score[i]);
        float w[16], z = 0.0f;
        for (uint i = 0u; i < 16u; i++) { w[i] = exp((score[i] - mx) / SP->temp); z += w[i]; }
        // the proposal first (what the host's accept uses), then the draw from exactly these probabilities
        for (uint i = 0u; i < 16u; i++) { q[2u * i] = sl[((ulong)p * 16u + i) * 2u]; q[2u * i + 1u] = w[i] / z; }
        const float u = float(dfl_pcg(SP->seed ^ dfl_pcg(SP->round * 64u + p)) >> 8u) * (1.0f / 16777216.0f);
        float cum = 0.0f;
        pick = 15u;
        for (uint i = 0u; i < 15u; i++) {
          cum += w[i] / z;
          if (u < cum) { pick = i; break; }
        }
      } else {  // greedy: the argmax; q = softmax(score) only as a confidence for the host's gate (no accept uses it)
        for (uint i = 1u; i < 16u; i++) if (score[i] > score[pick]) pick = i;
        float z = 0.0f;
        for (uint i = 0u; i < 16u; i++) z += exp(score[i] - score[pick]);
        for (uint i = 0u; i < 16u; i++) { q[2u * i] = sl[((ulong)p * 16u + i) * 2u]; q[2u * i + 1u] = exp(score[i] - score[pick]) / z; }
      }
      const uint id = as_type<uint>(sl[((ulong)p * 16u + pick) * 2u]);
      tok[1u + p] = id;
      dr[p] = id;
      prev_t = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
