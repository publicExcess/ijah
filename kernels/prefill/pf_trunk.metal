// Prefill (chunk of n consecutive tokens) pointwise / row kernels around the NAX GEMMs (pf_gemm.metal): hyper-connection
// read and write, GDN conv / gates / q-k norm / the delta-rule scan / gated norm, MoE routing + shared expert glue,
// PLE. Owner: lab-prefill. Design: docs/prefill.md.
//
// The arithmetic mirrors lab-forward's decode kernels (kernels/fn_trunk.metal, fn_moe.metal, fn_model.metal) term
// by term, with the same lane layouts where a reduction order matters (rms sums, the GDN recurrence, the router
// top-k), so that a chunk ends in the state N decode steps leave (bit-identical where the inputs are; the GEMM
// outputs differ from the decode GEMVs by bf16 rounding of the GEMM inputs only). Activations between kernels are
// bf16 (ushort) unless noted; the residual streams h stay f32 [n][4][2560] (bf16 values), as in the decode path.
#include "kernels/ne_common.h"

constant constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID;
constant constexpr uint NKH = 16u, NVH = 48u, DK = 128u, DV = 128u;
constant constexpr uint KEYD = NKH * DK, VALD = NVH * DV, CONVD = 2u * KEYD + VALD;  // 2048, 6144, 10240
constant constexpr uint NE = 512u, TOPK = 10u, INTER = 640u;

static inline ushort f2bf(float x) { return ushort(as_type<uint>(bf16r(x)) >> 16u); }

// ------------------------------------------------------------------------------------------------ HC read
// One threadgroup (256 threads) per row t. Per stream s: inv_s = rsqrt(mean(h_s^2) + 1e-6) (the sum in fn_hc_pre's
// order: thread-strided, simd_sum, 8 simdgroup partials in order); normed = rr(h * inv_s * w) (fn_hc_mix's
// expression) -> bf16 [n][10240]; inject logits = sum_s inv_s * ((h_s * w_s) . W_inj[j][s]) (fn_hc_pre's order: one
// simdgroup per (j, s), lane-chunked dot8bf, streams summed in order) -> inj[t][j] = rr(2 sigmoid(rr(v) / 4));
// inj[t][4 + s] = inv_s. f32 [n][8].
template <bool WR>
static inline void hc_norm_body(device const float *h, device const ushort *br, device float *hw_out,
                                device const ushort *nw, device const ushort *iw, device ushort *normed,
                                device float *inj, uint t, uint lid, uint sg, uint lane,
                                threadgroup float (*red)[8], threadgroup float *invs,
                                threadgroup float (*ip)[HC] /* [j][s] */) {
  // WR: first the previous mixer's HC write (pf_hc_write's expression) of row t: hw_out = rr(h + rr(br * inj_s))
  // with the old inject gates, then the norm of hw_out (re-read after a device barrier: same threadgroup)
  const device float *hr = WR ? (const device float *)(hw_out + (size_t)t * HCD) : h + (size_t)t * HCD;
  float ginj[HC];
  if (WR)
    for (uint s = 0u; s < HC; s++) ginj[s] = inj[t * 2u * HC + s];
  for (uint s = 0u; s < HC; s++) {
    float ss = 0.0f;
    for (uint i = lid; i < HID; i += 256u) {
      float v;
      if (WR) {
        const size_t o = (size_t)t * HCD + s * HID + i;
        v = rr(h[o] + rr(bf(br[(size_t)t * HID + i]) * ginj[s]));
        hw_out[o] = v;
      } else {
        v = hr[s * HID + i];
      }
      ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[s][sg] = ss;
  }
  threadgroup_barrier(WR ? (mem_flags::mem_threadgroup | mem_flags::mem_device) : mem_flags::mem_threadgroup);
  if (lid < HC) {
    float tt = 0.0f;
    for (uint g = 0u; g < 8u; g++) tt += red[lid][g];
    invs[lid] = rsqrt(tt / float(HID) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lid; i < HCD; i += 256u) {
    const uint s = i / HID;
    normed[(size_t)t * HCD + i] = f2bf(rr(hr[i] * invs[s] * bf(nw[i])));
  }
  // inject: simdgroup sg handles (j, s) pairs sg and sg + 8 of the 16
  for (uint p = sg; p < HC * HC; p += 8u) {
    const uint j = p / HC, s = p % HC;
    const device uint4 *w4 = (const device uint4 *)(iw + (size_t)j * HCD + s * HID);
    float acc = 0.0f;
    for (uint jj = 0u; jj < HID / 8u / 32u; jj++) {
      const uint c = lane + 32u * jj;
      const uint o = s * HID + 8u * c;
      const float4 x0 = float4(hr[o], hr[o + 1u], hr[o + 2u], hr[o + 3u]) *
                        float4(bf(nw[o]), bf(nw[o + 1u]), bf(nw[o + 2u]), bf(nw[o + 3u]));
      const float4 x1 = float4(hr[o + 4u], hr[o + 5u], hr[o + 6u], hr[o + 7u]) *
                        float4(bf(nw[o + 4u]), bf(nw[o + 5u]), bf(nw[o + 6u]), bf(nw[o + 7u]));
      acc += dot8bf(w4[c], x0, x1);
    }
    const float v = simd_sum(acc) * invs[s];
    if (lane == 0u) ip[j][s] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < HC) {
    const float v = ((ip[lid][0] + ip[lid][1]) + ip[lid][2]) + ip[lid][3];
    inj[t * 2u * HC + lid] = rr(2.0f * sigm(rr(v) * 0.25f));
    inj[t * 2u * HC + HC + lid] = invs[lid];
  }
}

kernel void pf_hc_norm(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                       device const ushort *iw [[buffer(2)]], device ushort *normed [[buffer(3)]],
                       device float *inj [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[HC][8], invs[HC], ip[HC][HC];
  hc_norm_body<false>(h, nullptr, nullptr, nw, iw, normed, inj, t, lid, sg, lane, red, invs, ip);
}
// pf_hc_write (h, br, the inject gates in inj) fused into the next HC read's pf_hc_norm: hout = the written stream
// (stored, it is the MoE's residual), then normed / inj of hout (lab-pflevers; Sushi's hc_prefill_norm WR=1 /
// oMLX's norm_inject idea). Bit-identical to pf_hc_write + pf_hc_norm. One threadgroup (256) per row.
kernel void pf_hc_norm_wr(device const float *h [[buffer(0)]], device const ushort *br [[buffer(1)]],
                          device float *inj [[buffer(2)]], device float *hout [[buffer(3)]],
                          device const ushort *nw [[buffer(4)]], device const ushort *iw [[buffer(5)]],
                          device ushort *normed [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[HC][8], invs[HC], ip[HC][HC];
  hc_norm_body<true>(h, br, hout, nw, iw, normed, inj, t, lid, sg, lane, red, invs, ip);
}

// mix = rr(silu(down / 4)) in place (down = the bf16 GEMM output). n * 320 threads.
kernel void pf_hc_act(device ushort *d [[buffer(0)]], constant uint &cnt [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= cnt) return;
  d[i] = f2bf(rr(silu(bf(d[i]) * 0.25f)));
}

// block_in[t][c] = rr(mean_s rr(rr(sigmoid(up_s)) * normed_s)) (fn_hc_mix: streams summed (s0 + s1) + (s2 + s3)).
// n * 2560 threads.
kernel void pf_hc_blockin(device const ushort *up [[buffer(0)]], device const ushort *normed [[buffer(1)]],
                          device ushort *out [[buffer(2)]], constant uint &cnt [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= cnt) return;
  const uint t = i / HID, c = i % HID;
  float v[HC];
  for (uint s = 0u; s < HC; s++) {
    const size_t o = (size_t)t * HCD + s * HID + c;
    v[s] = rr(rr(sigm(bf(up[o]))) * bf(normed[o]));
  }
  out[i] = f2bf(rr(((v[0] + v[1]) + (v[2] + v[3])) * 0.25f));
}

// HC write: hout[t][s][c] = rr(h[t][s][c] + rr(br[t][c] * inj[t][s])) (br = the bf16 branch). n * 2560 threads.
kernel void pf_hc_write(device const float *h [[buffer(0)]], device const ushort *br [[buffer(1)]],
                        device const float *inj [[buffer(2)]], device float *hout [[buffer(3)]],
                        constant uint &cnt [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= cnt) return;
  const uint t = i / HID, c = i % HID;
  const float b = bf(br[i]);
  for (uint s = 0u; s < HC; s++) {
    const size_t o = (size_t)t * HCD + s * HID + c;
    hout[o] = rr(h[o] + rr(b * inj[t * 2u * HC + s]));
  }
}

// ------------------------------------------------------------------------------------------------ GDN
// conv1d K=4 over [conv state (3 rows, oldest first); raw qkv rows 0..n-1] + silu: conv[t][c] =
// rr(silu(rr(s[t] w0 + s[t+1] w1 + s[t+2] w2 + s[t+3] w3))) (fn_gdn_inproj's epilogue). n * 10240 threads.
kernel void pf_gdn_conv(device const ushort *raw [[buffer(0)]], device const ushort *convw [[buffer(1)]],
                        device const float *convst [[buffer(2)]], device ushort *conv [[buffer(3)]],
                        constant uint &n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n * CONVD) return;
  const uint t = i / CONVD, c = i % CONVD;
  float sq[4];
  for (uint j = 0u; j < 4u; j++) {
    const int tt = int(t) + int(j) - 3;
    sq[j] = tt >= 0 ? bf(raw[(size_t)tt * CONVD + c]) : convst[uint(3 + tt) * CONVD + c];
  }
  const device ushort *cw = convw + (size_t)c * 4u;
  const float w0 = bf(cw[0]), w1 = bf(cw[1]), w2 = bf(cw[2]), w3 = bf(cw[3]);
  conv[i] = f2bf(rr(silu(rr(sq[0] * w0 + sq[1] * w1 + sq[2] * w2 + sq[3] * w3))));
}
// conv state <- the last 3 inputs of [state; raw] (after pf_gdn_conv). 10240 threads.
kernel void pf_gdn_convstate(device const ushort *raw [[buffer(0)]], device float *convst [[buffer(1)]],
                             constant uint &n [[buffer(2)]], uint c [[thread_position_in_grid]]) {
  if (c >= CONVD) return;
  float sq[3];
  for (uint j = 0u; j < 3u; j++) {
    const int tt = int(n) + int(j) - 3;  // input index (state rows are -3..-1)
    sq[j] = tt >= 0 ? bf(raw[(size_t)tt * CONVD + c]) : convst[uint(3 + tt) * CONVD + c];
  }
  for (uint j = 0u; j < 3u; j++) convst[j * CONVD + c] = sq[j];
}
// gb[t] = [g 48 | beta 48] from the bf16 a / b GEMM outputs (fn_gdn_inproj's epilogue). n * 48 threads.
kernel void pf_gdn_gates(device const ushort *araw [[buffer(0)]], device const ushort *braw [[buffer(1)]],
                         device const ushort *alog [[buffer(2)]], device const ushort *dtb [[buffer(3)]],
                         device float *gb [[buffer(4)]], constant uint &cnt [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= cnt) return;
  const uint t = i / NVH, r = i % NVH;
  gb[t * 2u * NVH + NVH + r] = rr(sigm(bf(braw[i])));
  const float e = exp(bf(alog[r])), dt = bf(dtb[r]);
  const float u = rr(bf(araw[i]) + dt);
  const float sp = rr(u > 20.0f ? u : log(1.0f + exp(u)));
  gb[t * 2u * NVH + r] = exp(-e * sp);
}
// q, k l2 norm (fn_gdn_step's expressions, lane = 4 consecutive dk elements): qk[t][0][hk*128 + d] = q normed *
// 1/sqrt(128), qk[t][1][...] = k normed; f32. One simdgroup per (t, q|k, key head): grid n * 32 simdgroups
// (threadgroups of 256 = 8 of them).
kernel void pf_gdn_qknorm(device const ushort *conv [[buffer(0)]], device float4 *qk [[buffer(1)]],
                          constant uint &n [[buffer(2)]], uint gid [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint w = gid * 8u + sg;
  if (w >= n * 2u * NKH) return;
  const uint t = w / (2u * NKH), which = (w / NKH) % 2u, hk = w % NKH;
  const device ushort *src = conv + (size_t)t * CONVD + which * KEYD + hk * DK + 4u * lane;
  const float4 q = float4(bf(src[0]), bf(src[1]), bf(src[2]), bf(src[3]));
  float4 o;
  if (NE_REFR) {
    const float qn = bf16r(rsqrt(bf16r(simd_sum(dot(q, q))) + 1e-6f));
    const float4 q1 = q * qn;
    if (which == 0u)
      o = float4(bf16r(bf16r(q1.x) * 0.08838834764831845f), bf16r(bf16r(q1.y) * 0.08838834764831845f),
                 bf16r(bf16r(q1.z) * 0.08838834764831845f), bf16r(bf16r(q1.w) * 0.08838834764831845f));
    else
      o = float4(bf16r(q1.x), bf16r(q1.y), bf16r(q1.z), bf16r(q1.w));
  } else {
    const float qn = which == 0u ? rsqrt(simd_sum(dot(q, q)) + 1e-6f) * rsqrt(float(DK)) : rsqrt(simd_sum(dot(q, q)) + 1e-6f);
    o = q * qn;
  }
  qk[((size_t)t * 2u + which) * (KEYD / 4u) + hk * (DK / 4u) + lane] = o;
}
// The gated delta rule over the chunk, token by token (fn_gdn_step's recurrence, same lane layout: a simdgroup
// owns 4 value rows i of head hv, lane = dk elements 4 lane .. 4 lane + 3). The value rows of a head are
// independent, so the 6144 rows spread over 1536 simdgroups (decode: 48 threadgroups). y[t][hv*128 + i] =
// rr(S_i . q) (f32 buffer). The next token's q / k / v / g / beta are loaded before the current update.
// grid 48 * 4 threadgroups x 256 threads (8 simdgroups x 4 rows = 32 rows per threadgroup).
kernel void pf_gdn_scan(device const float4 *qk [[buffer(0)]], device const ushort *conv [[buffer(1)]],
                        device const float *gb [[buffer(2)]], device float *state [[buffer(3)]],
                        device float *y [[buffer(4)]], constant uint &n [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint hv = tg / 4u, i0 = (tg % 4u) * 32u + sg * 4u, hk = hv / (NVH / NKH);
  device float4 *S4 = (device float4 *)(state + (size_t)hv * DV * DK);
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(i0 + r) * (DK / 4u) + lane];
  float4 qq = qk[0u * (KEYD / 4u) + hk * (DK / 4u) + lane];
  float4 kk = qk[1u * (KEYD / 4u) + hk * (DK / 4u) + lane];
  float g = gb[hv], beta = gb[NVH + hv];
  float v[4];
  for (uint r = 0u; r < 4u; r++) v[r] = bf(conv[2u * KEYD + hv * DV + i0 + r]);
  for (uint t = 0u; t < n; t++) {
    const uint tn = min(t + 1u, n - 1u);
    const float4 qn = qk[((size_t)tn * 2u + 0u) * (KEYD / 4u) + hk * (DK / 4u) + lane];
    const float4 kn = qk[((size_t)tn * 2u + 1u) * (KEYD / 4u) + hk * (DK / 4u) + lane];
    const float gn = gb[tn * 2u * NVH + hv], bn = gb[tn * 2u * NVH + NVH + hv];
    float vn[4];
    for (uint r = 0u; r < 4u; r++) vn[r] = bf(conv[(size_t)tn * CONVD + 2u * KEYD + hv * DV + i0 + r]);
    float yv[4];
    for (uint r = 0u; r < 4u; r++) {
      float4 s = S[r] * g;
      const float kv = simd_sum(dot(s, kk));
      const float delta = (v[r] - kv) * beta;
      s += kk * delta;
      yv[r] = simd_sum(dot(s, qq));
      S[r] = s;
    }
    if (lane < 4u) y[(size_t)t * VALD + hv * DV + i0 + lane] = rr(yv[lane]);
    qq = qn; kk = kn; g = gn; beta = bn;
    for (uint r = 0u; r < 4u; r++) v[r] = vn[r];
  }
  for (uint r = 0u; r < 4u; r++) S4[(i0 + r) * (DK / 4u) + lane] = S[r];
}
// gated norm: o[t][hv*128 + i] = rr(rr(y * inv * w) * sigmoid(z)) (fn_gdn_step's tail; lane = elements lane + 32 m).
// One simdgroup per (t, hv): threadgroups of 256 = 8 of them, n * 48 / 8 threadgroups.
kernel void pf_gdn_gnorm(device const float *y [[buffer(0)]], device const ushort *z [[buffer(1)]],
                         device const ushort *normw [[buffer(2)]], device ushort *o [[buffer(3)]],
                         constant uint &n [[buffer(4)]], uint gid [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint w = gid * 8u + sg;
  if (w >= n * NVH) return;
  const uint t = w / NVH, hv = w % NVH;
  const size_t base = (size_t)t * VALD + hv * DV;
  float yy[4];
  float ss = 0.0f;
  for (uint m = 0u; m < 4u; m++) { yy[m] = y[base + lane + 32u * m]; ss += yy[m] * yy[m]; }
  const float inv = rsqrt(simd_sum(ss) / float(DV) + 1e-6f);
  for (uint m = 0u; m < 4u; m++) {
    const uint i = lane + 32u * m;
    o[base + i] = f2bf(rr(rr(yy[m] * inv * bf(normw[i])) * sigm(bf(z[base + i]))));
  }
}

// ------------------------------------------------------------------------------------------------ MoE glue
// Routing per row (fn_moe_router's finisher): top-10 of the 512 bf16 logits (value desc, lowest id on ties: per
// simdgroup top-10 of 64, then of the 80 candidates), weights = renormalised softmax over the top-10; route[t] =
// [10 ids | 10 f32 weights (bits)]; the shared-expert gate logit rr(x . w_sg) (fn_moe_router's dot8bf order) ->
// sgl[t]. One threadgroup of 256 threads per row; x = the MoE input (bf16).
kernel void pf_moe_route(device const ushort *logits [[buffer(0)]], device const ushort *x [[buffer(1)]],
                         device const ushort *sgw [[buffer(2)]], device uint *route [[buffer(3)]],
                         device float *sgl [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float cval[80];
  threadgroup uint cidx[80];
  const device ushort *lg = logits + (size_t)t * NE;
  {
    const uint ia = sg * 64u + lane, ib = ia + 32u;
    float a = bf(lg[ia]), b = bf(lg[ib]);
    for (uint k = 0u; k < TOPK; k++) {
      const float m = max(a, b);
      const uint im = (a >= b) ? ia : ib;
      const float mm = simd_max(m);
      const uint win = simd_min(m == mm ? im : 0xFFFFFFFFu);
      if (lane == 0u) { cval[sg * TOPK + k] = mm; cidx[sg * TOPK + k] = win; }
      if (ia == win) a = -INFINITY;
      if (ib == win) b = -INFINITY;
    }
  }
  if (sg == 1u) {  // shared expert gate logit
    const device uint4 *w4 = (const device uint4 *)sgw;
    const device ushort *xr = x + (size_t)t * HID;
    float acc = 0.0f;
    for (uint j = 0u; j < HID / 256u; j++) {
      const uint c = lane + 32u * j;
      const uint o = 8u * c;
      acc += dot8bf(w4[c], float4(bf(xr[o]), bf(xr[o + 1u]), bf(xr[o + 2u]), bf(xr[o + 3u])),
                    float4(bf(xr[o + 4u]), bf(xr[o + 5u]), bf(xr[o + 6u]), bf(xr[o + 7u])));
    }
    const float a = rr(simd_sum(acc));
    if (lane == 0u) sgl[t] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  float v[3];
  uint ix[3];
  for (uint j = 0u; j < 3u; j++) {
    const uint c = lane + 32u * j;
    v[j] = c < 80u ? cval[c] : -INFINITY;
    ix[j] = c < 80u ? cidx[c] : 0xFFFFFFFFu;
  }
  float myv = 0.0f, top = 0.0f;
  uint myi = 0u;
  for (uint k = 0u; k < TOPK; k++) {
    const float m = max(v[0], max(v[1], v[2]));
    const uint im = (v[0] == m) ? ix[0] : ((v[1] == m) ? ix[1] : ix[2]);
    const float mm = simd_max(m);
    const uint win = simd_min(m == mm ? im : 0xFFFFFFFFu);
    if (k == 0u) top = mm;
    if (lane == k) { myv = mm; myi = win; }
    for (uint j = 0u; j < 3u; j++) if (ix[j] == win) v[j] = -INFINITY;
  }
  const float p = lane < TOPK ? exp(myv - top) : 0.0f;
  const float s = simd_sum(p);
  if (lane < TOPK) { route[t * 20u + lane] = myi & (NE - 1u); route[t * 20u + TOPK + lane] = as_type<uint>(p / s); }
}
// shared expert activation: a[t][j] = rr(silu(g) * u) (gu = [n][gate 640 | up 640] bf16). n * 640 threads.
kernel void pf_shared_act(device const ushort *gu [[buffer(0)]], device ushort *a [[buffer(1)]],
                          constant uint &cnt [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= cnt) return;
  const uint t = i / INTER, j = i % INTER;
  a[i] = f2bf(rr(silu(bf(gu[(size_t)t * 2u * INTER + j])) * bf(gu[(size_t)t * 2u * INTER + INTER + j])));
}

// ------------------------------------------------------------------------------------------------ PLE (layer 1)
static inline float tg_sum1024(float v, threadgroup float *red, uint sg, uint lane) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0u; i < 32u; i++) t += red[i];
  return t;
}
static inline float sigm_bf(float x) { return NE_REFR ? bf16r(1.0f / bf16r(1.0f + bf16r(exp(-x)))) : sigm(x); }
static inline float silu_bf(float x) { return NE_REFR ? bf16r(x * bf16r(sigm(x))) : silu(x); }
constant constexpr uint PLE_KV = HCD + HID, PLE_ST = 9u;
// Part 1 per (row t, stream s) threadgroup of 1024 threads (fn_ple_mix's per-row math, same layout): gv =
// rr(sigmoid(gate) * values) -> gv bf16 [n][10240], normed = rr(gv * inv * norm_conv) -> nrm bf16 [n][10240].
// kv = [n][key 10240 | value 2560] bf16 (the PLE projections). grid n * 4 threadgroups.
kernel void pf_ple_gate(device const float *h [[buffer(0)]], device const ushort *kv [[buffer(1)]],
                        device const ushort *nk [[buffer(2)]], device const ushort *nq [[buffer(3)]],
                        device const ushort *nc [[buffer(4)]], device ushort *gvo [[buffer(5)]],
                        device ushort *nrmo [[buffer(6)]], uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const uint t = gid / HC, s = gid % HC;
  constexpr uint NI = (HID + 1023u) / 1024u;
  float kp[NI], hv[NI], vv[NI];
  float ssk = 0.0f, ssh = 0.0f;
  for (uint j = 0u; j < NI; j++) {
    const uint i = lid + 1024u * j;
    kp[j] = hv[j] = vv[j] = 0.0f;
    if (i >= HID) continue;
    kp[j] = bf(kv[(size_t)t * PLE_KV + s * HID + i]);
    vv[j] = bf(kv[(size_t)t * PLE_KV + HCD + i]);
    hv[j] = h[(size_t)t * HCD + s * HID + i];
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
    gvo[(size_t)t * HCD + s * HID + i] = f2bf(gv[j]);
    nrmo[(size_t)t * HCD + s * HID + i] = f2bf(rr(gv[j] * ig * bf(nc[s * HID + i])));
  }
}
// Part 2 per (t, column): the dilated (3) K=4 conv over [state (9 rows); normed rows] + silu, hp = rr(h + rr(gv +
// silu(conv))). n * 10240 threads. State update in pf_ple_state (after a barrier).
kernel void pf_ple_conv(device const float *h [[buffer(0)]], device const ushort *gvo [[buffer(1)]],
                        device const ushort *nrm [[buffer(2)]], device const ushort *cw [[buffer(3)]],
                        device const float *state [[buffer(4)]], device float *hp [[buffer(5)]],
                        constant uint &n [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= n * HCD) return;
  const uint t = i / HCD, c = i % HCD;
  float sq[4];
  for (uint j = 0u; j < 4u; j++) {
    const int tt = int(t) + 3 * int(j) - int(PLE_ST);
    sq[j] = tt >= 0 ? bf(nrm[(size_t)tt * HCD + c]) : state[uint(int(PLE_ST) + tt) * HCD + c];
  }
  const float4 w4 = float4(bf(cw[c * 4u]), bf(cw[c * 4u + 1u]), bf(cw[c * 4u + 2u]), bf(cw[c * 4u + 3u]));
  const float co = rr(sq[0] * w4.x + sq[1] * w4.y + sq[2] * w4.z + sq[3] * w4.w);
  const float o = rr(bf(gvo[i]) + silu_bf(co));
  hp[i] = rr(h[i] + o);
}
kernel void pf_ple_state(device const ushort *nrm [[buffer(0)]], device float *state [[buffer(1)]],
                         constant uint &n [[buffer(2)]], uint c [[thread_position_in_grid]]) {
  if (c >= HCD) return;
  float sq[PLE_ST];
  for (uint j = 0u; j < PLE_ST; j++) {
    const int tt = int(n) + int(j) - int(PLE_ST);
    sq[j] = tt >= 0 ? bf(nrm[(size_t)tt * HCD + c]) : state[uint(int(PLE_ST) + tt) * HCD + c];
  }
  for (uint j = 0u; j < PLE_ST; j++) state[j * HCD + c] = sq[j];
}

// ------------------------------------------------------------------------------------------------ misc
// f32 -> bf16 values (the host-gathered PLE rows). cnt threads.
kernel void pf_f2bf(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]], constant uint &cnt [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < cnt) y[i] = f2bf(x[i]);
}
// pos[0] = base[0] + add (attention slices: the shared position of lab-attn's caches). 1 thread.
kernel void pf_set_pos(device uint *pos [[buffer(0)]], device const uint *base [[buffer(1)]],
                       constant uint &add [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i == 0u) pos[0] = base[0] + add;
}
// pos[4 j] = base[0] + add + 8 j for j < cnt (the per-slice positions of the concurrent attention slices).
kernel void pf_set_pos_multi(device uint *pos [[buffer(0)]], device const uint *base [[buffer(1)]],
                             constant uint &add [[buffer(2)]], constant uint &cnt [[buffer(3)]],
                             uint j [[thread_position_in_grid]]) {
  if (j < cnt) pos[4u * j] = base[0] + add + 8u * j;
}
// pf_gdn_scan with the inputs of TB tokens staged in threadgroup memory per step (q / k of the head's key head, the
// threadgroup's 32 value rows, g, beta): the recurrence then reads them at threadgroup latency. Same math / lane
// layout / op order as pf_gdn_scan (= fn_gdn_step). grid 48 * 4 threadgroups x 256 threads.
constant constexpr uint SCAN_TB = 16u;
kernel void pf_gdn_scan_tb(device const float4 *qk [[buffer(0)]], device const ushort *conv [[buffer(1)]],
                           device const float *gb [[buffer(2)]], device float *state [[buffer(3)]],
                           device float *y [[buffer(4)]], constant uint &n [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 sq[SCAN_TB][32], sk[SCAN_TB][32];
  threadgroup float sv[SCAN_TB][32], sgb[SCAN_TB][2];
  const uint hv = tg / 4u, r0 = (tg % 4u) * 32u, i0 = r0 + sg * 4u, hk = hv / (NVH / NKH);
  device float4 *S4 = (device float4 *)(state + (size_t)hv * DV * DK);
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(i0 + r) * (DK / 4u) + lane];
  for (uint t0 = 0u; t0 < n; t0 += SCAN_TB) {
    const uint tb = min(SCAN_TB, n - t0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < 2u * SCAN_TB * 32u; e += 256u) {
      const uint which = e / (SCAN_TB * 32u), j = (e / 32u) % SCAN_TB, l = e % 32u;
      if (j < tb) {
        const float4 v = qk[((size_t)(t0 + j) * 2u + which) * (KEYD / 4u) + hk * (DK / 4u) + l];
        if (which == 0u) sq[j][l] = v; else sk[j][l] = v;
      }
    }
    for (uint e = lid; e < SCAN_TB * 32u; e += 256u) {
      const uint j = e / 32u, rr_ = e % 32u;
      if (j < tb) sv[j][rr_] = bf(conv[(size_t)(t0 + j) * CONVD + 2u * KEYD + hv * DV + r0 + rr_]);
    }
    if (lid < 2u * SCAN_TB) {
      const uint j = lid / 2u, w = lid % 2u;
      if (j < tb) sgb[j][w] = gb[(t0 + j) * 2u * NVH + w * NVH + hv];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < tb; j++) {
      const float4 qq = sq[j][lane], kk = sk[j][lane];
      const float g = sgb[j][0], beta = sgb[j][1];
      float yv[4];
      for (uint r = 0u; r < 4u; r++) {
        float4 s = S[r] * g;
        const float kv = simd_sum(dot(s, kk));
        const float delta = (sv[j][sg * 4u + r] - kv) * beta;
        s += kk * delta;
        yv[r] = simd_sum(dot(s, qq));
        S[r] = s;
      }
      if (lane < 4u) y[(size_t)(t0 + j) * VALD + hv * DV + i0 + lane] = rr(yv[lane]);
    }
  }
  for (uint r = 0u; r < 4u; r++) S4[(i0 + r) * (DK / 4u) + lane] = S[r];
}
