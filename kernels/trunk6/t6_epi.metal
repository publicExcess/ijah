// lab-t6wire: epilogue kernels for the layer graph's EXL3 / bf16 trunk Linears (models/flash_next/layer_t6.mm).
// The q8 path fuses each Linear with its epilogue (kernels/fn_trunk.metal, fn_attn.metal, fn_moe.metal, nax_q8.metal
// EPI 1..6); with EXL3 weights the Linear is t6::record into a raw f32 [R][out] buffer and these kernels apply the same
// epilogue math and rounding points (rr = bf16 rounding when NE_REFR) on it. R = rows (consecutive tokens), runtime.
#include "kernels/ne_common.h"

constant constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID, LR = 320u;
constant constexpr uint NVH = 48u, VALD = 6144u, CONVD = 10240u, KVD = 512u, IQD = 640u, INTER = 640u;

// ---------------------------------------------------------------- HC read, part 1 (before the down Linear)
// Threadgroup (row r, stream s, 256-element chunk ch), 256 threads, R * 4 * 10 threadgroups: inv = 1/rms(h_s) (every
// threadgroup sums the whole stream: 10 KB from L2); xn[r][s*HID + i] = h_s[i] * w_s[i] * inv for its chunk (the down
// Linear's input: the q8 path's per-stream 1/rms is a row scale of its per-stream partial dots); ip[r][s][ch][j] =
// (W_inj[j][chunk] . (h_s * w_s)[chunk]) * inv for the 4 inject rows (bf16, when has_inj), ip[r][s][0][4] = inv.
constant constexpr uint HC_CH = HID / 256u;  // 10 chunks per stream
kernel void t6e_hc_prep(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                        device const ushort *iw [[buffer(2)]], device float *xn [[buffer(3)]],
                        device float *ip [[buffer(4)]], constant uint &has_inj [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8][5];
  const uint ch = tg % HC_CH, rs = tg / HC_CH, r = rs / HC, s = rs % HC;
  const device float *hr = h + (size_t)r * HCD + s * HID;
  float ss = 0.0f;
  for (uint j = 0u; j < HC_CH; j++) {
    const float v = hr[lid + 256u * j];
    ss += v * v;
  }
  const uint i = ch * 256u + lid;
  const float xv = hr[i] * bf(nw[s * HID + i]);
  float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (has_inj != 0u)
    for (uint k = 0u; k < 4u; k++) a[k] = bf(iw[(size_t)k * HCD + s * HID + i]) * xv;
  ss = simd_sum(ss);
  for (uint k = 0u; k < 4u; k++) a[k] = simd_sum(a[k]);
  if (lane == 0u) {
    red[sg][0] = ss;
    for (uint k = 0u; k < 4u; k++) red[sg][1u + k] = a[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = 0u; g < 8u; g++)
    for (uint k = 0u; k < 5u; k++) t[k] += red[g][k];
  const float inv = rsqrt(t[0] / float(HID) + 1e-6f);
  xn[(size_t)r * HCD + s * HID + i] = xv * inv;
  device float *o = ip + ((r * HC + s) * HC_CH + ch) * 8u;
  if (lid < 4u) o[lid] = t[1u + lid] * inv;
  if (lid == 4u) o[4] = inv;
}

// HC read, part 2 (after the down Linear): mix[r][k] = rr(silu(rr(mraw) / 4)); inj[r][j] = rr(2 sigmoid(rr(sum_s
// sum_ch ip[r][s][ch][j]) / 4)) (fixed order), inj[r][4 + s] = 1/rms_s. Grid ceil(R * 328 / 256).
kernel void t6e_hc_act(device const float *mraw [[buffer(0)]], device const float *ip [[buffer(1)]],
                       device float *mix [[buffer(2)]], device float *inj [[buffer(3)]], constant uint &R [[buffer(4)]],
                       constant uint &do_mix [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  constexpr uint W = LR + 2u * HC;
  if (gid >= R * W) return;
  const uint r = gid / W, k = gid % W;
  const device float *p = ip + r * HC * HC_CH * 8u;
  if (k < LR) {
    if (do_mix != 0u)  // else the down Linear's finisher writes mix (fused epilogue 2)
    mix[r * LR + k] = rr(silu(rr(mraw[r * LR + k]) * 0.25f));
  } else if (k < LR + HC) {
    const uint j = k - LR;
    float v = 0.0f;
    for (uint s = 0u; s < HC; s++) {
      float vs = 0.0f;
      for (uint c = 0u; c < HC_CH; c++) vs += p[(s * HC_CH + c) * 8u + j];
      v += vs;
    }
    inj[r * 2u * HC + j] = rr(2.0f * sigm(rr(v) * 0.25f));
  } else {
    const uint s = k - LR - HC;
    inj[r * 2u * HC + HC + s] = p[s * HC_CH * 8u + 4u];
  }
}

// HC read, part 3 (after the up Linear): out[r][c] = rr(mean_s rr(rr(sigmoid(rr(u[s*HID + c]))) * rr(h * inv_s * w))),
// streams summed (0 + 1) + (2 + 3) as fn_hc_mix's shuffles. Grid ceil(R * HID / 256).
kernel void t6e_hc_out(device const float *u [[buffer(0)]], device const float *h [[buffer(1)]],
                       device const ushort *nw [[buffer(2)]], device const float *inj [[buffer(3)]],
                       device float *out [[buffer(4)]], constant uint &R [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= R * HID) return;
  const uint r = gid / HID, c = gid % HID;
  float v[4];
  for (uint s = 0u; s < HC; s++) {
    const uint row = s * HID + c;
    v[s] = rr(rr(sigm(rr(u[(size_t)r * HCD + row]))) * rr(h[(size_t)r * HCD + row] * inj[r * 2u * HC + HC + s] * bf(nw[row])));
  }
  out[r * HID + c] = rr(((v[0] + v[1]) + (v[2] + v[3])) * 0.25f);
}

// ---------------------------------------------------------------- GDN in_proj epilogue (nax_q8 EPI 1 / 5 / 6)
// One thread per column over qkv (10240) | z (6144) | b (48) | a (48), all R rows. qkv: raw in place -> conv1d K4 +
// silu over [conv state; rows] (state updated); mode 5 also writes hist [3 + R][10240] = [state in; raw rows]; mode 6
// (lazy commit) convolves over [state; P pending raw rows (pend [8][10240], P = keep[0]); new rows], commits only the
// pending rows to the state and leaves the new raw rows in pend. z: rr in place. b -> beta, a -> g into gb [R][96].
kernel void t6e_gdn_in(device float *qkv [[buffer(0)]], device float *z [[buffer(1)]],
                       device const float *braw [[buffer(2)]], device const float *araw [[buffer(3)]],
                       device float *gb [[buffer(4)]], device const ushort *convw [[buffer(5)]],
                       device float *cst [[buffer(6)]], device const ushort *alog [[buffer(7)]],
                       device const ushort *dtb [[buffer(8)]], device float *hist [[buffer(9)]],
                       device const uint *keep [[buffer(10)]], constant uint &R [[buffer(11)]],
                       constant uint &mode [[buffer(12)]], uint n [[thread_position_in_grid]]) {
  constexpr uint R1 = CONVD, R2 = R1 + VALD, R3 = R2 + NVH, R4 = R3 + NVH;
  if (n >= R4) return;
  if (n < R1) {
    const device ushort *cw = convw + (size_t)n * 4u;
    const float c0 = bf(cw[0]), c1 = bf(cw[1]), c2 = bf(cw[2]), c3 = bf(cw[3]);
    const uint P = mode == 6u ? min(keep[0], 8u) : 0u;
    float sq[3u + 8u + 16u];
    sq[0] = cst[n]; sq[1] = cst[CONVD + n]; sq[2] = cst[2u * CONVD + n];
    for (uint p = 0u; p < P; p++) sq[3u + p] = hist[(size_t)p * CONVD + n];
    for (uint t = 0u; t < R; t++) sq[3u + P + t] = rr(qkv[(size_t)t * CONVD + n]);
    for (uint t = 0u; t < R; t++)
      qkv[(size_t)t * CONVD + n] = rr(silu(rr(sq[P + t] * c0 + sq[P + t + 1u] * c1 + sq[P + t + 2u] * c2 + sq[P + t + 3u] * c3)));
    if (mode == 5u)
      for (uint t = 0u; t < 3u + R; t++) hist[(size_t)t * CONVD + n] = sq[t];
    const uint e = mode == 6u ? P : R;  // state <- the last 3 of [state; pending] (lazy) or of [state; rows]
    cst[n] = sq[e]; cst[CONVD + n] = sq[e + 1u]; cst[2u * CONVD + n] = sq[e + 2u];
    if (mode == 6u)
      for (uint t = 0u; t < R; t++) hist[(size_t)t * CONVD + n] = sq[3u + P + t];
  } else if (n < R2) {
    const uint j = n - R1;
    for (uint t = 0u; t < R; t++) z[(size_t)t * VALD + j] = rr(z[(size_t)t * VALD + j]);
  } else if (n < R3) {
    const uint j = n - R2;
    for (uint t = 0u; t < R; t++) gb[t * 2u * NVH + NVH + j] = rr(sigm(rr(braw[t * NVH + j])));
  } else {
    const uint j = n - R3;
    const float ex = exp(bf(alog[j])), dt = bf(dtb[j]);
    for (uint t = 0u; t < R; t++) {
      const float u = rr(rr(araw[t * NVH + j]) + dt);
      const float sp = rr(u > 20.0f ? u : log(1.0f + exp(u)));
      gb[t * 2u * NVH + j] = exp(-ex * sp);
    }
  }
}

// ---------------------------------------------------------------- out_proj / o_proj + HC write (EPI 2)
// br raw in place -> rr; hout[r][s][c] = rr(h + rr(br * inj[r][s])). Grid ceil(R * HID / 256).
kernel void t6e_out_hc(device float *br [[buffer(0)]], device const float *h [[buffer(1)]],
                       device const float *inj [[buffer(2)]], device float *hout [[buffer(3)]],
                       constant uint &R [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= R * HID) return;
  const uint r = gid / HID, c = gid % HID;
  const float b = rr(br[gid]);
  br[gid] = b;
  for (uint s = 0u; s < HC; s++)
    hout[(size_t)r * HCD + s * HID + c] = rr(h[(size_t)r * HCD + s * HID + c] + rr(b * inj[r * 2u * HC + s]));
}

// ---------------------------------------------------------------- attention projections (EPI 3 / fn_attn_proj stores)
// q and the indexer qk were written in place by the Linears; kv[r] = [k | v]; the raw indexer key (iqk[r][512..640))
// -> ikeys[pos + r]. Grid ceil(R * (1024 + 128) / 256).
kernel void t6e_attn(device const float *kraw [[buffer(0)]], device const float *vraw [[buffer(1)]],
                     device const float *iqk [[buffer(2)]], device float *kv [[buffer(3)]],
                     device float *ikeys [[buffer(4)]], device const uint *pos [[buffer(5)]],
                     constant uint &R [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  constexpr uint W = 2u * KVD + 128u;
  if (gid >= R * W) return;
  const uint r = gid / W, j = gid % W;
  if (j < KVD) kv[r * 2u * KVD + j] = kraw[r * KVD + j];
  else if (j < 2u * KVD) kv[r * 2u * KVD + j] = vraw[r * KVD + j - KVD];
  else ikeys[(size_t)(pos[0] + r) * 128u + (j - 2u * KVD)] = iqk[r * IQD + 512u + (j - 2u * KVD)];
}

// ---------------------------------------------------------------- shared expert SwiGLU (fn_moe_router / gateup)
// gush[r] = [rr(gate) | rr(up)] (the q8 path's "shared_gu"), act[r][j] = rr(silu(rr gate) * rr up). Grid ceil(R*640/256).
kernel void t6e_swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                       device float *gush [[buffer(2)]], device float *act [[buffer(3)]],
                       constant uint &R [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= R * INTER) return;
  const uint r = gid / INTER, j = gid % INTER;
  const float gv = rr(g[gid]), uv = rr(u[gid]);
  gush[r * 2u * INTER + j] = gv;
  gush[r * 2u * INTER + INTER + j] = uv;
  act[gid] = rr(silu(gv) * uv);
}

// ---------------------------------------------------------------- strided copy (+ rr): dst[t*dld + doff + j] =
// rr?(src[t*sld + soff + j]), j < n, t < R (in place when src == dst with the same layout). Grid ceil(R * n / 256).
struct T6Copy { uint n, R, sld, soff, dld, doff, round; };
kernel void t6e_copy(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                     constant T6Copy &c [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= c.R * c.n) return;
  const uint t = gid / c.n, j = gid % c.n;
  const float v = src[(size_t)t * c.sld + c.soff + j];
  dst[(size_t)t * c.dld + c.doff + j] = c.round != 0u ? rr(v) : v;
}
