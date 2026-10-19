// Flash-Next decode kernels at R rows (R = 1 decode step, R = 4 MTP-verify shape): hyper-connection read (hc_pre +
// hc_mix), GDN in_proj (+ conv1d + gates), gated delta-rule step (+ gated norm), out_proj (+ HC write). Trunk weights
// are MLX 8-bit affine (bf16 scales and biases), HC mixes group 64, GDN projections group 128 on the K5 pack's layer 0
// (the group size is a template parameter, instantiated for 64 and 128). Every weight is read once for all R rows.
// Activations are row-major [R][...]; the R rows of a GDN step are consecutive tokens (the recurrence runs them in
// order inside the kernel).
//
// Reference math: mlx-vlm 0.7.7 qwen4_exp (Qwen4ExpGatedResidual, Qwen4ExpGatedDeltaNet, Qwen3_5GatedDeltaNet,
// gated_delta step ops), as used by fn_exl3.py. Design notes: docs/kernels.md.
#include "ne_common.h"

constant constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID, LR = 320u;
constant constexpr uint NKH = 16u, NVH = 48u, DK = 128u, DV = 128u;
constant constexpr uint KEYD = NKH * DK, VALD = NVH * DV, CONVD = 2u * KEYD + VALD;  // 2048, 6144, 10240


// ------------------------------------------------------------------------------------------------ HC read, part 1
// normed = rmsnorm_per_stream(h) * w (hc_norm, ones-centred in the pack -> multiply directly);
// mix[r][320] = silu((W_down . normed_r) / 4); inj[r][s] = 2 sigmoid((W_inj[s] . normed_r) / 4);
// inj[r][4 + s] = 1/rms(row r, stream s).
// Grid: (LR/8 down-row blocks + 1 inject block) x 4 streams threadgroups, 256 threads. Threadgroup (block b, stream
// s): stages x_s = h_s * w_s for the R rows (f32 at R = 1, f16 above: 32 KB of threadgroup memory) and the rms per
// row, each simdgroup dots one weight row's stream-s slice (q8 down rows; bf16 block_inject rows 0-3 in the inject
// block) with the R staged rows and scales by 1/rms (the dot is linear in each stream's slice). The last of the 4
// stream threadgroups of a block sums the 4 partials in fixed order (deterministic), applies the activation and
// resets the block counter. Partials go through atomics (plain stores were not visible to the finisher).
constant constexpr uint HC_NB = LR / 8u + 1u;  // row blocks: 40 down + 1 inject
template <uint G, uint R>
kernel void fn_hc_pre(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                      device const uchar *dw [[buffer(2)]], device const ushort *ds [[buffer(3)]],
                      device const ushort *db [[buffer(4)]], device const ushort *iw [[buffer(5)]],
                      device float *mix [[buffer(6)]], device float *inj [[buffer(7)]],
                      device atomic_float *part [[buffer(8)]], device atomic_uint *ctr [[buffer(9)]],
                      uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  typedef typename metal::conditional<R == 1u, float, half>::type XT;
  typedef typename metal::conditional<R == 1u, float4, half4>::type XT4;
  threadgroup XT xs[R * HID];
  threadgroup float red[R][8];
  threadgroup uint last;
  const uint blk = tg / HC, s = tg % HC;
  const bool injb = blk == HC_NB - 1u;
  const uint row = blk * 8u + sg;
  // weight loads first (in flight during the staging)
  constexpr uint NQ = HID / 16u / 32u;  // 5 uint4 (16 q8 inputs) per lane
  constexpr uint NB = HID / 8u / 32u;   // 10 uint4 (8 bf16 inputs) per lane
  uint4 wq[NB];
  if (!injb) {
    const device uint4 *w4 = (const device uint4 *)(dw + (size_t)row * HCD + s * HID);
    for (uint j = 0u; j < NQ; j++) wq[j] = w4[lane + 32u * j];
  } else if (sg < HC) {
    const device uint4 *w4 = (const device uint4 *)(iw + (size_t)sg * HCD + s * HID);
    for (uint j = 0u; j < NB; j++) wq[j] = w4[lane + 32u * j];
  }
  for (uint r = 0u; r < R; r++) {
    float ss = 0.0f;
    for (uint i = lid; i < HID; i += 256u) {
      const float v = h[r * HCD + s * HID + i];
      ss += v * v;
      xs[r * HID + i] = XT(v * bf(nw[s * HID + i]));
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[r][sg] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[R];
  for (uint r = 0u; r < R; r++) {
    float t = 0.0f;
    for (uint g = 0u; g < 8u; g++) t += red[r][g];
    inv[r] = rsqrt(t / float(HID) + 1e-6f);
  }
  if (blk == 0u && lid < R) inj[lid * 2u * HC + HC + s] = inv[lid];
  float acc[R];
  for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
  if (!injb) {
    const device ushort *sc = ds + (size_t)row * (HCD / G) + s * (HID / G);
    const device ushort *bi = db + (size_t)row * (HCD / G) + s * (HID / G);
    for (uint j = 0u; j < NQ; j++) {
      const uint c = lane + 32u * j;
      const uint g = (16u * c) / G;
      const float sgv = bf(sc[g]), bgv = bf(bi[g]);
      for (uint r = 0u; r < R; r++) {
        const threadgroup XT4 *x4 = (const threadgroup XT4 *)(xs + r * HID);
        const float2 d = dot16q(wq[j], float4(x4[4u * c]), float4(x4[4u * c + 1u]), float4(x4[4u * c + 2u]),
                                float4(x4[4u * c + 3u]));
        acc[r] += sgv * d.x + bgv * d.y;
      }
    }
  } else if (sg < HC) {
    for (uint j = 0u; j < NB; j++) {
      const uint c = lane + 32u * j;
      for (uint r = 0u; r < R; r++) {
        const threadgroup XT4 *x4 = (const threadgroup XT4 *)(xs + r * HID);
        acc[r] += dot8bf(wq[j], float4(x4[2u * c]), float4(x4[2u * c + 1u]));
      }
    }
  }
  for (uint r = 0u; r < R; r++) {
    const float v = simd_sum(acc[r]) * inv[r];
    if (lane == 0u) atomic_store_explicit(&part[((blk * HC + s) * 8u + sg) * R + r], v, memory_order_relaxed);
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (lid == 0u) last = atomic_fetch_add_explicit(&ctr[blk], 1u, memory_order_relaxed) == HC - 1u ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last == 0u || sg != 0u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (lane < 8u) {
    for (uint r = 0u; r < R; r++) {
      device atomic_float *p = part + (blk * HC * 8u + lane) * R + r;
      constexpr uint SS = 8u * R;  // stream stride
      const float v = ((atomic_load_explicit(&p[0], memory_order_relaxed) +
                        atomic_load_explicit(&p[SS], memory_order_relaxed)) +
                       atomic_load_explicit(&p[2u * SS], memory_order_relaxed)) +
                      atomic_load_explicit(&p[3u * SS], memory_order_relaxed);
      if (!injb) mix[r * LR + blk * 8u + lane] = rr(silu(rr(v) * 0.25f));
      else if (lane < HC) inj[r * 2u * HC + lane] = rr(2.0f * sigm(rr(v) * 0.25f));
    }
  }
  if (lane == 0u) atomic_store_explicit(&ctr[blk], 0u, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------ HC read, part 2
// out[r][c] = mean_s sigmoid(W_up[s*HID + c] . mix_r) * normed_r[s*HID + c]. Grid HID/8 threadgroups, 256 threads:
// simdgroup = one column c, 8 lanes per stream row (a 320-input q8 row = 40 x 8 bytes, 5 per lane, 1 per group).
template <uint G, uint R>
kernel void fn_hc_mix(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                      device const uchar *uw [[buffer(2)]], device const ushort *us [[buffer(3)]],
                      device const ushort *ub [[buffer(4)]], device const float *mix [[buffer(5)]],
                      device const float *inj [[buffer(6)]], device float *out [[buffer(7)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  static_assert(LR == 320u, "8 lanes x 5 x 8 bytes");
  const uint c = tg * 8u + sg;
  const uint s = lane >> 3u, l8 = lane & 7u;
  const uint row = s * HID + c;
  const device uint2 *w2 = (const device uint2 *)(uw + (size_t)row * LR);
  float acc[R];
  for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
  for (uint i = 0u; i < 5u; i++) {
    const uint ch = l8 + 8u * i;  // 8-byte chunk
    const uint g = (8u * ch) / G;
    const float sc = bf(us[(size_t)row * (LR / G) + g]), bi = bf(ub[(size_t)row * (LR / G) + g]);
    const uint2 u = w2[ch];
    for (uint r = 0u; r < R; r++) {
      const device float4 *m4 = (const device float4 *)(mix + r * LR);
      const float2 d = dot8q(u, m4[2u * ch], m4[2u * ch + 1u]);
      acc[r] += sc * d.x + bi * d.y;
    }
  }
  const float w = bf(nw[row]);
  for (uint r = 0u; r < R; r++) {
    float a = acc[r];
    a += simd_shuffle_xor(a, 1u);
    a += simd_shuffle_xor(a, 2u);
    a += simd_shuffle_xor(a, 4u);
    // reference: up, normed, sigmoid and the product are bf16
    float v = rr(rr(sigm(rr(a))) * rr(h[r * HCD + row] * inj[r * 2u * HC + HC + s] * w));
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 16u);
    if (lane == 0u) out[r * HID + c] = rr(v * 0.25f);
  }
}

// ------------------------------------------------------------------------------------------------ HC read v2 (lab-speed)
// Same math, same rounding, bit-identical outputs; no cross-threadgroup finisher (atomics + fences + counter +
// a dependent second load chain cost ~5 us in fn_hc_pre). fn_hc_pre2: as fn_hc_pre up to the per-stream partial
// dots, which are plain-stored to part[blk][s][sg][R] (1/rms_s in the inject block's unused slot sg = 4).
// fn_hc_mix2 (next dispatch, so plain stores are visible): every threadgroup sums the 4 stream partials of the 320
// mix rows (+ 4 inject rows) in the same fixed order, applies silu / 2 sigmoid into threadgroup memory, and runs
// fn_hc_mix's body on it; threadgroup 0 also stores mix [R][320] and inj [R][8] (4 gates for out_proj / moe_down,
// 4 x 1/rms). fn_hc_mix2 reads nothing that it writes (inj is written by threadgroup 0 only, read by nobody here).
template <uint G, uint R>
kernel void fn_hc_pre2(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                       device const uchar *dw [[buffer(2)]], device const ushort *ds [[buffer(3)]],
                       device const ushort *db [[buffer(4)]], device const ushort *iw [[buffer(5)]],
                       device float *inj [[buffer(6)]], device float *part [[buffer(7)]],
                       uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  typedef typename metal::conditional<R == 1u, float, half>::type XT;
  typedef typename metal::conditional<R == 1u, float4, half4>::type XT4;
  threadgroup XT xs[R * HID];
  threadgroup float red[R][8];
  const uint blk = tg / HC, s = tg % HC;
  const bool injb = blk == HC_NB - 1u;
  const uint row = blk * 8u + sg;
  constexpr uint NQ = HID / 16u / 32u;
  constexpr uint NB = HID / 8u / 32u;
  uint4 wq[NB];
  if (!injb) {
    const device uint4 *w4 = (const device uint4 *)(dw + (size_t)row * HCD + s * HID);
    for (uint j = 0u; j < NQ; j++) wq[j] = w4[lane + 32u * j];
  } else if (sg < HC) {
    const device uint4 *w4 = (const device uint4 *)(iw + (size_t)sg * HCD + s * HID);
    for (uint j = 0u; j < NB; j++) wq[j] = w4[lane + 32u * j];
  }
  float sgv[NQ], bgv[NQ];
  if (!injb) {
    const device ushort *sc = ds + (size_t)row * (HCD / G) + s * (HID / G);
    const device ushort *bi = db + (size_t)row * (HCD / G) + s * (HID / G);
    for (uint j = 0u; j < NQ; j++) {
      const uint g = (16u * (lane + 32u * j)) / G;
      sgv[j] = bf(sc[g]);
      bgv[j] = bf(bi[g]);
    }
  }
  for (uint r = 0u; r < R; r++) {
    float ss = 0.0f;
    for (uint i = lid; i < HID; i += 256u) {
      const float v = h[r * HCD + s * HID + i];
      ss += v * v;
      xs[r * HID + i] = XT(v * bf(nw[s * HID + i]));
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[r][sg] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[R];
  for (uint r = 0u; r < R; r++) {
    float t = 0.0f;
    for (uint g = 0u; g < 8u; g++) t += red[r][g];
    inv[r] = rsqrt(t / float(HID) + 1e-6f);
  }
  if (injb && lid < R) part[((blk * HC + s) * 8u + HC) * R + lid] = inv[lid];  // 1/rms_s: unused inject slot sg = 4
  if (injb && sg >= HC) return;
  float acc[R];
  for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
  if (!injb) {
    for (uint j = 0u; j < NQ; j++) {
      const uint c = lane + 32u * j;
      for (uint r = 0u; r < R; r++) {
        const threadgroup XT4 *x4 = (const threadgroup XT4 *)(xs + r * HID);
        const float2 d = dot16q(wq[j], float4(x4[4u * c]), float4(x4[4u * c + 1u]), float4(x4[4u * c + 2u]),
                                float4(x4[4u * c + 3u]));
        acc[r] += sgv[j] * d.x + bgv[j] * d.y;
      }
    }
  } else {
    for (uint j = 0u; j < NB; j++) {
      const uint c = lane + 32u * j;
      for (uint r = 0u; r < R; r++) {
        const threadgroup XT4 *x4 = (const threadgroup XT4 *)(xs + r * HID);
        acc[r] += dot8bf(wq[j], float4(x4[2u * c]), float4(x4[2u * c + 1u]));
      }
    }
  }
  for (uint r = 0u; r < R; r++) {
    const float v = simd_sum(acc[r]) * inv[r];
    if (lane == 0u) part[((blk * HC + s) * 8u + sg) * R + r] = v;
  }
}

// fn_hc_pre3 (lab-speed): fn_hc_pre2 without the staging: simdgroup = one (row, stream) slice, h and hc_norm read from
// L1/L2 by every simdgroup, each simdgroup computes the stream's sum of squares itself (lane-strided over 16-float
// chunks, then simd_sum; fn_hc_pre sums 256-thread-strided + 8 partials: 1/rms can differ in the last bit).
// No threadgroup memory: any R. Same outputs / layout as fn_hc_pre2 (-> fn_hc_mix2).
// lab-fuse (16 Oct): the q8 path's and the bf16 inject path's weight registers are separate (the shared uint4 wq[10]
// sized every simdgroup for the inject rows: -0.12..-0.16 ms per step, bit-identical).
template <uint G, uint R>
kernel void fn_hc_pre3(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                      device const uchar *dw [[buffer(2)]], device const ushort *ds [[buffer(3)]],
                      device const ushort *db [[buffer(4)]], device const ushort *iw [[buffer(5)]],
                      device float *inj [[buffer(6)]], device float *part [[buffer(7)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg / HC, s = tg % HC;
  const bool injb = blk == HC_NB - 1u;
  if (injb && sg >= HC) return;
  const uint row = blk * 8u + sg;
  constexpr uint NQ = HID / 16u / 32u;
  constexpr uint NB = HID / 8u / 32u;
  const device float4 *h4 = (const device float4 *)(h + s * HID);
  const device uint2 *n2 = (const device uint2 *)(nw + s * HID);
  float acc[R], ss[R];
  for (uint r = 0u; r < R; r++) { acc[r] = 0.0f; ss[r] = 0.0f; }
  if (!injb) {
    uint4 wq[NQ];
    float sgv[NQ], bgv[NQ];
    const device uint4 *w4 = (const device uint4 *)(dw + (size_t)row * HCD + s * HID);
    for (uint j = 0u; j < NQ; j++) wq[j] = w4[lane + 32u * j];
    const device ushort *sc = ds + (size_t)row * (HCD / G) + s * (HID / G);
    const device ushort *bi = db + (size_t)row * (HCD / G) + s * (HID / G);
    for (uint j = 0u; j < NQ; j++) {
      const uint g = (16u * (lane + 32u * j)) / G;
      sgv[j] = bf(sc[g]);
      bgv[j] = bf(bi[g]);
    }
    for (uint j = 0u; j < NQ; j++) {
      const uint c = lane + 32u * j;  // 16 inputs 16c..16c+15
      float4 nv[4];
      for (uint e = 0u; e < 4u; e++) nv[e] = bf4(n2[4u * c + e]);
      for (uint r = 0u; r < R; r++) {
        const device float4 *hr = h4 + r * (HCD / 4u);
        float4 xv[4];
        for (uint e = 0u; e < 4u; e++) { const float4 v = hr[4u * c + e]; ss[r] += dot(v, v); xv[e] = v * nv[e]; }
        const float2 d = dot16q(wq[j], xv[0], xv[1], xv[2], xv[3]);
        acc[r] += sgv[j] * d.x + bgv[j] * d.y;
      }
    }
  } else {
    uint4 wq[NB];
    const device uint4 *w4 = (const device uint4 *)(iw + (size_t)sg * HCD + s * HID);
    for (uint j = 0u; j < NB; j++) wq[j] = w4[lane + 32u * j];
    for (uint j = 0u; j < NB; j++) {
      const uint c = lane + 32u * j;  // 8 inputs 8c..8c+7
      const float4 n0 = bf4(n2[2u * c]), n1 = bf4(n2[2u * c + 1u]);
      for (uint r = 0u; r < R; r++) {
        const device float4 *hr = h4 + r * (HCD / 4u);
        const float4 v0 = hr[2u * c], v1 = hr[2u * c + 1u];
        ss[r] += dot(v0, v0) + dot(v1, v1);
        acc[r] += dot8bf(wq[j], v0 * n0, v1 * n1);
      }
    }
  }
  for (uint r = 0u; r < R; r++) {
    const float inv = rsqrt(simd_sum(ss[r]) / float(HID) + 1e-6f);
    const float v = simd_sum(acc[r]) * inv;
    if (lane == 0u) part[((blk * HC + s) * 8u + sg) * R + r] = v;
    if (injb && sg == 0u && lane == 0u) part[((blk * HC + s) * 8u + HC) * R + r] = inv;  // 1/rms_s (slot sg = 4)
  }
}
template <uint G, uint R>
kernel void fn_hc_mix2(device const float *h [[buffer(0)]], device const ushort *nw [[buffer(1)]],
                       device const uchar *uw [[buffer(2)]], device const ushort *us [[buffer(3)]],
                       device const ushort *ub [[buffer(4)]], device float *mix [[buffer(5)]],
                       device float *inj [[buffer(6)]], device float *out [[buffer(7)]],
                       device const float *part [[buffer(8)]],
                       uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  static_assert(LR == 320u, "8 lanes x 5 x 8 bytes");
  threadgroup float ms[R * LR];
  const uint c = tg * 8u + sg;
  const uint s = lane >> 3u, l8 = lane & 7u;
  const uint row = s * HID + c;
  // this simdgroup's weights first (in flight during the mix prologue)
  const device uint2 *w2 = (const device uint2 *)(uw + (size_t)row * LR);
  uint2 wu[5];
  float wsc[5], wbi[5];
  for (uint i = 0u; i < 5u; i++) {
    const uint ch = l8 + 8u * i;
    const uint g = (8u * ch) / G;
    wu[i] = w2[ch];
    wsc[i] = bf(us[(size_t)row * (LR / G) + g]);
    wbi[i] = bf(ub[(size_t)row * (LR / G) + g]);
  }
  const float w = bf(nw[row]);
  float hv[R], iv[R];
  constexpr uint SS = 8u * R;  // stream stride
  constexpr uint IVB = ((HC_NB - 1u) * HC * 8u + HC) * R;  // 1/rms_s at part[IVB + s * SS + r]
  for (uint r = 0u; r < R; r++) { hv[r] = h[r * HCD + row]; iv[r] = part[IVB + s * SS + r]; }
  if (tg == 0u && lid < R * HC) inj[(lid / HC) * 2u * HC + HC + lid % HC] = part[IVB + (lid % HC) * SS + lid / HC];
  // mix (and the inject gates) from the stream partials, fixed order as fn_hc_pre's finisher
  for (uint i = lid; i < R * (LR + HC); i += 256u) {
    const uint r = i / (LR + HC), k = i % (LR + HC);  // k < 320: mix row k; else inject row k - 320
    const uint blk = k / 8u, j = k % 8u;
    const device float *p = part + (blk * HC * 8u + j) * R + r;
    const float v = ((p[0] + p[SS]) + p[2u * SS]) + p[3u * SS];
    if (k < LR) {
      const float m = rr(silu(rr(v) * 0.25f));
      ms[r * LR + k] = m;
      if (tg == 0u) mix[r * LR + k] = m;
    } else if (tg == 0u) {
      inj[r * 2u * HC + (k - LR)] = rr(2.0f * sigm(rr(v) * 0.25f));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[R];
  for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
  for (uint i = 0u; i < 5u; i++) {
    const uint ch = l8 + 8u * i;
    for (uint r = 0u; r < R; r++) {
      const threadgroup float4 *m4 = (const threadgroup float4 *)(ms + r * LR);
      const float2 d = dot8q(wu[i], m4[2u * ch], m4[2u * ch + 1u]);
      acc[r] += wsc[i] * d.x + wbi[i] * d.y;
    }
  }
  for (uint r = 0u; r < R; r++) {
    float a = acc[r];
    a += simd_shuffle_xor(a, 1u);
    a += simd_shuffle_xor(a, 2u);
    a += simd_shuffle_xor(a, 4u);
    float v = rr(rr(sigm(rr(a))) * rr(hv[r] * iv[r] * w));
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 16u);
    if (lane == 0u) out[r * HID + c] = rr(v * 0.25f);
  }
}

// ------------------------------------------------------------------------------------------------ GDN in_proj
// One dispatch for qkv (CONVD rows), z (VALD), b (NVH), a (NVH): 8-bit affine over HID inputs (group GQ for qkv, G for
// z / b / a: the K5 pack has qkv at g64 and z/a/b at g128 on most GDN layers, everything g128 on 7 layers incl. 0),
// one weight row per simdgroup, 8 per threadgroup, R input rows. Epilogues: qkv -> depthwise causal conv1d (kernel 4) over
// [state (3 previous inputs); the R new inputs] + silu, state <- the last 3 (updated in place); z stored;
// b -> beta = sigmoid(b); a -> g = exp(-exp(A_log) softplus(a + dt_bias)). gb = [R][g 48 | beta 48].
template <uint GQ, uint G, uint R>
kernel void fn_gdn_inproj(device const float *x [[buffer(0)]],
                          device const uchar *qw [[buffer(1)]], device const ushort *qs [[buffer(2)]], device const ushort *qb [[buffer(3)]],
                          device const uchar *zw [[buffer(4)]], device const ushort *zs [[buffer(5)]], device const ushort *zb [[buffer(6)]],
                          device const uchar *bw [[buffer(7)]], device const ushort *bs [[buffer(8)]], device const ushort *bb [[buffer(9)]],
                          device const uchar *aw [[buffer(10)]], device const ushort *as [[buffer(11)]], device const ushort *ab [[buffer(12)]],
                          device const ushort *convw [[buffer(13)]], device float *convst [[buffer(14)]],
                          device const ushort *alog [[buffer(15)]], device const ushort *dtb [[buffer(16)]],
                          device float *qkv [[buffer(17)]], device float *z [[buffer(18)]], device float *gb [[buffer(19)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 8u + sg;
  constexpr uint R1 = CONVD, R2 = R1 + VALD, R3 = R2 + NVH, R4 = R3 + NVH;
  if (row >= R4) return;
  constexpr uint NG = HID / G, NGQ = HID / GQ;
  uint r;
  device const uchar *w;
  device const ushort *s, *b;
  if (row < R1) { r = row; w = qw; s = qs; b = qb; }
  else if (row < R2) { r = row - R1; w = zw; s = zs; b = zb; }
  else if (row < R3) { r = row - R2; w = bw; s = bs; b = bb; }
  else { r = row - R3; w = aw; s = as; b = ab; }
  // the epilogue's inputs (conv weights + state, A_log / dt_bias), loaded before the GEMV (lab-speed)
  float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f, s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
  if (lane == 0u) {
    if (row < R1) {
      const device ushort *cw = convw + (size_t)r * 4u;
      e0 = bf(cw[0]); e1 = bf(cw[1]); e2 = bf(cw[2]); e3 = bf(cw[3]);
      s0 = convst[r]; s1 = convst[CONVD + r]; s2 = convst[2u * CONVD + r];
    } else if (row >= R3) {
      e0 = exp(bf(alog[r])); e1 = bf(dtb[r]);
    }
  }
  float v[R];
  if (row < R1) q8rows_dev<HID, GQ, R>(w + (size_t)r * HID, s + (size_t)r * NGQ, b + (size_t)r * NGQ, x, lane, v);
  else q8rows_dev<HID, G, R>(w + (size_t)r * HID, s + (size_t)r * NG, b + (size_t)r * NG, x, lane, v);
  if (lane != 0u) return;
  if (row < R1) {
    const float w0 = e0, w1 = e1, w2 = e2, w3 = e3;
    float seq[3u + R];
    seq[0] = s0; seq[1] = s1; seq[2] = s2;
    for (uint t = 0u; t < R; t++) seq[3u + t] = rr(v[t]);  // reference: qkv rows (and the conv state) are bf16
    for (uint t = 0u; t < R; t++)
      qkv[t * CONVD + r] = rr(silu(rr(seq[t] * w0 + seq[t + 1u] * w1 + seq[t + 2u] * w2 + seq[t + 3u] * w3)));
    convst[r] = seq[R];
    convst[CONVD + r] = seq[R + 1u];
    convst[2u * CONVD + r] = seq[R + 2u];
  } else if (row < R2) {
    for (uint t = 0u; t < R; t++) z[t * VALD + r] = rr(v[t]);
  } else if (row < R3) {
    for (uint t = 0u; t < R; t++) gb[t * 2u * NVH + NVH + r] = rr(sigm(rr(v[t])));
  } else {
    const float e = e0, dt = e1;
    for (uint t = 0u; t < R; t++) {
      const float u = rr(rr(v[t]) + dt);
      const float sp = rr(u > 20.0f ? u : log(1.0f + exp(u)));
      gb[t * 2u * NVH + r] = exp(-e * sp);
    }
  }
}

// ------------------------------------------------------------------------------------------------ GDN step
// Per value head hv (one threadgroup of 1024 threads = 32 simdgroups x 4 value rows, state rows in registers), for
// each of the R tokens in order: q, k = l2-normalised conv outputs of key head hv / 3 (q also * 1/sqrt(DK)); per
// value row i: S_i *= g; kv = S_i . k; S_i += k (v_i - kv) beta; y_i = S_i . q. Then per token
// o = rmsnorm_DV(y) * w * sigmoid(z) (gated norm). State f32 [NVH][DV][DK] (MLX [B, H, Dv, Dk]), updated in place.
template <uint R>
kernel void fn_gdn_step(device const float *qkv [[buffer(0)]], device const float *gb [[buffer(1)]],
                        device const float *z [[buffer(2)]], device const ushort *normw [[buffer(3)]],
                        device float *state [[buffer(4)]], device float *o [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float ys[R][DV];
  const uint hv = tg, hk = hv / (NVH / NKH);
  device float4 *S4 = (device float4 *)(state + (size_t)hv * DV * DK);
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(sg * 4u + r) * (DK / 4u) + lane];
  for (uint t = 0u; t < R; t++) {
    const device float *qt = qkv + t * CONVD;
    const float4 q = ((const device float4 *)(qt + hk * DK))[lane];
    const float4 k = ((const device float4 *)(qt + KEYD + hk * DK))[lane];
    float4 qq, kk;
    if (NE_REFR) {  // reference (bf16 ops): sum of squares, its rsqrt, the products and the scale are bf16
      const float qn = bf16r(rsqrt(bf16r(simd_sum(dot(q, q))) + 1e-6f));
      const float kn = bf16r(rsqrt(bf16r(simd_sum(dot(k, k))) + 1e-6f));
      const float4 q1 = q * qn, k1 = k * kn;
      qq = float4(bf16r(bf16r(q1.x) * 0.08838834764831845f), bf16r(bf16r(q1.y) * 0.08838834764831845f),
                  bf16r(bf16r(q1.z) * 0.08838834764831845f), bf16r(bf16r(q1.w) * 0.08838834764831845f));
      kk = float4(bf16r(k1.x), bf16r(k1.y), bf16r(k1.z), bf16r(k1.w));
    } else {
      const float qn = rsqrt(simd_sum(dot(q, q)) + 1e-6f) * rsqrt(float(DK));
      const float kn = rsqrt(simd_sum(dot(k, k)) + 1e-6f);
      qq = q * qn;
      kk = k * kn;
    }
    const float g = gb[t * 2u * NVH + hv], beta = gb[t * 2u * NVH + NVH + hv];
    const device float *v = qt + 2u * KEYD + hv * DV;
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
  }
  for (uint r = 0u; r < 4u; r++) S4[(sg * 4u + r) * (DK / 4u) + lane] = S[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= R) return;
  const uint t = sg;  // simdgroup t: the gated norm of token t
  float y[4];
  float ss = 0.0f;
  for (uint m = 0u; m < 4u; m++) { y[m] = ys[t][lane + 32u * m]; ss += y[m] * y[m]; }
  const float inv = rsqrt(simd_sum(ss) / float(DV) + 1e-6f);
  for (uint m = 0u; m < 4u; m++) {
    const uint i = lane + 32u * m;
    o[t * VALD + hv * DV + i] = rr(rr(y[m] * inv * bf(normw[i])) * sigm(z[t * VALD + hv * DV + i]));
  }
}

// ------------------------------------------------------------------------------------------------ lazy GDN state store
// lab-gdnlazy: fn_gdn_step with the state store deferred (docs/kernels.md 5d). The 3 MB f32 state write is ~10 us of
// the step's ~19 us per GDN layer; at 1 row the step instead keeps the token's rank-1 update inputs (normalised k,
// v, g, beta: 1 KB per value head) as a pending row and stores S only every K-th step. The next read (the next step,
// or fn_gdn_materialise) applies the pending rows first with the same expressions (gdn_upd), so the state and every
// output are bit-identical to fn_gdn_step's. Per value head (= threadgroup): lz[hv] = pending rows c (< K), pend
// f32 [NVH][GDN_KP][GDN_PST] = rows (kk[128], v[128], g, beta). Each threadgroup owns its head's lz / pend, so no
// cross-threadgroup sync: reads happen before the threadgroup barrier, writes after it. R > 1 (verify / prefill
// rows): apply the pending rows, run the R rows, always store (c -> 0). Every other reader of the state (MTP verify,
// chunked prefill, snapshots, the layer API, tests) runs fn_gdn_materialise first (host: ne::fn::gdn_materialise).
constant constexpr uint GDN_KP = 15u, GDN_PST = 2u * DK + 4u;  // up to 15 pending rows (K <= 16); row stride
static inline void gdn_upd(thread float4 &s, float4 kk, float g, float vi, float beta) {
  s = s * g;
  const float kv = simd_sum(dot(s, kk));
  const float delta = (vi - kv) * beta;
  s += kk * delta;
}
template <uint R, uint K>
kernel void fn_gdn_step_lz(device const float *qkv [[buffer(0)]], device const float *gb [[buffer(1)]],
                           device const float *z [[buffer(2)]], device const ushort *normw [[buffer(3)]],
                           device float *state [[buffer(4)]], device float *o [[buffer(5)]],
                           device uint *lz [[buffer(6)]], device float *pend [[buffer(7)]],
                           uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float ys[R][DV];
  const uint hv = tg, hk = hv / (NVH / NKH);
  device float4 *S4 = (device float4 *)(state + (size_t)hv * DV * DK);
  device float *ph = pend + (size_t)hv * GDN_KP * GDN_PST;
  const uint c = min(lz[hv], GDN_KP);
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(sg * 4u + r) * (DK / 4u) + lane];
  for (uint p = 0u; p < c; p++) {  // pending rows (no outputs)
    const device float *pr = ph + p * GDN_PST;
    const float4 kk = ((const device float4 *)pr)[lane];
    const float g = pr[2u * DK], beta = pr[2u * DK + 1u];
    for (uint r = 0u; r < 4u; r++) gdn_upd(S[r], kk, g, pr[DK + sg * 4u + r], beta);
  }
  float4 kk;
  for (uint t = 0u; t < R; t++) {
    const device float *qt = qkv + t * CONVD;
    const float4 q = ((const device float4 *)(qt + hk * DK))[lane];
    const float4 k = ((const device float4 *)(qt + KEYD + hk * DK))[lane];
    float4 qq;
    if (NE_REFR) {
      const float qn = bf16r(rsqrt(bf16r(simd_sum(dot(q, q))) + 1e-6f));
      const float kn = bf16r(rsqrt(bf16r(simd_sum(dot(k, k))) + 1e-6f));
      const float4 q1 = q * qn, k1 = k * kn;
      qq = float4(bf16r(bf16r(q1.x) * 0.08838834764831845f), bf16r(bf16r(q1.y) * 0.08838834764831845f),
                  bf16r(bf16r(q1.z) * 0.08838834764831845f), bf16r(bf16r(q1.w) * 0.08838834764831845f));
      kk = float4(bf16r(k1.x), bf16r(k1.y), bf16r(k1.z), bf16r(k1.w));
    } else {
      const float qn = rsqrt(simd_sum(dot(q, q)) + 1e-6f) * rsqrt(float(DK));
      const float kn = rsqrt(simd_sum(dot(k, k)) + 1e-6f);
      qq = q * qn;
      kk = k * kn;
    }
    const float g = gb[t * 2u * NVH + hv], beta = gb[t * 2u * NVH + NVH + hv];
    const device float *v = qt + 2u * KEYD + hv * DV;
    for (uint r = 0u; r < 4u; r++) {
      const uint i = sg * 4u + r;
      gdn_upd(S[r], kk, g, v[i], beta);
      const float y = simd_sum(dot(S[r], qq));
      if (lane == 0u) ys[t][i] = rr(y);
    }
  }
  const bool defer = R == 1u && c + 1u < K;
  // the gated norm's z / weight loads go before any store (a load behind a store waits for it)
  const uint tn = min(sg, R - 1u);
  float zl[4], wl[4];
  for (uint m = 0u; m < 4u; m++) {
    const uint i = lane + 32u * m;
    zl[m] = z[tn * VALD + hv * DV + i];
    wl[m] = bf(normw[i]);
  }
  if (!defer)
    for (uint r = 0u; r < 4u; r++) S4[(sg * 4u + r) * (DK / 4u) + lane] = S[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);  // every read of lz / pend of this head is done
  if (sg == 31u) {  // the pending row / counter: a simdgroup without a gated norm to do
    if (defer) {
      device float *pw = ph + c * GDN_PST;
      ((device float4 *)pw)[lane] = kk;
      ((device float4 *)(pw + DK))[lane] = ((const device float4 *)(qkv + 2u * KEYD + hv * DV))[lane];
      if (lane == 0u) {
        pw[2u * DK] = gb[hv];
        pw[2u * DK + 1u] = gb[NVH + hv];
        lz[hv] = c + 1u;
      }
    } else if (lane == 0u && c) {
      lz[hv] = 0u;
    }
  }
  if (sg >= R) return;
  const uint t = sg;
  float y[4];
  float ss = 0.0f;
  for (uint m = 0u; m < 4u; m++) { y[m] = ys[t][lane + 32u * m]; ss += y[m] * y[m]; }
  const float inv = rsqrt(simd_sum(ss) / float(DV) + 1e-6f);
  for (uint m = 0u; m < 4u; m++) {
    const uint i = lane + 32u * m;
    o[t * VALD + hv * DV + i] = rr(rr(y[m] * inv * wl[m]) * sigm(zl[m]));
  }
}
// Applies a GDN layer's pending rows to its state and stores it (lz -> 0). Grid NVH x 1024. Heads without pending
// rows return at once.
kernel void fn_gdn_materialise(device float *state [[buffer(0)]], device uint *lz [[buffer(1)]],
                               device const float *pend [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint hv = tg;
  const uint c = min(lz[hv], GDN_KP);
  if (c == 0u) return;
  device float4 *S4 = (device float4 *)(state + (size_t)hv * DV * DK);
  const device float *ph = pend + (size_t)hv * GDN_KP * GDN_PST;
  float4 S[4];
  for (uint r = 0u; r < 4u; r++) S[r] = S4[(sg * 4u + r) * (DK / 4u) + lane];
  for (uint p = 0u; p < c; p++) {
    const device float *pr = ph + p * GDN_PST;
    const float4 kk = ((const device float4 *)pr)[lane];
    const float g = pr[2u * DK], beta = pr[2u * DK + 1u];
    for (uint r = 0u; r < 4u; r++) gdn_upd(S[r], kk, g, pr[DK + sg * 4u + r], beta);
  }
  for (uint r = 0u; r < 4u; r++) S4[(sg * 4u + r) * (DK / 4u) + lane] = S[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) lz[hv] = 0u;
}

// ------------------------------------------------------------------------------------------------ out_proj + HC write
// y = W_out . o (8-bit affine group G, VALD inputs, HID weight rows, one per simdgroup; R input rows);
// hout[r][s*HID + c] = h[r][s*HID + c] + y[r][c] * inj[r][s] (the hyper-connection write).
template <uint G, uint R>
kernel void fn_gdn_outproj(device const float *o [[buffer(0)]], device const uchar *w [[buffer(1)]],
                           device const ushort *sc [[buffer(2)]], device const ushort *bi [[buffer(3)]],
                           device const float *h [[buffer(4)]], device const float *inj [[buffer(5)]],
                           device float *hout [[buffer(6)]], device float *branch [[buffer(7)]],
                           uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg * 8u + sg;
  constexpr uint NG = VALD / G;
  float hl[R], il[R];  // the HC write's inputs, loaded before the GEMV (lab-speed: off the kernel's tail)
  for (uint r = 0u; r < R; r++) {
    hl[r] = lane < HC ? h[r * HCD + lane * HID + c] : 0.0f;
    il[r] = lane < HC ? inj[r * 2u * HC + lane] : 0.0f;
  }
  float v[R];
  q8rows_dev<VALD, G, R>(w + (size_t)c * VALD, sc + (size_t)c * NG, bi + (size_t)c * NG, o, lane, v);
  for (uint r = 0u; r < R; r++) {
    const float br = rr(v[r]);  // reference: branch, injection and the sum are bf16
    if (lane < HC) hout[r * HCD + lane * HID + c] = rr(hl[r] + rr(br * il[r]));
    if (lane == 0u) branch[r * HID + c] = br;
  }
}

#define NE_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define NE_INST_GR(base, func, G, R) NE_INST(base "_g" #G "_r" #R, func, G, R)
#define NE_INST_ALL(R)                                \
  NE_INST_GR("fn_hc_pre2", fn_hc_pre2, 64, R)         \
  NE_INST_GR("fn_hc_pre2", fn_hc_pre2, 128, R)        \
  NE_INST_GR("fn_hc_pre3", fn_hc_pre3, 64, R)         \
  NE_INST_GR("fn_hc_pre3", fn_hc_pre3, 128, R)        \
  NE_INST_GR("fn_hc_mix2", fn_hc_mix2, 64, R)         \
  NE_INST_GR("fn_hc_mix2", fn_hc_mix2, 128, R)        \
  NE_INST_GR("fn_hc_pre", fn_hc_pre, 64, R)           \
  NE_INST_GR("fn_hc_pre", fn_hc_pre, 128, R)          \
  NE_INST_GR("fn_hc_mix", fn_hc_mix, 64, R)           \
  NE_INST_GR("fn_hc_mix", fn_hc_mix, 128, R)          \
  NE_INST("fn_gdn_inproj_g64_r" #R, fn_gdn_inproj, 64, 64, R)          \
  NE_INST("fn_gdn_inproj_g128_r" #R, fn_gdn_inproj, 128, 128, R)       \
  NE_INST("fn_gdn_inproj_q64_g128_r" #R, fn_gdn_inproj, 64, 128, R)    \
  NE_INST_GR("fn_gdn_outproj", fn_gdn_outproj, 64, R) \
  NE_INST_GR("fn_gdn_outproj", fn_gdn_outproj, 128, R)\
  NE_INST("fn_gdn_step_r" #R, fn_gdn_step, R)          \
  NE_INST("fn_gdn_step_lz1_r" #R, fn_gdn_step_lz, R, 1) \
  NE_INST("fn_gdn_step_lz2_r" #R, fn_gdn_step_lz, R, 2) \
  NE_INST("fn_gdn_step_lz3_r" #R, fn_gdn_step_lz, R, 3) \
  NE_INST("fn_gdn_step_lz4_r" #R, fn_gdn_step_lz, R, 4) \
  NE_INST("fn_gdn_step_lz6_r" #R, fn_gdn_step_lz, R, 6) \
  NE_INST("fn_gdn_step_lz8_r" #R, fn_gdn_step_lz, R, 8) \
  NE_INST("fn_gdn_step_lz12_r" #R, fn_gdn_step_lz, R, 12) \
  NE_INST("fn_gdn_step_lz16_r" #R, fn_gdn_step_lz, R, 16)
NE_INST_ALL(1)
NE_INST_ALL(2)
NE_INST_ALL(3)
NE_INST_ALL(4)
NE_INST_ALL(5)
NE_INST_ALL(6)
NE_INST_ALL(8)  // fn_hc_pre_*_r8 compiles but needs 40 KB of staging: callers split it into 2 x 4 rows (docs/mtp.md)
