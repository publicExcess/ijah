// lab-trunk6: EXL3 dense trunk Linears (K6 / K7 / K8, MCG window 15, Hadamard + suh / svh folded in) and bf16 dense.
//
//   y[r] = x[r] @ W,  W [in, out] = diag(suh) . H_hi . W' . H_ho . diag(svh)   (r < R rows, x / y f32 row-major)
//   W' = the trellis: U16 [in/16][out/16][16 K] (tile (tk, tn) = W'[16 tk.., 16 tn..], tensor-core slot order);
//   H_n = block-diagonal normalised Sylvester Hadamard, n = 128, 64, 32 or 16 (HC 320 -> 64, GDN a/b 48 -> 16).
//
// t6_simd  (1..8 rows): threadgroup = 8 simdgroups = 128 output columns (8 tiles) x one k split of S; prepares its k
//          span of x * suh -> H_hi (half, k-permuted per tile) in threadgroup memory; simdgroup = one output tile,
//          a lane decodes 8 weights per k tile (tensor-core order: rows 2q, 2q+1, 2q+8, 2q+9 x cols c, c + 8);
//          S = 1 finishes in place (H_ho * svh), S > 1 through per-split partials + a last-arriving finisher.
// t6_prep + t6_nax (any rows <= 16, meant for >= 4): t6_prep writes xh[r][in] = half(H_hi (x * suh)), k-permuted per
//          tile, to device scratch (one dispatch, then a barrier); in t6_nax a threadgroup = KS simdgroups on one
//          32-column block (2 tiles) x one k split of S; each simdgroup decodes the two tiles of its k steps straight
//          into matmul2d's right operand (16x32x16; lane l reads tensor-core thread 4 c0 + ((l & 1) | ((l >> 3) & 1)
//          << 1): lab lead 6's "remap", no re-encode), x rows from xh into the left; KS partials reduce in threadgroup
//          memory, the 128-column block's pieces (4 column blocks x S splits) through partials + a last-arriving
//          finisher. (Preparing x inside every threadgroup, as t6_simd does, cost 2x at 8 rows: R x span halves of
//          threadgroup memory per split and little work per simdgroup.)
// t6_bf16  (1..8 rows): plain bf16 GEMV, HF layout W [out][in] (the final hyper-connection mixer).
// Every finisher sums its partials in a fixed order: results are deterministic; counters reset (replayable).
//
// Attribution: the slot order, MCG decode and the simd-shuffle FWHT follow sushi (github.com/beamivalice/sushi, MIT,
// (c) 2026 Theinruj Toranavikrai, David Dalcu) and exllamav3 (turboderp, MIT); the lane funnel reader and the dense
// GEMV shape are lab leads 1 / 4 (engine/roofline, engine/fused), the NAX operand layout lab lead 6 (engine/nax).
// MIT notice: permission is hereby granted, free of charge, to any person obtaining a copy of this software ... THE
// SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND (full text in each upstream LICENSE).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "kernels/ne_common.h"
using namespace mpp::tensor_ops;

constant uint T6_K [[function_constant(10)]];    // trellis bits per weight: 6, 7 or 8
constant uint T6_IN [[function_constant(11)]];   // inputs (multiple of 16 and of HI)
constant uint T6_OUT [[function_constant(12)]];  // outputs (multiple of 16 and of HO)
constant uint T6_HI [[function_constant(13)]];   // input Hadamard size (16..128)
constant uint T6_HO [[function_constant(14)]];   // output Hadamard size
constant uint T6_S [[function_constant(15)]];    // k splits across threadgroups (span = IN / 16 / S tiles)
constant uint T6_R [[function_constant(16)]];    // rows (SIMD <= 8, NAX <= 16)
constant uint T6_KS [[function_constant(17)]];   // NAX: simdgroups per threadgroup (k split inside)

// normalised Hadamard of size N (16..128) on a 128-element chunk held as (lane, lane+32, lane+64, lane+96):
// blocks of N never straddle the chunk; elements outside the matrix are zero and stay in their own block.
static inline float4 t6_fwht(float4 v, uint lane, uint N) {
  for (ushort bit = 1u; bit < 32u && bit < N; bit <<= 1u) {
    const float4 p = float4(simd_shuffle_xor(v.x, bit), simd_shuffle_xor(v.y, bit), simd_shuffle_xor(v.z, bit),
                            simd_shuffle_xor(v.w, bit));
    v = ((lane & bit) == 0u) ? v + p : p - v;
  }
  if (N >= 64u) v = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
  if (N >= 128u) v = float4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);
  return v * rsqrt(float(N));
}

// lab-t6fuse: input preparation folded into the GEMV (T6_PRE, FC 23; the e0..e3 buffers belong to it):
//   0 plain   x
//   1 HC norm x = h (f32 [R][10240]); input i = (h[i] * bf(e0[i])) * inv[r] (e0 = hc_norm bf16, inv = 1/rms of the
//             threadgroup's stream, computed in the threadgroup: the separate t6e_hc_prep's math and order)
//   2 SwiGLU  x = shared gate raw, e0 = shared up raw (f32 [R][640]); input = rr(silu(rr g) * rr u) (= t6e_swiglu)
constant uint T6_PRE_FC [[function_constant(23)]];
constant uint T6_PRE = is_function_constant_defined(T6_PRE_FC) ? T6_PRE_FC : 0u;
static inline float t6_xin(device const float *x, uint r, uint gi, device const float *e0, thread const float *inv) {
  const size_t o = (size_t)r * T6_IN + gi;
  if (T6_PRE == 1u) return x[o] * bf(((device const ushort *)e0)[gi]) * inv[r];
  if (T6_PRE == 2u) return rr(silu(rr(x[o])) * rr(e0[o]));
  return x[o];
}

// x * suh -> H_hi for rows 0..R-1 and inputs [k0e, k0e + n) -> xs[r * n + kp(i)] (half); all threads of the group.
static inline void t6_prep(device const float *x, device const half *suh, threadgroup half *xs, uint k0e, uint n,
                           uint R, uint sg, uint nsg, uint lane, device const float *e0 = nullptr,
                           thread const float *inv = nullptr) {
  const uint nch = (n + 127u) / 128u;
  for (uint job = sg; job < R * nch; job += nsg) {
    const uint r = job / nch, c = (job - r * nch) * 128u;
    float4 v;
    for (uint j = 0u; j < 4u; j++) {
      const uint i = c + lane + 32u * j;
      v[j] = i < n ? t6_xin(x, r, k0e + i, e0, inv) * float(suh[k0e + i]) : 0.0f;
    }
    v = t6_fwht(v, lane, T6_HI);
    for (uint j = 0u; j < 4u; j++) {
      const uint i = c + lane + 32u * j;
      if (i < n) xs[r * n + kp(i)] = half(v[j]);
    }
  }
}

// ---------------------------------------------------------------- trellis reader, any K <= 8 (window 15)
// Lane `ln` (tensor-core thread) owns ring slots 8 ln .. 8 ln + 7. Only the low 15 bits of a codeword are read, so
// its 8 codewords span 15 + 7 K bits (<= 64 for K <= 7; 71 at K = 8), starting at bit (8 ln + 1) K - 15 of the ring:
// 3 words cover it. The word indices and the shift depend on the lane only (computed once per kernel).
struct T6Lane {
  uint i0, i1, i2, a;
};
static inline T6Lane t6_lane(uint ln) {
  const uint K = T6_K, NW = 8u * K, TOT = 256u * K;
  uint s = (8u * ln + 1u) * K + TOT - 15u;
  s = s >= TOT ? s - TOT : s;
  T6Lane L;
  L.i0 = s >> 5u;
  L.i1 = L.i0 + 1u >= NW ? L.i0 + 1u - NW : L.i0 + 1u;
  L.i2 = L.i1 + 1u >= NW ? L.i1 + 1u - NW : L.i1 + 1u;
  L.a = s & 31u;
  return L;
}
static inline uint3 t6_load(const device uint *wp, T6Lane L) { return uint3(wp[L.i0], wp[L.i1], wp[L.i2]); }
// 8 weights of the lane's slots: w0 = slots 0..3 (rows 2q, 2q+1, 2q+8, 2q+9 at col c), w1 = slots 4..7 (col c + 8)
static inline void t6_dec(uint3 ww, T6Lane L, thread float4 &w0, thread float4 &w1) {
  uint cw[8];
  if (T6_K == 8u) {  // window starts at bit 25 of word i0 (8 K = 64: lane-independent); codeword j ends at 40 + 8 j
    for (uint j = 0u; j < 8u; j++) {
      const uint E = 40u + 8u * j;
      cw[j] = E <= 64u ? cw_at(ww.xy, E) : cw_at(ww.yz, E - 32u);
    }
  } else {
    const uint a = L.a;
    const uint2 v = uint2((ww.x << a) | ((ww.y >> 1u) >> (31u - a)), (ww.y << a) | ((ww.z >> 1u) >> (31u - a)));
    for (uint j = 0u; j < 8u; j++) cw[j] = cw_at(v, j * T6_K + 15u);
  }
  float wv[8];
  for (uint p = 0u; p < 4u; p++) {
    const float2 d = dec_mcg(uint2(cw[2u * p], cw[2u * p + 1u]));
    wv[2u * p] = d.x;
    wv[2u * p + 1u] = d.y;
  }
  w0 = float4(wv[0], wv[1], wv[2], wv[3]);
  w1 = float4(wv[4], wv[5], wv[6], wv[7]);
}

// ---------------------------------------------------------------- finisher (shared)
// o [R][128] (threadgroup) holds this block's raw outputs (before H_ho); applies H_ho * svh and stores y (row stride
// T6_LDY). lab-t6wire: T6_EPI != 0 applies a fused epilogue instead (the layer graph's q8 epilogues): the finished
// values go back into o, then one thread per column runs the epilogue over the R rows (rr = bf16 rounding when the
// reference rounding FC 0 is on). e0..e3 = the epilogue's extra buffers:
//   1 out_hc  y = branch = rr(v); e2[r][s][c] = rr(e0[r][s][c] + rr(branch * e1[r][s]))   (h, inj [R][8], hout)
//   2 hc act  y = rr(silu(rr(v) / 4))
//   3 round   y = rr(v)
//   4 gdn qkv conv1d K4 + silu over [state e1 (3 rows); (pending e2, P = e3[0], mode 6); rows], state updated, mode 5:
//             e2 = [state in; raw rows] history (T6_MODE: 1 / 5 / 6; e0 = conv weight bf16 [OUT][4])
//   5 beta    y[r][48 + c] = rr(sigmoid(rr(v)))  (y = gb [R][96])
//   6 decay   y[r][c] = exp(-exp(A_log) softplus(rr(rr(v) + dt_bias)))  (e0 = A_log, e1 = dt_bias, bf16)
//   7 iqk     y = v, and for c >= 512 the raw indexer key -> e0[(e1[0] + r) * 128 + c - 512] (cache sink, pos u32)
constant uint T6_EPI_FC [[function_constant(20)]];
constant uint T6_EPI = is_function_constant_defined(T6_EPI_FC) ? T6_EPI_FC : 0u;
constant uint T6_MODE_FC [[function_constant(21)]];
constant uint T6_MODE = is_function_constant_defined(T6_MODE_FC) ? T6_MODE_FC : 1u;
constant uint T6_LDY_FC [[function_constant(22)]];
constant uint T6_LDY = is_function_constant_defined(T6_LDY_FC) ? T6_LDY_FC : T6_OUT;

static inline void t6_epi_col(threadgroup const float *o, uint l, uint c, uint R, device float *y,
                              device float *e0, device float *e1, device float *e2, device float *e3) {
  constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID;
  if (T6_EPI == 1u) {
    for (uint r = 0u; r < R; r++) {
      const float br = rr(o[r * 128u + l]);
      y[(size_t)r * T6_LDY + c] = br;
      for (uint s = 0u; s < HC; s++)
        e2[(size_t)r * HCD + s * HID + c] = rr(e0[(size_t)r * HCD + s * HID + c] + rr(br * e1[r * 8u + s]));
    }
  } else if (T6_EPI == 2u) {
    for (uint r = 0u; r < R; r++) y[(size_t)r * T6_LDY + c] = rr(silu(rr(o[r * 128u + l]) * 0.25f));
  } else if (T6_EPI == 3u) {
    for (uint r = 0u; r < R; r++) y[(size_t)r * T6_LDY + c] = rr(o[r * 128u + l]);
  } else if (T6_EPI == 4u) {
    constexpr uint CD = 10240u;
    const device ushort *cw = (const device ushort *)e0 + (size_t)c * 4u;
    const float c0 = bf(cw[0]), c1 = bf(cw[1]), c2 = bf(cw[2]), c3 = bf(cw[3]);
    const uint P = T6_MODE == 6u ? min(((const device uint *)e3)[0], 8u) : 0u;
    float sq[3u + 8u + 16u];
    sq[0] = e1[c]; sq[1] = e1[CD + c]; sq[2] = e1[2u * CD + c];
    for (uint p = 0u; p < P; p++) sq[3u + p] = e2[(size_t)p * CD + c];
    for (uint t = 0u; t < R; t++) sq[3u + P + t] = rr(o[t * 128u + l]);
    for (uint t = 0u; t < R; t++)
      y[(size_t)t * T6_LDY + c] = rr(silu(rr(sq[P + t] * c0 + sq[P + t + 1u] * c1 + sq[P + t + 2u] * c2 + sq[P + t + 3u] * c3)));
    if (T6_MODE == 5u)
      for (uint t = 0u; t < 3u + R; t++) e2[(size_t)t * CD + c] = sq[t];
    const uint e = T6_MODE == 6u ? P : R;
    e1[c] = sq[e]; e1[CD + c] = sq[e + 1u]; e1[2u * CD + c] = sq[e + 2u];
    if (T6_MODE == 6u)
      for (uint t = 0u; t < R; t++) e2[(size_t)t * CD + c] = sq[3u + P + t];
  } else if (T6_EPI == 5u) {
    for (uint r = 0u; r < R; r++) y[(size_t)r * 96u + 48u + c] = rr(sigm(rr(o[r * 128u + l])));
  } else if (T6_EPI == 6u) {
    const float ex = exp(bf(((const device ushort *)e0)[c])), dt = bf(((const device ushort *)e1)[c]);
    for (uint r = 0u; r < R; r++) {
      const float u = rr(rr(o[r * 128u + l]) + dt);
      const float sp = rr(u > 20.0f ? u : log(1.0f + exp(u)));
      y[(size_t)r * 96u + c] = exp(-ex * sp);
    }
  } else if (T6_EPI == 7u) {
    const uint T = ((const device uint *)e1)[0];
    for (uint r = 0u; r < R; r++) {
      const float v = o[r * 128u + l];
      y[(size_t)r * T6_LDY + c] = v;
      if (c >= 512u) e0[(size_t)(T + r) * 128u + (c - 512u)] = v;
    }
  }
}

static inline void t6_finish(threadgroup float *o, device const half *svh, device float *y, uint cb, uint R,
                             uint sg, uint nsg, uint lane, uint lid, uint nthr, device float *e0, device float *e1,
                             device float *e2, device float *e3) {
  for (uint r = sg; r < R; r += nsg) {
    float4 v = float4(o[r * 128u + lane], o[r * 128u + lane + 32u], o[r * 128u + lane + 64u], o[r * 128u + lane + 96u]);
    v = t6_fwht(v, lane, T6_HO);
    for (uint j = 0u; j < 4u; j++) {
      const uint c = cb * 128u + lane + 32u * j;
      const float f = c < T6_OUT ? v[j] * float(svh[c]) : 0.0f;
      if (T6_EPI == 0u) {
        if (c < T6_OUT) y[(size_t)r * T6_LDY + c] = f;
      } else {
        o[r * 128u + lane + 32u * j] = f;
      }
    }
  }
  if (T6_EPI == 0u) return;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (T6_EPI == 8u) return;  // t6_hc_out (needs the counters)
  for (uint l = lid; l < 128u; l += nthr) {
    const uint c = cb * 128u + l;
    if (c < T6_OUT) t6_epi_col(o, l, c, R, y, e0, e1, e2, e3);
  }
}
// lab-t6fuse, EPI 8: the HC read's part 3 (t6e_hc_out) fused into the up Linear's finisher. A finished 128-column
// block (o, after H_ho * svh) is one stream's columns [128 cs, 128 cs + 128), cs = cb % 20: it is published to y (= the
// up output [R][10240], atomics) and counted on ctr[T6_HCO_CTR + cs]; the 4th arrival (the 4 streams' blocks of those
// columns) computes e3[r][c] = rr(mean_s rr(rr(sigmoid(rr(u_s))) * rr(h_s * inv_s * w_s))) (e0 = h [R][4][2560],
// e1 = hc_norm bf16, e2 = inj [R][8], e3 = out [R][2560]; streams summed (0 + 1) + (2 + 3)) and resets the counter.
constant constexpr uint T6_HCO_CTR = 128u;  // counter offset (past every combine counter: OUT / 128 = 80)
static inline void t6_hc_out(threadgroup const float *o, device float *y, device atomic_uint *ctr, uint cb, uint R,
                             uint lid, uint nthr, threadgroup uint *flag, device const float *e0,
                             device const float *e1, device const float *e2, device float *e3) {
  constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID, NB = HID / 128u;
  device atomic_float *ya = (device atomic_float *)y;
  for (uint i = lid; i < R * 128u; i += nthr) {
    const uint r = i / 128u, l = i - r * 128u;
    atomic_store_explicit(&ya[(size_t)r * HCD + cb * 128u + l], o[i], memory_order_relaxed);
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  const uint cs = cb % NB;
  if (lid == 0u) *flag = atomic_fetch_add_explicit(&ctr[T6_HCO_CTR + cs], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*flag != HC - 1u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  const device ushort *nw = (const device ushort *)e1;
  for (uint i = lid; i < R * 128u; i += nthr) {
    const uint r = i / 128u, c = cs * 128u + (i - r * 128u);
    float v[4];
    for (uint s = 0u; s < HC; s++) {
      const uint row = s * HID + c;
      const float u = atomic_load_explicit(&ya[(size_t)r * HCD + row], memory_order_relaxed);
      v[s] = rr(rr(sigm(rr(u))) * rr(e0[(size_t)r * HCD + row] * e2[r * 2u * HC + HC + s] * bf(nw[row])));
    }
    e3[r * HID + c] = rr(((v[0] + v[1]) + (v[2] + v[3])) * 0.25f);
  }
  if (lid == 0u) atomic_store_explicit(&ctr[T6_HCO_CTR + cs], 0u, memory_order_relaxed);
}

// Cross-threadgroup combine: this threadgroup's piece (o [R][128], columns [c0p, c0p + nc) of block cb, split sp) ->
// scr[cb][sp][R][128]; the last of `arrivals` threadgroups of the block sums the S splits (fixed order) into o and
// returns true. Atomics for the partials (plain stores + fence read stale values: docs/kernels.md traps).
static inline bool t6_combine(threadgroup float *o, device atomic_float *scr, device atomic_uint *ctr, uint cb,
                              uint sp, uint c0p, uint nc, uint arrivals, uint R, uint lid, uint nthreads,
                              threadgroup uint *flag) {
  const uint S = T6_S;
  device atomic_float *p = scr + ((size_t)cb * S + sp) * R * 128u;
  for (uint i = lid; i < R * nc; i += nthreads) {
    const uint r = i / nc, c = c0p + (i - r * nc);
    atomic_store_explicit(&p[r * 128u + c], o[r * 128u + c], memory_order_relaxed);
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (lid == 0u) *flag = atomic_fetch_add_explicit(&ctr[cb], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*flag != arrivals - 1u) return false;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  device atomic_float *pb = scr + (size_t)cb * S * R * 128u;
  for (uint i = lid; i < R * 128u; i += nthreads) {
    float v = 0.0f;
    for (uint s = 0u; s < S; s++) v += atomic_load_explicit(&pb[(size_t)s * R * 128u + i], memory_order_relaxed);
    o[i] = v;
  }
  if (lid == 0u) atomic_store_explicit(&ctr[cb], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return true;
}

// lab-t6fuse, PRE 1 (the HC read's part 1, t6e_hc_prep, folded into the down Linear): the threadgroup's k span
// [k0e, k0e + n) lies in one stream s (host check: 2560 % n == 0); inv[r] = 1/rms(h_s) (the whole stream, 10 KB from
// L2, same sums and order as t6e_hc_prep). Threadgroups of column block 0 (inj) also compute the 4 inject rows'
// partials of their span's 256-element chunks: ip[r][s][ch][j] = (W_inj[j][chunk] . (h_s * w_s)[chunk]) * inv,
// ip[r][s][ch][4] = inv (atomics: block 0's last arrival sums them, t6_hc_inj). e0 = hc_norm, e1 = inject weights
// bf16 [4][10240], e2 = ip [R][4][10][8]. All 256 threads.
static inline void t6_hc_norm(device const float *h, device const float *e0, device const float *e1,
                              device atomic_float *ip, uint k0e, uint n, bool inj, uint R, uint lid, uint sg,
                              uint lane, thread float *inv, threadgroup float *red) {
  constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID, CH = HID / 256u, RMAX = 8u;
  threadgroup float *red5 = red + 8u * RMAX;  // [8][4]
  const uint s = k0e / HID;
  for (uint r = 0u; r < RMAX; r++) {
    if (r >= R) break;
    const device float *hr = h + (size_t)r * HCD + s * HID;
    float ss = 0.0f;
    for (uint j = 0u; j < CH; j++) {
      const float v = hr[lid + 256u * j];
      ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[sg * RMAX + r] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0u; r < RMAX; r++) {
    if (r >= R) break;
    float t = 0.0f;
    for (uint g = 0u; g < 8u; g++) t += red[g * RMAX + r];
    inv[r] = rsqrt(t / float(HID) + 1e-6f);
  }
  if (!inj) return;
  const device ushort *nw = (const device ushort *)e0, *iw = (const device ushort *)e1;
  const uint ib = k0e - s * HID;  // span start inside the stream (a multiple of 256: n % 256 == 0, host check)
  for (uint r = 0u; r < RMAX; r++) {
    if (r >= R) break;
    const device float *hr = h + (size_t)r * HCD + s * HID;
    for (uint c = 0u; c < n / 256u; c++) {
      const uint i = ib + c * 256u + lid;
      const float xv = hr[i] * bf(nw[s * HID + i]);
      float a[4];
      for (uint k = 0u; k < 4u; k++) a[k] = simd_sum(bf(iw[(size_t)k * HCD + s * HID + i]) * xv);
      if (lane == 0u)
        for (uint k = 0u; k < 4u; k++) red5[sg * 4u + k] = a[k];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid < 5u) {
        float t = 0.0f;
        if (lid < 4u)
          for (uint g = 0u; g < 8u; g++) t += red5[g * 4u + lid];
        const uint ch = ib / 256u + c;
        atomic_store_explicit(&ip[((r * HC + s) * CH + ch) * 8u + lid], lid < 4u ? t * inv[r] : inv[r],
                              memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
// PRE 1, column block 0's last arrival (after t6_combine's fences): inj[r][j] = rr(2 sigmoid(rr(sum_s sum_ch
// ip[r][s][ch][j]) / 4)), inj[r][4 + s] = 1/rms_s (= t6e_hc_act).
static inline void t6_hc_inj(device atomic_float *ip, device float *inj, uint R, uint lid) {
  constexpr uint HC = 4u, CH = 10u;
  if (lid >= R * 8u) return;
  const uint r = lid / 8u, k = lid % 8u;
  device atomic_float *p = ip + r * HC * CH * 8u;
  if (k < HC) {
    float v = 0.0f;
    for (uint s = 0u; s < HC; s++) {
      float vs = 0.0f;
      for (uint c = 0u; c < CH; c++) vs += atomic_load_explicit(&p[(s * CH + c) * 8u + k], memory_order_relaxed);
      v += vs;
    }
    inj[r * 2u * HC + k] = rr(2.0f * sigm(rr(v) * 0.25f));
  } else {
    inj[r * 2u * HC + k] = atomic_load_explicit(&p[(k - HC) * CH * 8u + 4u], memory_order_relaxed);
  }
}

// ---------------------------------------------------------------- SIMD (1..8 rows)
// grid: ceil(OUT / 128) * S threadgroups of 256; threadgroup memory = R * span * 16 halves + R * 128 floats + 16 B.
kernel void t6_simd(device const float *x [[buffer(0)]], device const uint *w [[buffer(1)]],
                    device const half *suh [[buffer(2)]], device const half *svh [[buffer(3)]],
                    device atomic_float *scr [[buffer(4)]], device atomic_uint *ctr [[buffer(5)]],
                    device float *y [[buffer(6)]], device float *e0 [[buffer(7)]], device float *e1 [[buffer(8)]],
                    device float *e2 [[buffer(9)]], device float *e3 [[buffer(10)]],
                    threadgroup uchar *tgm [[threadgroup(0)]],
                    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint RMAX = 8u;
  const uint K = T6_K, NW = 8u * K, IT = T6_IN / 16u, OT = T6_OUT / 16u, S = T6_S, R = T6_R;
  const uint span = IT / S;
  const uint cb = tg / S, sp = tg - cb * S;
  const uint k0 = sp * span;
  threadgroup half *xs = (threadgroup half *)tgm;
  threadgroup float *o = (threadgroup float *)(tgm + ((R * span * 32u + 15u) & ~15u));
  threadgroup uint *flag = (threadgroup uint *)(o + R * 128u);
  float inv[RMAX];
  threadgroup float hcred[8u * RMAX + 8u * 4u];
  if (T6_PRE == 1u)
    t6_hc_norm(x, e0, e1, (device atomic_float *)e2, k0 * 16u, span * 16u, cb == 0u, R, lid, sg, lane, inv, hcred);
  t6_prep(x, suh, xs, k0 * 16u, span * 16u, R, sg, 8u, lane, e0, inv);
  if (T6_PRE == 2u && cb == 0u)  // the q8 path's shared_gu [R][rr gate | rr up] (checks read it)
    for (uint i = lid; i < R * span * 16u; i += 256u) {
      const uint r = i / (span * 16u), gi = k0 * 16u + (i - r * span * 16u);
      e1[r * 2u * T6_IN + gi] = rr(x[(size_t)r * T6_IN + gi]);
      e1[r * 2u * T6_IN + T6_IN + gi] = rr(e0[(size_t)r * T6_IN + gi]);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ot = cb * 8u + sg;
  const uint q = lane & 3u, col = lane >> 2u;
  float a0[RMAX], a1[RMAX];
  for (uint r = 0u; r < RMAX; r++) { a0[r] = 0.0f; a1[r] = 0.0f; }
  if (ot < OT) {
    const T6Lane L = t6_lane(lane);
    const device uint *wp = w + ((size_t)k0 * OT + ot) * NW;
    const size_t step = (size_t)OT * NW;
    threadgroup const half4 *x4 = (threadgroup const half4 *)xs;
    uint3 cur = t6_load(wp, L);
    for (uint t = 0u; t < span; t++) {  // next tile's words in flight while this one decodes
      const uint3 nxt = t6_load(wp + (t + 1u < span ? (t + 1u) : t) * step, L);
      float4 w0, w1;
      t6_dec(cur, L, w0, w1);
      for (uint r = 0u; r < RMAX; r++) {
        if (r >= R) break;
        const float4 xv = float4(x4[r * span * 4u + t * 4u + q]);
        a0[r] += dot(xv, w0);
        a1[r] += dot(xv, w1);
      }
      cur = nxt;
    }
  }
  for (uint r = 0u; r < RMAX; r++) {
    if (r >= R) break;
    float v0 = a0[r], v1 = a1[r];
    v0 += simd_shuffle_xor(v0, 1u); v0 += simd_shuffle_xor(v0, 2u);
    v1 += simd_shuffle_xor(v1, 1u); v1 += simd_shuffle_xor(v1, 2u);
    if (q == 0u) { o[r * 128u + sg * 16u + col] = v0; o[r * 128u + sg * 16u + col + 8u] = v1; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (S > 1u && !t6_combine(o, scr, ctr, cb, sp, 0u, 128u, S, R, lid, 256u, flag)) return;
  if (T6_PRE == 1u && cb == 0u) t6_hc_inj((device atomic_float *)e2, e3, R, lid);
  t6_finish(o, svh, y, cb, R, sg, 8u, lane, lid, 256u, e0, e1, e2, e3);
  if (T6_EPI == 8u) t6_hc_out(o, y, ctr, cb, R, lid, 256u, flag, e0, e1, e2, e3);
}

// ---------------------------------------------------------------- NAX (matmul2d 16x32x16), rows <= 16
// t6_prep: grid R * ceil(IN / 128) / 8 threadgroups of 256 (a simdgroup per 128-input chunk of one row).
kernel void t6_prep(device const float *x [[buffer(0)]], device const half *suh [[buffer(1)]],
                    device half *xh [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint nch = (T6_IN + 127u) / 128u;
  const uint job = tg * 8u + sg;
  if (job >= T6_R * nch) return;
  const uint r = job / nch, c = (job - r * nch) * 128u;
  float4 v;
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    v[j] = i < T6_IN ? x[(size_t)r * T6_IN + i] * float(suh[i]) : 0.0f;
  }
  v = t6_fwht(v, lane, T6_HI);
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    if (i < T6_IN) xh[(size_t)r * T6_IN + kp(i)] = half(v[j]);
  }
}

// t6_nax: grid ceil(OT / 2) * S threadgroups of 32 * KS; threadgroup memory = max(KS * RMP * 32, R * 128) floats +
// 16 B. Column block cp = tiles 2 cp, 2 cp + 1 (the second absent when OT is odd). xh from t6_prep.
kernel void t6_nax(device const half *xh [[buffer(0)]], device const uint *w [[buffer(1)]],
                   device const half *suh [[buffer(2)]], device const half *svh [[buffer(3)]],
                   device atomic_float *scr [[buffer(4)]], device atomic_uint *ctr [[buffer(5)]],
                   device float *y [[buffer(6)]], device float *e0 [[buffer(7)]], device float *e1 [[buffer(8)]],
                   device float *e2 [[buffer(9)]], device float *e3 [[buffer(10)]],
                   threadgroup uchar *tgm [[threadgroup(0)]],
                   uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                   uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint K = T6_K, NW = 8u * K, IT = T6_IN / 16u, OT = T6_OUT / 16u, S = T6_S, R = T6_R, KS = T6_KS;
  const uint RMP = R > 8u ? 16u : 8u;
  const uint NCP = (OT + 1u) / 2u;
  const uint span = IT / S;
  const uint cp = tg / S, sp = tg - cp * S;
  const uint k0 = sp * span;
  threadgroup float *part = (threadgroup float *)tgm;
  const uint PART = max(KS * RMP * 32u, R * 128u);
  threadgroup uint *flag = (threadgroup uint *)(part + PART);

  const uint k0l = 4u * (lane & 1u) + 8u * ((lane >> 3u) & 1u);
  const uint c0 = 4u * (lane >> 4u) + ((lane >> 1u) & 3u);
  const uint qq = k0l >> 2u;
  const T6Lane L = t6_lane(4u * c0 + qq);  // tensor-core thread whose 8 slots are this lane's right elements
  const uint ta = 2u * cp, tb = 2u * cp + 1u;
  const bool hasb = tb < OT;
  constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto left = op.get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.get_right_input_cooperative_tensor<half, half, float>();
  auto d = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                  metal::remove_addrspace_t<decltype(right)>, float>();
  for (uint s = 0u; s < d.get_capacity(); s++) d[s] = 0.0f;
  for (short s = 0; s < 8; s++) left[s] = 0.0h;
  for (short s = 0; s < 16; s++) right[s] = 0.0h;
  const bool live = c0 < R, live2 = c0 + 8u < R;
  const device half4 *x4 = (const device half4 *)xh + (size_t)k0 * 4u;  // row stride IT * 4 half4
  const size_t step = (size_t)OT * NW;
  const device uint *wa = w + ((size_t)k0 * OT + ta) * NW;
  const device uint *wb = w + ((size_t)k0 * OT + (hasb ? tb : ta)) * NW;
  for (uint t = sgi; t < span; t += KS) {
    const uint3 ua = t6_load(wa + t * step, L);
    const uint3 ub = t6_load(wb + t * step, L);
    if (live) { const half4 xa = x4[(size_t)c0 * IT * 4u + t * 4u + qq]; for (short s = 0; s < 4; s++) left[s] = xa[s]; }
    if (RMP == 16u && live2) {
      const half4 xb = x4[(size_t)(c0 + 8u) * IT * 4u + t * 4u + qq];
      for (short s = 0; s < 4; s++) left[4 + s] = xb[s];
    }
    float4 w0, w1;
    t6_dec(ua, L, w0, w1);
    for (short e = 0; e < 4; e++) { right[e] = half(w0[e]); right[4 + e] = half(w1[e]); }
    if (hasb) {
      t6_dec(ub, L, w0, w1);
      for (short e = 0; e < 4; e++) { right[8 + e] = half(w0[e]); right[12 + e] = half(w1[e]); }
    }
    op.run(left, right, d);
  }
  // KS partials -> part[sgi][row][32] (R <= 8: rows c0 only, e in {0..3, 8..11})
  for (uint i = 0u; i < RMP; i++) {
    const uint e = RMP == 16u ? i : (((i >> 2u) << 3u) | (i & 3u));
    const uint row = c0 + 8u * ((e >> 2u) & 1u);
    part[(sgi * RMP + row) * 32u + 16u * (e >> 3u) + k0l + (e & 3u)] = d[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // sum the KS partials in order -> o [R][128] at this column block's 32 columns (reuses part after a barrier)
  const uint cb = cp / 4u, c0p = (cp - cb * 4u) * 32u;
  float v[16];  // ceil(R * 32 / (32 KS)) <= 16 per thread
  const uint NT = 32u * KS;
  for (uint j = 0u; j < 16u; j++) {
    const uint i = lid + j * NT;
    v[j] = 0.0f;
    if (i < R * 32u) {
      const uint r = i / 32u, c = i - r * 32u;
      for (uint k = 0u; k < KS; k++) v[j] += part[(k * RMP + r) * 32u + c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float *o = part;
  for (uint j = 0u; j < 16u; j++) {
    const uint i = lid + j * NT;
    if (i < R * 32u) { const uint r = i / 32u, c = i - r * 32u; o[r * 128u + c0p + c] = v[j]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ncb = min(4u, NCP - cb * 4u);  // column blocks in this 128-column block
  if (!t6_combine(o, scr, ctr, cb, sp, c0p, 32u, ncb * S, R, lid, 32u * KS, flag)) return;
  t6_finish(o, svh, y, cb, R, sgi, KS, lane, lid, 32u * KS, e0, e1, e2, e3);
  if (T6_EPI == 8u) t6_hc_out(o, y, ctr, cb, R, lid, 32u * KS, flag, e0, e1, e2, e3);
}

// ---------------------------------------------------------------- bf16 dense (final mixer), 1..8 rows
// W bf16 [OUT][IN] (HF layout), IN % 8 == 0. T6_S = 1: grid ceil(OUT / 8) threadgroups of 256, simdgroup = one output
// row; T6_S = 8 (few outputs, e.g. the mixer's 10240 -> 320): grid OUT threadgroups, the 8 simdgroups split IN.
kernel void t6_bf16(device const float *x [[buffer(0)]], device const ushort *w [[buffer(1)]],
                    device float *y [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint RMAX = 8u;
  const uint IN = T6_IN, OUT = T6_OUT, R = T6_R;
  const bool split = T6_S == 8u;
  const uint n = split ? tg : tg * 8u + sg;
  if (n >= OUT) return;
  const device uint4 *w4 = (const device uint4 *)(w + (size_t)n * IN);
  float acc[RMAX];
  for (uint r = 0u; r < RMAX; r++) acc[r] = 0.0f;
  for (uint c = split ? lid : lane; c < IN / 8u; c += split ? 256u : 32u) {
    const uint4 u = w4[c];
    for (uint r = 0u; r < RMAX; r++) {
      if (r >= R) break;
      const device float4 *x4 = (const device float4 *)(x + (size_t)r * IN);
      acc[r] += dot8bf(u, x4[2u * c], x4[2u * c + 1u]);
    }
  }
  threadgroup float red[8][RMAX];
  for (uint r = 0u; r < RMAX; r++) {
    if (r >= R) break;
    const float s = simd_sum(acc[r]);
    if (lane == 0u) {
      if (split) red[sg][r] = s;
      else y[(size_t)r * OUT + n] = s;
    }
  }
  if (!split) return;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < R) {
    float v = 0.0f;
    for (uint g = 0u; g < 8u; g++) v += red[g][lid];
    y[(size_t)lid * OUT + n] = v;
  }
}

// ---------------------------------------------------------------- chunked prefill (lab-t6wire), any rows M
// t6::record_prefill: t6_pf_prep + t6_dense, barrier, pf_h16_hf (kernels/prefill/pf_gemm.metal), barrier, t6_pf_post.
constant bool T6_XB_FC [[function_constant(18)]];  // prefill x is bf16 (else f32)
constant bool T6_XB = is_function_constant_defined(T6_XB_FC) ? T6_XB_FC : false;
constant bool T6_YB_FC [[function_constant(19)]];  // prefill y is bf16 (else f32)
constant bool T6_YB = is_function_constant_defined(T6_YB_FC) ? T6_YB_FC : false;

// xh[m][i] = half(H_hi (x[m] * suh))[i] (natural order). Grid (ceil(IN / 128), ceil(M / 8)), simdgroup = (row, chunk).
kernel void t6_pf_prep(device const uchar *xb [[buffer(0)]], device const half *suh [[buffer(1)]],
                       device half *xh [[buffer(2)]], constant uint &M [[buffer(3)]], constant uint &ldx [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint m = tg.y * 8u + sg;
  if (m >= M) return;
  const uint c = tg.x * 128u;
  float4 v;
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    float xv = 0.0f;
    if (i < T6_IN) xv = T6_XB ? bf(((device const ushort *)xb)[(size_t)m * ldx + i]) : ((device const float *)xb)[(size_t)m * ldx + i];
    v[j] = i < T6_IN ? xv * float(suh[i]) : 0.0f;
  }
  v = t6_fwht(v, lane, T6_HI);
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    if (i < T6_IN) xh[(size_t)m * T6_IN + i] = half(v[j]);
  }
}

// W' dense: wd[n][k] = W'[k][n] (half, HF orientation for pf_h16_hf), one simdgroup per 16 x 16 tile. Rows n >= OUT
// of the padded buffer are not written (their GEMM columns are never read).
kernel void t6_dense(device const uint *w [[buffer(0)]], device half *wd [[buffer(1)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const uint K = T6_K, NW = 8u * K, IT = T6_IN / 16u, OT = T6_OUT / 16u;
  const uint tile = tg * 8u + sg;
  if (tile >= IT * OT) return;
  const uint tk = tile / OT, tn = tile - tk * OT;
  const T6Lane L = t6_lane(lane);
  const uint3 ww = t6_load(w + (size_t)tile * NW, L);
  float4 w0, w1;
  t6_dec(ww, L, w0, w1);
  const uint q = lane & 3u, c = lane >> 2u;
  const uint kk[4] = {2u * q, 2u * q + 1u, 2u * q + 8u, 2u * q + 9u};
  device half *d0 = wd + (size_t)(16u * tn + c) * T6_IN + 16u * tk;
  device half *d1 = d0 + (size_t)8u * T6_IN;
  for (uint e = 0u; e < 4u; e++) { d0[kk[e]] = half(w0[e]); d1[kk[e]] = half(w1[e]); }
}

// ---------------------------------------------------------------- INT8 prefill path (pfkern, 18 Oct; kernels/trunk6/t6_a8.metal)
// xq[m][i] = int8 of H_hi(x[m] * suh)[i] with one scale per (row, 128-block): sx[i / 128][m] = absmax / 127
// (round to nearest, |q| <= 127). Grid (ceil(IN / 128), ceil(M / 8)), simdgroup = (row, 128-block).
kernel void t6_pf_prep_a8(device const uchar *xb [[buffer(0)]], device const half *suh [[buffer(1)]],
                          device char *xq [[buffer(2)]], device float *sx [[buffer(3)]], constant uint &M [[buffer(4)]],
                          constant uint &ldx [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint m = tg.y * 8u + sg;
  if (m >= M) return;
  const uint c = tg.x * 128u;
  float4 v;
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    float xv = 0.0f;
    if (i < T6_IN) xv = T6_XB ? bf(((device const ushort *)xb)[(size_t)m * ldx + i]) : ((device const float *)xb)[(size_t)m * ldx + i];
    v[j] = i < T6_IN ? xv * float(suh[i]) : 0.0f;
  }
  v = t6_fwht(v, lane, T6_HI);
  const float4 a = abs(v);
  const float amax = simd_max(max(max(a.x, a.y), max(a.z, a.w)));
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    if (i < T6_IN) xq[(size_t)m * T6_IN + i] = char(clamp(rint(v[j] * inv), -127.0f, 127.0f));
  }
  if (lane == 0u) sx[(size_t)tg.x * M + m] = amax / 127.0f;
}

// W' -> int8 wq[n][k] with one scale per (output row n, 128-block of k): sw[k / 128][n] = absmax / 127 (NP = the
// padded row count = sw's row stride). Grid (OUT / 16, ceil(IN / 128)) threadgroups of 8 simdgroups: simdgroup s
// decodes tile (k tile 8 tg.y + s, n tile tg.x) as t6_dense does; row maxima over the 8 tiles via threadgroup memory.
kernel void t6_dense_a8(device const uint *w [[buffer(0)]], device char *wq [[buffer(1)]], device float *sw [[buffer(2)]],
                        constant uint &NP [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8][16];
  const uint K = T6_K, NW = 8u * K, IT = T6_IN / 16u, OT = T6_OUT / 16u;
  const uint tn = tg.x, tk = tg.y * 8u + sg;
  const bool live = tk < IT;
  float4 w0 = 0.0f, w1 = 0.0f;
  if (live) {
    const T6Lane L = t6_lane(lane);
    const uint3 ww = t6_load(w + (size_t)(tk * OT + tn) * NW, L);
    t6_dec(ww, L, w0, w1);
  }
  const uint q = lane & 3u, c = lane >> 2u;
  float m0 = max(max(abs(w0.x), abs(w0.y)), max(abs(w0.z), abs(w0.w)));
  float m1 = max(max(abs(w1.x), abs(w1.y)), max(abs(w1.z), abs(w1.w)));
  m0 = max(m0, simd_shuffle_xor(m0, 1u)); m0 = max(m0, simd_shuffle_xor(m0, 2u));
  m1 = max(m1, simd_shuffle_xor(m1, 1u)); m1 = max(m1, simd_shuffle_xor(m1, 2u));
  if (q == 0u) { red[sg][c] = m0; red[sg][c + 8u] = m1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float a0 = 0.0f, a1 = 0.0f;
  for (uint s = 0u; s < 8u; s++) { a0 = max(a0, red[s][c]); a1 = max(a1, red[s][c + 8u]); }
  if (sg == 0u && q == 0u) {
    sw[(size_t)tg.y * NP + 16u * tn + c] = a0 / 127.0f;
    sw[(size_t)tg.y * NP + 16u * tn + c + 8u] = a1 / 127.0f;
  }
  if (!live) return;
  const float i0 = a0 > 0.0f ? 127.0f / a0 : 0.0f, i1 = a1 > 0.0f ? 127.0f / a1 : 0.0f;
  const uint kk[4] = {2u * q, 2u * q + 1u, 2u * q + 8u, 2u * q + 9u};
  device char *d0 = wq + (size_t)(16u * tn + c) * T6_IN + 16u * tk;
  device char *d1 = d0 + (size_t)8u * T6_IN;
  for (uint e = 0u; e < 4u; e++) {
    d0[kk[e]] = char(clamp(rint(w0[e] * i0), -127.0f, 127.0f));
    d1[kk[e]] = char(clamp(rint(w1[e] * i1), -127.0f, 127.0f));
  }
}

// ---------------------------------------------------------------- folded W (pfkern, 18 Oct)
// The chunked prefill's EXL3 Linear y = svh * H_ho((H_hi(x * suh)) W') becomes y = x Wf^T with the dense
// Wf[n][i] = svh[n] suh[i] (H_ho W'^T H_hi)[n][i] (half), built per chunk from t6_dense's W' in two passes, so the
// GEMM reads x (bf16) directly and writes y directly: no prep / post passes over the activations.
// t6_fold_k: wd[n][:] <- H_hi(wd[n][:]) * suh, per (row n, 128-block). Grid (ceil(IN / 128), ceil(OUT / 8)) x 256.
kernel void t6_fold_k(device half *wd [[buffer(0)]], device const half *suh [[buffer(1)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.y * 8u + sg;
  if (n >= T6_OUT) return;
  const uint c = tg.x * 128u;
  device half *r = wd + (size_t)n * T6_IN;
  float4 v;
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    v[j] = i < T6_IN ? float(r[i]) : 0.0f;
  }
  v = t6_fwht(v, lane, T6_HI);
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    if (i < T6_IN) r[i] = half(v[j] * float(suh[i]));
  }
}
// t6_dense + t6_fold_k in one pass (pfkern): threadgroup = 16 output rows (n tile tg.x) x one 128-block of k (8 k tiles,
// one per simdgroup, decoded as t6_dense does) through threadgroup memory, then per row H_hi along k, * suh, half
// stores (coalesced). Grid (OUT / 16, ceil(IN / 128)) x 256. Same values as t6_dense -> t6_fold_k except that the
// decoded W' enters the transform in f32 instead of half.
kernel void t6_dense_fold(device const uint *w [[buffer(0)]], device half *wd [[buffer(1)]],
                          device const half *suh [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float t[16][129];
  const uint K = T6_K, NW = 8u * K, IT = T6_IN / 16u, OT = T6_OUT / 16u;
  const uint tn = tg.x, tk = tg.y * 8u + sg;
  const uint q = lane & 3u, c = lane >> 2u;
  const uint kk[4] = {2u * q, 2u * q + 1u, 2u * q + 8u, 2u * q + 9u};
  float4 w0 = 0.0f, w1 = 0.0f;
  if (tk < IT) {
    const T6Lane L = t6_lane(lane);
    t6_dec(t6_load(w + (size_t)(tk * OT + tn) * NW, L), L, w0, w1);
  }
  for (uint e = 0u; e < 4u; e++) { t[c][16u * sg + kk[e]] = w0[e]; t[c + 8u][16u * sg + kk[e]] = w1[e]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k0 = tg.y * 128u;
  for (uint rr2 = 0u; rr2 < 2u; rr2++) {
    const uint r = sg * 2u + rr2, n = 16u * tn + r;
    float4 v;
    for (uint j = 0u; j < 4u; j++) v[j] = k0 + lane + 32u * j < T6_IN ? t[r][lane + 32u * j] : 0.0f;
    v = t6_fwht(v, lane, T6_HI);
    for (uint j = 0u; j < 4u; j++) {
      const uint i = k0 + lane + 32u * j;
      if (i < T6_IN) wd[(size_t)n * T6_IN + i] = half(v[j] * float(suh[i]));
    }
  }
}

// t6_fold_n: wd[:][k] <- H_ho(wd[:][k]) * svh along n, per (128-block of n, 32 columns k) through threadgroup memory.
// Grid (IN / 32, ceil(OUT / 128)) x 256; rows >= OUT are zeros in the transform and not written. wo = wd (in place),
// or (T6_PERM) another buffer: the permuted rows cross threadgroups.
constant bool T6_PERM_FC [[function_constant(24)]];  // fold_n: HC up row order (t6_fold_hcup), see below
constant bool T6_PERM = is_function_constant_defined(T6_PERM_FC) ? T6_PERM_FC : false;
// HC up (10240 = 4 streams x 2560) for t6_fold_hcup: output n = s * 2560 + c goes to Wf row (c / 16) * 64 + s * 16 + c % 16,
// so each 64-column GEMM tile holds 16 hidden columns of all 4 streams
static inline uint t6_perm(uint n) {
  const uint s = n / 2560u, c = n % 2560u;
  return (c / 16u) * 64u + s * 16u + c % 16u;
}
kernel void t6_fold_n(device half *wd [[buffer(0)]], device const half *svh [[buffer(1)]], device half *wo [[buffer(2)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float t[128][33];
  const uint k0 = tg.x * 32u, n0 = tg.y * 128u;
  for (uint e = tid; e < 128u * 32u; e += 256u) {
    const uint rn = e / 32u, kc = e % 32u, n = n0 + rn;
    t[rn][kc] = n < T6_OUT ? float(wd[(size_t)n * T6_IN + k0 + kc]) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = 0u; q < 4u; q++) {
    const uint kc = sg * 4u + q;
    float4 v = float4(t[lane][kc], t[lane + 32u][kc], t[lane + 64u][kc], t[lane + 96u][kc]);
    v = t6_fwht(v, lane, T6_HO);
    for (uint j = 0u; j < 4u; j++) {
      const uint n = n0 + lane + 32u * j;
      t[lane + 32u * j][kc] = n < T6_OUT ? v[j] * float(svh[n]) : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 128u * 32u; e += 256u) {
    const uint rn = e / 32u, kc = e % 32u, n = n0 + rn;
    if (n < T6_OUT) wo[(size_t)(T6_PERM ? t6_perm(n) : n) * T6_IN + k0 + kc] = half(t[rn][kc]);
  }
}

// HC up GEMM + block_in fused (pfkern): xin[m][c] = f2bf(rr(sum_s rr(rr(sigm(up_s)) * normed[m][s][c]) / 4)) with
// up_s = bf16r(x[m] . Wf[perm(s * 2560 + c)]) (= pf_hc_blockin over t6_fold_gemm's bf16 up, same rounding points), x
// = the HC down activations bf16 [M][320]. The [M][10240] up plane is never written. BM = 64 rows x 64 columns (16
// hidden columns x 4 streams) per threadgroup of 128 threads, the tile through threadgroup memory.
// Grid (2560 / 16, ceil(M / 64)).
struct T6HcUpP {
  uint K, M;
};
kernel void t6_fold_hcup(device const half *wf [[buffer(0)]], device const bfloat *x [[buffer(1)]],
                         device const ushort *normed [[buffer(2)]], device ushort *xin [[buffer(3)]],
                         constant T6HcUpP &p [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  threadgroup float t[64][65];
  const int K = p.K, M = p.M;
  const int row0 = tgp.y * 64, col0 = tgp.x * 64;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A((device bfloat *)x, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B((device half *)wf, dextents<int32_t, 2>(K, 10240));
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = A.slice(0, row0);
  auto mB = B.slice(0, col0);
  auto ct = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  op.run(mA, mB, ct);
  for (int i = 0; i < 32; i++) {
    const auto ix = ct.get_multidimensional_index(i);
    t[ix[1]][ix[0]] = bf16r(ct[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 64u * 16u; e += 128u) {
    const uint r = e / 16u, cc = e % 16u;
    const int m = row0 + int(r);
    if (m >= M) continue;
    const uint c = tgp.x * 16u + cc;
    float v[4];
    for (uint s = 0u; s < 4u; s++)
      v[s] = rr(rr(sigm(t[r][s * 16u + cc])) * bf(normed[(size_t)m * 10240u + s * 2560u + c]));
    xin[(size_t)m * 2560u + c] = ushort(as_type<uint>(bf16r(rr(((v[0] + v[1]) + (v[2] + v[3])) * 0.25f))) >> 16u);
  }
}

// y[m][n] = (H_ho yraw[m])[n] * svh[n] (f32, or bf16 rounded to nearest). Grid (ceil(OUT / 128), ceil(M / 8)).
kernel void t6_pf_post(device const float *yraw [[buffer(0)]], device const half *svh [[buffer(1)]],
                       device uchar *yb [[buffer(2)]], constant uint &M [[buffer(3)]], constant uint &ldr [[buffer(4)]],
                       constant uint &ldy [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint m = tg.y * 8u + sg;
  if (m >= M) return;
  const uint c = tg.x * 128u;
  float4 v;
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    v[j] = i < T6_OUT ? yraw[(size_t)m * ldr + i] : 0.0f;
  }
  v = t6_fwht(v, lane, T6_HO);
  for (uint j = 0u; j < 4u; j++) {
    const uint i = c + lane + 32u * j;
    if (i >= T6_OUT) continue;
    const float o = v[j] * float(svh[i]);
    if (T6_YB) ((device ushort *)yb)[(size_t)m * ldy + i] = ushort(as_type<uint>(bf16r(o)) >> 16u);
    else ((device float *)yb)[(size_t)m * ldy + i] = o;
  }
}
