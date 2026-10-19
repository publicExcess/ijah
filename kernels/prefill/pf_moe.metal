// Prefill MoE over a chunk of n tokens (n x top-10 slots), EXL3 trellis experts on the M5 neural accelerators.
// Owner: lab-prefill. Design and numbers: docs/prefill.md.
//
// The slots are sorted by expert on the GPU (no host sync): pf_moe_count (one threadgroup per expert: count + the
// slot's rank among that expert's slots, in slot order = deterministic), pf_moe_scan (one threadgroup: expert
// offsets, run-aligned windows of <= 32 rows), pf_moe_place (sorted order, inverse). Then prep (x * suh, H128 -> f16
// per sorted slot, gate and up), the NAX GEMMs per window (gate, up; down), mid (H128 * svh, SwiGLU, * down suh, H128)
// and the finish (H128 * svh * score through the inverse, fixed slot order, + sigmoid-gated shared expert + HC write).
// The per-slot math follows lab-forward's decode kernels (kernels/fn_moe.metal) term by term.
//
// Ported from the lab's EXL3 prefill (the lab tree's exl3/kernels/exl3_prefill.py, step 3 of
// kernels/NOTES.md), whose GEMM body and trellis fragment decode (exl3_end, exl3_window, nax_wfrag, nax_wfrag_k,
// nax_origin, nax_store, ...) are lifted from sushi src/exl3/expert_exl3_kernels.zig:
//   sushi (https://github.com/beamivalice/sushi), MIT License, Copyright (c) 2026 Theinruj Toranavikrai,
//   Copyright (c) 2026 David Dalcu. Permission is hereby granted, free of charge, to any person obtaining a copy of
//   this software and associated documentation files (the "Software"), to deal in the Software without restriction,
//   ... The above copyright notice and this permission notice shall be included in all copies or substantial
//   portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "kernels/ne_common.h"
using namespace mpp::tensor_ops;

constant constexpr uint HID = 2560u, HC = 4u, HCD = HC * HID, NE = 512u, TOPK = 10u, INTER = 640u;


// ------------------------------------------------------------------------------------------------ routing tables
// route [n][20] (fn_moe_route: 10 ids, 10 weights); slot = t * 10 + k. One threadgroup (256 threads) per expert e:
// cnt[e], rank[slot] for its slots (thread j scans a contiguous range; exclusive scan of the per-thread counts).
kernel void pf_moe_count(device const uint *route [[buffer(0)]], device uint *cnt [[buffer(1)]],
                         device uint *rank [[buffer(2)]], constant uint &n [[buffer(3)]],
                         uint e [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup uint part[256];
  const uint S = n * TOPK, per = (S + 255u) / 256u;
  const uint s0 = min(lid * per, S), s1 = min(s0 + per, S);
  uint c = 0u;
  for (uint s = s0; s < s1; s++) c += route[(s / TOPK) * 20u + s % TOPK] == e ? 1u : 0u;
  part[lid] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    uint acc = 0u;
    for (uint j = 0u; j < 256u; j++) { const uint v = part[j]; part[j] = acc; acc += v; }
    cnt[e] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint r = part[lid];
  for (uint s = s0; s < s1; s++)
    if (route[(s / TOPK) * 20u + s % TOPK] == e) rank[s] = r++;
}
// One threadgroup of 512 threads: off[e] (exclusive scan of cnt), windows: wtab[w] = (start, rows, expert) for
// every run of an expert cut into <= WIN-row pieces (WIN = 16 x the GEMM's row blocks); unused windows up to maxwin get rows = 0.
kernel void pf_moe_scan(device const uint *cnt [[buffer(0)]], device uint *off [[buffer(1)]],
                        device uint4 *wtab [[buffer(2)]], constant uint &maxwin [[buffer(3)]],
                        constant uint &WIN [[buffer(4)]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup uint c[NE], o[NE], wo[NE + 1u];
  c[lid] = cnt[lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    uint a = 0u, w = 0u;
    for (uint e = 0u; e < NE; e++) {
      o[e] = a; wo[e] = w;
      a += c[e];
      w += (c[e] + WIN - 1u) / WIN;
    }
    wo[NE] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  off[lid] = o[lid];
  const uint nw = (c[lid] + WIN - 1u) / WIN;
  for (uint j = 0u; j < nw; j++) wtab[wo[lid] + j] = uint4(o[lid] + j * WIN, min(WIN, c[lid] - j * WIN), lid, 0u);
  for (uint w = wo[NE] + lid; w < maxwin; w += NE) wtab[w] = uint4(0u);
}
// per slot: pos = off[e] + rank; order[pos] = slot, seid[pos] = e, inv[slot] = pos. n * 10 threads.
kernel void pf_moe_place(device const uint *route [[buffer(0)]], device const uint *off [[buffer(1)]],
                         device const uint *rank [[buffer(2)]], device uint *order [[buffer(3)]],
                         device uint *seid [[buffer(4)]], device uint *inv [[buffer(5)]], constant uint &S [[buffer(6)]],
                         uint s [[thread_position_in_grid]]) {
  if (s >= S) return;
  const uint e = route[(s / TOPK) * 20u + s % TOPK];
  const uint pos = off[e] + rank[s];
  order[pos] = s;
  seid[pos] = e;
  inv[s] = pos;
}

// ------------------------------------------------------------------------------------------------ glue
// prep: per sorted slot pos and 128-block b: x = the token's MoE input row (bf16), pg = H128(x * suh_gate) and pu =
// H128(x * suh_up) as f16 (fn_moe_gateup's prep, natural element order). grid (20, S) threadgroups of 32.
kernel void pf_moe_prep(device const ushort *x [[buffer(0)]], device const half *gsu [[buffer(1)]],
                        device const half *usu [[buffer(2)]], device const uint *order [[buffer(3)]],
                        device const uint *seid [[buffer(4)]], device half *pg [[buffer(5)]], device half *pu [[buffer(6)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint b = tg.x, pos = tg.y;
  const uint tok = order[pos] / TOPK, e = seid[pos];
  const uint o = b * 128u + lane;
  const device ushort *xr = x + (size_t)tok * HID;
  const float4 xv = float4(bf(xr[o]), bf(xr[o + 32u]), bf(xr[o + 64u]), bf(xr[o + 96u]));
  for (uint pj = 0u; pj < 2u; pj++) {
    const device half *su = (pj == 0u ? gsu : usu) + (size_t)e * HID;
    float4 v = xv * float4(float(su[o]), float(su[o + 32u]), float(su[o + 64u]), float(su[o + 96u]));
    v = fwht128(v, lane);
    device half *y = (pj == 0u ? pg : pu) + (size_t)pos * HID;
    y[o] = half(v.x); y[o + 32u] = half(v.y); y[o + 64u] = half(v.z); y[o + 96u] = half(v.w);
  }
}
// mid: gate/up GEMM outputs (f32) -> H128 * svh, SwiGLU, * down suh, H128 -> f16 down input (fn_moe_down's prep).
// grid (5, S) threadgroups of 32. YT = the GEMM output plane type (f32, or f16: NE_PF_MOEH, lab-pflevers).
template <typename YT>
kernel void pf_moe_mid(device const YT *ig [[buffer(0)]], device const YT *iu [[buffer(1)]],
                       device const half *gsv [[buffer(2)]], device const half *usv [[buffer(3)]],
                       device const half *dsu [[buffer(4)]], device const uint *seid [[buffer(5)]],
                       device half *pd [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint b = tg.x, pos = tg.y, e = seid[pos];
  const uint o = b * 128u + lane;
  const device YT *g = ig + (size_t)pos * INTER, *u = iu + (size_t)pos * INTER;
  const device half *sg_ = gsv + (size_t)e * INTER, *su_ = usv + (size_t)e * INTER, *sd = dsu + (size_t)e * INTER;
  float4 gv = fwht128(float4(float(g[o]), float(g[o + 32u]), float(g[o + 64u]), float(g[o + 96u])), lane) *
              float4(float(sg_[o]), float(sg_[o + 32u]), float(sg_[o + 64u]), float(sg_[o + 96u]));
  float4 uv = fwht128(float4(float(u[o]), float(u[o + 32u]), float(u[o + 64u]), float(u[o + 96u])), lane) *
              float4(float(su_[o]), float(su_[o + 32u]), float(su_[o + 64u]), float(su_[o + 96u]));
  float4 hh = gv / (1.0f + exp(-gv)) * uv * float4(float(sd[o]), float(sd[o + 32u]), float(sd[o + 64u]), float(sd[o + 96u]));
  hh = fwht128(hh, lane);
  device half *y = pd + (size_t)pos * INTER;
  y[o] = half(hh.x); y[o + 32u] = half(hh.y); y[o + 64u] = half(hh.z); y[o + 96u] = half(hh.w);
}
// finish: per (token t, 128-column block): routed = sum_k (H128(dn[inv[t,k]]) * svh * w_k) in slot order k;
// branch = rr(rr(routed) + rr(rr(sigmoid(sgl)) * shared)); hout = rr(h + rr(branch * inj_s)) (fn_moe_down's finisher).
// grid (20, n) threadgroups of 32 * 10 (simdgroup = slot k).
template <typename YT>
kernel void pf_moe_finish(device const YT *dn [[buffer(0)]], device const half *dsv [[buffer(1)]],
                          device const uint *route [[buffer(2)]], device const uint *inv [[buffer(3)]],
                          device const float *sgl [[buffer(4)]], device const ushort *sho [[buffer(5)]],
                          device const float *h [[buffer(6)]], device const float *inj [[buffer(7)]],
                          device float *hout [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float vals[TOPK * 128u];
  const uint blk = tg.x, t = tg.y, k = sg;
  const uint e = route[t * 20u + k];
  const float sc = as_type<float>(route[t * 20u + TOPK + k]);
  const uint pos = inv[t * TOPK + k];
  const device YT *d = dn + (size_t)pos * HID + blk * 128u;
  const device half *sv = dsv + (size_t)e * HID + blk * 128u;
  const float4 v = fwht128(float4(float(d[lane]), float(d[lane + 32u]), float(d[lane + 64u]), float(d[lane + 96u])), lane) *
                   float4(float(sv[lane]), float(sv[lane + 32u]), float(sv[lane + 64u]), float(sv[lane + 96u])) * sc;
  vals[k * 128u + lane] = v.x; vals[k * 128u + lane + 32u] = v.y;
  vals[k * 128u + lane + 64u] = v.z; vals[k * 128u + lane + 96u] = v.w;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  const float sgt = sigm(sgl[t]);
  const device float *ip = inj + t * 2u * HC;
  for (uint j = 0u; j < 4u; j++) {
    const uint cl = lane + 32u * j, c = blk * 128u + cl;
    float s = 0.0f;
    for (uint kk = 0u; kk < TOPK; kk++) s += vals[kk * 128u + cl];
    const float br = rr(rr(s) + rr(rr(sgt) * bf(sho[(size_t)t * HID + c])));
    for (uint st = 0u; st < HC; st++) {
      const size_t o = (size_t)t * HCD + st * HID + c;
      hout[o] = rr(h[o] + rr(br * ip[st]));
    }
  }
}

// ------------------------------------------------------------------------------------------------ NAX GEMM
static inline half2 exl3_pairh(uint2 cw) {  // MCG, window 15 (= ne_common.h dec_mcg)
  cw &= uint2(0x7FFFu);
  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
  const half4 h = as_type<half4>(r);
  return half2(h.x + h.y, h.z + h.w);
}
static inline ulong exl3_window(const device uint *words, uint end, uint nwords, bool full) {
  const uint last = (end - 1u) >> 5u;
  const uint prev = last == 0u ? nwords - 1u : last - 1u;
  const uint s = (0u - end) & 31u;
  ulong bits = (((ulong)words[prev] << 32u) | (ulong)words[last]) >> s;
  if (full) bits |= ((ulong)words[prev == 0u ? nwords - 1u : prev - 1u] << 32u) << (32u - s);
  return bits;
}
using nfrag = vec<half, 8>;
static inline nfrag nax_wfrag(const device uint *words, uint lane) {
  const uint source0 = (lane & 16u) + ((lane & 7u) << 1u);
  const uint2 current = *(const device uint2 *)(words + source0);
  const uint previous = words[(source0 + 31u) & 31u];
  const uint slot = ((lane >> 3u) & 1u) * 2u;
  const ulong w0 = ((ulong)previous << 32) | (ulong)current.x;
  const ulong w1 = ((ulong)current.x << 32) | (ulong)current.y;
  const uint s0 = 28u - slot * 4u;
  const uint s1 = 28u - (slot + 4u) * 4u;
  const half2 p00 = exl3_pairh(uint2(uint(w0 >> s0) & 0xffffu, uint(w0 >> (s0 - 4u)) & 0xffffu));
  const half2 p01 = exl3_pairh(uint2(uint(w1 >> s0) & 0xffffu, uint(w1 >> (s0 - 4u)) & 0xffffu));
  const half2 p10 = exl3_pairh(uint2(uint(w0 >> s1) & 0xffffu, uint(w0 >> (s1 - 4u)) & 0xffffu));
  const half2 p11 = exl3_pairh(uint2(uint(w1 >> s1) & 0xffffu, uint(w1 >> (s1 - 4u)) & 0xffffu));
  return nfrag(p00.x, p00.y, p01.x, p01.y, p10.x, p10.y, p11.x, p11.y);
}
static inline constexpr uint nax_end(uint N, uint b, uint j) { return ((2u * b + j + 1u) * N) >> 4u; }
static inline constexpr bool nax_quads_fit(uint N) {
  return nax_end(N, 0u, 5u) - nax_end(N, 0u, 0u) <= 16u && nax_end(N, 1u, 5u) - nax_end(N, 1u, 0u) <= 16u;
}
static inline uint nax_sh(uint N, uint b, uint j) {
  const uint s0 = nax_end(N, 0u, 5u) - nax_end(N, 0u, j);
  return s0 + b * (nax_end(N, 1u, 5u) - nax_end(N, 1u, j) - s0);
}
template <typename F>
static inline nfrag nax_quads(F lo, F hi, uint s0, uint s1, uint s4) {
  const half2 p0 = exl3_pairh(uint2(uint(lo >> s0) & 0xffffu, uint(lo >> s1) & 0xffffu));
  const half2 p1 = exl3_pairh(uint2(uint(lo >> s4) & 0xffffu, uint(lo) & 0xffffu));
  const half2 p2 = exl3_pairh(uint2(uint(hi >> s0) & 0xffffu, uint(hi >> s1) & 0xffffu));
  const half2 p3 = exl3_pairh(uint2(uint(hi >> s4) & 0xffffu, uint(hi) & 0xffffu));
  return nfrag(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
}
template <uint N>
static inline nfrag nax_wfrag_k(const device uint *words, uint lane) {
  if (N == 64u) return nax_wfrag(words, lane);
  constexpr uint W = N / 2u;
  const uint end = 8u * N * (lane >> 4u) + N * (lane & 7u) +
                   (nax_end(N, 1u, 5u) - nax_end(N, 0u, 5u)) * ((lane >> 3u) & 1u) + nax_end(N, 0u, 5u);
  const uint b = (lane >> 3u) & 1u;
  const uint s0 = nax_sh(N, b, 0u);
  const uint s1 = nax_sh(N, b, 1u);
  const uint s4 = nax_sh(N, b, 4u);
  if (nax_quads_fit(N)) return nax_quads(uint(exl3_window(words, end, W, false)), uint(exl3_window(words, end + W, W, false)), s0, s1, s4);
  return nax_quads(exl3_window(words, end, W, true), exl3_window(words, end + W, W, true), s0, s1, s4);
}
static inline short2 nax_origin(uint lane) {
  const short qid = short(lane >> 2u);
  return short2(short(((qid & 2) | short(lane & 1u)) * 4), short((qid & 4) | short((lane >> 1u) & 3u)));
}
template <typename D>
static inline void nax_zero(thread D &dst) {
  for (uint s = 0u; s < dst.get_capacity(); s++) dst[s] = 0.0f;
}
template <typename L>
static inline void nax_left_rows(thread L &left, const device half *p0, const device half *p1) {
  const half4 v0 = *(const device half4 *)p0;
  const half4 v1 = *(const device half4 *)p1;
  for (uint c = 0u; c < 4u; c++) { left[c] = v0[c]; left[4 + c] = v1[c]; }
}
template <typename D, typename YT>
static inline void nax_store(const thread D &dst, device YT *y, uint row, uint odim, uint obase, short2 origin, uint rem) {
  const size_t o0 = (size_t)(row + uint(origin.y)) * (size_t)odim;
  const size_t o1 = o0 + 8u * (size_t)odim;
  const bool l0 = uint(origin.y) < rem;
  const bool l1 = uint(origin.y) + 8u < rem;
  for (uint ct = 0u; ct < 2u; ct++)
    for (uint c = 0u; c < 4u; c++) {
      const uint col = obase + ct * 16u + uint(origin.x) + c;
      if (l0) y[o0 + col] = YT(dst[ct * 8u + c]);
      if (l1) y[o1 + col] = YT(dst[ct * 8u + 4u + c]);
    }
}
// y[pos][ODIM] (f32) = x[pos][IDIM] (f16, sorted slots) @ W_e for the <= 16 NB rows of window w (one expert e): each
// decoded 16 x 32 weight fragment feeds up to NB 16-row NAX blocks. trellis u16 [E][IDIM/16][ODIM/16][N] (N = 16 K
// halfwords per 16x16 tile). grid (ODIM / 128, maxwin) threadgroups of 128 threads (4 simdgroups x 32 columns).
// DBG (timing only, wrong outputs; pfkern 18 Oct, NE_PF_MOEDBG): 1 = no trellis read / decode (a constant
// fragment: MMA + activation loads only), 2 = decode only (no MMA, no activation loads).
template <uint IDIM, uint ODIM, uint N, uint NB, typename YT, uint DBG = 0u>
kernel void pf_moe_gemm(device const half *x [[buffer(0)]], device const ushort *trellis [[buffer(1)]],
                        device const uint4 *wtab [[buffer(2)]], device YT *y [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint TILE = 16u, PACKED_W = N / 2u, IT = IDIM / TILE, OT = ODIM / TILE;
  const uint4 wt = wtab[tgp.y];
  const uint run0 = wt.x, nlive = wt.y, eid = wt.z;
  if (nlive == 0u) return;
  constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
  typedef metal::remove_addrspace_t<decltype(left)> LT;
  typedef metal::remove_addrspace_t<decltype(right)> RT;
  auto d0 = op.template get_destination_cooperative_tensor<LT, RT, float>();
  auto d1 = op.template get_destination_cooperative_tensor<LT, RT, float>();
  auto d2 = op.template get_destination_cooperative_tensor<LT, RT, float>();
  auto d3 = op.template get_destination_cooperative_tensor<LT, RT, float>();
  const short2 origin = nax_origin(lane);
  const uint output_base = tgp.x * 128u + sg * 32u;
  const uint last_row = run0 + nlive - 1u;
  const uint nb = (nlive + TILE - 1u) / TILE;  // live 16-row blocks (<= NB)
  nax_zero(d0);
  if (nb > 1u) nax_zero(d1);
  if (NB > 2u && nb > 2u) nax_zero(d2);
  if (NB > 2u && nb > 3u) nax_zero(d3);
  const device uint *wp = (const device uint *)(trellis + (size_t)eid * IT * OT * N) + (size_t)(output_base / TILE) * PACKED_W;
  const device half *xl[NB][2];
  for (uint b = 0u; b < NB; b++) {
    xl[b][0] = x + (size_t)min(run0 + b * TILE + uint(origin.y), last_row) * IDIM + uint(origin.x);
    xl[b][1] = x + (size_t)min(run0 + b * TILE + uint(origin.y) + 8u, last_row) * IDIM + uint(origin.x);
  }
#pragma clang loop unroll_count(2)
  for (uint tk = 0u; tk < IT; tk++) {
    const device uint *words = wp;
    wp += (size_t)OT * PACKED_W;
    nfrag w0, w1;
    if (DBG == 1u) {
      w0 = nfrag(half(lane) * 0.001h);
      w1 = nfrag(half(tk) * 0.001h);
    } else {
      w0 = nax_wfrag_k<N>(words, lane);
      w1 = nax_wfrag_k<N>(words + PACKED_W, lane);
    }
    if (DBG == 2u) {
      for (short s = 0; s < 8; s++) { d0[s] += float(w0[s]); d0[8 + s] += float(w1[s]); }
      continue;
    }
    for (short s = 0; s < 8; s++) { right[s] = w0[s]; right[8 + s] = w1[s]; }
    nax_left_rows(left, xl[0][0] + tk * TILE, xl[0][1] + tk * TILE);
    op.run(left, right, d0);
    if (nb > 1u) { nax_left_rows(left, xl[1][0] + tk * TILE, xl[1][1] + tk * TILE); op.run(left, right, d1); }
    if (NB > 2u && nb > 2u) { nax_left_rows(left, xl[NB > 2u ? 2 : 0][0] + tk * TILE, xl[NB > 2u ? 2 : 0][1] + tk * TILE); op.run(left, right, d2); }
    if (NB > 2u && nb > 3u) { nax_left_rows(left, xl[NB > 3u ? 3 : 0][0] + tk * TILE, xl[NB > 3u ? 3 : 0][1] + tk * TILE); op.run(left, right, d3); }
  }
  nax_store(d0, y, run0, ODIM, output_base, origin, min(nlive, TILE));
  if (nb > 1u) nax_store(d1, y, run0 + TILE, ODIM, output_base, origin, min(nlive - TILE, TILE));
  if (NB > 2u && nb > 2u) nax_store(d2, y, run0 + 2u * TILE, ODIM, output_base, origin, min(nlive - 2u * TILE, TILE));
  if (NB > 2u && nb > 3u) nax_store(d3, y, run0 + 3u * TILE, ODIM, output_base, origin, nlive - 3u * TILE);
}

#define PF_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define PF_MOE_K(K)                                                                        \
  PF_INST("pf_moe_gemm_gu_k" #K "_w32", pf_moe_gemm, HID, INTER, 16u * K, 2u, float)       \
  PF_INST("pf_moe_gemm_dn_k" #K "_w32", pf_moe_gemm, INTER, HID, 16u * K, 2u, float)       \
  PF_INST("pf_moe_gemm_gu_k" #K "_w64", pf_moe_gemm, HID, INTER, 16u * K, 4u, float)       \
  PF_INST("pf_moe_gemm_dn_k" #K "_w64", pf_moe_gemm, INTER, HID, 16u * K, 4u, float)       \
  PF_INST("pf_moe_gemm_gu_k" #K "_w32_h", pf_moe_gemm, HID, INTER, 16u * K, 2u, half)      \
  PF_INST("pf_moe_gemm_dn_k" #K "_w32_h", pf_moe_gemm, INTER, HID, 16u * K, 2u, half)      \
  PF_INST("pf_moe_gemm_gu_k" #K "_w64_h", pf_moe_gemm, HID, INTER, 16u * K, 4u, half)      \
  PF_INST("pf_moe_gemm_dn_k" #K "_w64_h", pf_moe_gemm, INTER, HID, 16u * K, 4u, half)
PF_MOE_K(4)
PF_MOE_K(5)
PF_INST("pf_moe_gemm_gu_k5_w32_h_nodec", pf_moe_gemm, HID, INTER, 80u, 2u, half, 1u)
PF_INST("pf_moe_gemm_dn_k5_w32_h_nodec", pf_moe_gemm, INTER, HID, 80u, 2u, half, 1u)
PF_INST("pf_moe_gemm_gu_k5_w32_h_nomma", pf_moe_gemm, HID, INTER, 80u, 2u, half, 2u)
PF_INST("pf_moe_gemm_dn_k5_w32_h_nomma", pf_moe_gemm, INTER, HID, 80u, 2u, half, 2u)
PF_MOE_K(6)
PF_INST("pf_moe_mid", pf_moe_mid, float)
PF_INST("pf_moe_mid_h", pf_moe_mid, half)
PF_INST("pf_moe_finish", pf_moe_finish, float)
PF_INST("pf_moe_finish_h", pf_moe_finish, half)
