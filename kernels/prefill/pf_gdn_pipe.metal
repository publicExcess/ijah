// Pipelined gated-delta recurrence for the chunked prefill (lab-pflevers, 16 Oct 2026; docs/prefill.md 4b).
//
// Ported from Sushi src/kernels/gdn_pipelined.metal (github.com/beamivalice/sushi, MIT License, Copyright (c) 2026
// Theinruj Toranavikrai, Copyright (c) 2026 David Dalcu), which is itself ported from oMLX
// omlx/custom_kernels/qwen35_prefill/gdn.py `gated_delta_pipelined` (jundot/omlx, Apache-2.0; SPDX header of the
// Sushi file: Apache-2.0). Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. Licensed under the Apache License,
// Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0) for the oMLX-derived parts.
//
// Changes from the source: our layouts (q / k f32 [n][2][2048] already l2-normed and q-scaled by pf_gdn_qknorm,
// v bf16 inside the conv rows, g / beta f32 [n][2 * 48]); the 12 unrolled steps written as one constant-trip loop;
// y written f32 with rr() (the decode path's rounding point); state f32 in place.
//
// Layout: a threadgroup of 128 threads owns 16 value rows of one v head; 8 lanes own one row (seg = lane % 8 holds
// dk channels 4 (seg + 8 i) .. + 3, i < 4). Blocks of TB = 12 tokens of q / k / v / g / beta are staged in
// threadgroup memory; the next block is fetched into registers while the current one runs; the q . S reductions of
// 4 steps are batched (one 3-level exchange yields 4 outputs) off the k . S dependency chain.
// Summation order differs from fn_gdn_step (8-lane xor tree of fma pairs vs a 32-lane simd_sum of float4 dots).
#include "kernels/ne_common.h"

namespace pfgp {
constant constexpr uint NKH = 16u, NVH = 48u, DK = 128u, DV = 128u;
constant constexpr uint KEYD = NKH * DK, VALD = NVH * DV, CONVD = 2u * KEYD + VALD;
constant constexpr int TB = 12, DB = 16, NT = 128;
constant constexpr int KQ4 = TB * (int)DK / 4, NKQ = (KQ4 + NT - 1) / NT, V4 = TB * DB / 4, NV = (V4 + NT - 1) / NT;
}  // namespace pfgp

// grid (DV / 16, 48) threadgroups x 128 threads
kernel void pf_gdn_scan_pipe(device const float4 *qk [[buffer(0)]], device const ushort *conv [[buffer(1)]],
                             device const float *gb [[buffer(2)]], device float *state [[buffer(3)]],
                             device float *y [[buffer(4)]], constant uint &n [[buffer(5)]],
                             uint2 tgp [[threadgroup_position_in_grid]], uint tidu [[thread_index_in_threadgroup]]) {
  using namespace pfgp;
  const int T = (int)n;
  const int tid = (int)tidu, hv = (int)tgp.y, hk = hv / (int)(NVH / NKH), dv0 = (int)tgp.x * DB;
  const int rg = tid / 8, seg = tid % 8;
  threadgroup float4 k_s[TB][DK / 4];
  threadgroup float4 q_s[TB][DK / 4];
  threadgroup float v_s[TB][DB];
  threadgroup float g_s[TB], b_s[TB];
  threadgroup float y_s[TB][DB];
  // qk float4 index of (t, which, c4): (t * 2 + which) * 512 + hk * 32 + c4
  const device float4 *q_base = qk + hk * (int)(DK / 4u);
  const device float4 *k_base = qk + (KEYD / 4u) + hk * (int)(DK / 4u);
  const device ushort *v_base = conv + 2u * KEYD + hv * (int)DV + dv0;
  device float4 *S4 = (device float4 *)(state + ((size_t)hv * DV + dv0 + rg) * DK);
  float4 st[4];
  for (int i = 0; i < 4; ++i) st[i] = S4[seg + 8 * i];

  float4 pk[NKQ], pq[NKQ];
  float4 pv[NV];
  float pg = 0.0f, pb = 0.0f;
  auto fetch = [&](int t0n) {
    const int ttn = min(TB, T - t0n);
    for (int j = 0; j < NKQ; ++j) {
      const int p = tid + j * NT;
      if (p < ttn * (int)(DK / 4)) {
        const int r = p / (int)(DK / 4), c4 = p % (int)(DK / 4);
        const size_t off = (size_t)(t0n + r) * (2u * KEYD / 4u) + c4;
        pk[j] = k_base[off];
        pq[j] = q_base[off];
      }
    }
    for (int j = 0; j < NV; ++j) {
      const int p = tid + j * NT;
      if (p < ttn * (DB / 4)) {
        const int r = p / (DB / 4), c4 = p % (DB / 4);
        const ushort4 u = *(const device ushort4 *)(v_base + (size_t)(t0n + r) * CONVD + 4 * c4);
        pv[j] = float4(bf(u.x), bf(u.y), bf(u.z), bf(u.w));
      }
    }
    if (tid < ttn) {
      pg = gb[(size_t)(t0n + tid) * 2u * NVH + hv];
      pb = gb[(size_t)(t0n + tid) * 2u * NVH + NVH + hv];
    }
  };
  fetch(0);

  float4 k_next[4];
  float g_next = 0.0f, b_next = 0.0f, v_next = 0.0f;
  float pq_prev = 0.0f;
  float pq4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int t0 = 0; t0 < T; t0 += TB) {
    const int tt = min(TB, T - t0);
    for (int j = 0; j < NKQ; ++j) {
      const int p = tid + j * NT;
      if (p < tt * (int)(DK / 4)) {
        const int r = p / (int)(DK / 4), c4 = p % (int)(DK / 4);
        k_s[r][c4] = pk[j];
        q_s[r][c4] = pq[j];
      }
    }
    for (int j = 0; j < NV; ++j) {
      const int p = tid + j * NT;
      if (p < tt * (DB / 4)) {
        const int r = p / (DB / 4), c4 = p % (DB / 4);
        *(threadgroup float4 *)(&v_s[r][4 * c4]) = pv[j];
      }
    }
    if (tid < tt) { g_s[tid] = pg; b_s[tid] = pb; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t0 + TB < T) fetch(t0 + TB);
    for (int i = 0; i < 4; ++i) k_next[i] = k_s[0][seg + 8 * i];
    g_next = g_s[0]; b_next = b_s[0]; v_next = v_s[0][rg];

    if (tt == TB) {
#pragma unroll
      for (int t = 0; t < TB; ++t) {
        float4 kc[4];
        for (int i = 0; i < 4; ++i) kc[i] = k_next[i];
        const float gc = g_next, bc = b_next, vc = v_next;
        if (t + 1 < TB) {
          for (int i = 0; i < 4; ++i) k_next[i] = k_s[t + 1][seg + 8 * i];
          g_next = g_s[t + 1]; b_next = b_s[t + 1]; v_next = v_s[t + 1][rg];
        }
        float2 a2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
          st[i] = st[i] * gc;
          a2 = fma(st[i].xy, kc[i].xy, a2);
          a2 = fma(st[i].zw, kc[i].zw, a2);
        }
        if (t == 4 || t == 8) {  // flush the q . S partials of steps t - 4 .. t - 1 (off the k . S chain)
          const bool qb2 = (seg & 4) != 0, qb1 = (seg & 2) != 0;
          float qa = qb2 ? pq4[2] : pq4[0];
          float qb = qb2 ? pq4[3] : pq4[1];
          qa += simd_shuffle_xor(qb2 ? pq4[0] : pq4[2], 4);
          qb += simd_shuffle_xor(qb2 ? pq4[1] : pq4[3], 4);
          float qx = qb1 ? qb : qa;
          qx += simd_shuffle_xor(qb1 ? qa : qb, 2);
          qx += simd_shuffle_xor(qx, 1);
          if ((seg & 1) == 0) y_s[t - 4 + (qb2 ? 2 : 0) + (qb1 ? 1 : 0)][rg] = qx;
        }
        float p = a2.x + a2.y;
        p += simd_shuffle_xor(p, 4);
        p += simd_shuffle_xor(p, 2);
        p += simd_shuffle_xor(p, 1);
        float4 qc[4];
        for (int i = 0; i < 4; ++i) qc[i] = q_s[t][seg + 8 * i];
        const float delta = (vc - p) * bc;
        float2 o2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
          st[i] = fma(kc[i], float4(delta), st[i]);
          o2 = fma(st[i].xy, qc[i].xy, o2);
          o2 = fma(st[i].zw, qc[i].zw, o2);
        }
        pq4[t % 4] = o2.x + o2.y;
      }
      {
        const bool qb2 = (seg & 4) != 0, qb1 = (seg & 2) != 0;
        float qa = qb2 ? pq4[2] : pq4[0];
        float qb = qb2 ? pq4[3] : pq4[1];
        qa += simd_shuffle_xor(qb2 ? pq4[0] : pq4[2], 4);
        qb += simd_shuffle_xor(qb2 ? pq4[1] : pq4[3], 4);
        float qx = qb1 ? qb : qa;
        qx += simd_shuffle_xor(qb1 ? qa : qb, 2);
        qx += simd_shuffle_xor(qx, 1);
        if ((seg & 1) == 0) y_s[8 + (qb2 ? 2 : 0) + (qb1 ? 1 : 0)][rg] = qx;
      }
    } else {
      for (int t = 0; t < tt; ++t) {
        float4 kc[4];
        for (int i = 0; i < 4; ++i) kc[i] = k_next[i];
        const float gc = g_next, bc = b_next, vc = v_next;
        if (t + 1 < tt) {
          for (int i = 0; i < 4; ++i) k_next[i] = k_s[t + 1][seg + 8 * i];
          g_next = g_s[t + 1]; b_next = b_s[t + 1]; v_next = v_s[t + 1][rg];
        }
        float2 a2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
          st[i] = st[i] * gc;
          a2 = fma(st[i].xy, kc[i].xy, a2);
          a2 = fma(st[i].zw, kc[i].zw, a2);
        }
        if (t > 0) {
          float yq = pq_prev;
          yq += simd_shuffle_down(yq, 4);
          yq += simd_shuffle_down(yq, 2);
          yq += simd_shuffle_down(yq, 1);
          if (seg == 0) y_s[t - 1][rg] = yq;
        }
        float p = a2.x + a2.y;
        p += simd_shuffle_xor(p, 4);
        p += simd_shuffle_xor(p, 2);
        p += simd_shuffle_xor(p, 1);
        float4 qc[4];
        for (int i = 0; i < 4; ++i) qc[i] = q_s[t][seg + 8 * i];
        const float delta = (vc - p) * bc;
        float2 o2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
          st[i] = fma(kc[i], float4(delta), st[i]);
          o2 = fma(st[i].xy, qc[i].xy, o2);
          o2 = fma(st[i].zw, qc[i].zw, o2);
        }
        pq_prev = o2.x + o2.y;
      }
      float yq = pq_prev;
      yq += simd_shuffle_down(yq, 4);
      yq += simd_shuffle_down(yq, 2);
      yq += simd_shuffle_down(yq, 1);
      if (seg == 0) y_s[tt - 1][rg] = yq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = tid; p < tt * (DB / 4); p += NT) {
      const int r = p / (DB / 4), c4 = p % (DB / 4);
      const float4 o = *(threadgroup float4 *)(&y_s[r][4 * c4]);
      *(device float4 *)(y + (size_t)(t0 + r) * VALD + hv * (int)DV + dv0 + 4 * c4) =
          float4(rr(o.x), rr(o.y), rr(o.z), rr(o.w));
    }
  }
  for (int i = 0; i < 4; ++i) S4[seg + 8 * i] = st[i];
}
