// QSA sparse attention core for a prefill chunk (lab-attn): every row of the chunk in ONE dispatch on the NAX
// (matmul2d) units, over the row's own QSA selection (lab-attn's score / select, docs/attn.md section 8).
//
//   qsa_chunk_attn  TG = (row, kv group): S [16 x 32] = Q [12 heads + 4 zero rows] . K^T per 32 keys (8 selected
//                   blocks), online softmax, O [16 x 256] += P . V; the tail [4 comp, pos] is a 513th, partial block.
//                   4 simdgroups split the head dim (64 dims each); the S partials are summed in a fixed order
//                   through threadgroup memory (deterministic). No split-K, no merge pass: the gate is applied in
//                   the store (qsa_merge's formula).
//
// Measured on real 32K text (docs/attn.md 8): neighbouring rows do NOT share most blocks (the union of 8 rows'
// selections is 2.4x / 3.1x of 512 at 10K / 30K tokens), so a union tile would multiply the work; one row per
// threadgroup with the 12 heads as matrix rows reads every selected K / V row once per (row, group) instead of once
// per head and takes the per-row split-K + merge (two dispatches, 2 x 65 partials per row) out of the chain.
//
// The NAX fragment helpers (BaseNAXFrag: get_coord / mma / row_reduce) are MLX 0.32.3's steel NAX code, vendored by
// lab-prefill in kernels/prefill/mlx/gemm_nax.h (MIT, Copyright (c) 2024-26 Apple Inc.; LICENSE.mlx); the loop
// follows MLX's attention_nax_dsplit (head-dim split, fixed-order S exchange) with gathered K / V rows and QSA masks.
#include "kernels/prefill/mlx/gemm_nax.h"
#include "kernels/ne_common.h"

using namespace mlx::steel;

namespace qch {
constant constexpr uint NQH = 24u, QPG = 12u, HD = 256u, QD = 12288u, AOD = 6144u, KB = 512u, RATIO = 4u;
constant constexpr float NEG = -3.40282347e+38F;  // MLX's finite_min (masked scores)
}  // namespace qch
using namespace qch;

typedef BaseNAXFrag Frag;
typedef Frag::dtype_frag_t<float> ffrag;
typedef Frag::dtype_frag_t<bfloat> bfrag;

struct MaxOpC {
  static float apply(float x, float y) { return metal::max(x, y); }
};
struct SumOpC {
  static float apply(float x, float y) { return x + y; }
};

// grid (n_rows, 2 kv groups) x 128 threads. Row i of the grid = chunk row rb + i at position posbase[0] + rb + i.
//   qr    f32 [C][24][256]  prepped queries (qsa_prep: normed, roped, bf16 values, * 1/16)
//   kc/vc bf16 [2][tmax][256] the layer's cache (the chunk's own rows already appended by qsa_prep)
//   sel   u32 [C][512] ascending block lists, nsel u32 [C] (qsa_select)
//   q     f32 [C][12288] q_proj rows (the gate half), out f32 [C][6144]
kernel void qsa_chunk_attn_bf16(device const float *qr [[buffer(0)]], device const bfloat *kc [[buffer(1)]],
                                device const bfloat *vc [[buffer(2)]], device const uint *sel [[buffer(3)]],
                                device const uint *nsel [[buffer(4)]], device const uint *posbase [[buffer(5)]],
                                constant uint &rb [[buffer(6)]], constant uint &tmax [[buffer(7)]],
                                device const float *q [[buffer(8)]], device float *out [[buffer(9)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[4][2 * 8 * 32];  // S partials of the 4 head-dim quarters (2 frags x 8 values x 32 lanes)
  const uint row = rb + tg.x, g = tg.y;
  const uint p = posbase[0] + row, comp = (p + 1u) / RATIO, n = nsel[row];
  const bool tail = comp * RATIO <= p;
  const uint nb = n + (tail ? 1u : 0u);  // blocks incl. the partial tail block `comp`
  const short2 sc = Frag::get_coord();
  const short sm = sc.y, sn = sc.x;  // fragment row (head / key) and first column of this lane
  const uint d0 = sg * 64u;
  const float LOG2E = 1.44269504089f;

  // Q quarter: 4 fragments of 16 heads x 16 dims (heads 12..15 zero), bf16 (exact: qr holds bf16 values / 16)
  bfrag Qf[4];
  for (short id = 0; id < 4; id++)
    for (short i = 0; i < 2; i++) {
      const uint h = sm + 8 * i;
      for (short j = 0; j < 4; j++)
        Qf[id][i * 4 + j] = h < QPG ? bfloat(qr[((size_t)row * NQH + g * QPG + h) * HD + d0 + id * 16 + sn + j]) : bfloat(0.0f);
    }
  ffrag Of[4];
  for (short id = 0; id < 4; id++) Of[id] = ffrag(0.0f);
  float mx[2] = {NEG, NEG}, sum[2] = {0.0f, 0.0f};

  const device uint *sl = sel + (size_t)row * KB;
  const device bfloat *K = kc + (size_t)g * tmax * HD + d0 + sn;
  const device bfloat *V = vc + (size_t)g * tmax * HD + d0 + sn;
  const uint e0 = uint(sm) / 4u, jk = uint(sm) % 4u;  // this lane's key rows in a 32-key tile: entries e0 + 2u
  const uint ptail = p - comp * RATIO;  // tail keys j <= ptail are attended (if tail)
  typedef vec<bfloat, 4> bf4v;
  // key row (cache row) of entry e, key jk; -1 = not attended (beyond the list, or tail key past pos)
  auto krow = [&](uint e) -> int {
    const uint b = e < n ? sl[e] : (e == n && tail ? comp : 0xFFFFFFFFu);
    if (b == 0xFFFFFFFFu || (e == n && jk > ptail)) return -1;
    return int(b * RATIO + jk);
  };
  auto ld = [&](const device bfloat *base, int r, uint off) -> bf4v {
    return r >= 0 ? *(const device bf4v *)(base + (size_t)r * HD + off) : bf4v(bfloat(0.0f));
  };
  for (uint kb = 0u; kb * 8u < nb; kb++) {
    int kr[4];
    for (uint u = 0u; u < 4u; u++) kr[u] = krow(kb * 8u + e0 + 2u * u);
    // S partial over this simdgroup's 64 dims (K loaded per 16-dim slice, used at once: few live registers)
    ffrag S[2] = {ffrag(0.0f), ffrag(0.0f)};
    for (short id = 0; id < 4; id++) {
      bfrag K0, K1;
      for (short i = 0; i < 2; i++) {
        const bf4v a0 = ld(K, kr[i], id * 16), a1 = ld(K, kr[2 + i], id * 16);
        for (short j = 0; j < 4; j++) {
          K0[i * 4 + j] = a0[j];
          K1[i * 4 + j] = a1[j];
        }
      }
      Frag::mma(S[0], S[1], Qf[id], metal::false_type{}, K0, K1, metal::true_type{});
    }
    // fixed-order sum of the 4 quarters
    for (short f = 0; f < 2; f++)
      for (short k = 0; k < 8; k++) xs[sg][(f * 8 + k) * 32 + lane] = S[f][k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short f = 0; f < 2; f++)
      for (short k = 0; k < 8; k++) {
        const uint o = (f * 8 + k) * 32 + lane;
        S[f][k] = ((xs[0][o] + xs[1][o]) + xs[2][o]) + xs[3][o];
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // mask (lane's columns: key f*16 + sn + j = entry kb*8 + f*4 + sn/4, key j) and scale to log2 units
    for (short f = 0; f < 2; f++) {
      const uint e = kb * 8u + f * 4u + uint(sn) / 4u;
      for (short j = 0; j < 4; j++) {
        const bool ok = e < n || (e == n && tail && uint(j) <= ptail);
        for (short i = 0; i < 2; i++) S[f][i * 4 + j] = ok ? S[f][i * 4 + j] * LOG2E : NEG;
      }
    }
    float nm[2] = {mx[0], mx[1]};
    Frag::row_reduce<MaxOpC>(S[0], nm);
    Frag::row_reduce<MaxOpC>(S[1], nm);
    for (short f = 0; f < 2; f++)
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < 4; j++) S[f][i * 4 + j] = fast::exp2(S[f][i * 4 + j] - nm[i]);
    float fac[2];
    for (short i = 0; i < 2; i++) {
      fac[i] = fast::exp2(mx[i] - nm[i]);
      mx[i] = nm[i];
      sum[i] *= fac[i];
    }
    Frag::row_reduce<SumOpC>(S[0], sum);
    Frag::row_reduce<SumOpC>(S[1], sum);
    for (short id = 0; id < 4; id++)
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < 4; j++) Of[id][i * 4 + j] *= fac[i];
    // O += P . V (keys f*16 + sm + 8i = key row u = 2f + i; dims id*16 .., (id+1)*16 ..)
    for (short id = 0; id < 4; id += 2)
      for (short f = 0; f < 2; f++) {
        bfrag V0, V1;
        for (short i = 0; i < 2; i++) {
          const bf4v a0 = ld(V, kr[2 * f + i], id * 16), a1 = ld(V, kr[2 * f + i], (id + 1) * 16);
          for (short j = 0; j < 4; j++) {
            V0[i * 4 + j] = a0[j];
            V1[i * 4 + j] = a1[j];
          }
        }
        Frag::mma(Of[id], Of[id + 1], S[f], metal::false_type{}, V0, V1, metal::false_type{});
      }
  }
  // out = rr(rr(O / l) * sigmoid(gate)) with qsa_merge's bf16 op chain
  for (short i = 0; i < 2; i++) {
    const uint h = sm + 8 * i;
    if (h >= QPG) continue;
    const uint hh = g * QPG + h;
    for (short id = 0; id < 4; id++)
      for (short j = 0; j < 4; j++) {
        const uint d = d0 + id * 16 + sn + j;
        const float o = Of[id][i * 4 + j] / sum[i];
        const float gq = rr(q[(size_t)row * QD + hh * 512u + HD + d]);
        const float sg2 = NE_REFR ? bf16r(1.0f / bf16r(1.0f + bf16r(precise::exp(-gq)))) : sigm(gq);
        out[(size_t)row * AOD + hh * HD + d] = rr(rr(o) * sg2);
      }
  }
}

