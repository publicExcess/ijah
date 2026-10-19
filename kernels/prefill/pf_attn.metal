// Prefill attention for the dense rows of a chunk (rows whose QSA selection is "all blocks" = plain causal attention:
// complete blocks <= 512, positions < ~2051): MLX's NAX flash attention (head-dim split kernel, 8 simdgroups) over
// lab-attn's bf16 KV cache, then the output gate. Owner: lab-prefill. Design: docs/prefill.md.
//
// The q rows are lab-attn's prepped queries (qsa_prep: q_norm, rope, bf16, * 1/16; copied out of each slice's step
// scratch by pf_attn_qcopy), so the scores are exactly lab-attn's q . k; the gate is qsa_merge's formula.
// attention_nax_dsplit is vendored from MLX 0.32.3 (kernels/prefill/mlx/attention_nax.h; MIT, Copyright (c) 2024-26
// Apple Inc.; LICENSE.mlx).
#include "kernels/prefill/mlx/attention_nax.h"
#include "kernels/ne_common.h"

template [[host_name("pf_attn_nax_d256")]] [[kernel]] decltype(attention_nax_dsplit<bfloat, 64, 32, 256, 4, 2>)
    attention_nax_dsplit<bfloat, 64, 32, 256, 4, 2>;

constant constexpr uint PA_NQH = 24u, PA_HD = 256u, PA_QD = 12288u;

// qb[r0 + t][h][d] = qr[t][h][d] (f32 values already bf16-exact) for t < R. R * 24 * 256 threads.
kernel void pf_attn_qcopy(device const float *qr [[buffer(0)]], device bfloat *qb [[buffer(1)]],
                          constant uint &r0 [[buffer(2)]], constant uint &cnt [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < cnt) qb[(size_t)r0 * PA_NQH * PA_HD + i] = bfloat(qr[i]);
}
// out[t][h*256 + d] = rr(rr(o) * sigmoid_bf(rr(gate))) (qsa_merge's tail; o = the bf16 attention output). n * 6144
// threads.
kernel void pf_attn_gate(device const bfloat *o [[buffer(0)]], device const float *q [[buffer(1)]],
                         device float *out [[buffer(2)]], constant uint &cnt [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= cnt) return;
  const uint t = i / (PA_NQH * PA_HD), h = (i / PA_HD) % PA_NQH, d = i % PA_HD;
  const float gq = rr(q[(size_t)t * PA_QD + h * 512u + PA_HD + d]);
  const float sg2 = NE_REFR ? bf16r(1.0f / bf16r(1.0f + bf16r(precise::exp(-gq)))) : sigm(gq);
  out[i] = rr(rr(float(o[i])) * sg2);
}
