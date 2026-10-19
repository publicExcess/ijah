// NAX (M5 neural accelerator, MPP matmul2d 16x32x16) multi-row GEMM over MLX 8-bit (or 4 / 2-bit) affine weights, for MTP verify
// widths (R = 1..8 rows). Owner: lab-mtp. Design and numbers: docs/mtp.md.
//
// One simdgroup = one 32-column block x a run of 64-wide k chunks. Per chunk a lane loads 16 weight bytes of each of
// its 4 weight rows (MLX layout W[n][k], 64 contiguous bytes per row across the 4 lanes that share a row) and one
// (scale, bias) per row, dequantises to half in f32 (w = s q + b), and feeds 4 matmul2d steps (one per 16-k
// sub-step). The k order inside a chunk is permuted (logical (sub-step j, k0 + e) = physical 16 (k0 / 4) + 4 j + e),
// and x is read with the same permutation, so both operands are contiguous 16-byte / 64-byte loads. Weights are read
// once for all rows (rows 0..R-1 of the 16-row M, R <= 16; rows >= R are zero), so the time is flat in R until the ALU binds.
// KS simdgroups split k; their partial 8 (16 when R > 8) x 32 tiles reduce through threadgroup memory in fixed order
// (deterministic); 32 threads then run a column-wise epilogue over the R rows (EPI):
//   EPI 0 store      : out_s[r * ld_s + n] = rr(v)  (rr = bf16 rounding when NE_REFR; ld_s = segment rows)
//   EPI 1 gdn in     : segments qkv / z / b / a of GDN in_proj; = fn_gdn_inproj's epilogue (conv1d K4 + silu with the
//                      conv state updated in place, z, beta = sigmoid(b), g = exp(-exp(A_log) softplus(a + dt_bias)))
//   EPI 5 gdn in + history: EPI 1 and o3 [3 + R][10240] = [conv state in; raw qkv rows] (MTP rollback, docs/mtp.md)
//   EPI 6 gdn in, lazy : as EPI 1 for z / b / a; qkv conv over [committed state; pending rows; new rows] (rollback v2)
//   EPI 2 hc write   : one segment (out_proj 6144 -> 2560): branch = rr(v), hout_s = rr(h_s + rr(branch * inj_s))
//   EPI 3 attn proj  : segments q / k / v / indexer qk; = fn_attn_proj's stores (+ raw indexer key -> its cache)
//   EPI 4 row bias   : out[r][n] = rr(rr(v) + e0[r / 4][n])  (the MTP head's embedding + hidden fuse)
// Lane layout of matmul2d<16,32,16> operands (right transposed): see lab lead 6 (the lab tree's
// dev/m5/flashnext/engine/nax/NOTES.md), derived there from sushi's nax_wfrag (MIT, (c) 2026 Theinruj Toranavikrai,
// David Dalcu): lane l: k0 = 4 (l & 1) + 8 ((l >> 3) & 1), c0 = 4 (l >> 4) + ((l >> 1) & 3); right element s ->
// (k0 + (s & 3), n = c0 + 8 ((s >> 2) & 1) + 16 (s >> 3)); left element s < 4 -> (row c0, k0 + s); destination
// element e -> (row c0 + 8 ((e >> 2) & 1), col 16 (e >> 3) + k0 + (e & 3)).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "kernels/ne_common.h"
using namespace mpp::tensor_ops;

constant uint NQ_IN [[function_constant(1)]];   // inputs (multiple of 64)
constant uint NQ_R [[function_constant(3)]];    // live rows 1..16 (KS * (R > 8 ? 16 : 8) <= 128)
constant uint NQ_KS [[function_constant(4)]];   // k splits (simdgroups per threadgroup)
constant uint NQ_EPI [[function_constant(5)]];  // epilogue
constant uint NQ_BITS_FC [[function_constant(6)]];  // weight bits: 8 (the pack's trunk), 4 or 2 (requantised draft heads)
constant uint NQ_BITS = is_function_constant_defined(NQ_BITS_FC) ? NQ_BITS_FC : 8u;

struct NaxSegs {     // up to 4 weight matrices in one dispatch (same IN); column blocks of 32 per segment
  uint nseg;
  uint rows[4];
  uint blk0[5];      // first column block of each segment (blk0[nseg] = total)
  uint gsh[4];       // log2 of each segment's group size (64 or 128: one (scale, bias) per 64-wide k chunk or two)
};

constant constexpr uint NX_HID = 2560u, NX_HC = 4u, NX_HCD = NX_HC * NX_HID;
constant constexpr uint NX_NVH = 48u, NX_VALD = 6144u, NX_CONVD = 10240u;
constant constexpr uint NX_QD = 12288u, NX_KVD = 512u, NX_IQD = 640u;

static inline half4 h4(float4 v) { return half4(v); }

kernel void nax_q8_gemm(device const float *x [[buffer(0)]],
                        device const uchar *w0 [[buffer(1)]], device const ushort *s0 [[buffer(2)]], device const ushort *b0 [[buffer(3)]],
                        device const uchar *w1 [[buffer(4)]], device const ushort *s1 [[buffer(5)]], device const ushort *b1 [[buffer(6)]],
                        device const uchar *w2 [[buffer(7)]], device const ushort *s2 [[buffer(8)]], device const ushort *b2 [[buffer(9)]],
                        device const uchar *w3 [[buffer(10)]], device const ushort *s3 [[buffer(11)]], device const ushort *b3 [[buffer(12)]],
                        device float *o0 [[buffer(13)]], device float *o1 [[buffer(14)]],
                        device float *o2 [[buffer(15)]], device float *o3 [[buffer(16)]],
                        constant NaxSegs &sg_ [[buffer(17)]],
                        device float *e0 [[buffer(18)]], device float *e1 [[buffer(19)]],
                        device float *e2 [[buffer(20)]], device float *e3 [[buffer(21)]],
                        device float *e4 [[buffer(22)]], device float *e5 [[buffer(23)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                        uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[16u * 8u * 32u];  // [KS][RMP rows][32 cols], KS * RMP <= 128
  const uint IN = NQ_IN, R = NQ_R, KS = NQ_KS;
  const uint RMP = NQ_R > 8u ? 16u : 8u;  // M rows in use (rows >= R are zero)
  // segment of this column block
  uint seg = 0u;
  for (uint i = 1u; i < sg_.nseg; i++) seg += (tg >= sg_.blk0[i]) ? 1u : 0u;
  const uint nrows = sg_.rows[seg];
  const uint n0 = (tg - sg_.blk0[seg]) * 32u;
  const device uchar *W = seg == 0u ? w0 : seg == 1u ? w1 : seg == 2u ? w2 : w3;
  const device ushort *S = seg == 0u ? s0 : seg == 1u ? s1 : seg == 2u ? s2 : s3;
  const device ushort *B = seg == 0u ? b0 : seg == 1u ? b1 : seg == 2u ? b2 : b3;

  const uint k0 = 4u * (lane & 1u) + 8u * ((lane >> 3u) & 1u);
  const uint c0 = 4u * (lane >> 4u) + ((lane >> 1u) & 3u);
  const uint q = k0 >> 2u;  // 0..3: this lane's 16-byte slice of a 64-k chunk
  const uint gsh = sg_.gsh[seg];
  const uint NG = IN >> gsh;
  uint nj[4];
  for (uint j = 0u; j < 4u; j++) nj[j] = min(n0 + c0 + 8u * (j & 1u) + 16u * (j >> 1u), nrows - 1u);

  constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto left = op.get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.get_right_input_cooperative_tensor<half, half, float>();
  auto d = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                  metal::remove_addrspace_t<decltype(right)>, float>();
  for (uint s = 0u; s < d.get_capacity(); s++) d[s] = 0.0f;
  for (short s = 0; s < 8; s++) left[s] = 0.0h;
  const bool live = c0 < R, live2 = c0 + 8u < R;  // left rows c0 and c0 + 8
  // Chunk = 4 lanes x 16 bytes per row: VPL values per lane (16 at 8 bits, 32 at 4, 64 at 2), CW = 4 VPL k per chunk,
  // SS = VPL / 4 matmul steps. Logical (step jj, k0 + e) = physical CW t + VPL (k0 / 4) + 4 jj + e; a lane's VPL
  // values sit in one group (g64 / g128), so one (scale, bias) per row per chunk.
  const uint VPL = 128u / NQ_BITS, CW = 4u * VPL, SS = VPL / 4u;
  const device float *xr = x + (size_t)min(c0, R - 1u) * IN + VPL * q;
  const device float *xr2 = x + (size_t)min(c0 + 8u, R - 1u) * IN + VPL * q;
  const uint NCH = IN / CW;
  const uint RB = IN * NQ_BITS / 8u;  // row bytes
  for (uint t = sgi; t < NCH; t += KS) {
    const uint kb = CW * t;
    uint4 wv[4];  // this lane's 16 bytes of row j
    float sc[4], bi[4];
    const uint g = (kb + VPL * q) >> gsh;
    for (uint j = 0u; j < 4u; j++) {
      wv[j] = *(const device uint4 *)(W + (size_t)nj[j] * RB + (kb + VPL * q) * NQ_BITS / 8u);
      sc[j] = bf(S[(size_t)nj[j] * NG + g]);
      bi[j] = bf(B[(size_t)nj[j] * NG + g]);
    }
    for (uint jj = 0u; jj < SS; jj++) {
      const half4 xa = h4(live ? ((const device float4 *)(xr + kb))[jj] : float4(0.0f));
      for (short s = 0; s < 4; s++) left[s] = xa[s];
      if (RMP == 16u) {
        const half4 xb = h4(live2 ? ((const device float4 *)(xr2 + kb))[jj] : float4(0.0f));
        for (short s = 0; s < 4; s++) left[4 + s] = xb[s];
      }
      for (uint j = 0u; j < 4u; j++) {
        half4 wq;
        if (NQ_BITS == 8u) wq = h4(fma(float4(as_type<uchar4>(wv[j][jj])), float4(sc[j]), float4(bi[j])));
        else {  // draft-only low-bit heads: dequantise in half (the shortlist is re-scored exactly)
          ushort4 qv;
          if (NQ_BITS == 4u) {  // values 4 jj .. 4 jj + 3 = nibbles 4 (jj & 1) .. of word jj / 2
            const uint w = wv[j][jj >> 1u] >> (16u * (jj & 1u));
            qv = ushort4(w & 15u, (w >> 4u) & 15u, (w >> 8u) & 15u, (w >> 12u) & 15u);
          } else {              // values 4 jj .. = 2-bit fields 4 (jj & 3) .. of word jj / 4
            const uint w = wv[j][jj >> 2u] >> (8u * (jj & 3u));
            qv = ushort4(w & 3u, (w >> 2u) & 3u, (w >> 4u) & 3u, (w >> 6u) & 3u);
          }
          wq = fma(half4(qv), half4(half(sc[j])), half4(half(bi[j])));
        }
        for (short e = 0; e < 4; e++) right[4u * j + e] = wq[e];
      }
      op.run(left, right, d);
    }
  }
  // KS partials -> threadgroup memory (R <= 8: rows c0 only, e in {0..3, 8..11}; else all 16 elements)
  for (uint i = 0u; i < (RMP == 16u ? 16u : 8u); i++) {
    const uint e = RMP == 16u ? i : (((i >> 2u) << 3u) | (i & 3u));
    const uint row = c0 + 8u * ((e >> 2u) & 1u);
    part[(sgi * RMP + row) * 32u + 16u * (e >> 3u) + k0 + (e & 3u)] = d[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid >= 32u) return;
  const uint col = lid, n = n0 + col;
  if (n >= nrows) return;
  float v[16];
  for (uint r = 0u; r < R; r++) {
    float a = 0.0f;
    for (uint k = 0u; k < KS; k++) a += part[(k * RMP + r) * 32u + col];
    v[r] = a;
  }
  device float *O = seg == 0u ? o0 : seg == 1u ? o1 : seg == 2u ? o2 : o3;
  if (NQ_EPI == 0u) {
    for (uint r = 0u; r < R; r++) O[(size_t)r * nrows + n] = rr(v[r]);
  } else if (NQ_EPI == 4u) {  // add a row bias shared by groups of 4 rows: O[r][n] = rr(rr(v) + e0[r / 4][n])
    // (MTP fuse: rows = (token, stream) pairs, e0 = fc_embedding(normed embed) per token)
    for (uint r = 0u; r < R; r++) O[(size_t)r * nrows + n] = rr(rr(v[r]) + e0[(size_t)(r >> 2u) * nrows + n]);
  } else if (NQ_EPI == 6u && seg == 0u) {  // GDN in_proj, lazy commit (rollback v2): conv input = [committed conv
    // state (3); the P pending raw rows of the previous round (e4 [8][10240], P = e5[0] as u32 = rows kept); the R new
    // raw rows]; the new rows' conv outputs -> o0; conv state <- the last 3 of [state; pending] (committed only);
    // e4 <- the R new raw rows (each column is one thread: read pending, then overwrite).
    const device ushort *cw = (const device ushort *)e0 + (size_t)n * 4u;
    const float c0w = bf(cw[0]), c1w = bf(cw[1]), c2w = bf(cw[2]), c3w = bf(cw[3]);
    device float *cst = e1;
    const uint P = min(as_type<uint>(e5[0]), 8u);
    float sq[3u + 8u + 16u];
    sq[0] = cst[n]; sq[1] = cst[NX_CONVD + n]; sq[2] = cst[2u * NX_CONVD + n];
    for (uint p = 0u; p < P; p++) sq[3u + p] = e4[(size_t)p * NX_CONVD + n];
    for (uint t = 0u; t < R; t++) sq[3u + P + t] = rr(v[t]);
    for (uint t = 0u; t < R; t++)
      o0[t * NX_CONVD + n] = rr(silu(rr(sq[P + t] * c0w + sq[P + t + 1u] * c1w + sq[P + t + 2u] * c2w + sq[P + t + 3u] * c3w)));
    cst[n] = sq[P]; cst[NX_CONVD + n] = sq[P + 1u]; cst[2u * NX_CONVD + n] = sq[P + 2u];
    for (uint t = 0u; t < R; t++) e4[(size_t)t * NX_CONVD + n] = sq[3u + P + t];
  } else if (NQ_EPI == 1u || NQ_EPI == 5u || NQ_EPI == 6u) {  // GDN in_proj: o0 qkv (conv + silu), o1 z, o2 gb [R][g 48 | beta 48]; e0 conv weight
    // (bf16 [10240][4] as ushort), e1 conv state f32 [3][10240] (in place), e2 A_log (bf16), e3 dt_bias (bf16)
    if (seg == 0u) {
      const device ushort *cw = (const device ushort *)e0 + (size_t)n * 4u;
      const float c0w = bf(cw[0]), c1w = bf(cw[1]), c2w = bf(cw[2]), c3w = bf(cw[3]);
      device float *cst = e1;
      float sq[3u + 16u];
      sq[0] = cst[n]; sq[1] = cst[NX_CONVD + n]; sq[2] = cst[2u * NX_CONVD + n];
      for (uint t = 0u; t < R; t++) sq[3u + t] = rr(v[t]);
      for (uint t = 0u; t < R; t++)
        o0[t * NX_CONVD + n] = rr(silu(rr(sq[t] * c0w + sq[t + 1u] * c1w + sq[t + 2u] * c2w + sq[t + 3u] * c3w)));
      if (NQ_EPI == 5u)  // rollback history: o3 [3 + R][10240] = conv state in, then the R raw (bf16) qkv rows
        for (uint t = 0u; t < 3u + R; t++) o3[(size_t)t * NX_CONVD + n] = sq[t];
      cst[n] = sq[R]; cst[NX_CONVD + n] = sq[R + 1u]; cst[2u * NX_CONVD + n] = sq[R + 2u];
    } else if (seg == 1u) {
      for (uint t = 0u; t < R; t++) o1[t * NX_VALD + n] = rr(v[t]);
    } else if (seg == 2u) {
      for (uint t = 0u; t < R; t++) o2[t * 2u * NX_NVH + NX_NVH + n] = rr(sigm(rr(v[t])));
    } else {
      const device ushort *alog = (const device ushort *)e2, *dtb = (const device ushort *)e3;
      const float ex = exp(bf(alog[n])), dt = bf(dtb[n]);
      for (uint t = 0u; t < R; t++) {
        const float u = rr(rr(v[t]) + dt);
        const float sp = rr(u > 20.0f ? u : log(1.0f + exp(u)));
        o2[t * 2u * NX_NVH + n] = exp(-ex * sp);
      }
    }
  } else if (NQ_EPI == 2u) {  // out_proj + HC write: o0 hout [R][4][2560], o1 branch [R][2560]; e0 h, e1 inj [R][8]
    for (uint r = 0u; r < R; r++) {
      const float br = rr(v[r]);
      o1[r * NX_HID + n] = br;
      for (uint s = 0u; s < NX_HC; s++)
        o0[r * NX_HCD + s * NX_HID + n] = rr(e0[r * NX_HCD + s * NX_HID + n] + rr(br * e1[r * 2u * NX_HC + s]));
    }
  } else {  // attention projections: o0 q [R][12288], o1 kv [R][2][512] (k then v: seg 1, 2), o2 iqk [R][640];
    // e0 index-key cache f32 [tmax][128], e1 pos (u32, as float bits)
    const uint T = as_type<uint>(e1[0]);
    for (uint t = 0u; t < R; t++) {
      if (seg == 0u) o0[t * NX_QD + n] = v[t];
      else if (seg == 1u) o1[t * 2u * NX_KVD + n] = v[t];
      else if (seg == 2u) o1[t * 2u * NX_KVD + NX_KVD + n] = v[t];
      else {
        o2[t * NX_IQD + n] = v[t];
        if (n >= 512u) e0[(size_t)(T + t) * 128u + (n - 512u)] = v[t];
      }
    }
  }
}
