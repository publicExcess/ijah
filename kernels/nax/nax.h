// NAX (matmul2d) multi-row kernels for MTP verify widths (R = 1..8 rows). Owner: lab-mtp. Design: docs/mtp.md.
//
// Clean recording interface for lab-forward: every function records ONE dispatch (no barrier; the caller adds it)
// on the layer's own weight structs (models/flash_next/layer.h) and buffers with the layer's layouts, so a layer's
// multi-row path switches by replacing the corresponding record_stage case (see the request in docs/kernels.md).
#pragma once
#include "models/flash_next/layer.h"

namespace ne::nax {

constexpr int kMaxRows = 8;

// 8-bit affine linear by prefix ("<prefix>.weight/.scales/.biases"); misaligned tensors via Pack::aligned.
fn::Q8 load_q8(const Pack &p, const std::string &prefix);

// Generic: y_s = W_s x for up to 4 8-bit affine matrices with the same input width (group 64 or 128 each), R <= 16
// rows; x f32 [R][in],
// y_s f32 [R][rows_s] (bf16-rounded when the reference rounding is on). ks = k splits (0 = auto).
void record_q8(Recorder &r, int R, const Tensor &x, const std::vector<const fn::Q8 *> &w,
               const std::vector<Tensor> &y, int ks = 0);
// One matrix at 8, 4 or 2 bits (MLX affine layout: 32 / bits values per u32, little-endian; q.in = inputs).
void record_qn(Recorder &r, int R, const Tensor &x, const fn::Q8 &w, int bits, const Tensor &y, int ks = 0);
// y[r][n] = rr(rr((W x_r)[n]) + bias[r / 4][n]) for R <= 16 rows (the MTP fuse: x = 4 streams per token).
void record_q8_rowbias(Recorder &r, int R, const Tensor &x, const fn::Q8 &w, const Tensor &bias, const Tensor &y,
                       int ks = 0);
// GDN in_proj (qkv, z, b, a) + conv1d / silu / beta / g epilogue: drop-in for fn_gdn_inproj (same buffers).
void record_gdn_in(Recorder &r, int R, const fn::LayerW &w, fn::GdnState &st, fn::Scratch &s, int ks = 0,
                   const Tensor *hist = nullptr);  // hist: also write the rollback history (EPI 5)
// out_proj (GDN or attention o_proj) + HC write: drop-in for fn_gdn_outproj.
void record_out_hc(Recorder &r, int R, const fn::Q8 &ow, const Tensor &o, const Tensor &h_in, const Tensor &inj,
                   const Tensor &h_out, const Tensor &branch, int ks = 0);
// attention q / k / v / indexer projections (+ raw indexer key -> cache): drop-in for fn_attn_proj.
void record_attn_proj(Recorder &r, int R, const fn::LayerW &w, fn::GdnState &st, fn::Scratch &s, int ks = 0);

// The MTP token loop's PLE key / value projections at R rows (= s.rows on a T6 pack), split outputs: emb f32 [R][2560]
// -> kb f32 [R][HCD], vb f32 [R][HID], bf16 values. q8 pack: record_q8 (2 segments); T6 pack: the K8 t6 Linears at R
// rows (NAX path from 4 rows) rounded in place. No trailing barrier. mtp-loop-check (2) compares it with
// fn::record_ple_proj at 1 row (the decode's arithmetic) on either pack.
void record_ple_proj(Recorder &r, int R, fn::Scratch &s, const fn::PleW &e, const Tensor &emb, const Tensor &kb,
                     const Tensor &vb);

// ---- a whole layer step at R = 1..8 rows with the NAX trunk GEMMs (lab-forward's kernels for everything else)
// Scratch / state for any R (the 1, 2, 4, 6 and 8-row kernels are instantiated in kernels/fn_{trunk,moe,attn}.metal).
fn::Scratch alloc_scratch_rows(int rows);
fn::GdnState alloc_state_rows(const fn::LayerW &w, int rows, int tmax = 4096);
struct LayerExtra {  // fn_hc_pre at R = 8 runs as 2 x 4 rows (its staging needs 40 KB at 8): the second half's partials
  Tensor hc_part2, hc_ctr2;
  // MTP rollback points (GDN layers, rollback v1, docs/mtp.md), allocated when alloc_layer_extra(rows > 1):
  Tensor gdn_snap;  // f32 [R-1][48][128][128]: recurrent state after verify rows 0 .. R-2
  Tensor gdn_hist;  // f32 [3 + R][10240]: conv state in + the R raw qkv rows (conv state after row a = rows a+1..a+3)
  // MTP rollback v2 (lazy commit, docs/mtp.md 5.4), allocated by alloc_layer_extra(rows, true) for a GDN layer:
  // the previous round's rows (conv'd qkv, g / beta, raw qkv) stay pending until the next step commits keep[0] of
  // them; `keep` = the token loop's u32 GPU buffer (rows kept by the last accept; 0 at the start; = lab-attn's).
  Tensor gdn_pq, gdn_pg, gdn_praw, keep;
};
LayerExtra alloc_layer_extra(int rollback_rows = 0, bool lazy = false, const Tensor *keep = nullptr);
// After a verify step recorded with rollback buffers: restore the GDN layer's ssm / conv state to "after verify row
// keep[0] - 1" (keep = u32 GPU buffer: rows kept = accepted drafts + 1, 1 .. R; R = keep everything), the same
// buffer as lab-attn's attn::record_rollback, so the token loop stays on the GPU. One dispatch.
void record_gdn_rollback(Recorder &r, fn::GdnState &st, LayerExtra &ex, int rows, const Tensor &keep);
// fn_hc_pre (lab-forward's) for R rows: one dispatch at R <= 6, 2 x 4 rows at R = 8 (second half: ex's partials)
void record_hc_pre(Recorder &r, int R, const fn::HcW &hw, const Tensor &h, const Tensor &mix, const Tensor &inj,
                   fn::Scratch &s, LayerExtra &ex);
// nax_trunk: GDN in_proj / attention projections and out_proj + HC write on NAX (else lab-forward's SIMD kernels)
// kv_only (QSA attention layers only; lab-headlazy): stop after lab-attn's prep dispatch (q / k norm + rope, cache
// append, indexer keys, new block keys): the layer's KV cache rows are written exactly as by the full layer (the
// same dispatches up to there), nothing after (attention core, o_proj, MLP / MoE; h_out is not written)
void record_layer(Recorder &r, const fn::LayerW &w, fn::GdnState &st, fn::Scratch &s, LayerExtra &ex,
                  const Tensor &h_in, const Tensor &h_out, bool nax_trunk = true, bool kv_only = false);

}  // namespace ne::nax
