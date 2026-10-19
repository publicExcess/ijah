// Flash-Next decoder layer graph (1 row decode step), recorded through the core Recorder. Owner: lab-layer.
//
// C entry points (ctypes): see the bottom of this file. Layout conventions:
//   hidden state h: f32 [hc_count=4][2560] (the 4 hyper-connection streams, row-major)
//   GDN conv state: f32 [3][10240] (the previous 3 in_proj_qkv outputs, oldest first = MLX conv_state[0, t, :])
//   GDN recurrent state: f32 [48][128 Dv][128 Dk] (MLX state [B, Hv, Dv, Dk])
#pragma once
#include "core/engine.hpp"
#include "kernels/attn/qsa.h"
#include "kernels/trunk6/trunk6.h"

#include <atomic>
#include <memory>

namespace ne::fn {

constexpr int HID = 2560, HC = 4, HCD = HC * HID, LR = 320;
constexpr int NKH = 16, NVH = 48, DK = 128, DV = 128, KEYD = NKH * DK, VALD = NVH * DV, CONVD = 2 * KEYD + VALD;
constexpr int NE = 512, TOPK = 10, INTER = 640;

struct Q8 {  // MLX 8-bit affine: weight U32 [rows][in/4] (bytes row-major), scales/biases BF16 [rows][in/G]
  Tensor w, s, b;
  int rows = 0, in = 0, group = 0;
  // lab-t6wire: set when the pack holds this Linear as EXL3 (K6/K7/K8) or dense bf16 (t6::detect; w/s/b empty,
  // group 0): the layer graph then records it with t6:: kernels + an epilogue kernel (layer_t6.mm, docs/trunk6.md 8)
  std::shared_ptr<const t6::Linear> t6;
};
struct Exl3 {  // trellis U16 [E][in/16][out/16][16K], suh F16 [E][in], svh F16 [E][out]
  Tensor t, suh, svh;
  int k = 0;
};
struct HcW { Tensor norm, inject; Q8 down, up; };
struct GdnW { Q8 qkv, z, a, b, out; Tensor conv, alog, dtb, norm; };
// full attention (QSA layers): q_proj = 24 x [256 query | 256 gate], k/v 2 x 256, indexer qk 5 x 128, o_proj
struct AttnW { Q8 q, k, v, iqk, o; Tensor qnorm, knorm, iqnorm, iknorm; };
constexpr int NQH = 24, NKV = 2, AHD = 256, AQD = NQH * 2 * AHD, AKVD = NKV * AHD, AIQD = 5 * 128, AOD = NQH * AHD;
struct MoeW { Tensor router, sgate; Q8 sg, su, sd; Exl3 gate, up, down; };
// PLE (layer 1 only): n-gram key/value projections (8-bit), grouped norms (ones-centred, multiply as stored), the
// dilated (3) depthwise conv K=4 weight [10240][4], and the n-gram id constants (CPU side: the ids pick rows of the
// 95 GB ngram_table.bin, gathered on the host).
constexpr int PLE_NG = 16, PLE_W = 160, PLE_ST = 9, PLE_KV = HCD + HID;
struct PleW {
  bool on = false;
  Q8 key, value;
  Tensor nk, nq, nc, conv;
  int64_t mult[3] = {0, 0, 0}, sizes[PLE_NG] = {}, offsets[PLE_NG] = {};
};
struct LayerW {
  int layer = 0;
  bool linear = true;
  HcW hca, hcm;
  GdnW gdn;    // linear layers
  AttnW attn;  // full-attention layers
  MoeW moe;
  PleW ple;    // layer 1
  attn::QsaW qsa;  // attention layers when g_attn_qsa (lab-attn's QSA step replaces fn_attn_core + merge)
  bool t6 = false;  // lab-t6wire: the layer's trunk Linears are EXL3 (T6 pack): record_stage -> t6_record_stage
};
LayerW load_layer(const Pack &pack, int layer);  // throws ne::Error on a missing / unexpected tensor
// Any decoder-layer prefix, e.g. "language_model.mtp.layers.0." (the MTP layer: attention + MoE); layer = the id
// stored in LayerW::layer (informational).
LayerW load_layer_prefix(const Pack &pack, const std::string &prefix, int layer = -1);
Q8 load_q8(const Pack &pack, const std::string &prefix);       // prefix.weight/.scales/.biases (8-bit, g64/g128)
Tensor load_tensor(const Pack &pack, const std::string &name);  // 16-byte aligned view (or an aligned copy)
// pipeline with the reference-rounding constant (g_ref_round); checks maxTotalThreadsPerThreadgroup >= min_threads
id<MTLComputePipelineState> pso(const std::string &name, int min_threads = 256);
extern size_t g_aligned_copy_bytes;  // bytes of misaligned pack tensors copied into aligned buffers so far

// Per-layer recurrent state. GDN layers: conv f32 [3][10240], ssm f32 [48][128][128]. Attention layers: KV cache
// kc / vc f32 [2][tmax][256] (keys roped), the QSA indexer's raw keys ikeys f32 [tmax][128], pos u32 [4]
// (pos[0] = T = tokens already in the cache; ne_fn_layer_step advances it by the rows of the step).
struct GdnState {
  Tensor conv, ssm;
  // lab-gdnlazy (lazy state store, docs/kernels.md 5d): GDN layers' pending rank-1 rows not yet applied to ssm:
  // lz u32 [48] = pending rows per value head, pend f32 [48][3][260] = rows (normalised k, v, g, beta). ssm alone
  // is the state only when no head has pending rows: readers call gdn_materialise first, writers gdn_clear_pending.
  Tensor lz, pend;
  Tensor kc, vc, ikeys, pos;
  Tensor apart;  // fn_attn_core chunk partials [2][tmax/32][R<=4][12][258]
  Tensor ple_conv;  // PLE layers: conv state f32 [9][10240] (bf16 values, oldest first)
  attn::QsaCache qc;  // attention layers with g_attn_qsa: lab-attn's caches (qc.pos = pos); kc/vc/apart unused
  int tmax = 0;
};
GdnState alloc_gdn_state();
// Lazy GDN state store (lab-gdnlazy): the 1-row decode step stores the 3 MB ssm only every K-th step (K =
// g_gdn_lazy, env NE_GDN_LAZY=2..4, default 2; 0 = fn_gdn_step, store every step) and keeps the deferred rows in
// lz / pend; the step's outputs and the materialised state are bit-identical to storing every step. Read when a
// layer is recorded.
extern int g_gdn_lazy;
bool gdn_pending(const GdnState &st);              // host: any pending rows (GPU idle)
void gdn_clear_pending(GdnState &st);              // host: drop pending rows (after ssm was overwritten)
void record_gdn_materialise(Recorder &r, GdnState &st);  // one dispatch, no barrier (layers are independent)
// Host: apply every pending row of these states to their ssm (one command buffer, waits); no-op if none pending.
// Must run (with no GPU work in flight) before anything reads ssm other than the decode step itself.
void gdn_materialise(std::vector<GdnState> &sts);
void gdn_materialise(GdnState &st);
GdnState alloc_attn_state(int tmax, int rows = 4);  // rows: the most rows per step (dense path partials)
// share (QSA path): another attention layer's state of the same model whose QSA step scratch and f32 indexer-key
// sink are reused (the layers run one after another), lab-attn's request.
GdnState alloc_state(const LayerW &w, int tmax = 4096, int rows = 4, const GdnState *share = nullptr);

// Per-step scratch (shared by all layers of a forward; every buffer is fully rewritten each step, the counters are
// reset by the kernels that use them).
struct Scratch {
  int rows = 1;  // R: rows per step (1 decode, 4 verify); every activation is [R][...]
  Tensor mix_a, inj_a, xa, qkv, z, gb, o, br_a, h2;
  Tensor mix_m, inj_m, xm, logits, gush, route, ctr1, guo, sho, scr3, ctr3, br_m;
  Tensor hc_part, hc_ctr;  // fn_hc_pre partials [41][4][8][R] + per-block counters
  Tensor aq, akv, aiqk;
  Tensor ple_emb, ple_kv, hp;  // PLE: gathered n-gram rows f32 [R][2560] (host-written), key|value proj [R][12800],
                               // the layer input after PLE [R][4][2560]
  std::shared_ptr<struct T6Buf> t6;  // lab-t6wire: EXL3 trunk raw outputs + t6 scratch slots (made on first use)
  Tensor uq, guo_u;        // R > 1: unique experts [1 + 40 * (2 + R)], gate/up partials [R*TOPK][R][2][INTER]    // attention layers: q_proj [R][12288], k|v [R][2*512], indexer qk [R][640]
};
Scratch alloc_scratch(int rows = 1);  // any rows 1..16 (kernels: 1, 2, 4 here; 6, 8 via kernels/nax/rows68_*)

// The layer step, one stage per dispatch (each can be recorded alone for per-sub-block checks and timing):
//   AttnHCPre: h_in -> attn_mix, attn_inj (4 gates + 4 inv-rms); AttnHCMix: h_in, attn_mix, attn_inj -> attn_in;
//   GdnIn: attn_in + conv state -> qkv, z, gb; GdnStep: qkv, gb, z + ssm state -> core; GdnOut: core, h_in,
//   attn_inj -> attn_out, h_mid; MlpHCPre / MlpHCMix: h_mid -> mlp_mix, mlp_inj, mlp_in; MoeRouter: mlp_in -> logits,
//   route, shared_gu; MoeGateUp: mlp_in, route, shared_gu -> experts_gu, shared_down; MoeDown: experts_gu, route,
//   logits, shared_down, h_mid, mlp_inj -> mlp_out, h_out.
enum class Stage { AttnHCPre = 0, AttnHCMix, GdnIn, GdnStep, GdnOut, MlpHCPre, MlpHCMix, MoeRouter, MoeGateUp, MoeDown, N };
extern const char *const kStageNames[(int)Stage::N];
constexpr int kNumBlocks = 6;  // sub-blocks of ne_fn_layer_run_stage: AttnHC GdnIn GdnStep GdnOut MlpHC Moe
// Reference rounding points (kernels/ne_common.h NE_REFR): false (default) = plain f32; true = round to bf16 where
// the MLX reference does (inject / down logits, GDN a/b gate chain), only when asked for: env NE_FN_ROUND=ref (at
// library load) or ne_fn_set_ref_rounding(1) (the golden / KLD-vs-MLX checks). Read when a layer / model records;
// a Model keeps the mode it was loaded with (Model::ref_round) and the setter refuses to change it while a Model lives.
extern bool g_ref_round;
extern std::atomic<int> g_models_live;  // live Models (models/flash_next/model.h): ne_fn_set_ref_rounding's guard
// Attention layers: true (default) = lab-attn's QSA step (kernels/attn/qsa.h: KV format from NE_KV, default bf16;
// sparse past 512 complete blocks, as the reference); false (env NE_FN_ATTN=dense) = lab-layer's dense path
// (fn_attn_core + fn_attn_merge, f32 KV). Read when a layer is loaded / its state allocated.
extern bool g_attn_qsa;
// R >= 4 steps: GDN in_proj / attention projections / out_proj + HC write on lab-mtp's NAX GEMMs (kernels/nax/nax.h,
// default; env NE_FN_NAX=0 = the SIMD q8 kernels). Read when a layer is recorded.
extern bool g_nax_rows;
// R > 1: routed experts deduplicated across rows at 1 < R <= g_moe_dedupe (env NE_FN_MOE_DEDUPE: 0 off, 1 every R,
// N = up to N rows; default 3 since lab-verify: on real text 3 consecutive tokens use 21.7 distinct experts of 30 per
// layer (layer 0 alone shares least: the old "39 of 40" on L00_verify4), but the R-row kernels lose from R = 4 on).
extern int g_moe_dedupe;
// lab-verify (opt-in, NE_FN_MOE_SPLIT=1; T6 layers, R 2-5): routed gate / up split by multiplicity: experts used by one
// row through the (row, slot) kernel, the others once for all their rows (dedupe kernel), down per (row, slot)
extern bool g_moe_split;
bool moe_split(int R, const LayerW &w);
// HC read v2 (lab-speed's fn_hc_pre3 + fn_hc_mix2, kernels/fn_rec.h: no atomics, -0.30 ms per decode step; default;
// env NE_FN_HC=v1 = fn_hc_pre + fn_hc_mix). hc_v2(R): used at R rows (not at R = 8, where nax::record_layer still
// splits fn_hc_pre into 2 x 4 rows; the mix must match the pre). Read when a layer is recorded.
extern bool g_hc_v2;
bool hc_v2(int R);
// lab-fuse: experiment switch for A/B runs (env NE_FUSE=mask, default 0): guard a candidate change in record_stage
// with `(g_fuse & bit)` and time it against the old recording in one process with `NE_PROF_AB=1 NE_FUSE=bit
// build/fn-prof` (docs/kernels.md 5c). No bit is in use. Read when a layer is recorded.
extern int g_fuse;
void record_stage(Recorder &r, Stage stage, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in,
                  const Tensor &h_out);
const Tensor *scratch_tensor(Scratch &s, const std::string &name);  // by the names of ne_fn_layer_read; or nullptr

// For attention layers the stages GdnIn / GdnStep / GdnOut are attn_proj / attn_core / out_proj (+ HC write).
// Records one layer step (GDN or full attention) for s.rows rows (consecutive tokens): h_in -> h_out (both f32
// [R][4][2560]; may not alias). 10 dispatches (11 attention; + 2 PLE on layer 1: h_in + PLE -> s.hp, then the layer
// runs on s.hp; s.ple_emb must hold the step's gathered n-gram rows).
void record_gdn_layer(Recorder &r, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in, const Tensor &h_out);

// ---- lab-t6wire: EXL3 / bf16 trunk Linears in the layer graph (models/flash_next/layer_t6.mm, docs/trunk6.md 8).
// Raw [R][out] outputs of the t6 Linears and 4 scratch slots (Linears of one stage run concurrently).
struct T6Buf {
  int rows = 0;
  t6::Scratch slot[4];
  Tensor xn, mraw, ip, uraw;      // HC read: normed input [R][10240], down raw [R][320], inject partials [R][4][10][8], up raw
  Tensor braw, araw, kraw, vraw;  // GDN b / a [R][48]; attention k / v [R][512]
  Tensor graw, uraw2, act;        // shared expert gate / up raw [R][640], SwiGLU [R][640]
  Tensor pk, pv;                  // PLE key / value raw [R][10240] / [R][2560]
};
T6Buf &t6_buf(Scratch &s);
// Records one stage of an EXL3 layer (w.t6) and returns true; false = not a t6 stage (GdnStep, MoeDown: the q8
// graph's kernels, unchanged). GdnIn of a GDN layer: hist = nax's rollback history (EPI 5), or with lazy = the lazy
// commit (EPI 6: hist = pending rows [8][10240], keep = rows kept).
struct T6GdnOpt { const Tensor *hist = nullptr; const Tensor *keep = nullptr; bool lazy = false; };
bool t6_record_stage(Recorder &r, Stage stage, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in,
                     const Tensor &h_out, const T6GdnOpt &opt = {});
// HC read with t6 down / up Linears (layer HC: inject = true; the final mixer: bf16, no inject): h [R][4][2560] ->
// mix [R][320], inj [R][8] (4 gates, 4 x 1/rms), out [R][2560]. 5 dispatches, each followed by a barrier.
void t6_hc_read(Recorder &r, const HcW &hw, int R, const Tensor &h, bool inject, Scratch &s, const Tensor &mix,
                const Tensor &inj, const Tensor &out);
// Linears with t6 weights recorded concurrently (slots of s), then a barrier; round = rr the outputs (q8 EPI 0).
// ep: fused finisher epilogues per Linear (t6::Epi; EXL3 only). g_t6_fuse: the layer stages use them (default; env
// NE_T6_FUSE=0 = the separate t6e_* epilogue kernels).
extern bool g_t6_fuse;
extern bool g_t6_fuse2;  // lab-t6fuse folded HC read / shared SwiGLU (env NE_T6_FUSE2=0 = off)
extern int g_t6_x;       // lab-t6fuse experiment bits (env NE_T6_X; fn-prof NE_PROF_ABENV A/B)
void t6_linears(Recorder &r, int R, Scratch &s, const std::vector<const Q8 *> &w, const std::vector<Tensor> &x,
                const std::vector<Tensor> &y, const std::vector<t6::Epi> *ep = nullptr);
void t6_round(Recorder &r, int R, const Tensor &src, int n, int sld, int soff, const Tensor &dst, int dld, int doff,
              bool round);
// The step path's PLE key / value projections at s.rows rows: s.ple_emb -> s.ple_kv [R][12800] (key | value, bf16
// values; q8 pack: fn_ple_proj_g64; T6 pack: the K8 t6 Linears), then a barrier. record_ple = this + fn_ple_mix.
void record_ple_proj(Recorder &r, const PleW &e, Scratch &s);
void record_ple(Recorder &r, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in, const Tensor &h_out);
// The 16 n-gram row ids of token `t` given the two previous tokens (p1 = previous, p2 = the one before; EOS when
// none / across an EOS: once going back hits an EOS every earlier position reads EOS), mlx-vlm's math exactly
// (int64 wrapping products, XOR, floor mod) -- and the gathered row (f32 [2560], bf16 values) from the table.
void ngram_ids(const PleW &p, int64_t p2, int64_t p1, int64_t t, int64_t eos, int64_t ids[PLE_NG]);
void ngram_gather(const Pack &pack, const int64_t ids[PLE_NG], float *emb);

// GPU time per layer step (us), differential over reps1 vs reps2 steps in one command buffer; with flush_mb > 0 a
// streaming read of flush_mb MB between steps evicts the SLC (its own differential cost is subtracted, us_flush).
void time_layer(const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in, const Tensor &h_out, Backend be,
                int reps1, int reps2, int flush_mb, double *us_per_step, double *us_flush);

}  // namespace ne::fn

extern "C" {
typedef struct ne_fn_layer ne_fn_layer;
// Loads layer `layer` of an open pack (GDN layers only for now), allocates its state (zeros) and scratch, and
// records the step for 1 row. NULL on error (ne_last_error()).
NE_API ne_fn_layer *ne_fn_layer_create(ne_pack *pack, int layer);
// Same for `rows` consecutive tokens per step (1, 2 or 4; 4 = the MTP-verify shape). Every activation passed in or
// read out is then [rows][...] (h: [rows][4*2560]); the states are per layer as before.
NE_API ne_fn_layer *ne_fn_layer_create_rows(ne_pack *pack, int layer, int rows);
NE_API void ne_fn_layer_free(ne_fn_layer *l);
// Attention layers: KV cache in / out for T cached tokens: k, v f32 [2][T][256], ikeys f32 [T][128] (raw indexer
// keys); set also sets pos = T; get copies T = the current pos (n_tokens may be NULL).
NE_API int ne_fn_layer_set_kv(ne_fn_layer *l, const float *k, const float *v, const float *ikeys, int T);
NE_API int ne_fn_layer_get_kv(ne_fn_layer *l, float *k, float *v, float *ikeys, int *n_tokens);
// State in / out: conv f32 [3*10240], ssm f32 [48*128*128] (either pointer may be NULL to skip).
NE_API int ne_fn_layer_set_state(ne_fn_layer *l, const float *conv, const float *ssm);
NE_API int ne_fn_layer_get_state(ne_fn_layer *l, float *conv, float *ssm);
// One decode step: h_in f32 [4*2560] -> h_out f32 [4*2560]; updates the state. gpu_ms may be NULL. h_in NULL:
// keep the current input buffer (e.g. after ne_fn_layer_write("h_in")); h_out NULL: don't copy out.
NE_API int ne_fn_layer_step(ne_fn_layer *l, const float *h_in, float *h_out, double *gpu_ms);
// Run one sub-block alone on the current buffers (set its inputs first with ne_fn_layer_write):
//   0 AttnHC (h_in -> attn_mix, attn_inj, attn_in)   1 GdnIn (attn_in, conv_state -> qkv, z, gb, conv_state)
//   2 GdnStep (qkv, gb, z, ssm_state -> core)        3 GdnOut (core, h_in, attn_inj -> attn_out, h_mid)
//   4 MlpHC (h_mid -> mlp_mix, mlp_inj, mlp_in)      5 Moe (mlp_in, h_mid, mlp_inj -> logits ... h_out)
NE_API int ne_fn_layer_run_stage(ne_fn_layer *l, int block);
// Run one dispatch alone (ne::fn::Stage: 0 attn_hc_pre, 1 attn_hc_mix, 2 gdn_in, 3 gdn_step, 4 gdn_out,
// 5 mlp_hc_pre, 6 mlp_hc_mix, 7 moe_router, 8 moe_gateup, 9 moe_down).
NE_API int ne_fn_layer_run_dispatch(ne_fn_layer *l, int stage);
// Reference rounding points on (1) / off (0, default) for layers / models created afterwards (see g_ref_round);
// refused (error) as a change while any model lives -- free the models first.
NE_API int ne_fn_set_ref_rounding(int on);
// Overwrite an intermediate (names as ne_fn_layer_read, plus "h_in", "h_out", "conv_state", "ssm_state").
NE_API int ne_fn_layer_write(ne_fn_layer *l, const char *name, const float *src, int n);
// Intermediates of the last step by name (f32 unless noted): "attn_mix"[320] "attn_inj"[8: 4 gates, 4 inv-rms]
// "attn_in"[2560] "qkv"[10240, conv+silu] "z"[6144] "gb"[96: g[48], beta[48]] "core"[6144, gated-norm output]
// "attn_out"[2560] "h_mid"[10240] "mlp_mix"[320] "mlp_inj"[8] "mlp_in"[2560] "logits"[513]
// "route"[20: 10 u32 ids, 10 f32 scores] "shared_gu"[1280] "experts_gu"[10*2*640] "shared_down"[2560]
// "mlp_out"[2560]. Copies n floats (n <= size); returns the size, -1 if unknown.
NE_API int ne_fn_layer_read(ne_fn_layer *l, const char *name, float *dst, int n);
// GPU time per layer step (us), differential over reps1 vs reps2 steps in one command buffer, with an SLC flush
// (flush_mb streamed between steps, its own cost subtracted) when flush_mb > 0. Leaves the state advanced.
NE_API int ne_fn_layer_time(ne_fn_layer *l, int reps1, int reps2, int flush_mb, double *us_per_step,
                            double *us_flush);
}
