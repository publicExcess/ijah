// Flash-Next QSA full attention (layers 3, 7, ..., 47) for R = 1..8 rows: KV cache (f32 | bf16 | kv8 g64) + the
// indexer's own caches, the recorded attention step, rollback. Owner: lab-attn. Details: docs/attn.md.
//
// The step consumes fn_attn_proj's outputs and produces what fn_gdn_outproj (o_proj + HC write) reads:
//   q   f32 [R][12288]  q_proj rows (per head [256 query | 256 gate])
//   kv  f32 [R][1024]   k_proj [512] | v_proj [512]
//   iqk f32 [R][640]    index_qk_proj (4 query heads + raw key, x 128)
//   out f32 [R][6144]   softmax(q k^T / 16) v over the QSA-selected keys, * sigmoid(gate)
// pos[0] = T = tokens in the cache before the step (GPU buffer, so recordings replay).
#pragma once
#include "core/engine.hpp"

namespace ne::attn {

enum class KvFmt : int { F32 = 0, BF16 = 1, KV8 = 2 };
const char *kv_fmt_name(KvFmt f);  // "f32" "bf16" "kv8"
KvFmt kv_fmt_from_env(KvFmt dflt = KvFmt::BF16);  // NE_KV=f32|bf16|kv8
size_t kv_bytes_per_token(KvFmt f);            // K + V, both kv heads: f32 4096, bf16 2048, kv8 1088 B

constexpr int NKV = 2, HD = 256, NQH = 24, QD = 12288, KVD = 512, IQD = 640, IHD = 128, RATIO = 4, KB = 512;
constexpr int MAX_ROWS = 8;
constexpr int SLICE = 2048;  // blocks per local-select slice
constexpr int NCH = KB / 8 + 1;  // attention chunks per (kv group, row): 64 block chunks + tail

struct QsaW { Tensor qnorm, knorm, iqnorm, iknorm; };  // bf16 [256], [256], [128], [128] (pack weights as stored)
// From an open pack: language_model.model.layers.<L>.self_attn.{q_norm,k_norm,indexer.q_layernorm,k_layernorm}.
QsaW load_qsa_weights(const Pack &p, int layer);

struct QsaCache {
  KvFmt fmt = KvFmt::BF16;
  int tmax = 0;  // capacity in tokens (multiple of 64)
  // kv: f32/bf16 [NKV][tmax][256]; kv8 codes u8 [NKV][tmax][256] + ksb/vsb ushort2 [NKV][tmax][4] (bf16 scale, bias)
  Tensor kc, vc, ksb, vsb;
  Tensor ikeys;  // bf16 [tmax][128] raw indexer keys
  Tensor bkeys;  // bf16 [tmax/4][128] pooled + normed + roped block keys (block b valid iff 4b + 3 < T)
  Tensor pos;    // u32 [4]: pos[0] = T
  Tensor invf;   // f32 [32] rope inverse frequencies 1e7^(-2d/64) + i32 [4] {mlen, delta, 0, 0} + i32 [mlen][4]
                 // multimodal (t, h, w) positions of rows < mlen (qsa.metal mrope_pos; lab-vision). A model shares
                 // one such buffer across its attention layers (fn::Model::mrope).
  // per-step scratch (rows <= MAX_ROWS)
  Tensor qr;      // f32 [R][24][256] normed + roped q * 1/16
  Tensor iq;      // f32 [R][4][128] indexer queries (bf16 values)
  Tensor scores;  // f32 [R][tmax/4]
  Tensor sel;     // u32 [R][512] block list of each row (ascending)
  Tensor nsel;    // u32 [R]
  Tensor cand;    // u32 [R][tmax/4/SLICE][512] per-slice top 512 (qsa_select_local)
  Tensor part;    // f32 [NKV][R][NCH][12][258]
  int scratch_tmax = 0;  // tmax the scratch was sized for
  int T() const { return (int)((const uint32_t *)pos.host())[0]; }
};
// scratch_from: share the per-step scratch (qr .. cand, ~7 MB + 36 B per token) of another cache with tmax >= this
// one (the 12 attention layers of a forward run one after another).
QsaCache alloc_cache(KvFmt fmt, int tmax, const QsaCache *scratch_from = nullptr);
size_t cache_bytes(const QsaCache &c);  // persistent bytes (kv + indexer), excl. scratch

// Records the attention step for R rows (6 dispatches + barriers): prep (q/k norm + rope, cache append, indexer
// queries, new block keys), score, select_local (top 512 per slice of 2048 blocks), select, attention (split-K),
// merge. advance: append pos[0] += R.
constexpr int kNumDispatches = 6;
void record_step(Recorder &r, const QsaW &w, QsaCache &c, int R, const Tensor &q, const Tensor &kv, const Tensor &iqk,
                 const Tensor &out, bool advance = false);
// The separate dispatches (prep/score/select_local/select/attn/merge) for timing: which = 0..5.
void record_one(Recorder &r, int which, const QsaW &w, QsaCache &c, int R, const Tensor &q, const Tensor &kv,
                const Tensor &iqk, const Tensor &out);
// MTP: after a verify step that advanced pos by `rows`, keep `accepted[0]` of them (u32 GPU buffer).
void record_rollback(Recorder &r, QsaCache &c, const Tensor &accepted, int rows);
void record_advance(Recorder &r, QsaCache &c, int n);

// Host-synchronous helpers (no GPU work in flight on c).
// Cache rows [0, T) from f32 k, v [NKV][T][256] (roped keys) and raw indexer keys [T][128]; rebuilds every block key
// on the GPU and sets pos = T. Any of k / v / ikeys may be null (left as is).
void import_cache(QsaCache &c, const QsaW &w, const float *k, const float *v, const float *ikeys, int T);
// Dequantized rows [0, n): k, v f32 [NKV][n][256], ikeys f32 [n][128], bkeys f32 [n/4][128] (each may be null).
void export_cache(QsaCache &c, int n, float *k, float *v, float *ikeys, float *bkeys);
// Rollback for rejected MTP drafts: T' <= T. Cache rows >= T' and block keys of blocks with 4b + 3 >= T' become
// stale and are rewritten by the steps that re-append them (nothing else to undo: kv8 groups are per row).
void truncate(QsaCache &c, int T);

}  // namespace ne::attn

// C API (ctypes tests: tests/attn/test_qsa.py). A handle = weights of one layer + a cache + recorded steps.
extern "C" {
typedef struct ne_attn ne_attn;
NE_API ne_attn *ne_attn_create(ne_pack *pack, int layer, int fmt, int tmax);
NE_API void ne_attn_free(ne_attn *a);
NE_API int ne_attn_import(ne_attn *a, const float *k, const float *v, const float *ikeys, int T);
NE_API int ne_attn_export(ne_attn *a, int n, float *k, float *v, float *ikeys, float *bkeys);
NE_API int ne_attn_truncate(ne_attn *a, int T);
NE_API int ne_attn_pos(ne_attn *a);
// One step of R rows: q [R][12288], kv [R][1024], iqk [R][640] -> out [R][6144]; advances pos by R.
NE_API int ne_attn_step(ne_attn *a, int R, const float *q, const float *kv, const float *iqk, float *out,
                        double *gpu_ms);
// Scratch of the last step: "qr" [R*6144] "iq" [R*512] "scores" [R*tmax/4] "sel" [R*512] (u32 as float bits)
// "nsel" [R] (u32). Copies n values; returns the element count, -1 if unknown.
NE_API int ne_attn_read(ne_attn *a, const char *name, void *dst, int n);
// Overwrite "sel" / "nsel" (u32) before timing one dispatch alone (experiments).
NE_API int ne_attn_write(ne_attn *a, const char *name, const void *src, int n);
// GPU us per step at the current pos (pos is restored after every replay), differential reps1 vs reps2 steps,
// flush_mb streamed between steps (its cost subtracted) when > 0. which = -1: whole step, 0..5: one dispatch.
NE_API int ne_attn_time(ne_attn *a, int R, int which, int reps1, int reps2, int flush_mb, double *us);
}
