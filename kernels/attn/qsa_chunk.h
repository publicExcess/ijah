// QSA attention core of a prefill chunk (lab-attn): the cache append / indexer / selection of every row (lab-attn's
// step dispatches in 8-row slices, 16 slices concurrently) and then the attention of all rows at once:
//   dense rows (complete blocks <= 512, positions < ~2051; bf16 KV, T0 known): MLX's NAX flash attention
//     (kernels/prefill/pf_attn.metal, lab-prefill's) + the gate;
//   the other rows: qsa_chunk_attn_bf16 (kernels/attn/qsa_chunk.metal), one (row, kv group) per threadgroup on the
//     NAX, no split-K / merge. kv8 / f32 caches (or NE_QSA_CHUNK=0): the per-slice qsa_attn + qsa_merge as before.
// Design and numbers: docs/attn.md section 8. Inputs / outputs = record_step's, for n rows of a chunk.
#pragma once
#include "kernels/attn/qsa.h"

#include <functional>
#include <string>
#include <vector>

namespace ne::attn {

// The per-step scratch of alloc_cache alone (qr .. part, rows <= MAX_ROWS), for callers that run several steps
// concurrently on one cache (lab-prefill's request): `c.kc` .. `c.bkeys` stay empty, pos is not allocated.
QsaCache alloc_step_scratch(KvFmt fmt, int tmax);

constexpr int kChunkSlots = 16;  // concurrent 8-row slices per group

struct ChunkScratch {
  KvFmt fmt = KvFmt::BF16;
  int C = 0, tmax = 0;
  std::vector<QsaCache> slot;  // step scratch per concurrent slice
  Tensor pslot;                // u32 [kChunkSlots][4] slice positions
  Tensor qr;                   // f32 [C + 8][24][256] prepped queries of every chunk row
  Tensor sel, nsel;            // u32 [C + 8][512], [C + 8] block lists of every chunk row
  Tensor qb, ob;               // bf16 [C][24][256] dense rows: queries, NAX attention output
  // batched selection (the multi-row core): every row's indexer queries, scores, per-slice candidates; positions
  Tensor iq, scores, cand;     // f32 [C + 8][4][128], f32 [C + 8][tmax / 4], u32 [C + 8][tmax / 4 / 2048][512]
  Tensor spos;                 // u32 [C / 8 + 1][4]: slice j at posbase[0] + 8 j
};
ChunkScratch alloc_chunk_scratch(KvFmt fmt, int tmax, int C);

// The attention core of n <= C chunk rows at positions posbase[0] + 0 .. n - 1 (posbase: u32 GPU buffer, set by the
// caller before each replay). T0 = the chunk's first position if known when recording (enables the dense-row NAX
// path for rows with <= 512 complete blocks; the recording is then specific to T0), else -1 (every row through the
// sparse kernels: still exact). c = the layer's cache (its pos is not used or changed). q / kv / iqk / out as
// record_step, n rows. Leaves barriers around itself. core: -1 = the policy (chunk_core_on), 0 = per-slice step path,
// 1 = the multi-row core where the format has it (tests).
void record_chunk(Recorder &r, const QsaW &w, const QsaCache &c, ChunkScratch &s, int n, int T0,
                  const Tensor &posbase, const Tensor &q, const Tensor &kv, const Tensor &iqk, const Tensor &out,
                  int core = -1,
                  const std::function<Recorder &(const char *)> *phase = nullptr);
// lab-pfpath: only the cache side of record_chunk's batched path for n rows at posbase[0] .. + n - 1: one prep (q / k
// norm + rope, cache append, indexer keys) and the chunk's block keys; no selection, no attention core (the MTP
// head's KV-only rows of a prefill chunk). The multi-row format only (chunk_core_on(c.fmt, -1)).
void record_chunk_kv(Recorder &r, const QsaW &w, const QsaCache &c, ChunkScratch &s, int n, const Tensor &posbase,
                     const Tensor &q, const Tensor &kv, const Tensor &iqk);
// rows [0, dense_rows(T0, n)) of a chunk take the dense path (multiple of 8; 0 if T0 < 0 or not bf16)
int chunk_dense_rows(const ChunkScratch &s, int T0, int n);
// the multi-row kernel serves this format (bf16) and (force < 0) NE_QSA_CHUNK != 0 / (force >= 0) force != 0
bool chunk_core_on(KvFmt fmt, int force = -1);

}  // namespace ne::attn

extern "C" {
// Tests: n rows (one prefill chunk, n <= 2048) on the handle's cache at its current pos T0 through record_chunk;
// mode 0 = the per-slice step path (qsa_attn + qsa_merge per 8 rows: the reference), 1 = the multi-row core;
// dense = 1 lets rows with <= 512 complete blocks take the NAX dense path. Advances pos by n. gpu_ms may be NULL.
NE_API int ne_attn_chunk(ne_attn *a, int n, const float *q, const float *kv, const float *iqk, float *out, int mode,
                         int dense, double *gpu_ms);
// The selections of the last ne_attn_chunk call: sel u32 [n][512], nsel u32 [n] (either may be NULL).
NE_API int ne_attn_chunk_sel(ne_attn *a, int n, uint32_t *sel, uint32_t *nsel);
// GPU ms of the chunk core at the current pos (pos restored), median of reps replays; part: 0 = whole core,
// 1 = attention only (selection already in the scratch from the last ne_attn_chunk call at this pos).
NE_API int ne_attn_chunk_time(ne_attn *a, int n, int mode, int part, int reps, double *ms);
}
