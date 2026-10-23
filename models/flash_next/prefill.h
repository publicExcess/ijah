// Flash-Next chunked prefill (lab-prefill): a prompt chunk of n <= C tokens goes through the whole model as matrix
// work -- NAX GEMMs for the trunk (kernels/prefill/pf_gemm.metal), the sorted-window EXL3 MoE (pf_moe.metal), the
// GDN recurrence as one scan per chunk and lab-attn's QSA step in 8-row slices for the attention core (pf_trunk.metal,
// kernels/attn/qsa.h) -- and leaves every cache / state of the Model (GDN conv + ssm, PLE conv, QSA KV + indexer
// caches, T, the n-gram history) as n decode steps would. No head: the last prompt token goes through the decode
// step (ne::fn::step(m, token, true)) for its logits. Design, numbers and traps: docs/prefill.md.
#pragma once
#include "models/flash_next/model.h"

#include <map>
#include <string>
#include <vector>

namespace ne::fn {

// struct Prefill / PrefillDeleter: declared in model.h (the Model owns one)
using PrefillPtr = std::unique_ptr<Prefill, PrefillDeleter>;
// Buffers for chunks of up to `chunk` tokens (multiple of 8; ~0.6 MB per token of chunk). profile = record every
// (layer, stage) as its own command buffer (GPU time per stage; slower end to end).
// donor: another model's Prefill whose chunk buffers this one uses instead of its own when the geometry matches (same
// chunk, attention window, KV format, EXL3 trunk sizes); its own recordings. Only for models whose prefills never run
// at the same time (one engine thread). lab-srvperf.
PrefillPtr prefill_create(Model &m, int chunk, bool profile = false, const Prefill *donor = nullptr);
// tokens[0, n) at positions m.T .. m.T + n - 1 (any n >= 1; chunked): every state advanced, m.T += n, history set.
// ahead (lab-pfpath; may be NULL): the tokens the NEXT call will start with (its first chunk). Their n-gram rows (the
// PLE gather: ~60 ms per 1K tokens of new text, SSD-bound) are read while this call's last chunk runs on the GPU and
// kept for that call, which uses them when its history and first tokens match (else it gathers as before): the same
// rows either way, so results are unchanged.
void prefill_tokens(Prefill &p, const int *tokens, int n, const int *ahead = nullptr, int n_ahead = 0);
// prompt[0, n - 1) chunked, then the last token through the decode step with the head. Returns its sample.
StepOut prefill_prompt(Prefill &p, const int *prompt, int n);

// lab-pfpath: the MTP head's KV-only rows for the positions of the last prefill_tokens chunk, as matrix work (the
// lazy head's catch-up runs them as 8-row decode-kernel steps: ~29 us per row, ~120 ms per 4096-row chunk). Row r
// (r < n <= that chunk's length) pairs the chunk's layer-47 pre-mixer streams of row r (prefill_hidden) with
// next[r] (the token after it) and is written to the head's QSA cache at position pos + r: fuse_pre ->
// fc_embedding / fc_hidden GEMMs (+ the embedding row as the bias, bf16) -> the head layer's HC read -> k / v /
// indexer GEMMs -> lab-attn's chunk prep (attn::record_chunk_kv). Rounding: the prefill trunk's (bf16 activations
// into the GEMMs), not the 8-row steps'. Commits and waits; uses the chunk buffers (call between chunks).
struct HeadKvW {
  const Q8 *fc_e = nullptr, *fc_h = nullptr, *embed = nullptr;
  Tensor norm_e, norm_h;
  const LayerW *layer = nullptr;
  uint32_t vocab = 0;
};
void prefill_head_kv(Prefill &p, const HeadKvW &w, GdnState &head_st, const int *next, int n, int pos);
// whether prefill_head_kv serves this head cache (the batched attention prep: bf16 KV, chunk scratch geometry)
bool prefill_head_kv_ok(const Prefill &p, const GdnState &head_st);

// lab-pfpath: Recorder::touch every recording (the engine's idle keep-warm, ne_fn_gpu_ping); -> recorders touched
int prefill_touch(Prefill &p);

// Stage groups of the profile (GPU ms summed over the last prefill_tokens call)
enum class PfStage { Embed = 0, Ple, HcGlue, TrunkGemm, Gdn, Attn, Moe, MoeGemm, N };  // Moe = router, shared expert, sort, glue
extern const char *const kPfStageNames[(int)PfStage::N];
struct PfStats {
  double wall_ms = 0, gpu_ms = 0, gather_ms = 0;  // last prefill_tokens call
  double rec_ms = 0;                              // of which: recording new chunk lengths (lab-ttft)
  double ahead_ms = 0;                            // gathering the next call's first chunk (in the GPU's shadow)
  int ahead_hit = 0;                              // first-chunk rows the previous call had gathered (lab-pfpath)
  double stage_ms[(int)PfStage::N] = {};          // profile mode only
  int chunks = 0, dispatches = 0;
  // NE_PF_PROF=fine (profile mode, pfkern 18 Oct): GPU ms per kernel label (every label its own command buffer;
  // T6 Linears split into prep / dense / gemm / post) and per-chunk GPU / attention ms
  std::map<std::string, double> fine_ms;
  std::vector<double> chunk_gpu_ms, chunk_attn_ms;
};
const PfStats &prefill_stats(const Prefill &p);
// tests: the chunk's hidden streams after `layers` layers (f32 [C][4][2560], rows of the last chunk)
const Tensor &prefill_hidden(const Prefill &p, int layers);
// lab-dflash (docs/mtp.md 13): the Prefill's DFlash2 feature ring (f32 [2048][12800], row = position % 2048), made on the
// first call (which clears the chunk recordings once: they bind it from then on). Chunks write the features of their
// last <= 2048 rows into it only while prefill_dfl_capture(p, true) (lab-dflfix: a Prefill shared by several models --
// lab-srvperf -- must not let a model without the drafter overwrite rows; the drafter copies its rows out after its
// call).
const Tensor &prefill_dfl_ring(Prefill &p);
void prefill_dfl_capture(Prefill &p, bool on);

}  // namespace ne::fn

extern "C" {
typedef struct ne_fn_model ne_fn_model;
typedef struct ne_fn_prefill ne_fn_prefill;
NE_API ne_fn_prefill *ne_fn_prefill_create(ne_fn_model *m, int chunk, int profile);
NE_API void ne_fn_prefill_free(ne_fn_prefill *p);
// tokens at the model's current position (no head); stats (may be NULL, 4 + 8 doubles): wall ms, GPU ms, gather ms,
// chunks, then the 8 stage GPU ms (profile mode).
NE_API int ne_fn_prefill_tokens(ne_fn_prefill *p, const int *tokens, int n, double *stats);
// prompt: chunked prefill of the first n - 1 tokens + the decode step (head) on the last; logits f32 [248320] / next
// (may be NULL).
NE_API int ne_fn_prefill_prompt(ne_fn_prefill *p, const int *prompt, int n, float *logits, int *next, double *stats);
// lab-srvperf: dst's chunked prefill (the one prefill() / the MTP feed use) borrows src's chunk buffers (~0.6 MB per
// chunk row + the attention window's scratch: ~3.5 GB at 4096 rows and a 262K window) -- one set for several models /
// slots of a process whose prefills never run at the same time (the server's engine thread). Creates src's Prefill if
// needed and replaces dst's (its recordings). 0 = shared, 1 = geometry differs (dst keeps its own buffers), -1 = error.
NE_API int ne_fn_model_share_prefill(ne_fn_model *dst, ne_fn_model *src);
}
