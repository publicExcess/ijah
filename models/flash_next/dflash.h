// DFlash2 block drafter for Flash-Next on the engine (lab-dflash). Kernels: dflash.metal; design, numbers, traps:
// docs/mtp.md section 13. Drafter: incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) after z-lab/dflash (DFlash, ICML
// 2026) and Splash's DFlash2 port (splash-m5 runtime/model/DFlashDraft.cpp).
//
// One draft = one recording (record_block): the context rows the last verify committed (its target features ->
// fc -> hidden_norm -> c -> per layer k / v -> the bf16 ring), then the block [anchor, mask x (R - 1)] at positions
// T .. T + R - 1 through 5 layers (two-tap dynamic convolutions around attention and MLP; attention over the ring +
// the block, non-causal, window 2048), the final norm, the readout (the MTP loop's pruned 4-bit draft head + exact
// 8-bit re-score of the top 16 per row) and the candidate selector -> R - 1 drafts straight into the verify token
// buffer + their 16-way proposals q.
#pragma once
#include "models/flash_next/mtp.h"

namespace ne::dfl {

constexpr int H = 2560, NL = 5, NQ = 32, NKV = 8, HD = 128, FF = 7680, DYN = 640, FEAT = 5 * H, W = 2048, RMAX = 8;
constexpr int SELR = 256, NC = 16, CROWS = 16;  // selector rank, candidates per position, context rows per GEMM step
constexpr int SPLITS = 16;                      // attention splits over the context keys
extern const int kLayers[5];                    // target layers whose outputs are the features: 5 15 25 35 45

struct LayerW {
  Tensor in_ln, post_ln, qn, kn;  // bf16
  Tensor base_att, base_mlp;      // f32 [2 stages][2 taps][2560]
  fn::Q8 kp_att, q, k, v, o, kp_mlp, gate, up, down;
};
struct Weights {
  std::shared_ptr<const Pack> pack;  // the drafter checkpoint (mapped)
  std::vector<LayerW> L;
  fn::Q8 fc, sel;
  Tensor hnorm, norm, pcb, scb;  // bf16; codebooks [V][256] (mapped)
  Tensor invf;                   // f32 [64] rope inverse frequencies (theta 1e4)
  int mask = 0, block = 8, bits = 8;
};
// dir: the checkpoint (config.json + model.safetensors); bits 8 (or 4: the big MLP / attention matrices at 4 bits,
// k / v / dynamic kernels / fc at 8). Quantises on the CPU at load (~1-2 s).
Weights load(const std::string &dir, int bits = 8);

// The drafter's context: target positions [T - n, T) in the rings (slot = pos % 2048).
struct Ctx {
  Tensor cr;              // bf16 [W][H]: c_j = hidden_norm(fc(f_j)) (bf16 values): the persistent state
  Tensor kr[NL], vr[NL];  // bf16 [NKV][W][HD]: keys (k_norm + rope) / values, a function of cr
  int T = 0, n = 0;
};
Ctx alloc_ctx();

struct Scratch {
  int R = 0;  // block rows (4 or 8)
  Tensor tok;                                         // u32 [RMAX]: [anchor, mask ...]
  Tensor h, nrm, dyn, x, q, k, v, kb, vb, part, att, o, g, u, a, hn, sh;  // block activations
  Tensor coarse, cand, sl, amax, qo, dr;              // readout + selector
  Tensor P;    // u32 [8]: [0] T = the block's first position, [1] context rows to add, [2] their first position,
               // [3] the ring's first visible position
  Tensor SP;   // DflSelP (dflash.metal)
  Tensor cx, cc, ck, cv;  // context rows: fc out, c, k, v (CROWS rows)
};
Scratch alloc_scratch(int R);
struct SelP { uint32_t anchor, P, sampled, seed, round, pad0, pad1, pad2; float temp, pad3, pad4, pad5; };  // = DflSelP

// Context rows feat f32 [rows][FEAT] (row r = position P[2] + r0 + r, only r0 + r < P[1] are written) -> rings.
// rows <= CROWS. No trailing barrier issues: ends with one.
void record_ctx_rows(Recorder &r, const Weights &w, Ctx &c, Scratch &s, const Tensor &feat, int rows, uint32_t r0);
// its two halves: rows -> fc -> hidden_norm -> c (s.cc, bf16 values) -> the c ring; and s.cc -> k / v -> the rings
void record_ctx_c(Recorder &r, const Weights &w, Ctx &c, Scratch &s, const Tensor &feat, int rows, uint32_t r0);
void record_ctx_kv(Recorder &r, const Weights &w, Ctx &c, Scratch &s, int rows, uint32_t r0);
// the key / value rings from the c ring: positions P[2] .. P[2] + P[1] - 1 (a restored state; 128 x 16 rows)
void record_rebuild(Recorder &r, const Weights &w, Ctx &c, Scratch &s);
// The block draft (s.R rows; s.P / s.SP / s.tok host-filled): drafts -> vtok[1 .. R - 1] (and s.dr), proposals s.qo
// [R - 1][16][2]. mw: the MTP loop's draft head (4-bit pruned rows + exact 8-bit re-score) and lm_head.
void record_block(Recorder &r, const Weights &w, Ctx &c, Scratch &s, const mtp::MtpW &mw, const fn::Q8 &embed,
                  const Tensor &vtok);

}  // namespace ne::dfl
