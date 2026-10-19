// lab-trunk6: per-Linear weight format + recording of the EXL3 trunk GEMV (kernels/trunk6/trunk6.metal). Design and
// numbers: docs/trunk6.md.
//
// The interface for the layer graph (lab-integrate): a Linear knows its format; the layer code dispatches on
// `fmt` (Q8 -> its existing fused q8 kernels; EXL3 / BF16 -> t6::record into an f32 [R][out] buffer + its epilogue).
//
//   const Pack &p = ...;  auto lin = t6::load(p, "language_model.model.layers.0.linear_attn.in_proj_qkv");
//   t6::Scratch s = t6::alloc_scratch();                 // shared by every Linear of a recording (serialised)
//   t6::record(r, lin, R, x, y, s);  r.barrier();        // x f32 [R][in], y f32 [R][out]
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "core/pack.h"
#include "core/recorder.h"
#include "core/tensor.h"

namespace ne::t6 {

enum class Fmt { None, Q8, EXL3, BF16 };
const char *fmt_name(Fmt f);

struct Linear {
  std::string name;  // pack prefix
  Fmt fmt = Fmt::None;
  int in = 0, out = 0;
  int k = 0;                  // EXL3: trellis bits per weight (6, 7, 8)
  int had_in = 0, had_out = 0;  // EXL3: Hadamard sizes (128, else the largest power of two <= 64 dividing the dim)
  Tensor trellis, suh, svh;   // EXL3: U16 [in/16][out/16][16 k], F16 [in], F16 [out]
  Tensor w;                   // BF16: [out][in] (HF layout)
};

// "<prefix>.trellis" -> EXL3, "<prefix>.scales" -> Q8 (8-bit affine, loaded by fn / nax), "<prefix>.weight" BF16 -> BF16.
Fmt detect(const Pack &p, const std::string &prefix);
// EXL3 or BF16 Linear (throws for Q8 / missing). Misaligned tensors are copied (Pack::aligned); T6 packs are aligned.
Linear load(const Pack &p, const std::string &prefix);
int had_size(int d);

enum class Path { Auto, Simd, Nax };
struct Plan {
  Path path = Path::Simd;
  int R = 1, S = 1, KS = 1;   // rows, k splits across threadgroups, NAX simdgroups per threadgroup
  uint32_t tg_mem = 0;        // threadgroup memory bytes
  MTLSize grid{}, tg{};
  size_t part_floats = 0, counters = 0;  // scratch needed
};
// Auto: SIMD below 4 rows, NAX from 4 (env T6_PATH=simd|nax forces one; T6_S / T6_KS override the splits).
Plan plan(const Linear &l, int R, Path path = Path::Auto, int S = 0, int KS = 0);

struct Scratch {
  Tensor part;  // f32 partials
  Tensor ctr;   // u32 counters (zero; finishers reset them)
  Tensor xh;    // NAX: prepared input half [R][in]
};
Scratch alloc_scratch(size_t part_floats = 4u << 20, size_t counters = 4096, size_t xh_halves = 16u * 16384u);

// y[R][out] = x[R][in] @ W, no trailing barrier. EXL3 SIMD and BF16: one dispatch; EXL3 NAX: two (t6_prep, barrier,
// t6_nax).
void record(Recorder &r, const Linear &l, int R, const Tensor &x, const Tensor &y, Scratch &s,
            const Plan *p = nullptr);
// lab-t6wire: several Linears recorded to run concurrently, Linear i on its own scratch slots[i]: the NAX ones'
// t6_prep first, one barrier (if any), then every GEMV dispatch. No trailing barrier.
// Fused epilogue in the finisher (EXL3 only; kernels/trunk6/trunk6.metal t6_finish: kind 1 out_hc, 2 hc act,
// 3 round, 4 GDN conv (mode 1 / 5 / 6), 5 beta, 6 decay, 7 iqk + raw key sink); e = its extra buffers; ldy = y row
// stride (0 = out); refr = the reference rounding (FC 0).
// lab-t6fuse: pre = the input preparation folded into the GEMV (SIMD path only; FC 23): 1 HC norm (x = h, e = hc_norm,
// inject weights, inject partials, inj; with kind 2 = the whole HC read part 1), 2 SwiGLU (x = shared gate raw, e0 =
// shared up raw, e1 = shared_gu out). Kind 8 = HC read part 3 (e = h, hc_norm, inj, out; y = the up output scratch).
struct Epi {
  int kind = 0;
  uint32_t pre = 0;
  uint32_t mode = 1;
  int ldy = 0;
  bool refr = true;
  Tensor e[4];
};
void record_multi(Recorder &r, int R, const std::vector<const Linear *> &l, const std::vector<Tensor> &x,
                  const std::vector<Tensor> &y, Scratch *slots, const std::vector<Epi> *epi = nullptr);

// lab-t6wire, chunked prefill (M rows, any M): y[M][out] (row stride ldy, f32 or bf16) = x[M][in] (stride ldx, f32 or
// bf16) @ W through a dense decode of W' (half [ceil64(out)][in], no Hadamards) and lab-prefill's NAX GEMM:
// t6_pf_prep (x * suh -> H_hi, half) and t6_dense, barrier, pf_h16_hf GEMM -> yraw f32 [M][ceil64(out)], barrier,
// t6_pf_post (H_ho * svh -> y). The buffers are shared by consecutive calls (safe without a trailing barrier: the next
// call's first barrier orders it). xh half [M][in], wd half [ceil64(out)][in], yraw f32 [M][ceil64(out)].
// lab-ttft: part f32 (split-K partials for chunks of <= splitk_max_rows() rows; empty = always the NAX GEMM).
struct PfBuf { Tensor xh, wd, yraw, part, xq, sx, wq, sw; };  // xq..sw: the INT8 path (pfkern, a8_min_rows)
// INT8 prefill GEMM (pfkern 18 Oct, kernels/trunk6/t6_a8.metal): chunks of >= this many rows run the EXL3 trunk
// Linears as int8 activations x int8 W' (per (row, 128-block) scales) on the M5 tensor ops; 0 = off. NE_T6_A8.
int a8_min_rows();

constexpr int kA8Block = 128;
constexpr int kSplitKRows = 32, kSplitKGroups = 1024;  // NE_T6_SK = max rows of the split-K path (0 = off, <= 128)
constexpr size_t kSplitKPartFloats = 8u << 20;          // 32 MB of partials
int splitk_max_rows();
// phase (profiling only, NE_PF_PROF=fine): returns the recorder for each phase ("prep", "dense", "gemm", "post")
using PfPhase = std::function<Recorder &(const char *)>;
void record_prefill(Recorder &r, const Linear &l, int M, const Arg &x, bool xf, int ldx, const Arg &y, bool yf, int ldy,
                    PfBuf &b, const PfPhase *phase = nullptr);
// pfkern: the HC up Linear (320 -> 10240, EXL3, M >= 64 rows) with pf_hc_blockin fused: xin bf16 [M][2560] from
// x = HC down activations bf16 [M][320] and normed bf16 [M][10240]. false = not applicable (caller runs gemm + blockin).
bool record_prefill_hcup(Recorder &r, const Linear &l, int M, const Arg &x, const Arg &normed, const Arg &xin, PfBuf &b,
                         const PfPhase *phase = nullptr);

}  // namespace ne::t6
