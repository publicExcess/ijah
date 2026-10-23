// lab-trunk6: EXL3 / bf16 trunk Linears, host side (docs/trunk6.md).
#include "kernels/trunk6/trunk6.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

#include "core/error.h"
#include "core/pipelines.h"

namespace ne::t6 {

const char *fmt_name(Fmt f) {
  switch (f) {
    case Fmt::Q8: return "q8";
    case Fmt::EXL3: return "exl3";
    case Fmt::BF16: return "bf16";
    default: return "none";
  }
}

int had_size(int d) {
  for (int n : {128, 64, 32, 16})
    if (d % n == 0) return n;
  throw Error(strfmt("t6: no Hadamard size for %d", d));
}

Fmt detect(const Pack &p, const std::string &prefix) {
  if (p.has(prefix + ".trellis")) return Fmt::EXL3;
  if (p.has(prefix + ".scales")) return Fmt::Q8;
  const TensorInfo *w = p.info(prefix + ".weight");
  if (w && w->t.dtype == DType::BF16) return Fmt::BF16;
  return Fmt::None;
}

Linear load(const Pack &p, const std::string &prefix) {
  Linear l;
  l.name = prefix;
  l.fmt = detect(p, prefix);
  if (l.fmt == Fmt::EXL3) {
    l.trellis = p.aligned(prefix + ".trellis", 16);
    l.suh = p.aligned(prefix + ".suh", 16);
    l.svh = p.aligned(prefix + ".svh", 16);
    NE_CHECK(l.trellis.ndim == 3 && l.trellis.dtype == DType::U16, "t6: %s.trellis %s", prefix.c_str(),
             l.trellis.desc().c_str());
    l.in = (int)l.trellis.dim(0) * 16;
    l.out = (int)l.trellis.dim(1) * 16;
    l.k = (int)l.trellis.dim(2) / 16;
    NE_CHECK(l.k >= 6 && l.k <= 8 && l.trellis.dim(2) == 16 * l.k, "t6: %s: K %d (6..8)", prefix.c_str(), l.k);
    NE_CHECK(l.suh.numel() == l.in && l.svh.numel() == l.out, "t6: %s: suh/svh sizes", prefix.c_str());
    l.had_in = had_size(l.in);
    l.had_out = had_size(l.out);
  } else if (l.fmt == Fmt::BF16) {
    l.w = p.aligned(prefix + ".weight", 16);
    NE_CHECK(l.w.ndim == 2, "t6: %s.weight %s", prefix.c_str(), l.w.desc().c_str());
    l.out = (int)l.w.dim(0);
    l.in = (int)l.w.dim(1);
    NE_CHECK(l.in % 8 == 0, "t6: bf16 in %d %% 8", l.in);
  } else {
    throw Error(strfmt("t6: %s is %s (not EXL3 / BF16)", prefix.c_str(), fmt_name(l.fmt)));
  }
  return l;
}

static int env_int(const char *k) {
  const char *e = getenv(k);
  return e ? atoi(e) : 0;
}

static uint32_t align16(size_t n) { return (uint32_t)((n + 15) & ~(size_t)15); }

// threadgroup memory of a split: xs (R * span * 16 halves) + partial / output floats + flag
static uint32_t tgmem(int R, int span, size_t floats) { return align16(align16((size_t)R * span * 32) + floats * 4 + 16); }

// lab-verify: rows from which Auto picks NAX (T6_NAX_MIN, default 4; read once)
static int t6_nax_min() {
  static const int v = getenv("T6_NAX_MIN") ? std::max(1, atoi(getenv("T6_NAX_MIN"))) : 4;
  return v;
}

Plan plan(const Linear &l, int R, Path path, int S, int KS) {
  NE_CHECK(l.fmt == Fmt::EXL3, "t6::plan: %s is not EXL3", l.name.c_str());
  Plan p;
  p.R = R;
  if (path == Path::Auto) {
    const char *e = getenv("T6_PATH");
    if (e && !strcmp(e, "simd")) path = Path::Simd;
    else if (e && !strcmp(e, "nax")) path = Path::Nax;
    else path = R >= t6_nax_min() ? Path::Nax : Path::Simd;
    // lab-verify: per-shape override T6_P_<in>x<out>=simd|nax (A/B runs; read per plan). Default: the 2560 -> 640
    // Linears (shared-expert gate / up, attention indexer qk) stay SIMD up to 5 rows (verify R 4 / 5: -0.4 / -1.3 ms
    // in fn-prof, docs/mtp.md 12)
    const std::string pk = "T6_P_" + std::to_string(l.in) + "x" + std::to_string(l.out);
    const char *pe = getenv(pk.c_str());
    if (pe && !strcmp(pe, "simd") && R <= 8) path = Path::Simd;
    else if (pe && !strcmp(pe, "nax")) path = Path::Nax;
    else if (!pe && !e && l.in == 2560 && l.out == 640 && R <= 5) path = Path::Simd;
  }
  p.path = path;
  const int IT = l.in / 16, OT = l.out / 16, NCB = (l.out + 127) / 128;
  // lab-t6fuse: per-shape overrides for A/B runs: T6_S_<in>x<out> (k splits of that shape; a 2-8 deep
  // SIMD prefetch was slower in the whole model: +0.2 ms)
  const std::string shp = std::to_string(l.in) + "x" + std::to_string(l.out);
  if (!S) S = env_int(("T6_S_" + shp).c_str());
  if (!S) S = env_int("T6_S");
  if (!KS) KS = env_int("T6_KS");
  // threadgroup memory budget per threadgroup (occupancy: 28 KB at 8 NAX rows halved the speed)
  const uint32_t budget = env_int("T6_TGMEM") ? (uint32_t)env_int("T6_TGMEM") : (path == Path::Nax ? 12288u : 16384u);
  if (path == Path::Simd) {
    NE_CHECK(R >= 1 && R <= 8, "t6: SIMD rows %d (1..8)", R);
    p.KS = 1;
    auto ok = [&](int s) {
      if (IT % s) return false;
      const int span = IT / s;
      return (span * 16) % l.had_in == 0 && tgmem(R, span, (size_t)R * 128) <= budget;
    };
    if (!S && R <= 3) {  // lab-t6fuse: decode shapes whose serial k chain (span tiles per simdgroup) was the cost;
      // in-process A/B (fn-prof, 4 T6 layers in the 48-layer pattern): together -0.33 ms per step (docs/trunk6.md 8b)
      if (l.in == 2560 && l.out == 640) S = 10;   // shared gate / up: 5 -> 10 (-0.11 ms)
      if (l.in == 640 && l.out == 2560) S = 5;    // shared down: 1 -> 5 (-0.11 ms)
      if (l.in == 6144 && l.out == 2560) S = 24;  // out_proj / o_proj: 12 -> 24 (-0.08 ms)
      if (S && !ok(S)) S = 0;
    }
    if (!S) {  // smallest split giving >= 320 threadgroups with >= 32 k tiles per split (sweep in docs/trunk6.md)
      int best = 0;
      for (int s = 1; s <= IT; s++) {
        if (!ok(s)) continue;
        if (IT / s < 32 && best) break;
        best = s;
        if (NCB * s >= 320) break;
      }
      S = best;
    }
    NE_CHECK(S >= 1 && ok(S), "t6: %s: no valid SIMD split (S %d)", l.name.c_str(), S);
    p.S = S;
    p.tg_mem = tgmem(R, IT / S, (size_t)R * 128);
    p.grid = MTLSizeMake((NSUInteger)(NCB * S), 1, 1);
    p.tg = MTLSizeMake(256, 1, 1);
  } else {
    NE_CHECK(R >= 1 && R <= 16, "t6: NAX rows %d (1..16)", R);
    if (!KS) KS = 8;
    const int RMP = R > 8 ? 16 : 8, NCP = (OT + 1) / 2;
    const size_t floats = std::max((size_t)KS * RMP * 32, (size_t)R * 128);
    auto ok = [&](int s) { return IT % s == 0 && (IT / s) % KS == 0; };
    if (!S) {  // smallest split giving >= 640 threadgroups with >= 4 k steps per simdgroup
      int best = 0;
      for (int s = 1; s <= IT; s++) {
        if (!ok(s)) continue;
        if (IT / s < 4 * KS && best) break;
        best = s;
        if (NCP * s >= 640) break;
      }
      S = best ? best : 1;
    }
    NE_CHECK(S >= 1 && IT % S == 0, "t6: %s: no valid NAX split (S %d, R %d)", l.name.c_str(), S, R);
    p.S = S;
    p.KS = KS;
    p.tg_mem = align16(floats * 4 + 16);
    p.grid = MTLSizeMake((NSUInteger)(NCP * S), 1, 1);
    p.tg = MTLSizeMake((NSUInteger)(32 * KS), 1, 1);
  }
  p.part_floats = (size_t)NCB * p.S * R * 128;
  p.counters = (size_t)NCB;
  return p;
}

Scratch alloc_scratch(size_t part_floats, size_t counters, size_t xh_halves) {
  Scratch s;
  s.xh = alloc(DType::F16, {(int64_t)xh_halves});
  s.part = alloc(DType::F32, {(int64_t)part_floats});
  s.ctr = alloc(DType::U32, {(int64_t)counters});
  return s;
}

static FunctionConstants exl3_fc(const Linear &l, const Plan &p, int R, const Epi *e = nullptr) {
  FunctionConstants fc = {FC::u32(10, (uint32_t)l.k),       FC::u32(11, (uint32_t)l.in),     FC::u32(12, (uint32_t)l.out),
                          FC::u32(13, (uint32_t)l.had_in),  FC::u32(14, (uint32_t)l.had_out), FC::u32(15, (uint32_t)p.S),
                          FC::u32(16, (uint32_t)R),         FC::u32(17, (uint32_t)p.KS)};
  if (e && (e->kind || e->ldy || e->pre)) {
    fc.push_back(FC::b(0, e->refr));
    fc.push_back(FC::u32(20, (uint32_t)e->kind));
    fc.push_back(FC::u32(21, e->mode));
    fc.push_back(FC::u32(22, (uint32_t)(e->ldy ? e->ldy : l.out)));
    if (e->pre) fc.push_back(FC::u32(23, e->pre));
  }
  return fc;
}

static void record_bf16(Recorder &r, const Linear &l, int R, const Tensor &x, const Tensor &y) {
  NE_CHECK(R >= 1 && R <= 8, "t6: bf16 rows %d (1..8)", R);
  const bool split = l.out < 2048;  // few rows: 8 simdgroups per row (10240 -> 320: 40 -> 320 threadgroups)
  auto pso = Pipelines::get("t6_bf16", {FC::u32(11, (uint32_t)l.in), FC::u32(12, (uint32_t)l.out),
                                        FC::u32(15, split ? 8u : 1u), FC::u32(16, (uint32_t)R)});
  r.dispatch(pso, {x, l.w, y}, MTLSizeMake((NSUInteger)(split ? l.out : (l.out + 7) / 8), 1, 1),
             MTLSizeMake(256, 1, 1));
}

static void check_scratch(const Linear &l, const Plan &p, int R, const Scratch &s) {
  NE_CHECK(p.R == R, "t6: plan rows %d != %d", p.R, R);
  NE_CHECK((size_t)s.part.numel() >= p.part_floats && (size_t)s.ctr.numel() >= p.counters,
           "t6: scratch too small for %s (%zu floats, %zu counters)", l.name.c_str(), p.part_floats, p.counters);
  if (p.path == Path::Nax)
    NE_CHECK((size_t)s.xh.numel() >= (size_t)R * l.in, "t6: scratch xh too small (%d x %d)", R, l.in);
}
static void record_prep(Recorder &r, const Linear &l, const Plan &p, int R, const Tensor &x, Scratch &s) {
  const int jobs = R * ((l.in + 127) / 128);
  r.dispatch(Pipelines::get("t6_prep", exl3_fc(l, p, R)), {x, l.suh, s.xh}, MTLSizeMake((NSUInteger)((jobs + 7) / 8), 1, 1),
             MTLSizeMake(256, 1, 1));
}
static void record_main(Recorder &r, const Linear &l, const Plan &p, int R, const Tensor &x, const Tensor &y,
                        Scratch &s, const Epi *e = nullptr) {
  const FunctionConstants fc = exl3_fc(l, p, R, e);
  if (e && e->pre) {  // lab-t6fuse folded input preparation: SIMD only; PRE 1 spans inside one 2560 stream, whole chunks
    NE_CHECK(p.path == Path::Simd, "t6: %s: folded input prep (pre %u) needs the SIMD path", l.name.c_str(), e->pre);
    const int n = l.in / p.S;
    NE_CHECK(e->pre != 1 || (l.in == 10240 && 2560 % n == 0 && n % 256 == 0), "t6: %s: HC prep span %d", l.name.c_str(), n);
  }
  NE_CHECK(!e || e->kind != 8 || (l.out == 10240 && (size_t)s.ctr.numel() >= 128 + 20), "t6: %s: HC out epilogue",
           l.name.c_str());
  auto eb = [&](int i) { return e && e->e[i].buf ? Arg(e->e[i]) : Arg(y); };
  if (p.path == Path::Nax)
    r.dispatch(Pipelines::get("t6_nax", fc), {s.xh, l.trellis, l.suh, l.svh, s.part, s.ctr, y, eb(0), eb(1), eb(2), eb(3)},
               p.grid, p.tg, p.tg_mem);
  else
    r.dispatch(Pipelines::get("t6_simd", fc), {x, l.trellis, l.suh, l.svh, s.part, s.ctr, y, eb(0), eb(1), eb(2), eb(3)},
               p.grid, p.tg, p.tg_mem);
}

void record(Recorder &r, const Linear &l, int R, const Tensor &x, const Tensor &y, Scratch &s, const Plan *pp) {
  if (l.fmt == Fmt::BF16) {
    record_bf16(r, l, R, x, y);
    return;
  }
  Plan own;
  if (!pp) {
    own = plan(l, R);
    pp = &own;
  }
  const Plan &p = *pp;
  check_scratch(l, p, R, s);
  if (p.path == Path::Nax) {
    record_prep(r, l, p, R, x, s);
    r.barrier();
  }
  record_main(r, l, p, R, x, y, s);
}

void record_multi(Recorder &r, int R, const std::vector<const Linear *> &l, const std::vector<Tensor> &x,
                  const std::vector<Tensor> &y, Scratch *slots, const std::vector<Epi> *epi) {
  NE_CHECK(!epi || epi->size() == l.size(), "t6::record_multi: %zu epilogues for %zu Linears", epi->size(), l.size());
  NE_CHECK(l.size() == x.size() && l.size() == y.size(), "t6::record_multi: %zu Linears, %zu x, %zu y", l.size(),
           x.size(), y.size());
  std::vector<Plan> p(l.size());
  bool nax = false;
  for (size_t i = 0; i < l.size(); i++) {
    if (l[i]->fmt != Fmt::EXL3) continue;
    p[i] = plan(*l[i], R);
    check_scratch(*l[i], p[i], R, slots[i]);
    if (p[i].path == Path::Nax) {
      record_prep(r, *l[i], p[i], R, x[i], slots[i]);
      nax = true;
    }
  }
  if (nax) r.barrier();
  for (size_t i = 0; i < l.size(); i++) {
    const Epi *e = epi ? &(*epi)[i] : nullptr;
    if (l[i]->fmt == Fmt::BF16) {
      NE_CHECK(!e || (!e->kind && !e->ldy), "t6: no fused epilogue for the bf16 Linear %s", l[i]->name.c_str());
      record_bf16(r, *l[i], R, x[i], y[i]);
    } else {
      record_main(r, *l[i], p[i], R, x[i], y[i], slots[i], e);
    }
  }
}

struct PfGemmP {  // = kernels/prefill/pf_gemm.metal PfGemm
  int K, N, M, ldx, ldy;
};
// lab-ttft: split-K GEMM for chunks of few rows (kernels/trunk6/t6_splitk.metal). Built into the metallib; a build
// without the Metal toolchain (a sandboxed agent) compiles that file at run time instead.
struct T6SkP {  // = t6_splitk.metal T6SkP
  uint32_t K, NP, M, Mp, ks, S;
};
static void need_splitk() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (Pipelines::has("t6_pf_sk2")) return;
    // __FILE__ is relative to the source root (the build runs there): resolve it against the library's location
    // (build/engine.metallib -> the root is build/..), not the process's working directory
    std::string rel = __FILE__;
    rel = rel.substr(0, rel.rfind('/') + 1);
    std::vector<std::string> dirs;
    if (!rel.empty() && rel[0] == '/') {
      dirs.push_back(rel);
    } else {
      std::string lib = Pipelines::library_path();
      lib = lib.rfind('/') == std::string::npos ? "./" : lib.substr(0, lib.rfind('/') + 1);
      dirs.push_back(lib + "../" + rel);
      dirs.push_back(lib);  // shipped next to the library
    }
    std::ifstream f;
    std::string tried;
    for (const std::string &d : dirs) {
      f = std::ifstream(d + "t6_splitk.metal");
      if (f.good()) break;
      tried += " " + d + "t6_splitk.metal";
    }
    NE_CHECK(f.good(), "t6: t6_pf_sk2 is not in the metallib and t6_splitk.metal is not at%s", tried.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    Pipelines::add_source(ss.str(), "t6_splitk.metal");
  });
}
struct T6A8P {  // = t6_a8.metal T6A8P
  uint32_t K, N, M, ldy;
};
static bool tensor_gemm() {  // NE_T6_TG=0: the 16x32x16 fragment GEMM pf_h16_hf (A/B)
  const char *e = getenv("NE_T6_TG");
  return !e || atoi(e) != 0;
}
int a8_min_rows() {  // read at record time
  const char *e = getenv("NE_T6_A8");
  return e ? atoi(e) : 0;
}
static bool fold_on() {  // NE_T6_FOLD=0: the prep / GEMM / post path for bf16-input Linears too
  const char *e = getenv("NE_T6_FOLD");
  return !e || atoi(e) != 0;
}
// pfast: the folded W costs a dense Wf build per Linear and chunk (dense + fold_n passes over the whole matrix), which
// only pays from ~1K rows: whole-model chunk GPU ms fold vs prep / GEMM / post at T 16K, 64 rows 185 vs 145, 128 216
// vs 181, 500 383 vs 346, 750 505 vs 483, 1000 619 vs 616, 1500 872 vs 889, 4096 2142 vs 2352. NE_T6_FOLD_MIN = the
// fewest rows that fold (read at record time). Default 64 = master: parked (23 Oct) until the night job
// tests/quality/pfast_null_gate.sh judges it against the teacher-KLD gate's null spread; 1024 = the faster setting.
static int fold_min_rows() {
  const char *e = getenv("NE_T6_FOLD_MIN");
  return std::max(64, e ? atoi(e) : 64);
}
static int tg_min_rows() {  // pfast A/B: the fewest rows on the f16 tensor-op GEMM (else pf_h16_hf); NE_T6_TG_MIN
  const char *e = getenv("NE_T6_TG_MIN");
  return std::max(1, e ? atoi(e) : 64);
}
static bool a8_skip(const std::string &name) {  // NE_T6_A8_SKIP=<comma list of Linear name parts>: those stay f16
  const char *e = getenv("NE_T6_A8_SKIP");
  if (!e || !*e) return false;
  const std::string list = std::string(",") + e + ",";
  for (size_t a = 1, b; (b = list.find(',', a)) != std::string::npos; a = b + 1)
    if (b > a && name.find(list.substr(a, b - a)) != std::string::npos) return true;
  return false;
}

// pfast: split-K rows for narrow Linears (read at record time; 0 = as wide ones = master, the default: parked like
// NE_T6_FOLD_MIN). 128: whole-model trunk stage at T 16K, 45 / 64 / 128 rows 53 / 62 / 65 -> 41 / 45 / 51 ms (the HC
// down and the shared expert's gate / up; at >= 64 rows only together with NE_T6_FOLD_MIN > rows).
static int narrow_max_rows() {
  const char *e = getenv("NE_T6_SK_NARROW");
  return e ? atoi(e) : 0;
}
int splitk_max_rows() {  // read at record time (A/B: a new Prefill per value)
  const char *e = getenv("NE_T6_SK");
  return e ? atoi(e) : kSplitKRows;
}

// W' -> H_hi along k, * suh (half, wd): t6_dense_fold in one pass (NE_T6_DFOLD=0: t6_dense, then t6_fold_k)
template <typename RF>
static void record_dense_fold(RF &R, const Linear &l, const FunctionConstants &fc, PfBuf &b) {
  static const bool fused = !getenv("NE_T6_DFOLD") || atoi(getenv("NE_T6_DFOLD"));
  if (fused) {
    R("dense").dispatch(Pipelines::get("t6_dense_fold", fc), {l.trellis, b.wd, l.suh},
                        MTLSizeMake((NSUInteger)(l.out / 16), (NSUInteger)((l.in + 127) / 128), 1), MTLSizeMake(256, 1, 1));
    R("dense").barrier();
    return;
  }
  const int tiles = (l.in / 16) * (l.out / 16);
  R("dense").dispatch(Pipelines::get("t6_dense", fc), {l.trellis, b.wd}, MTLSizeMake((NSUInteger)((tiles + 7) / 8), 1, 1),
                      MTLSizeMake(256, 1, 1));
  R("dense").barrier();
  R("fold").dispatch(Pipelines::get("t6_fold_k", fc), {b.wd, l.suh},
                     MTLSizeMake((NSUInteger)((l.in + 127) / 128), (NSUInteger)((l.out + 7) / 8), 1), MTLSizeMake(256, 1, 1));
  R("fold").barrier();
}

void record_prefill(Recorder &r, const Linear &l, int M, const Arg &x, bool xf, int ldx, const Arg &y, bool yf, int ldy,
                    PfBuf &b, const PfPhase *phase) {
  auto R = [&](const char *ph) -> Recorder & { return phase ? (*phase)(ph) : r; };
  NE_CHECK(l.fmt == Fmt::EXL3, "t6::record_prefill: %s is %s", l.name.c_str(), fmt_name(l.fmt));
  const int NP = (l.out + 63) / 64 * 64;
  NE_CHECK((size_t)b.xh.numel() >= (size_t)M * l.in && (size_t)b.wd.numel() >= (size_t)NP * l.in &&
               (size_t)b.yraw.numel() >= (size_t)M * NP,
           "t6: prefill buffers too small for %s at %d rows", l.name.c_str(), M);
  NE_CHECK(l.in % 32 == 0, "t6: prefill in %d %% 32", l.in);
  const FunctionConstants fc = {FC::u32(10, (uint32_t)l.k),      FC::u32(11, (uint32_t)l.in),
                                FC::u32(12, (uint32_t)l.out),    FC::u32(13, (uint32_t)l.had_in),
                                FC::u32(14, (uint32_t)l.had_out), FC::b(18, !xf), FC::b(19, !yf)};
  const uint32_t uM = (uint32_t)M, ux = (uint32_t)ldx, uy = (uint32_t)ldy, unp = (uint32_t)NP;
  const int nchi = (l.in + 127) / 128, ncho = (l.out + 127) / 128;
  const int a8 = a8_min_rows();
  // in >= 1024 only: the HC up Linear (320 -> 10240, feeding the HC gates) carried most of the int8 error (4-layer
  // hidden rel L2 2.1e-2 with it, 9.4e-3 without; f16 4.8e-3), and the short-K Linears are cheap
  if (a8 > 0 && M >= a8 && b.xq.buf && l.in >= 1024 && l.in % 32 == 0 && !a8_skip(l.name)) {  // INT8 path (pfkern)
    const int KB = (l.in + kA8Block - 1) / kA8Block;
    NE_CHECK((size_t)b.xq.nbytes() >= (size_t)M * l.in && (size_t)b.sx.numel() >= (size_t)M * KB &&
                 (size_t)b.wq.nbytes() >= (size_t)NP * l.in && (size_t)b.sw.numel() >= (size_t)NP * KB,
             "t6: a8 buffers too small for %s at %d rows", l.name.c_str(), M);
    R("prep").dispatch(Pipelines::get("t6_pf_prep_a8", fc), {x, l.suh, b.xq, b.sx, Arg::value(uM), Arg::value(ux)},
                       MTLSizeMake((NSUInteger)nchi, (NSUInteger)((M + 7) / 8), 1), MTLSizeMake(256, 1, 1));
    R("dense").dispatch(Pipelines::get("t6_dense_a8", fc), {l.trellis, b.wq, b.sw, Arg::value(unp)},
                        MTLSizeMake((NSUInteger)(l.out / 16), (NSUInteger)((l.in / 16 + 7) / 8), 1), MTLSizeMake(256, 1, 1));
    R("dense").barrier();
    const bool tall = NP <= 640;  // few column tiles: 128-row tiles (the HC down 10240 -> 320)
    const T6A8P ap{(uint32_t)l.in, (uint32_t)NP, uM, unp};
    R("gemm").dispatch(Pipelines::get(tall ? "t6_a8_gemm_m128" : "t6_a8_gemm", {}), {b.wq, b.sw, b.xq, b.sx, b.yraw, Arg::value(ap)},
                       MTLSizeMake((NSUInteger)(NP / 64), (NSUInteger)((M + (tall ? 127 : 63)) / (tall ? 128 : 64)), 1),
                       MTLSizeMake(128, 1, 1));
    R("gemm").barrier();
    R("post").dispatch(Pipelines::get("t6_pf_post", fc), {b.yraw, l.svh, y, Arg::value(uM), Arg::value(unp), Arg::value(uy)},
                       MTLSizeMake((NSUInteger)ncho, (NSUInteger)((M + 7) / 8), 1), MTLSizeMake(256, 1, 1));
    return;
  }
  if (!xf && M >= fold_min_rows() && fold_on() && l.in % 32 == 0 && ldx == l.in) {  // folded W (pfkern): x bf16 straight into the GEMM, y straight out
    record_dense_fold(R, l, fc, b);
    R("fold").dispatch(Pipelines::get("t6_fold_n", fc), {b.wd, l.svh, b.wd},
                       MTLSizeMake((NSUInteger)(l.in / 32), (NSUInteger)ncho, 1), MTLSizeMake(256, 1, 1));
    R("fold").barrier();
    const T6A8P gp{(uint32_t)l.in, (uint32_t)l.out, uM, uy};
    R("gemm").dispatch(Pipelines::get(yf ? "t6_fold_gemm_f" : "t6_fold_gemm_b", {}), {b.wd, x, y, Arg::value(gp)},
                       MTLSizeMake((NSUInteger)(NP / 64), (NSUInteger)((M + 127) / 128), 1), MTLSizeMake(128, 1, 1));
    R("gemm").barrier();  // the next Linear's t6_dense rewrites wd
    return;
  }
  R("prep").dispatch(Pipelines::get("t6_pf_prep", fc), {x, l.suh, b.xh, Arg::value(uM), Arg::value(ux)},
             MTLSizeMake((NSUInteger)nchi, (NSUInteger)((M + 7) / 8), 1), MTLSizeMake(256, 1, 1));
  const int tiles = (l.in / 16) * (l.out / 16);
  const char *dbg = getenv("NE_T6_PFDBG");  // lab-ttft timing experiments only (wrong outputs): nodense / nogemm
  if (!(dbg && strstr(dbg, "nodense")))
    R("dense").dispatch(Pipelines::get("t6_dense", fc), {l.trellis, b.wd}, MTLSizeMake((NSUInteger)((tiles + 7) / 8), 1, 1),
               MTLSizeMake(256, 1, 1));
  R("dense").barrier();
  // few rows (lab-ttft): the NAX GEMM's grid is (NP / 64) x ceil(M / 64) threadgroups over the whole K -- 1 to 160 for
  // every trunk Linear at M <= 64 -- so split K over S slices until ~kSplitKGroups threadgroups run (partials in
  // b.part, reduced in a fixed order); bigger chunks keep the NAX GEMM.
  const int Mp = M <= 16 ? 16 : M <= 32 ? 32 : M <= 64 ? 64 : 128;
  int S = 0, ks = l.in;
  // pfast: narrow Linears (NP <= 640: the HC down 10240 -> 320, twice per layer) have NP / 64 = 5 column tiles over
  // K = 10240 on the NAX GEMM, latency-bound at any small M: split K for them up to NE_T6_SK_NARROW rows (<= 128)
  const bool narrow = NP <= 640 && M <= std::min(128, narrow_max_rows());
  if ((M <= std::min(128, splitk_max_rows()) || narrow) && b.part.buf) {
    const int tg0 = NP / 32;
    S = std::max(1, std::min((kSplitKGroups + tg0 - 1) / tg0, l.in / 128));
    ks = (l.in / S + 31) / 32 * 32;
    S = (l.in + ks - 1) / ks;
    while (S > 1 && (size_t)S * Mp * NP > (size_t)b.part.numel()) {  // partial buffer bound
      ks = (l.in / (S - 1) + 31) / 32 * 32;
      S = (l.in + ks - 1) / ks;
    }
    NE_CHECK((size_t)b.yraw.numel() >= (size_t)Mp * NP, "t6: yraw too small for %d rows", Mp);
  }
  if (S > 0) {
    need_splitk();
    const T6SkP sp{(uint32_t)l.in, (uint32_t)NP, uM, (uint32_t)Mp, (uint32_t)ks, (uint32_t)S};
    if (!(dbg && strstr(dbg, "nogemm")))
      R("gemm").dispatch(Pipelines::get("t6_pf_sk" + std::to_string(Mp / 8), {}), {b.wd, b.xh, S == 1 ? b.yraw : b.part, Arg::value(sp)},
                 MTLSizeMake((NSUInteger)(NP / 32), 1, (NSUInteger)S), MTLSizeMake(128, 1, 1));
    R("gemm").barrier();
    if (S > 1) {
      R("gemm").dispatch(Pipelines::get("t6_pf_skr", {}), {b.part, b.yraw, Arg::value(sp)},
                 MTLSizeMake((NSUInteger)((M * NP + 255) / 256), 1, 1), MTLSizeMake(256, 1, 1));
      R("gemm").barrier();
    }
  } else if (M >= tg_min_rows() && tensor_gemm()) {  // pfkern: f16 on the MPP tensor op (t6_a8.metal)
    // 128 x 64 tiles over 4 simdgroups: in the model the 64 x 64 / 8-simdgroup variant ran 1.5-2x slower than in
    // isolation on the wide Linears (qkv 6.9 vs 3.3 ms); NE_T6_TALL=0 selects it for N > 640
    static const bool all_tall = !getenv("NE_T6_TALL") || atoi(getenv("NE_T6_TALL"));
    const bool tall = all_tall || NP <= 640;
    const T6A8P gp{(uint32_t)l.in, (uint32_t)NP, uM, unp};
    if (!(dbg && strstr(dbg, "nogemm")))
      R("gemm").dispatch(Pipelines::get(tall ? "t6_h16_gemm_m128" : "t6_h16_gemm", {}), {b.wd, b.xh, b.yraw, Arg::value(gp)},
                         MTLSizeMake((NSUInteger)(NP / 64), (NSUInteger)((M + (tall ? 127 : 63)) / (tall ? 128 : 64)), 1),
                         MTLSizeMake(tall ? 128 : 256, 1, 1));
    R("gemm").barrier();
  } else {
    PfGemmP g{l.in, NP, M, l.in, NP};
    if (!(dbg && strstr(dbg, "nogemm")))
      R("gemm").dispatch(Pipelines::get("pf_h16_hf", {}), {b.wd, b.xh, b.yraw, Arg::value(g)},
                 MTLSizeMake((NSUInteger)(NP / 64), (NSUInteger)((M + 63) / 64), 1), MTLSizeMake(128, 1, 1));
    R("gemm").barrier();
  }
  R("post").dispatch(Pipelines::get("t6_pf_post", fc), {b.yraw, l.svh, y, Arg::value(uM), Arg::value(unp), Arg::value(uy)},
             MTLSizeMake((NSUInteger)ncho, (NSUInteger)((M + 7) / 8), 1), MTLSizeMake(256, 1, 1));
}

}  // namespace ne::t6

namespace ne::t6 {
struct T6HcUpP {  // = trunk6.metal T6HcUpP
  uint32_t K, M;
};
bool record_prefill_hcup(Recorder &r, const Linear &l, int M, const Arg &x, const Arg &normed, const Arg &xin, PfBuf &b,
                         const PfPhase *phase) {
  static const bool on = !getenv("NE_T6_HCUP") || atoi(getenv("NE_T6_HCUP"));  // NE_T6_HCUP=0: gemm + pf_hc_blockin
  if (!on || !fold_on() || M < fold_min_rows() || l.fmt != Fmt::EXL3 || l.out != 10240 || l.in % 32 != 0) return false;
  auto R = [&](const char *ph) -> Recorder & { return phase ? (*phase)(ph) : r; };
  const FunctionConstants fc = {FC::u32(10, (uint32_t)l.k),      FC::u32(11, (uint32_t)l.in),
                                FC::u32(12, (uint32_t)l.out),    FC::u32(13, (uint32_t)l.had_in),
                                FC::u32(14, (uint32_t)l.had_out), FC::b(18, true), FC::b(19, true), FC::b(24, true)};
  record_dense_fold(R, l, fc, b);
  // fold_n writes the permuted rows to yraw (reused as half [10240][in]): rows cross threadgroups, so not in place
  R("fold").dispatch(Pipelines::get("t6_fold_n", fc), {b.wd, l.svh, b.yraw},
                     MTLSizeMake((NSUInteger)(l.in / 32), (NSUInteger)((l.out + 127) / 128), 1), MTLSizeMake(256, 1, 1));
  R("fold").barrier();
  const T6HcUpP hp{(uint32_t)l.in, (uint32_t)M};
  R("gemm").dispatch(Pipelines::get("t6_fold_hcup", fc), {b.yraw, x, normed, xin, Arg::value(hp)},
                     MTLSizeMake(2560 / 16, (NSUInteger)((M + 63) / 64), 1), MTLSizeMake(128, 1, 1));
  R("gemm").barrier();
  return true;
}
}  // namespace ne::t6
