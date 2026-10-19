// Flash-Next QSA attention host side: cache allocation, recording, import / export / truncate, C API. Owner: lab-attn.
#include "kernels/attn/qsa.h"

#include <algorithm>
#include "kernels/attn/qsa_chunk.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/bench.h"

namespace ne::attn {

// Reference rounding (function constant 0 of every kernel, kernels/ne_common.h): the layer graph's switch
// ne::fn::g_ref_round (off by default; env NE_FN_ROUND=ref or ne_fn_set_ref_rounding(1) turn it on; models/flash_next/layer.mm, same
// library). Read when a step is recorded.
}  // namespace ne::attn
namespace ne::fn {
extern bool g_ref_round;
}
namespace ne::attn {
static bool ref_round() { return ne::fn::g_ref_round; }

const char *kv_fmt_name(KvFmt f) { return f == KvFmt::F32 ? "f32" : f == KvFmt::BF16 ? "bf16" : "kv8"; }
KvFmt kv_fmt_from_env(KvFmt dflt) {
  const char *e = getenv("NE_KV");
  if (!e || !*e) return dflt;
  const std::string s(e);
  if (s == "f32") return KvFmt::F32;
  if (s == "bf16" || s == "16") return KvFmt::BF16;
  if (s == "kv8" || s == "8") return KvFmt::KV8;
  NE_CHECK(false, "NE_KV=%s: expected f32, bf16 or kv8", e);
  return dflt;
}
size_t kv_bytes_per_token(KvFmt f) {  // K and V of both kv heads
  const size_t row = f == KvFmt::F32 ? HD * 4 : f == KvFmt::BF16 ? HD * 2 : HD + (HD / 64) * 4;
  return 2 * NKV * row;
}

static Tensor aligned_t(const Pack &p, const std::string &name) {
  const Tensor &t = p.at(name);
  if (t.gpu_address() % 16 == 0) return t;
  return p.aligned(name, 16);
}

QsaW load_qsa_weights(const Pack &p, int layer) {
  const std::string sa = "language_model.model.layers." + std::to_string(layer) + ".self_attn.";
  QsaW w;
  w.qnorm = aligned_t(p, sa + "q_norm.weight");
  w.knorm = aligned_t(p, sa + "k_norm.weight");
  w.iqnorm = aligned_t(p, sa + "indexer.q_layernorm.weight");
  w.iknorm = aligned_t(p, sa + "indexer.k_layernorm.weight");
  NE_CHECK(w.qnorm.dtype == DType::BF16 && w.qnorm.numel() == HD && w.knorm.numel() == HD, "layer %d: q/k norm", layer);
  NE_CHECK(w.iqnorm.dtype == DType::BF16 && w.iqnorm.numel() == IHD && w.iknorm.numel() == IHD,
           "layer %d: indexer norms", layer);
  return w;
}

// Rope inverse frequencies 1 / 1e7^(2d/64), d < 32, exactly as the MLX reference computes them on the GPU
// (mlx_vlm rope_utils.compute_inv_freq: 1.0 / (base ** (arange(0, 64, 2) / 64)) in f32). 19 of the 32 differ by one
// ulp from a correctly rounded table (and from numpy's): at position 131072 one ulp of inv_freq[0] moves the angle by
// 0.008 rad, so the table matters at long context.
static const uint32_t kInvFreqBits[32] = {
    0x3f800000u, 0x3f1ab32bu, 0x3ebaf81au, 0x3e61f835u, 0x3e088d77u, 0x3da50956u, 0x3d47763eu, 0x3cf11177u,
    0x3c91ad39u, 0x3c301051u, 0x3bd4ca14u, 0x3b80967du, 0x3b1b690du, 0x3abbd3edu, 0x3a6301e0u, 0x3a092e02u,
    0x39a5cb60u, 0x394860c1u, 0x38f22ce2u, 0x3892587eu, 0x3830df51u, 0x37d5c441u, 0x37812dabu, 0x371c1fc3u,
    0x36bcb0c1u, 0x36640cc6u, 0x3609cf4au, 0x35a68e4bu, 0x35494c57u, 0x34f3499bu, 0x3493048du, 0x3431af43u,
};

QsaCache alloc_cache(KvFmt fmt, int tmax, const QsaCache *scratch_from) {
  NE_CHECK(tmax > 0 && tmax % 64 == 0, "tmax %d must be a positive multiple of 64", tmax);
  QsaCache c;
  c.fmt = fmt;
  c.tmax = tmax;
  const int64_t rows = (int64_t)NKV * tmax;
  if (fmt == KvFmt::F32) {
    c.kc = alloc(DType::F32, {NKV, tmax, HD});
    c.vc = alloc(DType::F32, {NKV, tmax, HD});
  } else if (fmt == KvFmt::BF16) {
    c.kc = alloc(DType::BF16, {NKV, tmax, HD});
    c.vc = alloc(DType::BF16, {NKV, tmax, HD});
  } else {
    c.kc = alloc(DType::U8, {NKV, tmax, HD});
    c.vc = alloc(DType::U8, {NKV, tmax, HD});
  }
  // kv8 scales / biases (a 16-byte dummy otherwise, so the argument lists stay the same)
  c.ksb = fmt == KvFmt::KV8 ? alloc(DType::U32, {rows, HD / 64}) : alloc(DType::U32, {4});
  c.vsb = fmt == KvFmt::KV8 ? alloc(DType::U32, {rows, HD / 64}) : alloc(DType::U32, {4});
  c.ikeys = alloc(DType::BF16, {tmax, IHD});
  c.bkeys = alloc(DType::BF16, {tmax / RATIO, IHD});
  c.pos = alloc(DType::U32, {4});
  c.invf = alloc(DType::F32, {36});  // + the multimodal position header (mlen = delta = 0: text); see set_mrope
  memcpy(c.invf.host(), kInvFreqBits, sizeof(kInvFreqBits));
  if (scratch_from && scratch_from->tmax >= tmax) {  // layers run one after another: one set of step scratch
    c.qr = scratch_from->qr;
    c.iq = scratch_from->iq;
    c.scores = scratch_from->scores;
    c.sel = scratch_from->sel;
    c.nsel = scratch_from->nsel;
    c.part = scratch_from->part;
    c.cand = scratch_from->cand;
    c.scratch_tmax = scratch_from->scratch_tmax;
    return c;
  }
  c.scratch_tmax = tmax;
  c.qr = alloc(DType::F32, {MAX_ROWS, NQH, HD});
  c.iq = alloc(DType::F32, {MAX_ROWS, 4, IHD});
  c.scores = alloc(DType::F32, {MAX_ROWS, tmax / RATIO});
  c.sel = alloc(DType::U32, {MAX_ROWS, KB});
  c.nsel = alloc(DType::U32, {MAX_ROWS});
  c.cand = alloc(DType::U32, {MAX_ROWS, (tmax / RATIO + SLICE - 1) / SLICE, KB});
  c.part = alloc(DType::F32, {NKV, MAX_ROWS, NCH, 16, HD + 2});  // 16 head slots (12 used)
  return c;
}

size_t cache_bytes(const QsaCache &c) {
  return c.kc.nbytes() + c.vc.nbytes() + (c.fmt == KvFmt::KV8 ? c.ksb.nbytes() + c.vsb.nbytes() : 0) +
         c.ikeys.nbytes() + c.bkeys.nbytes();
}

static id<MTLComputePipelineState> pso(const std::string &name, int min_threads = 32) {
  id<MTLComputePipelineState> p = Pipelines::get(name, {FC::b(0, ref_round())});
  NE_CHECK((int)p.maxTotalThreadsPerThreadgroup >= min_threads, "%s: max %d threads per threadgroup < %d",
           name.c_str(), (int)p.maxTotalThreadsPerThreadgroup, min_threads);
  return p;
}
static MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
// qsa_attn shape: HPS q heads per simdgroup (12 / HPS simdgroups per threadgroup), 8 blocks per chunk (qsa.metal):
// HPS 1 for f32 / bf16, 2 for kv8 (decode once, two heads: kv8 is ALU-heavier). Swept at T = 8192 (docs/attn.md).
static int attn_hps(KvFmt f) { return f == KvFmt::KV8 ? 2 : 1; }
// qsa_attn_mma (simdgroup matrices) for kv8: 16 vs 23 us per 2048 keys at 1 row, 51 vs 73 at 4 rows; for bf16 it
// ties the SIMT kernel (15.7 vs 16.6 us), which keeps exact f32 softmax weights, so bf16 stays SIMT.
// NE_QSA_ATTN=mma | simt overrides (f32: always SIMT).
static bool attn_mma(KvFmt f) {
  const char *e = getenv("NE_QSA_ATTN");
  if (f == KvFmt::F32) return false;
  if (e && std::string(e) == "mma") return true;
  if (e && std::string(e) == "simt") return false;
  return f == KvFmt::KV8;
}
// block scores on the simdgroup matrix units (default) or the SIMT kernel (NE_QSA_SCORE=simt)
static bool score_simt() {
  const char *e = getenv("NE_QSA_SCORE");
  return e && std::string(e) == "simt";
}
static int nslices(const QsaCache &c) { return (c.tmax / RATIO + SLICE - 1) / SLICE; }
// kernels index scores / cand with the cache's own tmax: a shared scratch must have been sized for >= tmax
static int score_tgs(const QsaCache &c) {  // grid-stride: enough threadgroups to fill the GPU, at most one per 32 blocks
  return std::max(1, std::min(c.tmax / RATIO / 32, 320));
}

void record_one(Recorder &r, int which, const QsaW &w, QsaCache &c, int R, const Tensor &q, const Tensor &kv,
                const Tensor &iqk, const Tensor &out) {
  NE_CHECK(R >= 1 && R <= MAX_ROWS, "rows %d (1..%d)", R, MAX_ROWS);
  const std::string rs = "_r" + std::to_string(R), fs = std::string("_") + kv_fmt_name(c.fmt);
  const uint32_t tmax = (uint32_t)c.tmax;
  switch (which) {
    case 0:
      r.dispatch(pso("qsa_prep" + fs + rs),
                 {q, kv, iqk, w.qnorm, w.knorm, w.iqnorm, w.iknorm, c.invf, c.kc, c.vc, c.ksb, c.vsb, c.ikeys, c.bkeys,
                  c.pos, Arg::value(tmax), c.qr, c.iq},
                 sz(R * NQH + R * NKV + R * 4 + (R + 3) / 4), sz(32));
      break;
    case 1:
      r.dispatch(pso((score_simt() ? "qsa_score" : "qsa_score_mma") + rs, 256),
                 {c.iq, c.bkeys, c.pos, c.scores, Arg::value(tmax)}, sz(score_tgs(c)), sz(256));
      break;
    case 2:
      r.dispatch(pso("qsa_select_local" + rs, 1024), {c.scores, c.pos, c.cand, Arg::value(tmax)},
                 MTLSizeMake(R, nslices(c), 1), sz(1024));
      break;
    case 3:
      r.dispatch(pso("qsa_select" + rs, 1024), {c.scores, c.pos, c.sel, c.nsel, Arg::value(tmax), c.cand}, sz(R),
                 sz(1024));
      break;
    case 4:
      if (attn_mma(c.fmt))
        r.dispatch(pso("qsa_attn_mma" + fs + rs, 128),
                   {c.qr, c.kc, c.vc, c.ksb, c.vsb, c.sel, c.nsel, c.pos, Arg::value(tmax), c.part}, sz(NKV * R * NCH),
                   sz(128));
      else
        r.dispatch(pso("qsa_attn" + fs + rs, (NQH / NKV / attn_hps(c.fmt)) * 32),
                   {c.qr, c.kc, c.vc, c.ksb, c.vsb, c.sel, c.nsel, c.pos, Arg::value(tmax), c.part}, sz(NKV * R * NCH),
                   sz((NQH / NKV / attn_hps(c.fmt)) * 32));
      break;
    case 5:
      r.dispatch(pso("qsa_merge" + rs, 256), {q, c.part, c.nsel, c.pos, out}, sz(R * NQH), sz(256));
      break;
    default:
      NE_CHECK(false, "bad dispatch %d", which);
  }
}

void record_step(Recorder &r, const QsaW &w, QsaCache &c, int R, const Tensor &q, const Tensor &kv, const Tensor &iqk,
                 const Tensor &out, bool advance) {
  for (int i = 0; i < kNumDispatches; i++) {
    record_one(r, i, w, c, R, q, kv, iqk, out);
    r.barrier();
  }
  if (advance) record_advance(r, c, R);
}

void record_advance(Recorder &r, QsaCache &c, int n) {
  r.dispatch(pso("qsa_pos_add", 1), {c.pos, Arg::value((uint32_t)n)}, sz(1), sz(1));
  r.barrier();
}
void record_rollback(Recorder &r, QsaCache &c, const Tensor &accepted, int rows) {
  r.dispatch(pso("qsa_pos_rollback", 1), {c.pos, accepted, Arg::value((uint32_t)rows)}, sz(1), sz(1));
  r.barrier();
}

static uint16_t to_bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float from_bf16(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

void import_cache(QsaCache &c, const QsaW &w, const float *k, const float *v, const float *ikeys, int T) {
  NE_CHECK(T >= 0 && T + MAX_ROWS <= c.tmax, "T %d out of range (tmax %d)", T, c.tmax);
  Recorder r;
  r.begin();
  Tensor sk, sv;
  if (T > 0 && (k || v)) {
    // staging in chunks would save RAM; T * 2 KB per tensor is fine for tests and hand-overs
    sk = alloc(DType::F32, {NKV, T, HD});
    sv = alloc(DType::F32, {NKV, T, HD});
    if (k) upload(sk, k, sk.nbytes());
    if (v) upload(sv, v, sv.nbytes());
    NE_CHECK(k && v, "import_cache: k and v go together");
    r.dispatch(pso(std::string("qsa_import_") + kv_fmt_name(c.fmt)),
               {sk, sv, c.kc, c.vc, c.ksb, c.vsb, Arg::value(0u), Arg::value((uint32_t)T), Arg::value((uint32_t)c.tmax)},
               sz(NKV * T), sz(32));
    r.barrier();
  }
  if (ikeys && T > 0) {
    uint16_t *ik = (uint16_t *)c.ikeys.host();
    for (size_t i = 0; i < (size_t)T * IHD; i++) ik[i] = to_bf16(ikeys[i]);
  }
  const int nb = T / RATIO;
  if (ikeys && nb > 0) {
    r.dispatch(pso("qsa_build_blocks"), {c.ikeys, w.iknorm, c.invf, c.bkeys, Arg::value(0u)}, sz(nb), sz(32));
    r.barrier();
  }
  r.end();
  r.run();
  ((uint32_t *)c.pos.host())[0] = (uint32_t)T;
}

void export_cache(QsaCache &c, int n, float *k, float *v, float *ikeys, float *bkeys) {
  NE_CHECK(n >= 0 && n <= c.tmax, "n %d out of range", n);
  if ((k || v) && n > 0) {
    Tensor ok = alloc(DType::F32, {NKV, n, HD}), ov = alloc(DType::F32, {NKV, n, HD});
    Recorder r;
    r.begin();
    r.dispatch(pso(std::string("qsa_export_") + kv_fmt_name(c.fmt)),
               {c.kc, c.vc, c.ksb, c.vsb, ok, ov, Arg::value((uint32_t)n), Arg::value((uint32_t)c.tmax)}, sz(NKV * n),
               sz(32));
    r.end();
    r.run();
    if (k) download(ok, k, ok.nbytes());
    if (v) download(ov, v, ov.nbytes());
  }
  if (ikeys) {
    const uint16_t *ik = (const uint16_t *)c.ikeys.host();
    for (size_t i = 0; i < (size_t)n * IHD; i++) ikeys[i] = from_bf16(ik[i]);
  }
  if (bkeys) {
    const uint16_t *bk = (const uint16_t *)c.bkeys.host();
    for (size_t i = 0; i < (size_t)(n / RATIO) * IHD; i++) bkeys[i] = from_bf16(bk[i]);
  }
}

void truncate(QsaCache &c, int T) {
  NE_CHECK(T >= 0 && T <= c.T(), "truncate to %d > pos %d", T, c.T());
  ((uint32_t *)c.pos.host())[0] = (uint32_t)T;
}

}  // namespace ne::attn

// ------------------------------------------------------------------------------------------------ C entry points
using namespace ne;
using namespace ne::attn;

struct ne_attn {
  QsaW w;
  QsaCache c;
  Tensor q, kv, iqk, out;
  std::unique_ptr<Recorder> rec[MAX_ROWS + 1];
  int last_rows = 1;
  // prefill-chunk tests (ne_attn_chunk*)
  std::unique_ptr<ChunkScratch> cs;
  Tensor cq, ckv, ciqk, cout, cpos;
};

extern "C" {

NE_API ne_attn *ne_attn_create(ne_pack *pack, int layer, int fmt, int tmax) {
  ne_attn *a = nullptr;
  int rc = c_call([&] {
    const Pack *p = pack_from_c(pack);
    NE_CHECK(p, "null pack");
    NE_CHECK(fmt >= 0 && fmt <= 2, "fmt %d (0 f32, 1 bf16, 2 kv8)", fmt);
    auto u = std::make_unique<ne_attn>();
    u->w = load_qsa_weights(*p, layer);
    u->c = alloc_cache((KvFmt)fmt, tmax);
    u->q = alloc(DType::F32, {MAX_ROWS, QD});
    u->kv = alloc(DType::F32, {MAX_ROWS, 2 * KVD});
    u->iqk = alloc(DType::F32, {MAX_ROWS, IQD});
    u->out = alloc(DType::F32, {MAX_ROWS, NQH * HD});
    a = u.release();
  });
  return rc == 0 ? a : nullptr;
}

NE_API void ne_attn_free(ne_attn *a) { delete a; }

NE_API int ne_attn_import(ne_attn *a, const float *k, const float *v, const float *ikeys, int T) {
  return c_call([&] {
    NE_CHECK(a, "null handle");
    import_cache(a->c, a->w, k, v, ikeys, T);
  });
}
NE_API int ne_attn_export(ne_attn *a, int n, float *k, float *v, float *ikeys, float *bkeys) {
  return c_call([&] {
    NE_CHECK(a, "null handle");
    export_cache(a->c, n, k, v, ikeys, bkeys);
  });
}
NE_API int ne_attn_truncate(ne_attn *a, int T) {
  return c_call([&] {
    NE_CHECK(a, "null handle");
    truncate(a->c, T);
  });
}
NE_API int ne_attn_pos(ne_attn *a) { return a ? a->c.T() : -1; }

static Recorder &step_rec(ne_attn *a, int R) {
  NE_CHECK(R >= 1 && R <= MAX_ROWS, "rows %d (1..%d)", R, MAX_ROWS);
  if (!a->rec[R]) {
    a->rec[R] = std::make_unique<Recorder>();
    a->rec[R]->begin();
    record_step(*a->rec[R], a->w, a->c, R, a->q, a->kv, a->iqk, a->out, true);
    a->rec[R]->end();
  }
  return *a->rec[R];
}

NE_API int ne_attn_step(ne_attn *a, int R, const float *q, const float *kv, const float *iqk, float *out,
                        double *gpu_ms) {
  return c_call([&] {
    NE_CHECK(a, "null handle");
    Recorder &r = step_rec(a, R);
    NE_CHECK(a->c.T() + R <= a->c.tmax, "cache full (%d + %d > %d)", a->c.T(), R, a->c.tmax);
    if (q) upload(a->q, q, (size_t)R * QD * 4);
    if (kv) upload(a->kv, kv, (size_t)R * 2 * KVD * 4);
    if (iqk) upload(a->iqk, iqk, (size_t)R * IQD * 4);
    r.run();
    a->last_rows = R;
    if (out) download(a->out, out, (size_t)R * NQH * HD * 4);
    if (gpu_ms) *gpu_ms = r.gpu_ms();
  });
}

NE_API int ne_attn_read(ne_attn *a, const char *name, void *dst, int n) {
  int size = -1;
  int rc = c_call([&] {
    NE_CHECK(a && name, "null argument");
    const std::string s(name);
    const int R = a->last_rows;
    const Tensor *t = nullptr;
    if (s == "qr") { t = &a->c.qr; size = R * NQH * HD; }
    else if (s == "iq") { t = &a->c.iq; size = R * 4 * IHD; }
    else if (s == "scores") { t = &a->c.scores; size = R * (a->c.tmax / RATIO); }
    else if (s == "sel") { t = &a->c.sel; size = R * KB; }
    else if (s == "nsel") { t = &a->c.nsel; size = R; }
    else if (s == "out") { t = &a->out; size = R * NQH * HD; }
    NE_CHECK(t, "unknown name '%s'", name);
    NE_CHECK(n <= size, "%s: n %d > size %d", name, n, size);
    if (dst && n > 0) download(*t, dst, (size_t)n * 4);
  });
  return rc == 0 ? size : -1;
}

NE_API int ne_attn_write(ne_attn *a, const char *name, const void *src, int n) {
  return c_call([&] {
    NE_CHECK(a && name && src, "null argument");
    const std::string s(name);
    const Tensor *t = s == "sel" ? &a->c.sel : s == "nsel" ? &a->c.nsel : nullptr;
    NE_CHECK(t && n <= (int)t->numel(), "ne_attn_write: unknown '%s' or n %d too large", name, n);
    upload(*t, src, (size_t)n * 4);
  });
}

NE_API int ne_attn_time(ne_attn *a, int R, int which, int reps1, int reps2, int flush_mb, double *us) {
  return c_call([&] {
    NE_CHECK(a && reps2 > reps1 && reps1 >= 1, "bad arguments");
    NE_CHECK(a->c.T() + R <= a->c.tmax, "cache full");
    Tensor fl, flo;
    uint32_t n4 = 0;
    if (flush_mb > 0) {
      fl = alloc(DType::U8, {(int64_t)flush_mb << 20});
      n4 = (uint32_t)(((size_t)flush_mb << 20) / 16);
      flo = alloc(DType::U32, {1 << 16});
    }
    auto chain = [&](int reps, bool step, bool flush) {
      auto r = std::make_unique<Recorder>();
      r->begin();
      for (int i = 0; i < reps; i++) {
        if (step) {
          if (which < 0) record_step(*r, a->w, a->c, R, a->q, a->kv, a->iqk, a->out, false);
          else {
            record_one(*r, which, a->w, a->c, R, a->q, a->kv, a->iqk, a->out);
            r->barrier();
          }
        }
        if (flush) {
          r->dispatch(pso("ne_stream_read"), {fl, flo, Arg::value(n4)}, sz(2048), sz(256));
          r->barrier();
        }
      }
      r->end();
      return r;
    };
    auto best = [&](Recorder &r) {
      double b = 1e30;
      for (int i = 0; i < 5; i++) { r.run(); b = std::min(b, r.gpu_ms()); }
      return b;
    };
    const bool f = flush_mb > 0;
    auto a1 = chain(reps1, true, f), a2 = chain(reps2, true, f);
    best(*a1);  // warm-up
    const double t1 = best(*a1), t2 = best(*a2);
    double ff = 0;
    if (f) {
      auto f1 = chain(reps1, false, true), f2 = chain(reps2, false, true);
      ff = (best(*f2) - best(*f1)) * 1e3 / (reps2 - reps1);
    }
    *us = (t2 - t1) * 1e3 / (reps2 - reps1) - ff;
  });
}


static const int kChunkMax = 2048;
static void chunk_alloc(ne_attn *a) {
  if (a->cs) return;
  a->cs = std::make_unique<ChunkScratch>(alloc_chunk_scratch(a->c.fmt, a->c.tmax, kChunkMax));
  a->cq = alloc(DType::F32, {kChunkMax, QD});
  a->ckv = alloc(DType::F32, {kChunkMax, 2 * KVD});
  a->ciqk = alloc(DType::F32, {kChunkMax, IQD});
  a->cout = alloc(DType::F32, {kChunkMax, NQH * HD});
  a->cpos = alloc(DType::U32, {4});
}
static std::unique_ptr<Recorder> chunk_rec(ne_attn *a, int n, int mode, int dense) {
  auto r = std::make_unique<Recorder>();
  r->begin();
  record_chunk(*r, a->w, a->c, *a->cs, n, dense ? a->c.T() : -1, a->cpos, a->cq, a->ckv, a->ciqk, a->cout,
               mode ? 1 : 0);  // the mode as a parameter (was a setenv / unsetenv of NE_QSA_CHUNK: clobbered the caller's)
  r->end();
  return r;
}

NE_API int ne_attn_chunk(ne_attn *a, int n, const float *q, const float *kv, const float *iqk, float *out, int mode,
                         int dense, double *gpu_ms) {
  return c_call([&] {
    NE_CHECK(a && n >= 1 && n <= kChunkMax, "bad arguments");
    NE_CHECK(a->c.T() + n <= a->c.tmax, "cache full (%d + %d > %d)", a->c.T(), n, a->c.tmax);
    chunk_alloc(a);
    if (q) upload(a->cq, q, (size_t)n * QD * 4);
    if (kv) upload(a->ckv, kv, (size_t)n * 2 * KVD * 4);
    if (iqk) upload(a->ciqk, iqk, (size_t)n * IQD * 4);
    ((uint32_t *)a->cpos.host())[0] = (uint32_t)a->c.T();
    auto r = chunk_rec(a, n, mode, dense);
    r->run();
    if (out) download(a->cout, out, (size_t)n * NQH * HD * 4);
    if (gpu_ms) *gpu_ms = r->gpu_ms();
    ((uint32_t *)a->c.pos.host())[0] += (uint32_t)n;
  });
}

NE_API int ne_attn_chunk_sel(ne_attn *a, int n, uint32_t *sel, uint32_t *nsel) {
  return c_call([&] {
    NE_CHECK(a && a->cs && n >= 1 && n <= kChunkMax, "bad arguments (no chunk run yet?)");
    if (sel) download(a->cs->sel, sel, (size_t)n * KB * 4);
    if (nsel) download(a->cs->nsel, nsel, (size_t)n * 4);
  });
}

NE_API int ne_attn_chunk_time(ne_attn *a, int n, int mode, int part, int reps, double *ms) {
  return c_call([&] {
    NE_CHECK(a && n >= 1 && n <= kChunkMax && reps >= 1 && ms, "bad arguments");
    chunk_alloc(a);
    ((uint32_t *)a->cpos.host())[0] = (uint32_t)a->c.T();
    std::unique_ptr<Recorder> r;
    if (part == 0) r = chunk_rec(a, n, mode, 0);
    else {  // attention only, on the selection in the scratch
      r = std::make_unique<Recorder>();
      r->begin();
      if (mode) {
        r->dispatch(Pipelines::get("qsa_chunk_attn_bf16", {FC::b(0, ne::fn::g_ref_round)}),
                    {a->cs->qr, a->c.kc, a->c.vc, a->cs->sel, a->cs->nsel, a->cpos, Arg::value(0u),
                     Arg::value((uint32_t)a->c.tmax), a->cq, a->cout},
                    MTLSizeMake(n, NKV, 1), MTLSizeMake(128, 1, 1));
      } else {
        NE_CHECK(false, "part 1 needs mode 1");
      }
      r->end();
    }
    std::vector<double> t;
    r->run();  // warm-up
    for (int i = 0; i < reps; i++) { r->run(); t.push_back(r->gpu_ms()); }
    std::sort(t.begin(), t.end());
    *ms = t[t.size() / 2];
  });
}

}  // extern "C"
