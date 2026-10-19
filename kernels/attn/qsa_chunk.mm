// QSA attention core of a prefill chunk: host side (lab-attn). Kernels: kernels/attn/qsa.metal (prep / score /
// select per 8-row slice), kernels/attn/qsa_chunk.metal (all sparse rows at once), lab-prefill's
// kernels/prefill/pf_attn.metal + pf_trunk.metal (dense-row NAX flash attention, gate, slice positions).
#include "kernels/attn/qsa_chunk.h"

#include <cstring>
#include <string>

namespace ne::fn {
extern bool g_ref_round;
}
namespace ne::attn {

static id<MTLComputePipelineState> pso(const std::string &name) {
  return Pipelines::get(name, {FC::b(0, ne::fn::g_ref_round)});
}
static MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
static int cdiv(int a, int b) { return (a + b - 1) / b; }

QsaCache alloc_step_scratch(KvFmt fmt, int tmax) {
  NE_CHECK(tmax > 0 && tmax % 64 == 0, "tmax %d must be a positive multiple of 64", tmax);
  QsaCache c;
  c.fmt = fmt;
  c.tmax = tmax;
  c.scratch_tmax = tmax;
  c.qr = alloc(DType::F32, {MAX_ROWS, NQH, HD});
  c.iq = alloc(DType::F32, {MAX_ROWS, 4, IHD});
  c.scores = alloc(DType::F32, {MAX_ROWS, tmax / RATIO});
  c.sel = alloc(DType::U32, {MAX_ROWS, KB});
  c.nsel = alloc(DType::U32, {MAX_ROWS});
  c.cand = alloc(DType::U32, {MAX_ROWS, (tmax / RATIO + SLICE - 1) / SLICE, KB});
  c.part = alloc(DType::F32, {NKV, MAX_ROWS, NCH, 16, HD + 2});
  return c;
}

ChunkScratch alloc_chunk_scratch(KvFmt fmt, int tmax, int C) {
  NE_CHECK(C >= 8 && C % 8 == 0, "chunk %d must be a positive multiple of 8", C);
  ChunkScratch s;
  s.fmt = fmt;
  s.C = C;
  s.tmax = tmax;
  for (int j = 0; j < kChunkSlots; j++) s.slot.push_back(alloc_step_scratch(fmt, tmax));
  s.pslot = alloc(DType::U32, {kChunkSlots, 4});
  s.qr = alloc(DType::F32, {C + 8, NQH, HD});
  s.sel = alloc(DType::U32, {C + 8, KB});
  s.nsel = alloc(DType::U32, {C + 8});
  if (fmt == KvFmt::BF16) {
    s.qb = alloc(DType::BF16, {C, NQH * HD});
    s.ob = alloc(DType::BF16, {C, NQH * HD});
    s.iq = alloc(DType::F32, {C + 8, 4, IHD});
    s.scores = alloc(DType::F32, {C + 8, tmax / RATIO});
    s.cand = alloc(DType::U32, {C + 8, (tmax / RATIO + SLICE - 1) / SLICE, KB});
    s.spos = alloc(DType::U32, {C / 8 + 1, 4});
  }
  return s;
}

bool chunk_core_on(KvFmt fmt, int force) {
  if (force == 0) return false;
  const char *e = force < 0 ? getenv("NE_QSA_CHUNK") : nullptr;
  if (e && std::string(e) == "0") return false;
  return fmt == KvFmt::BF16;
}

int chunk_dense_rows(const ChunkScratch &s, int T0, int n) {
  if (T0 < 0 || s.fmt != KvFmt::BF16) return 0;
  const int nd = std::max(0, std::min(n, 4 * KB + 3 - T0));  // rows with (pos + 1) / 4 <= 512
  return nd / MAX_ROWS * MAX_ROWS;
}

// MLX AttnParams (kernels/prefill/mlx/attn_params.h)
struct NaxAttnParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

void record_chunk(Recorder &r, const QsaW &w, const QsaCache &c, ChunkScratch &s, int n, int T0,
                  const Tensor &posbase, const Tensor &q, const Tensor &kv, const Tensor &iqk, const Tensor &out,
                  int core_mode, const std::function<Recorder &(const char *)> *phase) {
  // phase (profiling only, pfkern's NE_PF_PROF=fine): the recorder per sub-step label
  auto PH = [&](const char *ph) -> Recorder & { return phase ? (*phase)(ph) : r; };
  NE_CHECK(n >= 1 && n <= s.C, "chunk rows %d (1..%d)", n, s.C);
  NE_CHECK(s.fmt == c.fmt && s.tmax >= c.tmax, "chunk scratch fmt / tmax do not match the cache");
  const bool core = chunk_core_on(c.fmt, core_mode);
  const int ns = cdiv(n, MAX_ROWS);
  const int ndr = chunk_dense_rows(s, T0, n), nds = ndr / MAX_ROWS;
  const size_t qrow = (size_t)NQH * HD * 4;
  if (core) {
    // batched: one prep for every row (cache append, indexer queries, raw keys), the chunk's block keys, then the
    // scores of every sparse 8-row slice, select_local and select over every sparse row (each one dispatch)
    const uint32_t un = (uint32_t)n, tm = (uint32_t)c.tmax;
    PH("attn_prep").dispatch(pso("pf_set_pos_multi"), {s.spos, posbase, Arg::value(0u), Arg::value((uint32_t)ns)}, sz(cdiv(ns, 32)),
               sz(32));
    PH("attn_prep").dispatch(pso(std::string("qsa_chunk_prep_") + kv_fmt_name(c.fmt)),
               {q, kv, iqk, w.qnorm, w.knorm, w.iqnorm, w.iknorm, c.invf, c.kc, c.vc, c.ksb, c.vsb, c.ikeys, c.bkeys,
                posbase, Arg::value(tm), s.qr, s.iq, Arg::value(un)},
               sz(n * (NQH + NKV + 4)), sz(32));
    PH("attn_prep").barrier();
    PH("attn_blocks").dispatch(pso("qsa_chunk_blocks"), {c.ikeys, w.iknorm, c.invf, c.bkeys, posbase, Arg::value(un)}, sz(n / RATIO + 1),
               sz(32));
    PH("attn_blocks").barrier();
    // pfkern: rows with <= 8192 complete blocks select straight from their scores (NE_QSA_DSEL=0: per-slice pass)
    static const bool dsel = !getenv("NE_QSA_DSEL") || atoi(getenv("NE_QSA_DSEL")) != 0;
    if (nds < ns) {
      const int nsp = ns - nds, nsl = (c.tmax / RATIO + SLICE - 1) / SLICE;
      const int G = std::max(1, std::min(c.tmax / RATIO / 64, 16));
      static const bool tsc = !getenv("NE_QSA_TSCORE") || atoi(getenv("NE_QSA_TSCORE")) != 0;  // pfkern: tensor op
      PH("attn_score").dispatch(pso(tsc ? "qsa_chunk_score_t" : "qsa_chunk_score"),
                                {s.iq, c.bkeys, s.spos, s.scores, Arg::value(tm), Arg::value((uint32_t)nds), Arg::value(un)},
                                MTLSizeMake(G, nsp, 1), sz(tsc ? 128 : 256));
      PH("attn_score").barrier();
      const Tensor p0 = s.spos.view((size_t)nds * 16, DType::U32, {4});  // posbase + ndr
      const Arg sc = Arg(s.scores.buf, s.scores.offset + (size_t)ndr * (c.tmax / RATIO) * 4);
      const Arg cd = Arg(s.cand.buf, s.cand.offset + (size_t)ndr * nsl * KB * 4);
      PH("attn_selloc").dispatch(pso(dsel ? "qsa_select_local_r8_d" : "qsa_select_local_r8"), {sc, p0, cd, Arg::value(tm)}, MTLSizeMake(n - ndr, nsl, 1), sz(1024));
      PH("attn_selloc").barrier();
      PH("attn_select").dispatch(pso(dsel ? "qsa_select_r8_d" : "qsa_select_r8"),
                 {sc, p0, Arg(s.sel.buf, s.sel.offset + (size_t)ndr * KB * 4), Arg(s.nsel.buf, s.nsel.offset + (size_t)ndr * 4),
                  Arg::value(tm), cd},
                 sz(n - ndr), sz(1024));
    }
  } else {
    // per group of kChunkSlots slices: the preps in order (cache append, block keys may span slices), then score /
    // select_local / select (and without the core: attn / merge) for all slices at once, each on its own position and
    // step scratch; qr / sel / nsel are views of the chunk-wide buffers.
    for (int g0 = 0; g0 < ns; g0 += kChunkSlots) {
      const int gs = std::min(kChunkSlots, ns - g0);
      PH("attn_step").dispatch(pso("pf_set_pos_multi"),
                 {s.pslot, posbase, Arg::value((uint32_t)(g0 * MAX_ROWS)), Arg::value((uint32_t)gs)}, sz(1), sz(32));
      PH("attn_step").barrier();
      std::vector<QsaCache> cs(gs);
      std::vector<int> R(gs), r0(gs);
      for (int j = 0; j < gs; j++) {
        QsaCache x = c;
        const QsaCache &d = s.slot[j];
        r0[j] = (g0 + j) * MAX_ROWS;
        R[j] = std::min(MAX_ROWS, n - r0[j]);
        x.iq = d.iq; x.scores = d.scores; x.cand = d.cand; x.part = d.part;
        x.qr = s.qr.view((size_t)r0[j] * qrow, DType::F32, {MAX_ROWS, NQH, HD});
        x.sel = s.sel.view((size_t)r0[j] * KB * 4, DType::U32, {MAX_ROWS, KB});
        x.nsel = s.nsel.view((size_t)r0[j] * 4, DType::U32, {MAX_ROWS});
        x.scratch_tmax = d.scratch_tmax;
        x.pos = s.pslot.view((size_t)j * 16, DType::U32, {4});
        cs[j] = x;
      }
      const int last = core ? 4 : kNumDispatches;  // the core replaces attn (4) + merge (5)
      for (int which = 0; which < last; which++) {
        bool any = false;
        for (int j = 0; j < gs; j++) {
          if (which > 0 && g0 + j < nds) continue;  // dense slices: prep only
          record_one(PH("attn_step"), which, w, cs[j], R[j], q.rows(r0[j], R[j]), kv.rows(r0[j], R[j]), iqk.rows(r0[j], R[j]),
                     out.rows(r0[j], R[j]));
          any = true;
          if (which == 0) PH("attn_step").barrier();  // preps in order
        }
        if (which > 0 && any) PH("attn_step").barrier();
      }
    }
  }
  PH(nds < ns ? "attn_select" : "attn_blocks").barrier();
  if (nds > 0) {  // dense rows: queries -> bf16, MLX NAX flash attention over the cache, gate (after the barrier)
    const uint32_t cnt = (uint32_t)ndr * NQH * HD;
    PH("attn_dense").dispatch(pso("pf_attn_qcopy"), {s.qr, s.qb, Arg::value(0u), Arg::value(cnt)}, sz(cdiv((int)cnt, 256)), sz(256));
    PH("attn_dense").barrier();
    const int qL = ndr, kL = T0 + ndr, tq = c.tmax;
    NaxAttnParams ap{};
    ap.B = 1; ap.H = NQH; ap.D = HD; ap.qL = qL; ap.kL = kL; ap.gqa_factor = NQH / NKV;
    ap.scale = 1.0f;  // the prepped q carries the 1/16
    ap.NQ = cdiv(qL, 64); ap.NK = cdiv(kL, 32); ap.NQ_aligned = qL / 64; ap.NK_aligned = kL / 32;
    ap.qL_rem = qL - ap.NQ_aligned * 64; ap.kL_rem = kL - ap.NK_aligned * 32; ap.qL_off = kL - qL;
    const int64_t qs[3] = {(int64_t)s.C * NQH * HD, HD, NQH * HD}, ks[3] = {0, (int64_t)tq * HD, HD};
    for (int i = 0; i < 3; i++) { ap.Q_strides[i] = ap.O_strides[i] = qs[i]; ap.K_strides[i] = ap.V_strides[i] = ks[i]; }
    auto ps = Pipelines::get("pf_attn_nax_d256", {FC::b(0, ne::fn::g_ref_round), FC::b(200, qL % 64 == 0),
                                                  FC::b(201, kL % 32 == 0), FC::b(300, false), FC::b(301, true),
                                                  FC::b(302, false)});
    PH("attn_dense").dispatch(ps, {s.qb, c.kc, c.vc, s.ob, Arg::value(ap)}, MTLSizeMake(ap.NQ, NQH, 1), sz(256));
  }
  if (core && ndr < n)  // sparse rows, concurrently with the dense flash attention (disjoint outputs)
    PH("attn_sparse").dispatch(pso("qsa_chunk_attn_bf16"),
               {s.qr, c.kc, c.vc, s.sel, s.nsel, posbase, Arg::value((uint32_t)ndr), Arg::value((uint32_t)c.tmax), q,
                out},
               MTLSizeMake(n - ndr, NKV, 1), sz(128));
  if (nds > 0) {
    PH("attn_dense").barrier();
    const uint32_t cnt = (uint32_t)ndr * NQH * HD;
    PH("attn_dense").dispatch(pso("pf_attn_gate"), {s.ob, q, out, Arg::value(cnt)}, sz(cdiv((int)cnt, 256)), sz(256));
  }
  PH("attn_sparse").barrier();
}

void record_chunk_kv(Recorder &r, const QsaW &w, const QsaCache &c, ChunkScratch &s, int n, const Tensor &posbase,
                     const Tensor &q, const Tensor &kv, const Tensor &iqk) {
  NE_CHECK(n >= 1 && n <= s.C, "chunk rows %d (1..%d)", n, s.C);
  NE_CHECK(s.fmt == c.fmt && s.tmax >= c.tmax, "chunk scratch fmt / tmax do not match the cache");
  NE_CHECK(chunk_core_on(c.fmt, -1), "record_chunk_kv: the batched prep needs the multi-row format");
  const uint32_t un = (uint32_t)n, tm = (uint32_t)c.tmax;
  // = record_chunk's first two dispatches (core path)
  r.dispatch(pso(std::string("qsa_chunk_prep_") + kv_fmt_name(c.fmt)),
             {q, kv, iqk, w.qnorm, w.knorm, w.iqnorm, w.iknorm, c.invf, c.kc, c.vc, c.ksb, c.vsb, c.ikeys, c.bkeys,
              posbase, Arg::value(tm), s.qr, s.iq, Arg::value(un)},
             sz(n * (NQH + NKV + 4)), sz(32));
  r.barrier();
  r.dispatch(pso("qsa_chunk_blocks"), {c.ikeys, w.iknorm, c.invf, c.bkeys, posbase, Arg::value(un)}, sz(n / RATIO + 1),
             sz(32));
  r.barrier();
}

}  // namespace ne::attn
