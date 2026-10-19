// Flash-Next chunked prefill (lab-prefill). Kernels: kernels/prefill/*.metal; design and numbers: docs/prefill.md.
//
// One recording per chunk length n (cached): embedding -> 48 layers -> nothing (no head). Per layer:
//   [PLE, layer 1: host-gathered n-gram rows -> key/value GEMMs -> gate rows -> dilated conv]
//   HC read: pf_hc_norm (per-row rms, normed bf16, inject gates) -> down GEMM -> silu(/4) -> up GEMM -> block_in
//   GDN: qkv / z / b / a GEMMs -> conv1d + silu (all rows at once), gates, q-k l2 norm -> pf_gdn_scan (the chunk's
//        recurrence, state in registers) -> gated norm -> out_proj GEMM -> HC write
//   attention: q / k / v / indexer GEMMs (f32 out) -> lab-attn's QSA step per 8-row slice (pos set on the GPU per
//        slice) -> o_proj GEMM -> HC write
//   MoE: router GEMM (bf16) -> top-10 per row + shared gate logit; shared gate / up GEMMs -> SwiGLU -> down GEMM;
//        GPU sort by expert (count / scan / place) -> prep -> NAX trellis GEMMs gate, up -> mid -> down GEMM ->
//        finish (H128 * svh * w, fixed slot order, + gated shared, HC write)
#include "models/flash_next/prefill.h"

#include "kernels/attn/qsa_chunk.h"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace ne::fn {

const char *const kPfStageNames[(int)PfStage::N] = {"embed", "ple", "hc_glue", "trunk_gemm", "gdn", "attn", "moe_glue", "moe_gemm"};

struct PfGemm {  // = pf_gemm.metal PfGemm
  int K, N, M, ldx, ldy;
};

struct Prefill {
  Model *m = nullptr;
  int C = 0;
  bool profile = false;
  bool fine = false;  // NE_PF_PROF=fine: one command buffer per kernel label (pfkern; implies profile)
  // chunk buffers (rows = C)
  Tensor tok, h[2], hm, hp;
  Tensor normed, inj, down, up, xin;               // HC read
  Tensor raw, z, araw, braw, conv, gb, qk, y, o, br;  // GDN (+ the bf16 branch of either mixer)
  Tensor aq, akv, aiqk, aout;                      // attention (f32, lab-attn's layouts)
  const char *scan = "pf_gdn_scan_pipe";  // NE_PF_SCAN=tb: pf_gdn_scan_tb (decode op order); =plain: pf_gdn_scan
  bool no_dense = false;                           // NE_PF_DENSE=0: every row through lab-attn's step
  Tensor logits, route, sgl, gu, sact, sho;        // MoE routing + shared expert
  Tensor cnt, rank, off, wtab, order, seid, inv, pg, pu, ig, iu, pd, dn;  // routed experts
  Tensor pemb_f, pemb, pkv, pgv, pnrm;             // PLE
  Tensor posbase;                                  // u32 [4]: T before the chunk
  Tensor dfl_fs;                                   // lab-dflash: f32 [2048][12800] feature ring by position (or empty)
  t6::PfBuf t6pf;                                  // lab-t6wire: EXL3 trunk (allocated when the model has t6 layers)
  attn::ChunkScratch ac;                           // lab-attn's chunk attention scratch (kernels/attn/qsa_chunk.h)
  bool shared = false;  // chunk buffers lent by another model's Prefill (prefill_create donor)
  bool hcf = true;     // NE_PF_HCF=0 off: HC write folded into the next norm, SiLU in the HC down GEMM epilogue (4b)
  bool moe_h = true;   // NE_PF_MOEH=0 off: f16 expert GEMM output planes (Sushi's nax_store to half; docs/prefill.md 4b)
  int maxwin = 0, win = 32;  // MoE GEMM windows: rows per window (32 or 64, NE_PF_WIN), max count
  // recordings by chunk length; profile mode: one recorder per (layer, stage) segment
  struct Rec {
    std::vector<std::pair<PfStage, std::unique_ptr<Recorder>>> parts;
    std::vector<std::string> labels;  // per part (fine profile)
    size_t dispatches = 0;
    uint64_t used = 0;  // LRU stamp
  };
  std::map<int64_t, std::unique_ptr<Rec>> recs;  // by (n, T0) for chunks with dense rows, (n, -1) else
  uint64_t use_ctr = 0;
  size_t max_recs = 32;  // LRU cap (a server sees every suffix length; lab-server's request); NE_PF_MAX_RECS
  PfStats stats;
  // lab-pfpath: n-gram rows gathered ahead for the next call's first chunk (prefill_tokens' `ahead`): history before
  // them, their tokens, the rows (f32 [n][HID]). NE_PF_AHEAD=0: never gathered ahead.
  struct Ahead {
    int64_t h2 = -1, h1 = -1;
    std::vector<int> toks;
    std::vector<float> emb;
  } ahead;
  bool ahead_on = true;
  // lab-pfpath: the MTP head's KV-only chunk (prefill_head_kv): recordings by rows, own small buffers (the rest are
  // the chunk buffers, free between chunks)
  std::map<std::pair<uintptr_t, int>, std::unique_ptr<Rec>> head_recs;  // by (the head cache's K buffer, rows): a
                                                                      // model may carry several MTP loops
  Tensor hk_tok, hk_en, hk_pe, hk_ones, hk_pos;
};

static MTLSize sz1(int n) { return MTLSizeMake(n, 1, 1); }
static int cdiv(int a, int b) { return (a + b - 1) / b; }
static Arg at(const Tensor &t, size_t byte_off = 0) { return Arg(t.buf, t.offset + byte_off); }

// Segmented recorder: in profile mode each stage change starts a new command buffer (queue-ordered).
// Fine profile (NE_PF_PROF=fine): a new command buffer at every label change as well.
struct Seg {
  Prefill::Rec &rec;
  bool split;
  bool fine = false;
  PfStage cur = PfStage::N;
  std::string lab;
  Recorder &r(PfStage s, const std::string &l = "") {
    const std::string nl = l.empty() ? kPfStageNames[(int)s] : l;
    if (rec.parts.empty() || (split && (s != cur || (fine && nl != lab)))) {
      if (!rec.parts.empty()) rec.parts.back().second->end();
      rec.parts.emplace_back(s, std::make_unique<Recorder>());
      rec.labels.push_back(nl);
      rec.parts.back().second->begin();
    }
    cur = s;
    lab = nl;
    return *rec.parts.back().second;
  }
};
static Seg *g_seg = nullptr;  // the recording in progress (fine profile: T6 Linear phases get their own labels)

// y[M][w.rows] (ld ldy) = x[M][w.in] (ld ldx) @ W^T; x / y f32 (xf / yf) or bf16
static t6::PfBuf *g_t6pf = nullptr;  // the prefill being recorded (lab-t6wire: EXL3 trunk buffers)
static void gemm(Recorder &r, const Q8 &w, const Arg &x, bool xf, int ldx, const Arg &y, bool yf, int ldy, int M,
                 bool act = false) {  // act: pf_hc_act in the epilogue (bf16 in / out, rows % 64 == 0, q8 only)
  if (const char *sk = getenv("NE_PF_SKIP"))  // lab-ttft timing experiments only (wrong outputs): skip Linears by name
    if (w.t6 && *sk) {
      std::string list = std::string(",") + sk + ",";
      for (size_t a = 1, b; (b = list.find(',', a)) != std::string::npos; a = b + 1)
        if (b > a && w.t6->name.find(list.substr(a, b - a)) != std::string::npos) return;
    }
  if (w.t6) {  // lab-t6wire: EXL3 trunk Linear (dense W' decode + NAX GEMM, kernels/trunk6/trunk6.h)
    NE_CHECK(g_t6pf && g_t6pf->xh.buf, "prefill: EXL3 Linear %s but no t6 buffers", w.t6->name.c_str());
    if (g_seg && g_seg->fine) {
      const std::string base = g_seg->lab.substr(0, g_seg->lab.find('/'));
      const PfStage stg = g_seg->cur;
      const t6::PfPhase ph = [base, stg](const char *s) -> Recorder & { return g_seg->r(stg, base + "/" + s); };
      t6::record_prefill(r, *w.t6, M, x, xf, ldx, y, yf, ldy, *g_t6pf, &ph);
    } else {
      t6::record_prefill(r, *w.t6, M, x, xf, ldx, y, yf, ldy, *g_t6pf);
    }
    return;
  }
  NE_CHECK(w.in % 64 == 0 && (w.group == 64 || w.group == 128), "pf gemm: in %d group %d", w.in, w.group);
  const std::string name = std::string("pf_q8_") + (xf ? "f" : "b") + (yf ? "f" : "b") + "_g" +
                           std::to_string(w.group) + (w.rows % 64 ? "_u" : "") + (act ? "_act" : "");
  NE_CHECK(!act || (!xf && !yf && w.rows % 64 == 0), "pf gemm act: bf16 in / out, aligned rows only");
  PfGemm p{w.in, w.rows, M, ldx, ldy};
  r.dispatch(pso(name, 128), {w.w, w.s, w.b, x, y, Arg::value(p)}, MTLSizeMake(cdiv(w.rows, 64), cdiv(M, 64), 1),
             sz1(128));
}
static void gemm_bf16(Recorder &r, const Tensor &w, const Arg &x, const Arg &y, int N, int K, int M) {
  PfGemm p{K, N, M, K, N};
  r.dispatch(pso("pf_bf16_bb", 128), {w, x, y, Arg::value(p)}, MTLSizeMake(N / 64, cdiv(M, 64), 1), sz1(128));
}
static void pointwise(Recorder &r, const char *fn, std::vector<Arg> args, uint32_t cnt) {
  args.push_back(Arg::value(cnt));
  r.dispatch(pso(fn, 256), args, sz1(cdiv((int)cnt, 256)), sz1(256));
}

// Every dispatch asks the Seg for its recorder by (stage, label): one recorder without profile, a command buffer per
// stage with profile, per label with NE_PF_PROF=fine. bar(sg) = barrier after the dispatches recorded so far.
static void bar(Seg &sg) { sg.rec.parts.back().second->barrier(); }

// HC read of hw on h (f32 [n][4][2560]) -> xout bf16 [n][2560], inj f32 [n][8]. wr_h (p.hcf): h = the HC write of
// the mixer branch p.br onto *wr_h (pf_hc_write), done by the norm kernel itself (pf_hc_norm_wr stores it in h).
static void record_hc(Prefill &p, Seg &sg, const HcW &hw, const Tensor &h, int n, const Tensor *wr_h = nullptr) {
  if (wr_h)
    sg.r(PfStage::HcGlue, "hc_norm").dispatch(pso("pf_hc_norm_wr", 256), {*wr_h, p.br, p.inj, h, hw.norm, hw.inject, p.normed}, sz1(n), sz1(256));
  else
    sg.r(PfStage::HcGlue, "hc_norm").dispatch(pso("pf_hc_norm", 256), {h, hw.norm, hw.inject, p.normed, p.inj}, sz1(n), sz1(256));
  bar(sg);
  const bool act = p.hcf && !hw.down.t6 && hw.down.rows % 64 == 0;
  gemm(sg.r(PfStage::TrunkGemm, "hc_down"), hw.down, at(p.normed), false, HCD, at(p.down), false, LR, n, act);
  bar(sg);
  if (!act) {
    pointwise(sg.r(PfStage::HcGlue, "hc_act"), "pf_hc_act", {p.down}, (uint32_t)n * LR);
    bar(sg);
  }
  if (hw.up.t6) {  // pfkern: HC up + block_in in one GEMM epilogue (t6_fold_hcup), no [n][10240] up plane
    Recorder &g = sg.r(PfStage::TrunkGemm, "hc_up");
    t6::PfPhase ph;
    if (p.fine) ph = [](const char *s) -> Recorder & { return g_seg->r(PfStage::TrunkGemm, std::string("hc_up/") + s); };
    if (t6::record_prefill_hcup(g, *hw.up.t6, n, at(p.down), at(p.normed), at(p.xin), p.t6pf, p.fine ? &ph : nullptr)) {
      bar(sg);
      return;
    }
  }
  gemm(sg.r(PfStage::TrunkGemm, "hc_up"), hw.up, at(p.down), false, LR, at(p.up), false, HCD, n);
  bar(sg);
  pointwise(sg.r(PfStage::HcGlue, "hc_blockin"), "pf_hc_blockin", {p.up, p.normed, p.xin}, (uint32_t)n * HID);
  bar(sg);
}
static void record_hc_write(Prefill &p, Seg &sg, const Tensor &h, const Tensor &hout, int n) {
  pointwise(sg.r(PfStage::HcGlue, "hc_write"), "pf_hc_write", {h, p.br, p.inj, hout}, (uint32_t)n * HID);
  bar(sg);
}

static void record_gdn(Prefill &p, Seg &sg, const LayerW &w, GdnState &st, int n) {
  const GdnW &g = w.gdn;
  gemm(sg.r(PfStage::TrunkGemm, "gdn_qkv"), g.qkv, at(p.xin), false, HID, at(p.raw), false, CONVD, n);
  gemm(sg.r(PfStage::TrunkGemm, "gdn_z"), g.z, at(p.xin), false, HID, at(p.z), false, VALD, n);
  gemm(sg.r(PfStage::TrunkGemm, "gdn_ba"), g.b, at(p.xin), false, HID, at(p.braw), false, NVH, n);
  gemm(sg.r(PfStage::TrunkGemm, "gdn_ba"), g.a, at(p.xin), false, HID, at(p.araw), false, NVH, n);
  bar(sg);
  const uint32_t un = (uint32_t)n;
  sg.r(PfStage::Gdn, "gdn_conv").dispatch(pso("pf_gdn_conv", 256), {p.raw, g.conv, st.conv, p.conv, Arg::value(un)}, sz1(cdiv(n * CONVD, 256)),
             sz1(256));
  pointwise(sg.r(PfStage::Gdn, "gdn_conv"), "pf_gdn_gates", {p.araw, p.braw, g.alog, g.dtb, p.gb}, un * NVH);
  bar(sg);
  sg.r(PfStage::Gdn, "gdn_conv").dispatch(pso("pf_gdn_convstate", 256), {p.raw, st.conv, Arg::value(un)}, sz1(CONVD / 256), sz1(256));
  sg.r(PfStage::Gdn, "gdn_qknorm").dispatch(pso("pf_gdn_qknorm", 256), {p.conv, p.qk, Arg::value(un)}, sz1(cdiv(n * 2 * NKH, 8)), sz1(256));
  bar(sg);
  if (std::string(p.scan) == "pf_gdn_scan_pipe")  // 16 value rows per threadgroup of 128 (docs/prefill.md 4b)
    sg.r(PfStage::Gdn, "gdn_scan").dispatch(pso(p.scan, 128), {p.qk, p.conv, p.gb, st.ssm, p.y, Arg::value(un)}, MTLSizeMake(DV / 16, NVH, 1), sz1(128));
  else
    sg.r(PfStage::Gdn, "gdn_scan").dispatch(pso(p.scan, 256), {p.qk, p.conv, p.gb, st.ssm, p.y, Arg::value(un)}, sz1(NVH * 4), sz1(256));
  bar(sg);
  sg.r(PfStage::Gdn, "gdn_gnorm").dispatch(pso("pf_gdn_gnorm", 256), {p.y, p.z, g.norm, p.o, Arg::value(un)}, sz1(cdiv(n * NVH, 8)), sz1(256));
  bar(sg);
  gemm(sg.r(PfStage::TrunkGemm, "gdn_out"), g.out, at(p.o), false, VALD, at(p.br), false, HID, n);
  bar(sg);
}

// rows [0, nd) of the chunk are dense (their QSA selection is every block: complete blocks (pos + 1) / 4 <= 512)

static void record_attn(Prefill &p, Seg &sg, const LayerW &w, GdnState &st, int n, int T0) {
  const AttnW &a = w.attn;
  NE_CHECK(st.qc.tmax > 0, "prefill needs lab-attn's QSA caches (NE_FN_ATTN=dense is not supported)");
  gemm(sg.r(PfStage::TrunkGemm, "attn_q"), a.q, at(p.xin), false, HID, at(p.aq), true, AQD, n);
  gemm(sg.r(PfStage::TrunkGemm, "attn_kvi"), a.k, at(p.xin), false, HID, at(p.akv), true, 2 * AKVD, n);
  gemm(sg.r(PfStage::TrunkGemm, "attn_kvi"), a.v, at(p.xin), false, HID, at(p.akv, (size_t)AKVD * 4), true, 2 * AKVD, n);
  gemm(sg.r(PfStage::TrunkGemm, "attn_kvi"), a.iqk, at(p.xin), false, HID, at(p.aiqk), true, AIQD, n);
  bar(sg);
  // lab-attn's chunk attention core: cache append / indexer / selection per 8-row slice, dense rows on the NAX
  // flash attention, the other rows in one multi-row dispatch (kernels/attn/qsa_chunk.h)
  const std::function<Recorder &(const char *)> aph = [&sg](const char *l) -> Recorder & { return sg.r(PfStage::Attn, l); };
  attn::record_chunk(sg.r(PfStage::Attn, "attn_core"), w.qsa, st.qc, p.ac, n, p.no_dense ? -1 : T0, p.posbase, p.aq, p.akv,
                     p.aiqk, p.aout, -1, p.fine ? &aph : nullptr);
  gemm(sg.r(PfStage::TrunkGemm, "attn_o"), a.o, at(p.aout), true, AOD, at(p.br), false, HID, n);
  bar(sg);
}

static void record_moe(Prefill &p, Seg &sg, const LayerW &w, const Tensor &h, const Tensor &hout, int n) {
  const MoeW &m = w.moe;
  const uint32_t un = (uint32_t)n, S = un * TOPK;
  gemm_bf16(sg.r(PfStage::Moe, "moe_router"), m.router, at(p.xin), at(p.logits), NE, HID, n);
  gemm(sg.r(PfStage::Moe, "moe_shared_gu"), m.sg, at(p.xin), false, HID, at(p.gu), false, 2 * INTER, n);
  gemm(sg.r(PfStage::Moe, "moe_shared_gu"), m.su, at(p.xin), false, HID, at(p.gu, (size_t)INTER * 2), false, 2 * INTER, n);
  bar(sg);
  sg.r(PfStage::Moe, "moe_route").dispatch(pso("pf_moe_route", 256), {p.logits, p.xin, m.sgate, p.route, p.sgl}, sz1(n), sz1(256));
  pointwise(sg.r(PfStage::Moe, "moe_route"), "pf_shared_act", {p.gu, p.sact}, un * INTER);
  bar(sg);
  sg.r(PfStage::Moe, "moe_sort").dispatch(pso("pf_moe_count", 256), {p.route, p.cnt, p.rank, Arg::value(un)}, sz1(NE), sz1(256));
  gemm(sg.r(PfStage::Moe, "moe_shared_dn"), m.sd, at(p.sact), false, INTER, at(p.sho), false, HID, n);
  bar(sg);
  sg.r(PfStage::Moe, "moe_sort").dispatch(pso("pf_moe_scan", 512), {p.cnt, p.off, p.wtab, Arg::value((uint32_t)p.maxwin), Arg::value((uint32_t)p.win)}, sz1(1), sz1(512));
  bar(sg);
  sg.r(PfStage::Moe, "moe_sort").dispatch(pso("pf_moe_place", 256), {p.route, p.off, p.rank, p.order, p.seid, p.inv, Arg::value(S)},
             sz1(cdiv((int)S, 256)), sz1(256));
  bar(sg);
  sg.r(PfStage::Moe, "moe_prep").dispatch(pso("pf_moe_prep", 32), {p.xin, m.gate.suh, m.up.suh, p.order, p.seid, p.pg, p.pu},
             MTLSizeMake(HID / 128, S, 1), sz1(32));
  bar(sg);
  const std::string ks = "_k" + std::to_string(m.gate.k);
  std::string ws = "_w" + std::to_string(p.win) + (p.moe_h ? "_h" : "");
  const std::string hs = p.moe_h ? "_h" : "";
  NE_CHECK(m.up.k == m.gate.k, "gate / up K differ");
  static const std::string dbg = getenv("NE_PF_MOEDBG") && *getenv("NE_PF_MOEDBG") ? std::string("_") + getenv("NE_PF_MOEDBG") : "";  // timing only
  ws += dbg;
  sg.r(PfStage::MoeGemm, "moe_gate").dispatch(pso("pf_moe_gemm_gu" + ks + ws, 128), {p.pg, m.gate.t, p.wtab, p.ig}, MTLSizeMake(INTER / 128, p.maxwin, 1),
             sz1(128));
  sg.r(PfStage::MoeGemm, "moe_up").dispatch(pso("pf_moe_gemm_gu" + ks + ws, 128), {p.pu, m.up.t, p.wtab, p.iu}, MTLSizeMake(INTER / 128, p.maxwin, 1),
             sz1(128));
  bar(sg);
  sg.r(PfStage::Moe, "moe_mid").dispatch(pso("pf_moe_mid" + hs, 32), {p.ig, p.iu, m.gate.svh, m.up.svh, m.down.suh, p.seid, p.pd},
             MTLSizeMake(INTER / 128, S, 1), sz1(32));
  bar(sg);
  sg.r(PfStage::MoeGemm, "moe_down").dispatch(pso("pf_moe_gemm_dn_k" + std::to_string(m.down.k) + ws, 128), {p.pd, m.down.t, p.wtab, p.dn},
              MTLSizeMake(HID / 128, p.maxwin, 1), sz1(128));
  bar(sg);
  sg.r(PfStage::Moe, "moe_finish").dispatch(pso("pf_moe_finish" + hs, 320), {p.dn, m.down.svh, p.route, p.inv, p.sgl, p.sho, h, p.inj, hout},
             MTLSizeMake(HID / 128, n, 1), sz1(32 * TOPK));
  bar(sg);
}

static void record_ple(Prefill &p, Seg &sg, const LayerW &w, GdnState &st, const Tensor &h, int n) {
  const PleW &q = w.ple;
  const uint32_t un = (uint32_t)n;
  pointwise(sg.r(PfStage::Ple), "pf_f2bf", {p.pemb_f, p.pemb}, un * HID);
  bar(sg);
  gemm(sg.r(PfStage::Ple), q.key, at(p.pemb), false, HID, at(p.pkv), false, PLE_KV, n);
  gemm(sg.r(PfStage::Ple), q.value, at(p.pemb), false, HID, at(p.pkv, (size_t)HCD * 2), false, PLE_KV, n);
  bar(sg);
  sg.r(PfStage::Ple).dispatch(pso("pf_ple_gate", 1024), {h, p.pkv, q.nk, q.nq, q.nc, p.pgv, p.pnrm}, sz1(n * HC), sz1(1024));
  bar(sg);
  sg.r(PfStage::Ple).dispatch(pso("pf_ple_conv", 256), {h, p.pgv, p.pnrm, q.conv, st.ple_conv, p.hp, Arg::value(un)},
             sz1(cdiv(n * HCD, 256)), sz1(256));
  bar(sg);
  sg.r(PfStage::Ple).dispatch(pso("pf_ple_state", 256), {p.pnrm, st.ple_conv, Arg::value(un)}, sz1(HCD / 256), sz1(256));
  bar(sg);
}

static Prefill::Rec &get_rec(Prefill &p, int n, int T0) {
  const int64_t key = ((int64_t)n << 32) | (uint32_t)T0;
  auto it = p.recs.find(key);
  if (it != p.recs.end()) {
    it->second->used = ++p.use_ctr;
    return *it->second;
  }
  while (p.recs.size() >= p.max_recs) {  // evict the least recently used recording (no chunk is in flight here)
    auto lru = p.recs.begin();
    for (auto j = p.recs.begin(); j != p.recs.end(); ++j)
      if (j->second->used < lru->second->used) lru = j;
    p.recs.erase(lru);
  }
  auto rec = std::make_unique<Prefill::Rec>();
  rec->used = ++p.use_ctr;
  g_t6pf = &p.t6pf;
  Model &m = *p.m;
  Seg sg{*rec, p.profile, p.fine};
  g_seg = &sg;
  sg.r(PfStage::Embed).dispatch(pso("fn_embed_g" + std::to_string(m.embed.group)),
                                {p.tok, m.embed.w, m.embed.s, m.embed.b, p.h[0], m.vfeat}, sz1(n), sz1(256));
  bar(sg);
  size_t nl = m.L.size();
  if (const char *e = getenv("NE_PF_LAYERS")) nl = std::min(nl, (size_t)atoi(e));  // debug: first k layers only
  for (size_t l = 0; l < nl; l++) {
    const LayerW &w = m.L[l];
    GdnState &st = m.st[l];
    const Tensor &hin0 = p.h[l % 2], &hout = p.h[(l + 1) % 2];
    const Tensor *hin = &hin0;
    if (w.ple.on) {
      record_ple(p, sg, w, st, hin0, n);
      hin = &p.hp;
    }
    record_hc(p, sg, w.hca, *hin, n);
    if (w.linear) record_gdn(p, sg, w, st, n);
    else record_attn(p, sg, w, st, n, T0);
    if (p.hcf) {
      record_hc(p, sg, w.hcm, p.hm, n, hin);
    } else {
      record_hc_write(p, sg, *hin, p.hm, n);
      record_hc(p, sg, w.hcm, p.hm, n);
    }
    record_moe(p, sg, w, p.hm, hout, n);
    if (p.dfl_fs.buf && dfl_slot((int)l) >= 0) {  // lab-dflash: the last <= 2048 rows' features -> the ring by position
      const uint32_t r0 = (uint32_t)std::max(0, n - 2048);
      struct { uint32_t x, y, z, w; } a{(uint32_t)n - r0, (uint32_t)dfl_slot((int)l), r0, 0};
      sg.r(PfStage::HcGlue).dispatch(pso("dfl_hc_mean_ring"), {hout, p.dfl_fs, p.posbase, Arg::value(a)},
                                     sz1(cdiv((int)(n - r0) * HID, 256)), sz1(256));
    }
  }
  rec->parts.back().second->end();
  g_seg = nullptr;
  for (auto &pr : rec->parts) rec->dispatches += pr.second->size();
  Prefill::Rec &out = *rec;
  p.recs[key] = std::move(rec);
  return out;
}

// ---- lab-pfpath: the MTP head's KV-only rows of a chunk (prefill.h prefill_head_kv)
bool prefill_head_kv_ok(const Prefill &p, const GdnState &hst) {
  return hst.qc.tmax > 0 && p.ac.C == p.C && p.ac.fmt == hst.qc.fmt && p.ac.tmax >= hst.qc.tmax &&
         attn::chunk_core_on(hst.qc.fmt, -1);
}

void prefill_head_kv(Prefill &p, const HeadKvW &w, GdnState &hst, const int *next, int n, int pos) {
  NE_CHECK(n >= 1 && n <= p.C, "head kv rows %d (1..%d)", n, p.C);
  NE_CHECK(prefill_head_kv_ok(p, hst), "prefill_head_kv: head cache / chunk scratch mismatch");
  Model &m = *p.m;
  const int C = p.C;
  if (!p.hk_tok.buf) {
    p.hk_tok = alloc(DType::U32, {C});
    p.hk_en = alloc(DType::F32, {C, HID});
    p.hk_pe = alloc(DType::BF16, {C, HID});
    p.hk_ones = alloc(DType::F32, {C, 8});
    float *o = (float *)p.hk_ones.host();
    for (int i = 0; i < C * 8; i++) o[i] = 1.0f;
    p.hk_pos = alloc(DType::U32, {4});
  }
  const std::pair<uintptr_t, int> key{(uintptr_t)(__bridge void *)hst.qc.kc.buf + hst.qc.kc.offset, n};
  auto it = p.head_recs.find(key);
  if (it == p.head_recs.end()) {
    while (p.head_recs.size() >= p.max_recs) p.head_recs.erase(p.head_recs.begin());
    auto rec = std::make_unique<Prefill::Rec>();
    g_t6pf = &p.t6pf;
    Seg sg{*rec, false};
    const Tensor &hid = p.h[m.L.size() % 2];  // the chunk's layer-47 output streams (prefill_hidden)
    // hn and h_in in chunk buffers free between chunks: hm (the MoE residual), hp (the PLE output)
    const Tensor &hn = p.hm, &hin = p.hp;
    const uint32_t un = (uint32_t)n;
    {
      Recorder &r = sg.r(PfStage::TrunkGemm);
      r.dispatch(Pipelines::get("mtp_fuse_pre", {FC::b(0, g_ref_round)}),
                 {p.hk_tok, w.embed->w, w.embed->s, w.embed->b, w.norm_e, hid, w.norm_h, p.hk_en, hn,
                  Arg::value(w.vocab)},
                 sz1(n), sz1(256));
      r.barrier();
      gemm(r, *w.fc_e, at(p.hk_en), true, HID, at(p.hk_pe), false, HID, n);
      gemm(r, *w.fc_h, at(hn), true, HID, at(hin), true, HID, 4 * n);  // the 4n (token, stream) rows
      r.barrier();
      pointwise(r, "pf_hc_write", {hin, p.hk_pe, p.hk_ones, hin}, un * HID);  // + pe of each token (in place)
      r.barrier();
    }
    const LayerW &lw = *w.layer;
    record_hc(p, sg, lw.hca, hin, n);  // -> p.xin
    {
      const AttnW &a = lw.attn;
      Recorder &g = sg.r(PfStage::TrunkGemm);  // (no q: the prep's queries are not kept)
      gemm(g, a.k, at(p.xin), false, HID, at(p.akv), true, 2 * AKVD, n);
      gemm(g, a.v, at(p.xin), false, HID, at(p.akv, (size_t)AKVD * 4), true, 2 * AKVD, n);
      gemm(g, a.iqk, at(p.xin), false, HID, at(p.aiqk), true, AIQD, n);
      g.barrier();
      attn::record_chunk_kv(g, lw.qsa, hst.qc, p.ac, n, p.hk_pos, p.aq, p.akv, p.aiqk);
    }
    rec->parts.back().second->end();
    for (auto &pr : rec->parts) rec->dispatches += pr.second->size();
    it = p.head_recs.emplace(key, std::move(rec)).first;
  }
  uint32_t *tk = (uint32_t *)p.hk_tok.host();
  for (int i = 0; i < n; i++) tk[i] = (uint32_t)next[i];
  ((uint32_t *)p.hk_pos.host())[0] = (uint32_t)pos;
  for (auto &pr : it->second->parts) pr.second->commit();
  for (auto &pr : it->second->parts) pr.second->wait();
}

int prefill_touch(Prefill &p) {
  int k = 0;
  auto one = [&](Prefill::Rec &r) {
    for (auto &pr : r.parts) {
      pr.second->touch();
      k++;
    }
  };
  for (auto &e : p.recs) one(*e.second);
  for (auto &e : p.head_recs) one(*e.second);
  return k;
}

void PrefillDeleter::operator()(Prefill *p) const { delete p; }

PrefillPtr prefill_create(Model &m, int chunk, bool profile, const Prefill *donor) {
  NE_CHECK(chunk >= 8 && chunk % 8 == 0, "chunk %d must be a positive multiple of 8", chunk);
  NE_CHECK(m.L.size() >= 2 || (m.L.size() == 1 && !m.L[0].ple.on), "model not loaded");  // tests: first k layers
  PrefillPtr p(new Prefill());
  p->m = &m;
  p->C = chunk;
  p->profile = profile;
  if (const char *e = getenv("NE_PF_PROF")) p->fine = std::string(e) == "fine";
  if (p->fine) p->profile = true;
  if (const char *e = getenv("NE_PF_MAX_RECS")) p->max_recs = std::max(1, atoi(e));
  const int C = chunk, S = C * TOPK;
  if (const char *e = getenv("NE_PF_WIN")) p->win = atoi(e) == 64 ? 64 : 32;
  if (const char *e = getenv("NE_PF_MOEH")) p->moe_h = atoi(e) != 0;
  if (const char *e = getenv("NE_PF_HCF")) p->hcf = atoi(e) != 0;
  p->maxwin = cdiv(S, p->win) + NE;
  // geometry: the attention caches' window (chunk attention scratch) and the largest EXL3 trunk Linear
  int tq = 0;
  attn::KvFmt fmt = attn::KvFmt::BF16;
  for (auto &st : m.st)
    if (st.qc.tmax > 0) { tq = std::max(tq, st.qc.tmax); fmt = st.qc.fmt; }
  int64_t mx_in = 0, mx_w = 0, mx_out = 0;
  auto see = [&](const Q8 &q) {
    if (!q.t6) return;
    const int64_t np = (q.t6->out + 63) / 64 * 64;
    mx_in = std::max<int64_t>(mx_in, q.t6->in);
    mx_out = std::max<int64_t>(mx_out, np);
    mx_w = std::max<int64_t>(mx_w, np * q.t6->in);
  };
  for (const LayerW &w : m.L) {
    if (!w.t6) continue;
    for (const Q8 *q : {&w.hca.down, &w.hca.up, &w.hcm.down, &w.hcm.up, &w.moe.sg, &w.moe.su, &w.moe.sd, &w.gdn.qkv,
                        &w.gdn.z, &w.gdn.a, &w.gdn.b, &w.gdn.out, &w.attn.q, &w.attn.k, &w.attn.v, &w.attn.iqk,
                        &w.attn.o, &w.ple.key, &w.ple.value})
      see(*q);
  }
  // donor (lab-srvperf): another model's Prefill of the same geometry lends its chunk buffers (they hold nothing
  // between calls; prefill_tokens waits for its command buffers): one set for every model / slot of a server whose
  // engine thread runs one prefill at a time. Never share between models used from different threads at once.
  auto nb = [](const Tensor &t) { return t.buf ? (int64_t)t.nbytes() : 0; };
  const Prefill *sh = donor && donor != p.get() && donor->C == C && donor->ac.tmax == tq && donor->ac.fmt == fmt &&
                              nb(donor->t6pf.xh) == C * mx_in * 2 && nb(donor->t6pf.wd) == mx_w * 2 &&
                              nb(donor->t6pf.yraw) == C * mx_out * 4
                          ? donor : nullptr;
  p->shared = sh != nullptr;
  p->tok = sh ? sh->tok : alloc(DType::U32, {C});
  p->h[0] = sh ? sh->h[0] : alloc(DType::F32, {C, HCD});
  p->h[1] = sh ? sh->h[1] : alloc(DType::F32, {C, HCD});
  p->hm = sh ? sh->hm : alloc(DType::F32, {C, HCD});
  p->hp = sh ? sh->hp : alloc(DType::F32, {C, HCD});
  p->normed = sh ? sh->normed : alloc(DType::BF16, {C, HCD});
  p->inj = sh ? sh->inj : alloc(DType::F32, {C, 8});
  if (mx_in) {  // lab-t6wire: EXL3 trunk buffers (x prepared, W' dense, raw y) sized for the largest Linear
    p->t6pf.xh = sh ? sh->t6pf.xh : alloc(DType::F16, {(int64_t)C * mx_in});
    p->t6pf.wd = sh ? sh->t6pf.wd : alloc(DType::F16, {mx_w});
    p->t6pf.yraw = sh ? sh->t6pf.yraw : alloc(DType::F32, {(int64_t)C * mx_out});
    p->t6pf.part = sh ? sh->t6pf.part : alloc(DType::F32, {(int64_t)t6::kSplitKPartFloats});  // lab-ttft split-K
    // pfkern: INT8 path (int8 x / W' + per-128-block scales); the donor check covers them through xh / wd
    const int64_t kb = (mx_in + t6::kA8Block - 1) / t6::kA8Block;
    p->t6pf.xq = sh ? sh->t6pf.xq : alloc(DType::U8, {(int64_t)C * mx_in});
    p->t6pf.sx = sh ? sh->t6pf.sx : alloc(DType::F32, {(int64_t)C * kb});
    p->t6pf.wq = sh ? sh->t6pf.wq : alloc(DType::U8, {mx_w});
    p->t6pf.sw = sh ? sh->t6pf.sw : alloc(DType::F32, {mx_out * kb});
  }
  p->down = sh ? sh->down : alloc(DType::BF16, {C, LR});
  p->up = sh ? sh->up : alloc(DType::BF16, {C, HCD});
  p->xin = sh ? sh->xin : alloc(DType::BF16, {C, HID});
  p->raw = sh ? sh->raw : alloc(DType::BF16, {C, CONVD});
  p->z = sh ? sh->z : alloc(DType::BF16, {C, VALD});
  p->araw = sh ? sh->araw : alloc(DType::BF16, {C, NVH});
  p->braw = sh ? sh->braw : alloc(DType::BF16, {C, NVH});
  p->conv = sh ? sh->conv : alloc(DType::BF16, {C, CONVD});
  p->gb = sh ? sh->gb : alloc(DType::F32, {C, 2 * NVH});
  p->qk = sh ? sh->qk : alloc(DType::F32, {C, 2, KEYD});
  p->y = sh ? sh->y : alloc(DType::F32, {C, VALD});
  p->o = sh ? sh->o : alloc(DType::BF16, {C, VALD});
  p->br = sh ? sh->br : alloc(DType::BF16, {C, HID});
  p->aq = sh ? sh->aq : alloc(DType::F32, {C, AQD});
  p->akv = sh ? sh->akv : alloc(DType::F32, {C, 2 * AKVD});
  p->aiqk = sh ? sh->aiqk : alloc(DType::F32, {C, AIQD});
  p->aout = sh ? sh->aout : alloc(DType::F32, {C, AOD});
  if (const char *e = getenv("NE_PF_DENSE")) p->no_dense = std::string(e) == "0";
  if (const char *e = getenv("NE_PF_AHEAD")) p->ahead_on = std::string(e) != "0";
  if (const char *e = getenv("NE_PF_SCAN"))
    p->scan = std::string(e) == "plain" ? "pf_gdn_scan" : std::string(e) == "tb" ? "pf_gdn_scan_tb" : "pf_gdn_scan_pipe";
  p->logits = sh ? sh->logits : alloc(DType::BF16, {C, NE});
  p->route = sh ? sh->route : alloc(DType::U32, {C, 20});
  p->sgl = sh ? sh->sgl : alloc(DType::F32, {C});
  p->gu = sh ? sh->gu : alloc(DType::BF16, {C, 2 * INTER});
  p->sact = sh ? sh->sact : alloc(DType::BF16, {C, INTER});
  p->sho = sh ? sh->sho : alloc(DType::BF16, {C, HID});
  p->cnt = sh ? sh->cnt : alloc(DType::U32, {NE});
  p->rank = sh ? sh->rank : alloc(DType::U32, {S});
  p->off = sh ? sh->off : alloc(DType::U32, {NE});
  p->wtab = sh ? sh->wtab : alloc(DType::U32, {p->maxwin, 4});
  p->order = sh ? sh->order : alloc(DType::U32, {S});
  p->seid = sh ? sh->seid : alloc(DType::U32, {S});
  p->inv = sh ? sh->inv : alloc(DType::U32, {S});
  p->pg = sh ? sh->pg : alloc(DType::F16, {S, HID});
  p->pu = sh ? sh->pu : alloc(DType::F16, {S, HID});
  p->ig = sh ? sh->ig : alloc(DType::F32, {S, INTER});
  p->iu = sh ? sh->iu : alloc(DType::F32, {S, INTER});
  p->pd = sh ? sh->pd : alloc(DType::F16, {S, INTER});
  p->dn = sh ? sh->dn : alloc(DType::F32, {S, HID});
  p->pemb_f = sh ? sh->pemb_f : alloc(DType::F32, {C, HID});
  p->pemb = sh ? sh->pemb : alloc(DType::BF16, {C, HID});
  p->pkv = sh ? sh->pkv : alloc(DType::BF16, {C, PLE_KV});
  p->pgv = sh ? sh->pgv : alloc(DType::BF16, {C, HCD});
  p->pnrm = sh ? sh->pnrm : alloc(DType::BF16, {C, HCD});
  p->posbase = sh ? sh->posbase : alloc(DType::U32, {4});
  if (tq > 0) p->ac = sh ? sh->ac : attn::alloc_chunk_scratch(fmt, tq, C);
  return p;
}

// n-gram rows of tokens[0, n) after history (p2, p1) -> emb f32 [n][2560] (bf16 values); parallel preads (the decode
// path's gather_row, model.mm, over all rows of the chunk at once)
// threads > 0 (lab-pfpath: the first chunk's gather, before the GPU can start): that many reader threads (queue depth
// for the SSD: 65K random 320 B rows per 4096 tokens, ~25% faster at 64 than at the core count)
static void gather_chunk(Model &m, int64_t p2, int64_t p1, const int *tokens, int n, float *emb, int threads = 0) {
  if (m.L.size() < 2 || !m.L[1].ple.on) return;  // tests with layer 0 only
  std::vector<int64_t> ids((size_t)n * PLE_NG);
  for (int t = 0; t < n; t++) {
    ngram_ids(m.L[1].ple, p2, p1, tokens[t], m.eos, &ids[(size_t)t * PLE_NG]);
    p2 = p1;
    p1 = tokens[t];
  }
  std::vector<uint16_t> raw((size_t)n * PLE_NG * PLE_W);
  __block int bad = 0;
  uint16_t *rp = raw.data();
  const int64_t *idp = ids.data();
  const int fd = m.ngram_fd;
  const uint64_t off = m.ngram_off;
  const size_t nrow = (size_t)n * PLE_NG, per = 64;
  if (threads > 1 && nrow >= (size_t)threads * per) {
    std::vector<std::thread> th;
    std::atomic<int> tbad{0};
    for (int t = 0; t < threads; t++)
      th.emplace_back([&, t] {
        for (size_t i = (size_t)t; i < nrow; i += (size_t)threads)
          if (pread(fd, rp + i * PLE_W, PLE_W * 2, (off_t)(off + (uint64_t)idp[i] * PLE_W * 2)) != PLE_W * 2) tbad = 1;
      });
    for (auto &x : th) x.join();
    if (tbad) bad = 1;
  } else {
    dispatch_apply((nrow + per - 1) / per, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t b) {
      for (size_t i = b * per; i < std::min(nrow, (b + 1) * per); i++) {
        const ssize_t k = pread(fd, rp + i * PLE_W, PLE_W * 2, (off_t)(off + (uint64_t)idp[i] * PLE_W * 2));
        if (k != PLE_W * 2) bad = 1;
      }
    });
  }
  NE_CHECK(!bad, "n-gram row pread failed");
  for (size_t i = 0; i < nrow * PLE_W; i++) {
    const uint32_t u = (uint32_t)raw[i] << 16;
    memcpy(&emb[i], &u, 4);
  }
}

void prefill_tokens(Prefill &p, const int *tokens, int n, const int *ahead, int n_ahead) {
  Model &m = *p.m;
  NE_CHECK(n >= 1, "empty");
  NE_CHECK(m.T + n <= m.tmax, "context full (%d + %d > %d)", m.T, n, m.tmax);
  NE_CHECK(!m.dirty, "model state is one token ahead after generate(): reset() first");
  for (int i = 0; i < n; i++) NE_CHECK(tokens[i] >= 0 && tokens[i] < VOCAB, "token %d out of range", tokens[i]);
  gdn_materialise(m.st);  // lab-gdnlazy: pf_gdn_scan reads ssm (deferred decode rows first)
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  PfStats st;
  // gather of chunk k + 1 overlaps the GPU work of chunk k
  std::vector<float> next_emb;
  auto gather = [&](int c0, int cn, int64_t p2, int64_t p1, std::vector<float> &dst) {
    const auto g0 = clk::now();
    dst.resize((size_t)cn * HID);
    gather_chunk(m, p2, p1, tokens + c0, cn, dst.data());
    st.gather_ms += std::chrono::duration<double, std::milli>(clk::now() - g0).count();
  };
  {
    // the first chunk's rows: gathered by the previous call (same history, same tokens), else now
    const int cn0 = std::min(p.C, n);
    Prefill::Ahead &A = p.ahead;
    if (!A.toks.empty() && A.h2 == m.hist[0] && A.h1 == m.hist[1] && (int)A.toks.size() >= cn0 &&
        std::equal(tokens, tokens + cn0, A.toks.begin())) {
      A.emb.resize((size_t)cn0 * HID);
      next_emb.swap(A.emb);
      st.ahead_hit = cn0;
    } else {  // (the GPU waits for these: more reader threads, NE_PF_GATHER_THREADS, read per call; 0 = the cores)
      const char *gt = getenv("NE_PF_GATHER_THREADS");
      const auto g0 = clk::now();
      next_emb.resize((size_t)cn0 * HID);
      gather_chunk(m, m.hist[0], m.hist[1], tokens, cn0, next_emb.data(), gt ? atoi(gt) : 64);
      st.gather_ms += std::chrono::duration<double, std::milli>(clk::now() - g0).count();
    }
    A.toks.clear();
    A.emb.clear();
  }
  for (int c0 = 0; c0 < n; c0 += p.C) {
    const int cn = std::min(p.C, n - c0);
    const bool has_dense = !p.no_dense && p.ac.C > 0 && attn::chunk_dense_rows(p.ac, m.T, cn) > 0;
    const auto r0 = clk::now();
    const size_t nrec = p.recs.size();
    const uint64_t uc = p.use_ctr;
    Prefill::Rec &rec = get_rec(p, cn, has_dense ? m.T : -1);
    const double rms = std::chrono::duration<double, std::milli>(clk::now() - r0).count();
    st.rec_ms += rms;
    static const bool reclog = getenv("NE_PF_RECLOG") != nullptr;
    if (reclog && (p.recs.size() != nrec || rec.used != uc + 1 || rms > 1.0))
      fprintf(stderr, "prefill: chunk %d rows at T %d: recording %.2f ms (%zu dispatches, %zu recordings)\n", cn, m.T,
              rms, rec.dispatches, p.recs.size());
    uint32_t *tk = (uint32_t *)p.tok.host();
    for (int i = 0; i < cn; i++) tk[i] = embed_id(m, m.T + i, tokens[c0 + i]);
    memcpy(p.pemb_f.host(), next_emb.data(), (size_t)cn * HID * 4);
    ((uint32_t *)p.posbase.host())[0] = (uint32_t)m.T;
    for (auto &pr : rec.parts) pr.second->commit();
    try {
      // history after this chunk; gather the next chunk while the GPU runs
      for (int i = 0; i < cn; i++) {
        m.hist[0] = m.hist[1];
        m.hist[1] = tokens[c0 + i];
      }
      if (c0 + cn < n) {
        gather(c0 + cn, std::min(p.C, n - c0 - cn), m.hist[0], m.hist[1], next_emb);
      } else if (ahead && n_ahead >= kChunkedMin && p.ahead_on) {  // the next call's first chunk (lab-pfpath)
        const int an = std::min(p.C, n_ahead);
        const auto g0 = clk::now();
        Prefill::Ahead &A = p.ahead;
        A.h2 = m.hist[0];
        A.h1 = m.hist[1];
        A.emb.resize((size_t)an * HID);
        gather_chunk(m, A.h2, A.h1, ahead, an, A.emb.data());
        A.toks.assign(ahead, ahead + an);
        st.ahead_ms += std::chrono::duration<double, std::milli>(clk::now() - g0).count();
      }
      double cg = 0, ca = 0;
      for (size_t i = 0; i < rec.parts.size(); i++) {
        auto &pr = rec.parts[i];
        pr.second->wait();
        const double g = pr.second->gpu_ms();
        st.gpu_ms += g;
        cg += g;
        st.stage_ms[(int)pr.first] += g;
        if (pr.first == PfStage::Attn) ca += g;
        if (p.fine) st.fine_ms[rec.labels[i]] += g;
        if (p.fine && pr.first == PfStage::Attn) {  // per chunk too: attention grows with the context
          char k[16];
          snprintf(k, sizeof k, "#%02d", st.chunks);
          st.fine_ms[rec.labels[i] + k] += g;
        }
      }
      st.chunk_gpu_ms.push_back(cg);
      st.chunk_attn_ms.push_back(ca);
    } catch (...) {  // this chunk's KV / state rows are (being) written, T does not count them
      for (auto &pr : rec.parts) abort_inflight(m, {pr.second.get()});
      throw;
    }
    m.T += cn;
    m.gen++;
    st.chunks++;
    st.dispatches += (int)rec.dispatches;
  }
  ((uint32_t *)m.pos.host())[0] = (uint32_t)m.T;
  st.wall_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  p.stats = st;
  static const bool reclog = getenv("NE_PF_RECLOG") != nullptr;
  if (reclog)
    fprintf(stderr, "prefill: %d tokens at T %d: wall %.1f ms, GPU %.1f, gather %.1f (first chunk's %s), ahead %.1f, "
            "recording %.1f\n", n, m.T - n, st.wall_ms, st.gpu_ms, st.gather_ms,
            st.ahead_hit ? "gathered before" : "now", st.ahead_ms, st.rec_ms);
}

StepOut prefill_prompt(Prefill &p, const int *prompt, int n) {
  NE_CHECK(n >= 1, "empty prompt");
  check_feed(*p.m, prompt, n, n - 1 + p.m->rows, "prefill");
  if (n > 1) prefill_tokens(p, prompt, n - 1);
  return step(*p.m, prompt[n - 1], true);
}

const PfStats &prefill_stats(const Prefill &p) { return p.stats; }
const Tensor &prefill_hidden(const Prefill &p, int layers) { return p.h[layers % 2]; }
void prefill_set_dfl(Prefill &p, const Tensor *fs) {
  const bool same = fs ? (p.dfl_fs.buf == fs->buf && p.dfl_fs.offset == fs->offset) : !p.dfl_fs.buf;
  if (same) return;
  p.dfl_fs = fs ? *fs : Tensor();
  p.recs.clear();  // the chunk recordings bind (or not) the feature ring
}

}  // namespace ne::fn

// ------------------------------------------------------------------------------------------------ C entry points
using namespace ne;
using namespace ne::fn;

struct ne_fn_prefill {
  PrefillPtr p;
};

static void fill_stats(const PfStats &s, double *out) {
  if (!out) return;
  out[0] = s.wall_ms;
  out[1] = s.gpu_ms;
  out[2] = s.gather_ms;
  out[3] = s.chunks;
  for (int i = 0; i < (int)PfStage::N; i++) out[4 + i] = s.stage_ms[i];
}

extern "C" {

NE_API ne_fn_prefill *ne_fn_prefill_create(ne_fn_model *m, int chunk, int profile) {
  ne_fn_prefill *out = nullptr;
  int rc = c_call([&] {
    NE_CHECK(m && m->m, "null model");
    auto u = std::make_unique<ne_fn_prefill>();
    u->p = prefill_create(*m->m, chunk, profile != 0);
    out = u.release();
  });
  return rc == 0 ? out : nullptr;
}
NE_API void ne_fn_prefill_free(ne_fn_prefill *p) { delete p; }
NE_API int ne_fn_model_share_prefill(ne_fn_model *dst, ne_fn_model *src) {
  bool shared = false;
  int rc = c_call([&] {
    NE_CHECK(dst && dst->m && src && src->m && dst != src, "bad arguments");
    Prefill &d = chunked(*src->m);
    dst->m->pf = prefill_create(*dst->m, d.C, false, &d);
    shared = dst->m->pf->shared;
  });
  return rc != 0 ? -1 : shared ? 0 : 1;
}
NE_API int ne_fn_prefill_tokens(ne_fn_prefill *p, const int *tokens, int n, double *stats) {
  return c_call([&] {
    NE_CHECK(p && tokens, "null argument");
    prefill_tokens(*p->p, tokens, n);
    fill_stats(p->p->stats, stats);
  });
}
NE_API int ne_fn_prefill_prompt(ne_fn_prefill *p, const int *prompt, int n, float *logits, int *next, double *stats) {
  return c_call([&] {
    NE_CHECK(p && prompt, "null argument");
    StepOut o = prefill_prompt(*p->p, prompt, n);
    if (logits) memcpy(logits, p->p->m->logits.host(), (size_t)VOCAB * 4);
    if (next) *next = o.token;
    fill_stats(p->p->stats, stats);
  });
}

}  // extern "C"
