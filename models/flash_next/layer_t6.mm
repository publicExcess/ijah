// lab-t6wire: the layer graph's EXL3 / bf16 trunk Linears (T6 pack, docs/trunk6.md section 8). A Linear whose pack
// tensors are EXL3 (or the final mixer's bf16) carries a t6::Linear (Q8::t6); a layer whose trunk is EXL3 (LayerW::t6)
// records its stages here instead of the fused q8 kernels: t6::record_multi (the stage's Linears concurrently, one
// scratch slot each; SIMD below 4 rows, NAX from 4) into raw f32 [R][out] buffers, then a small epilogue kernel
// (kernels/trunk6/t6_epi.metal) with the q8 path's epilogue math and rounding points. GdnStep, MoeDown, the router
// GEMV and the routed experts are the q8 graph's kernels unchanged (the router / gateup dispatches skip their fused
// shared-expert threadgroups: the shared expert runs as t6 Linears).
#include "models/flash_next/layer.h"

#include <string>
#include <vector>

namespace ne::fn {

static MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
static int cdiv(int a, int b) { return (a + b - 1) / b; }

T6Buf &t6_buf(Scratch &s) {
  if (!s.t6) {
    auto b = std::make_shared<T6Buf>();
    const int64_t R = s.rows;
    b->rows = s.rows;
    for (auto &sl : b->slot) sl = t6::alloc_scratch(1u << 20, 4096, (size_t)R * HCD);
    b->xn = alloc(DType::F32, {R, HCD});
    b->mraw = alloc(DType::F32, {R, LR});
    b->ip = alloc(DType::F32, {R, HC, HID / 256, 8});
    b->uraw = alloc(DType::F32, {R, HCD});
    b->braw = alloc(DType::F32, {R, NVH});
    b->araw = alloc(DType::F32, {R, NVH});
    b->kraw = alloc(DType::F32, {R, AKVD});
    b->vraw = alloc(DType::F32, {R, AKVD});
    b->graw = alloc(DType::F32, {R, INTER});
    b->uraw2 = alloc(DType::F32, {R, INTER});
    b->act = alloc(DType::F32, {R, INTER});
    b->pk = alloc(DType::F32, {R, HCD});
    b->pv = alloc(DType::F32, {R, HID});
    s.t6 = std::move(b);
  }
  NE_CHECK(s.t6->rows == s.rows, "t6 buffers for %d rows, scratch has %d", s.t6->rows, s.rows);
  return *s.t6;
}

bool g_t6_fuse = [] {  // fused finisher epilogues (default); NE_T6_FUSE=0 = separate epilogue kernels
  const char *e = getenv("NE_T6_FUSE");
  return !(e && std::string(e) == "0");
}();
// lab-t6fuse: the HC read's prep / act / out kernels and the shared SwiGLU folded into the Linears (t6::Epi pre /
// kind 8): HC read 2 dispatches + 2 barriers instead of 5 + 4, shared expert 2 phases instead of 3. Default on;
// env NE_T6_FUSE2=0 = lab-t6wire's recording (A/B).
bool g_t6_fuse2 = [] {
  const char *e = getenv("NE_T6_FUSE2");
  return !(e && std::string(e) == "0");
}();
int g_t6_x = getenv("NE_T6_X") ? (int)strtol(getenv("NE_T6_X"), nullptr, 0) : 0;
static bool simd_path(const Q8 &q, int R) {
  return q.t6 && q.t6->fmt == t6::Fmt::EXL3 && t6::plan(*q.t6, R).path == t6::Path::Simd;
}
static t6::Epi epi(int kind, std::vector<Tensor> e = {}, int ldy = 0, uint32_t mode = 1, uint32_t pre = 0) {
  t6::Epi x;
  x.kind = kind;
  x.pre = pre;
  x.mode = mode;
  x.ldy = ldy;
  x.refr = g_ref_round;
  for (size_t i = 0; i < e.size() && i < 4; i++) x.e[i] = e[i];
  return x;
}

void t6_linears(Recorder &r, int R, Scratch &s, const std::vector<const Q8 *> &w, const std::vector<Tensor> &x,
                const std::vector<Tensor> &y, const std::vector<t6::Epi> *ep) {
  T6Buf &b = t6_buf(s);
  NE_CHECK(w.size() <= 4, "t6_linears: %zu Linears (<= 4 slots)", w.size());
  std::vector<const t6::Linear *> l;
  for (const Q8 *q : w) {
    NE_CHECK(q->t6, "t6_linears: a q8 Linear in a t6 stage");
    l.push_back(q->t6.get());
  }
  t6::record_multi(r, R, l, x, y, b.slot, ep);
  r.barrier();
}

struct T6Copy { uint32_t n, R, sld, soff, dld, doff, round; };  // = t6_epi.metal
void t6_round(Recorder &r, int R, const Tensor &src, int n, int sld, int soff, const Tensor &dst, int dld, int doff,
              bool round) {
  const T6Copy c{(uint32_t)n, (uint32_t)R, (uint32_t)sld, (uint32_t)soff, (uint32_t)dld, (uint32_t)doff, round ? 1u : 0u};
  r.dispatch(pso("t6e_copy"), {src, dst, Arg::value(c)}, sz(cdiv(R * n, 256)), sz(256));
}

static void hc_pre(Recorder &r, const HcW &hw, int R, const Tensor &h, bool inject, Scratch &s, const Tensor &mix,
                   const Tensor &inj) {
  T6Buf &b = t6_buf(s);
  if (g_t6_fuse && g_t6_fuse2 && inject && simd_path(hw.down, R)) {  // prep + down + silu(/4) + inject gates: one dispatch
    const std::vector<t6::Epi> e = {epi(2, {hw.norm, hw.inject, b.ip, inj}, 0, 1, 1)};
    t6_linears(r, R, s, {&hw.down}, {h}, {mix}, &e);
    return;
  }
  const uint32_t hi = inject ? 1u : 0u, uR = (uint32_t)R;
  r.dispatch(pso("t6e_hc_prep"), {h, hw.norm, inject ? hw.inject : hw.norm, b.xn, b.ip, Arg::value(hi)},
             sz(R * HC * (HID / 256)), sz(256));
  r.barrier();
  if (g_t6_fuse && hw.down.t6->fmt == t6::Fmt::EXL3) {  // down + silu(/4) fused; the inject gates alongside
    const uint32_t nomix = 0;
    r.dispatch(pso("t6e_hc_act"), {b.mraw, b.ip, mix, inj, Arg::value(uR), Arg::value(nomix)},
               sz(cdiv(R * (LR + 2 * HC), 256)), sz(256));
    const std::vector<t6::Epi> e = {epi(2)};
    t6_linears(r, R, s, {&hw.down}, {b.xn}, {mix}, &e);
    return;
  }
  const uint32_t domix = 1;
  t6_linears(r, R, s, {&hw.down}, {b.xn}, {b.mraw});
  r.dispatch(pso("t6e_hc_act"), {b.mraw, b.ip, mix, inj, Arg::value(uR), Arg::value(domix)},
             sz(cdiv(R * (LR + 2 * HC), 256)), sz(256));
  r.barrier();
}
static void hc_mix(Recorder &r, const HcW &hw, int R, const Tensor &h, Scratch &s, const Tensor &mix, const Tensor &inj,
                   const Tensor &out) {
  T6Buf &b = t6_buf(s);
  const uint32_t uR = (uint32_t)R;
  if (g_t6_fuse && g_t6_fuse2 && hw.up.t6->fmt == t6::Fmt::EXL3) {  // up + sigmoid-mean in the finishers
    const std::vector<t6::Epi> e = {epi(8, {h, hw.norm, inj, out})};
    t6_linears(r, R, s, {&hw.up}, {mix}, {b.uraw}, &e);
    return;
  }
  t6_linears(r, R, s, {&hw.up}, {mix}, {b.uraw});
  r.dispatch(pso("t6e_hc_out"), {b.uraw, h, hw.norm, inj, out, Arg::value(uR)}, sz(cdiv(R * HID, 256)), sz(256));
  r.barrier();
}
void t6_hc_read(Recorder &r, const HcW &hw, int R, const Tensor &h, bool inject, Scratch &s, const Tensor &mix,
                const Tensor &inj, const Tensor &out) {
  NE_CHECK(hw.down.t6 && hw.up.t6, "t6_hc_read: q8 HC weights");
  hc_pre(r, hw, R, h, inject, s, mix, inj);
  hc_mix(r, hw, R, h, s, mix, inj, out);
}

bool t6_record_stage(Recorder &r, Stage stage, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in,
                     const Tensor &h_out, const T6GdnOpt &opt) {
  const int R = s.rows;
  const uint32_t uR = (uint32_t)R;
  const MoeW &m = w.moe;
  switch (stage) {
    case Stage::AttnHCPre: hc_pre(r, w.hca, R, h_in, true, s, s.mix_a, s.inj_a); return true;
    case Stage::AttnHCMix: hc_mix(r, w.hca, R, h_in, s, s.mix_a, s.inj_a, s.xa); return true;
    case Stage::MlpHCPre: hc_pre(r, w.hcm, R, s.h2, true, s, s.mix_m, s.inj_m); return true;
    case Stage::MlpHCMix: hc_mix(r, w.hcm, R, s.h2, s, s.mix_m, s.inj_m, s.xm); return true;
    case Stage::GdnIn: {
      T6Buf &b = t6_buf(s);
      if (!w.linear) {  // q / k / v / indexer: q and iqk land in place, k / v -> kv, raw indexer key -> its cache
        const AttnW &a = w.attn;
        if (g_t6_fuse) {  // q, iqk (+ raw key sink) in place, k / v into kv [R][k 512 | v 512]
          const std::vector<t6::Epi> e = {epi(0), epi(0, {}, 2 * AKVD), epi(0, {}, 2 * AKVD), epi(7, {st.ikeys, st.pos})};
          t6_linears(r, R, s, {&a.q, &a.k, &a.v, &a.iqk}, {s.xa, s.xa, s.xa, s.xa},
                     {s.aq, s.akv, s.akv.view((size_t)AKVD * 4, DType::F32, {R, 2 * AKVD - AKVD}), s.aiqk}, &e);
          return true;
        }
        t6_linears(r, R, s, {&a.q, &a.k, &a.v, &a.iqk}, {s.xa, s.xa, s.xa, s.xa}, {s.aq, b.kraw, b.vraw, s.aiqk});
        r.dispatch(pso("t6e_attn"), {b.kraw, b.vraw, s.aiqk, s.akv, st.ikeys, st.pos, Arg::value(uR)},
                   sz(cdiv(R * (2 * AKVD + 128), 256)), sz(256));
        r.barrier();
        return true;
      }
      const GdnW &g = w.gdn;
      const uint32_t mode = opt.lazy ? 6u : opt.hist ? 5u : 1u;
      NE_CHECK(!opt.lazy || (opt.hist && opt.keep), "t6 lazy GDN commit needs the pending rows and keep");
      if (g_t6_fuse) {  // conv1d + silu / z / beta / decay in the four finishers
        const Tensor hist = opt.hist ? *opt.hist : b.ip, keep = opt.keep ? *opt.keep : b.ip;
        const std::vector<t6::Epi> e = {epi(4, {g.conv, st.conv, hist, keep}, 0, mode), epi(3), epi(5),
                                        epi(6, {g.alog, g.dtb})};
        t6_linears(r, R, s, {&g.qkv, &g.z, &g.b, &g.a}, {s.xa, s.xa, s.xa, s.xa}, {s.qkv, s.z, s.gb, s.gb}, &e);
        return true;
      }
      t6_linears(r, R, s, {&g.qkv, &g.z, &g.b, &g.a}, {s.xa, s.xa, s.xa, s.xa}, {s.qkv, s.z, b.braw, b.araw});
      r.dispatch(pso("t6e_gdn_in"),
                 {s.qkv, s.z, b.braw, b.araw, s.gb, g.conv, st.conv, g.alog, g.dtb, opt.hist ? *opt.hist : b.ip,
                  opt.keep ? *opt.keep : b.ip, Arg::value(uR), Arg::value(mode)},
                 sz(cdiv(CONVD + VALD + 2 * NVH, 256)), sz(256));
      r.barrier();
      return true;
    }
    case Stage::GdnOut: {
      const Q8 &ow = w.linear ? w.gdn.out : w.attn.o;
      if (g_t6_fuse) {  // branch + HC write in the finisher
        const std::vector<t6::Epi> e = {epi(1, {h_in, s.inj_a, s.h2})};
        t6_linears(r, R, s, {&ow}, {s.o}, {s.br_a}, &e);
        return true;
      }
      t6_linears(r, R, s, {&ow}, {s.o}, {s.br_a});
      r.dispatch(pso("t6e_out_hc"), {s.br_a, h_in, s.inj_a, s.h2, Arg::value(uR)}, sz(cdiv(R * HID, 256)), sz(256));
      r.barrier();
      return true;
    }
    case Stage::MoeRouter: {  // router threadgroups only (the shared gate / up ones are t6 Linears), then SwiGLU
      T6Buf &b = t6_buf(s);
      const Tensor &d = m.router;  // dummy for the unused q8 shared-expert bindings
      r.dispatch(pso("fn_moe_router_r" + std::to_string(R)),
                 {s.xm, m.router, m.sgate, d, d, d, d, d, d, s.logits, s.gush, s.route, s.ctr1, s.uq,
                  Arg::value(moe_lists(R, w) | (uint32_t)g_moe_spread << 8)},
                 sz((NE + 1 + 7) / 8), sz(256));
      t6_linears(r, R, s, {&m.sg, &m.su}, {s.xm, s.xm}, {b.graw, b.uraw2});
      if (g_t6_fuse && g_t6_fuse2 && simd_path(m.sd, R)) return true;  // SwiGLU in the shared down's prep (MoeGateUp)
      r.dispatch(pso("t6e_swiglu"), {b.graw, b.uraw2, s.gush, b.act, Arg::value(uR)}, sz(cdiv(R * INTER, 256)),
                 sz(256));
      r.barrier();
      return true;
    }
    case Stage::MoeGateUp: {  // routed gate / up threadgroups only, the shared down as a t6 Linear alongside
      T6Buf &b = t6_buf(s);
      const std::string k = "_k" + std::to_string(m.gate.k) + "_r" + std::to_string(R);
      const Tensor &d = m.router;
      const int nxt = R * TOPK * 2 * (40 / 2);
      if (moe_group(R, w)) {  // lab-moegroup: expert tiles of 1 / 2 / 4 rows concurrently (no barrier between)
        record_moe_group(r, stage, w, s, h_out);
      } else if (moe_split(R, w)) {  // lab-verify: singles and multis concurrently (no barrier between)
        r.dispatch(pso("fn_moe_gateup_s" + k), {s.xm, m.gate.t, m.up.t, m.gate.suh, m.up.suh, s.route, s.guo, s.gush,
                                                d, d, d, s.sho, s.uq},
                   sz(nxt), sz(256));
        r.dispatch(pso("fn_moe_gateup_m" + k), {s.xm, m.gate.t, m.up.t, m.gate.suh, m.up.suh, s.uq, s.guo, s.gush, d,
                                                d, d, s.sho},
                   sz(R * TOPK / 2 * 2 * (40 / 2)), sz(256));
      } else if (R > 1 && R <= g_moe_dedupe)
        r.dispatch(pso("fn_moe_gateup_u" + k), {s.xm, m.gate.t, m.up.t, m.gate.suh, m.up.suh, s.uq, s.guo_u, s.gush, d,
                                                d, d, s.sho},
                   sz(nxt), sz(256));
      else
        r.dispatch(pso("fn_moe_gateup" + k), {s.xm, m.gate.t, m.up.t, m.gate.suh, m.up.suh, s.route, s.guo, s.gush,
                                              d, d, d, s.sho, s.uq},
                   sz(nxt), sz(256));
      if (g_t6_fuse && g_t6_fuse2 && simd_path(m.sd, R)) {  // SwiGLU folded into the prep (+ shared_gu)
        const std::vector<t6::Epi> e = {epi(3, {b.uraw2, s.gush}, 0, 1, 2)};
        t6_linears(r, R, s, {&m.sd}, {b.graw}, {s.sho}, &e);
        return true;
      }
      if (g_t6_fuse) {
        const std::vector<t6::Epi> e = {epi(3)};
        t6_linears(r, R, s, {&m.sd}, {b.act}, {s.sho}, &e);
        return true;
      }
      t6_linears(r, R, s, {&m.sd}, {b.act}, {s.sho});
      if (g_ref_round) {  // the q8 path stores rr(shared down)
        t6_round(r, R, s.sho, HID, HID, 0, s.sho, HID, 0, true);
        r.barrier();
      }
      return true;
    }
    default:
      return false;  // GdnStep, MoeDown: unchanged
  }
}

}  // namespace ne::fn
