// NAX multi-row kernels: host recording side. Owner: lab-mtp. Kernels: kernels/nax/nax_q8.metal, nax_moe.metal.
#include "kernels/nax/nax.h"

#include <string>

namespace ne::nax {
using namespace ne::fn;

namespace {
struct NaxSegs {  // = kernels/nax/nax_q8.metal
  uint32_t nseg;
  uint32_t rows[4];
  uint32_t blk0[5];
  uint32_t gsh[4];
};

id<MTLComputePipelineState> q8_pso(int in, int R, int ks, int epi, int bits = 8) {
  return Pipelines::get("nax_q8_gemm", {FC::b(0, g_ref_round), FC::u32(1, (uint32_t)in), FC::u32(3, (uint32_t)R),
                                         FC::u32(4, (uint32_t)ks), FC::u32(5, (uint32_t)epi),
                                         FC::u32(6, (uint32_t)bits)});
}

// k splits per shape (measured, docs/mtp.md: +-10% noise): lm_head-sized outputs KS 16, 6144-input out_proj KS 4
// (only 80 column blocks), the rest 8. NE_NAX_KS overrides.
int auto_ks(int in, int out_rows) {
  static const int env = [] { const char *e = getenv("NE_NAX_KS"); return e ? atoi(e) : 0; }();
  if (env) return env;
  if (out_rows >= 65536) return 16;
  if (in >= 6144) return 4;
  return 8;
}

// one dispatch over up to 4 segments; outs / extras: up to 4 each (missing ones bound to the first output)
void q8_dispatch(Recorder &r, int R, const Tensor &x, const std::vector<const Q8 *> &w, const std::vector<Tensor> &o,
                 const std::vector<Tensor> &e, int epi, int ks, int bits = 8) {
  NE_CHECK(R >= 1 && R <= 16, "nax: rows %d (1..16)", R);
  NE_CHECK(!w.empty() && w.size() <= 4 && !o.empty() && o.size() <= 4, "nax: 1..4 segments");
  const int in = w[0]->in;
  NE_CHECK(in % (512 / bits) == 0, "nax: in %d not a multiple of %d", in, 512 / bits);
  NaxSegs sg{};
  sg.nseg = (uint32_t)w.size();
  uint32_t blk = 0;
  for (size_t i = 0; i < w.size(); i++) {
    NE_CHECK(w[i]->in == in, "nax: segments differ in input width");
    NE_CHECK(w[i]->group == 64 || w[i]->group == 128, "nax: group %d (64 or 128)", w[i]->group);
    sg.rows[i] = (uint32_t)w[i]->rows;
    sg.gsh[i] = w[i]->group == 64 ? 6 : 7;
    sg.blk0[i] = blk;
    blk += (uint32_t)((w[i]->rows + 31) / 32);
  }
  sg.blk0[w.size()] = blk;
  if (!ks) ks = auto_ks(in, (int)blk * 32);
  if (R > 8) ks = std::min(ks, 8);  // threadgroup partials: KS * 16 rows * 32 cols <= 16 KB
  ks = std::min(ks, in * bits / 512);  // no more simdgroups than k chunks (a chunk = 4 lanes x 16 bytes per row)
  NE_CHECK(ks >= 1 && ks <= 16, "nax: ks %d", ks);
  std::vector<Arg> a;
  a.emplace_back(x);
  for (int i = 0; i < 4; i++) {
    const Q8 *q = w[std::min<size_t>(i, w.size() - 1)];
    a.emplace_back(q->w);
    a.emplace_back(q->s);
    a.emplace_back(q->b);
  }
  for (int i = 0; i < 4; i++) a.emplace_back(o[std::min<size_t>(i, o.size() - 1)]);
  a.push_back(Arg::value(sg));
  for (int i = 0; i < 6; i++) a.emplace_back(e.empty() ? o[0] : e[std::min<size_t>(i, e.size() - 1)]);
  r.dispatch(q8_pso(in, R, ks, epi, bits), a, MTLSizeMake(blk, 1, 1), MTLSizeMake(32 * ks, 1, 1));
}
}  // namespace

Q8 load_q8(const Pack &p, const std::string &pre) {
  auto T = [&](const std::string &n) {
    const Tensor &t = p.at(n);
    return t.gpu_address() % 16 == 0 ? t : p.aligned(n, 16);
  };
  Q8 q;
  q.w = T(pre + ".weight");
  q.s = T(pre + ".scales");
  q.b = T(pre + ".biases");
  NE_CHECK(q.w.dtype == DType::U32 && q.s.dtype == DType::BF16 && q.w.ndim == 2, "%s: not 8-bit affine", pre.c_str());
  q.rows = (int)q.w.dim(0);
  q.in = (int)q.w.dim(1) * 4;
  q.group = q.in / (int)q.s.dim(1);
  return q;
}

void record_q8(Recorder &r, int R, const Tensor &x, const std::vector<const Q8 *> &w, const std::vector<Tensor> &y,
               int ks) {
  NE_CHECK(w.size() == y.size(), "nax: one output per segment");
  q8_dispatch(r, R, x, w, y, {}, 0, ks);
}

void record_qn(Recorder &r, int R, const Tensor &x, const Q8 &w, int bits, const Tensor &y, int ks) {
  NE_CHECK(bits == 8 || bits == 4 || bits == 2, "nax: %d-bit weights", bits);
  q8_dispatch(r, R, x, {&w}, {y}, {}, 0, ks, bits);
}

void record_ple_proj(Recorder &r, int R, Scratch &s, const PleW &e, const Tensor &emb, const Tensor &kb,
                     const Tensor &vb) {
  if (e.key.t6) {  // lab-t6wire: K8 PLE projections, bf16-rounded in place (= record_q8's EPI 0)
    t6_linears(r, R, s, {&e.key, &e.value}, {emb, emb}, {kb, vb});
    t6_round(r, R, kb, HCD, HCD, 0, kb, HCD, 0, true);
    t6_round(r, R, vb, HID, HID, 0, vb, HID, 0, true);
  } else {
    record_q8(r, R, emb, {&e.key, &e.value}, {kb, vb});
  }
}

void record_q8_rowbias(Recorder &r, int R, const Tensor &x, const Q8 &w, const Tensor &bias, const Tensor &y, int ks) {
  q8_dispatch(r, R, x, {&w}, {y}, {bias}, 4, ks);
}

void record_gdn_in(Recorder &r, int R, const LayerW &w, GdnState &st, Scratch &s, int ks, const Tensor *hist) {
  const GdnW &g = w.gdn;
  if (hist) q8_dispatch(r, R, s.xa, {&g.qkv, &g.z, &g.b, &g.a}, {s.qkv, s.z, s.gb, *hist}, {g.conv, st.conv, g.alog, g.dtb}, 5, ks);
  else q8_dispatch(r, R, s.xa, {&g.qkv, &g.z, &g.b, &g.a}, {s.qkv, s.z, s.gb}, {g.conv, st.conv, g.alog, g.dtb}, 1, ks);
}

void record_out_hc(Recorder &r, int R, const Q8 &ow, const Tensor &o, const Tensor &h_in, const Tensor &inj,
                   const Tensor &h_out, const Tensor &branch, int ks) {
  q8_dispatch(r, R, o, {&ow}, {h_out, branch}, {h_in, inj}, 2, ks);
}

void record_attn_proj(Recorder &r, int R, const LayerW &w, GdnState &st, Scratch &s, int ks) {
  const AttnW &a = w.attn;
  q8_dispatch(r, R, s.xa, {&a.q, &a.k, &a.v, &a.iqk}, {s.aq, s.akv, s.aiqk}, {st.ikeys, st.pos}, 3, ks);
}

Scratch alloc_scratch_rows(int rows) {  // fn::alloc_scratch sizes every buffer at R (1..16)
  NE_CHECK(rows >= 1 && rows <= kMaxRows, "rows %d (1..%d)", rows, kMaxRows);
  return alloc_scratch(rows);
}

GdnState alloc_state_rows(const LayerW &w, int rows, int tmax) {
  GdnState st = alloc_state(w, tmax);
  if (!w.linear && rows > 4) st.apart = alloc(DType::F32, {NKV, tmax / 32, rows, NQH / NKV, AHD + 2});
  return st;
}

LayerExtra alloc_layer_extra(int rb, bool lazy, const Tensor *keep) {
  LayerExtra e;
  e.hc_part2 = alloc(DType::F32, {LR / 8 + 1, HC, 8, 4});
  e.hc_ctr2 = alloc(DType::U32, {64});
  if (lazy) {
    e.gdn_pq = alloc(DType::F32, {kMaxRows, CONVD});
    e.gdn_pg = alloc(DType::F32, {kMaxRows, 2 * NVH});
    e.gdn_praw = alloc(DType::F32, {kMaxRows, CONVD});
    e.keep = keep ? *keep : alloc(DType::U32, {4});
    return e;
  }
  if (rb > 1) {
    e.gdn_snap = alloc(DType::F32, {rb - 1, NVH, DV, DK});
    e.gdn_hist = alloc(DType::F32, {3 + rb, CONVD});
  }
  return e;
}

void record_gdn_rollback(Recorder &r, GdnState &st, LayerExtra &ex, int rows, const Tensor &keep) {
  NE_CHECK(ex.gdn_snap.buf && ex.gdn_snap.dim(0) >= rows - 1, "rollback: no snapshot buffers for %d rows", rows);
  const int n = NVH * DV * DK / 4 + 3 * CONVD;
  r.dispatch(Pipelines::get("mtp_gdn_rollback", {FC::b(0, g_ref_round)}),
             {ex.gdn_snap, st.ssm, ex.gdn_hist, st.conv, keep, Arg::value((uint32_t)rows)},
             MTLSizeMake((n + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1));
}

void record_hc_pre(Recorder &r, int R, const HcW &hw, const Tensor &h, const Tensor &mix, const Tensor &inj,
                   Scratch &s, LayerExtra &ex) {
  const int n = R == 8 ? 4 : R;
  NE_CHECK(R <= 7 || R == 8, "hc_pre rows %d", R);
  auto p = Pipelines::get("fn_hc_pre_g" + std::to_string(hw.down.group) + "_r" + std::to_string(n), {FC::b(0, g_ref_round)});
  for (int half = 0; half < (R == 8 ? 2 : 1); half++)
    r.dispatch(p, {h.rows(n * half, n), hw.norm, hw.down.w, hw.down.s, hw.down.b, hw.inject, mix.rows(n * half, n),
                   inj.rows(n * half, n), half ? ex.hc_part2 : s.hc_part, half ? ex.hc_ctr2 : s.hc_ctr},
               MTLSizeMake((LR / 8 + 1) * HC, 1, 1), MTLSizeMake(256, 1, 1));
  r.barrier();
}

void record_layer(Recorder &r, const LayerW &w, GdnState &st, Scratch &s, LayerExtra &ex, const Tensor &h_in,
                  const Tensor &h_out, bool nax_trunk, bool kv_only) {
  const int R = s.rows;
  NE_CHECK(!kv_only || (!w.linear && !w.t6 && st.qc.tmax > 0), "kv_only: QSA attention layers only");
  auto hc_pre8 = [&](const HcW &hw, const Tensor &h, const Tensor &mix, const Tensor &inj) {
    record_hc_pre(r, R, hw, h, mix, inj, s, ex);
  };
  for (int i = 0; i < (int)Stage::N; i++) {
    const Stage stg = (Stage)i;
    const bool lazy = w.linear && ex.gdn_pq.buf;
    const bool rb = w.linear && R > 1 && ex.gdn_snap.buf && ex.gdn_snap.dim(0) >= R - 1;
    if (w.t6) {  // lab-t6wire: EXL3 trunk layer (models/flash_next/layer_t6.mm; any R, its own HC read)
      T6GdnOpt o;
      if (lazy) o = {&ex.gdn_praw, &ex.keep, true};
      else if (rb) o.hist = &ex.gdn_hist;
      if (t6_record_stage(r, stg, w, st, s, h_in, h_out, o)) continue;
    }
    if (R == 8 && stg == Stage::AttnHCPre) { hc_pre8(w.hca, h_in, s.mix_a, s.inj_a); continue; }
    if (R == 8 && stg == Stage::MlpHCPre) { hc_pre8(w.hcm, s.h2, s.mix_m, s.inj_m); continue; }
    if (lazy && stg == Stage::GdnIn) {
      NE_CHECK(nax_trunk, "lazy GDN commit needs the NAX in_proj epilogue");
      const GdnW &g = w.gdn;
      q8_dispatch(r, R, s.xa, {&g.qkv, &g.z, &g.b, &g.a}, {s.qkv, s.z, s.gb},
                  {g.conv, st.conv, g.alog, g.dtb, ex.gdn_praw, ex.keep}, 6, 0);
      r.barrier();
      continue;
    }
    if (lazy && stg == Stage::GdnStep) {
      r.dispatch(Pipelines::get("mtp_gdn_step_lazy_r" + std::to_string(R), {FC::b(0, g_ref_round)}),
                 {s.qkv, s.gb, s.z, w.gdn.norm, st.ssm, s.o, ex.gdn_pq, ex.gdn_pg, ex.keep}, MTLSizeMake(NVH, 1, 1),
                 MTLSizeMake(1024, 1, 1));
      r.barrier();
      r.dispatch(Pipelines::get("mtp_gdn_pending_copy", {FC::b(0, g_ref_round)}),
                 {s.qkv, s.gb, ex.gdn_pq, ex.gdn_pg, Arg::value((uint32_t)R)},
                 MTLSizeMake((R * (CONVD + 2 * NVH) + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1));
      r.barrier();
      continue;
    }
    if (rb && stg == Stage::GdnStep) {
      r.dispatch(Pipelines::get("mtp_gdn_step_rows_r" + std::to_string(R), {FC::b(0, g_ref_round)}),
                 {s.qkv, s.gb, s.z, w.gdn.norm, st.ssm, s.o, ex.gdn_snap}, MTLSizeMake(NVH, 1, 1),
                 MTLSizeMake(1024, 1, 1));
      r.barrier();
      continue;
    }
    if (nax_trunk && stg == Stage::GdnIn) {
      if (w.linear) record_gdn_in(r, R, w, st, s, 0, rb ? &ex.gdn_hist : nullptr);
      else record_attn_proj(r, R, w, st, s);
      r.barrier();
      continue;
    }
    if (kv_only && stg == Stage::GdnStep) {  // = the first dispatch of attn::record_step (record_stage's path)
      attn::record_one(r, 0, w.qsa, st.qc, R, s.aq, s.akv, s.aiqk, s.o);
      r.barrier();
      return;
    }
    if (nax_trunk && stg == Stage::GdnOut) {
      record_out_hc(r, R, w.linear ? w.gdn.out : w.attn.o, s.o, h_in, s.inj_a, s.h2, s.br_a);
      r.barrier();
      continue;
    }
    record_stage(r, stg, w, st, s, h_in, h_out);
  }
}

}  // namespace ne::nax
