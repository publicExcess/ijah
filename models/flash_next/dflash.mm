// DFlash2 block drafter (lab-dflash). Header: dflash.h; docs/mtp.md section 13.
#include "models/flash_next/dflash.h"

#include <array>
#include <cmath>
#include <cstring>
#include <dispatch/dispatch.h>

namespace ne::dfl {
using fn::Q8;
using fn::pso;
const int kLayers[5] = {5, 15, 25, 35, 45};

namespace {
MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
struct U4 { uint32_t x, y, z, w; };
float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
// bf16 [rows][in] -> MLX affine `bits` (8 or 4), group 64 (min / max per group, bf16 scale / bias, as mtp::requantize)
Q8 quantize(const Tensor &src, int rows, int in, int bits) {
  NE_CHECK(in % 64 == 0 && rows % 32 == 0, "dflash quantize: %d x %d", rows, in);
  Q8 q;
  q.rows = rows;
  q.in = in;
  q.group = 64;
  const int64_t words = (int64_t)in * bits / 32, ng = in / 64;
  q.w = alloc(DType::U32, {rows, words});
  q.s = alloc(DType::BF16, {rows, ng});
  q.b = alloc(DType::BF16, {rows, ng});
  const uint16_t *x = (const uint16_t *)src.host();
  uint32_t *wo = (uint32_t *)q.w.host();
  uint16_t *so = (uint16_t *)q.s.host(), *bo = (uint16_t *)q.b.host();
  const int maxq = (1 << bits) - 1, per = 32 / bits;
  dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
    for (int g = 0; g < ng; g++) {
      const uint16_t *xg = x + (size_t)r * in + g * 64;
      float mn = bf2f(xg[0]), mx = mn;
      for (int k = 1; k < 64; k++) { mn = std::min(mn, bf2f(xg[k])); mx = std::max(mx, bf2f(xg[k])); }
      const uint16_t sb = f2bf(std::max((mx - mn) / maxq, 1e-8f)), bb = f2bf(mn);
      so[r * ng + g] = sb;
      bo[r * ng + g] = bb;
      const float sc = bf2f(sb), bi = bf2f(bb);
      for (int k = 0; k < 64; k += per) {
        uint32_t wd = 0;
        for (int j = 0; j < per; j++) {
          const int qv = std::clamp((int)std::lround((bf2f(xg[k + j]) - bi) / sc), 0, maxq);
          wd |= (uint32_t)qv << (bits * j);
        }
        wo[(size_t)r * words + (g * 64 + k) / per] = wd;
      }
    }
  });
  return q;
}
Tensor f32_of(const Tensor &bf, int64_t n) {
  Tensor t = alloc(DType::F32, {n});
  const uint16_t *s = (const uint16_t *)bf.host();
  float *d = (float *)t.host();
  for (int64_t i = 0; i < n; i++) d[i] = bf2f(s[i]);
  return t;
}
// 8-bit GEMM (record_q8) or a single matrix at 4 bits (record_qn)
void lin(Recorder &r, int R, const Tensor &x, const Q8 &w, const Tensor &y, int bits) {
  if (bits == 8) nax::record_q8(r, R, x, {&w}, {y});
  else nax::record_qn(r, R, x, w, bits, y);
}
}  // namespace

Weights load(const std::string &dir, int bits) {
  NE_CHECK(bits == 8 || bits == 4, "dflash: drafter bits %d (8 or 4)", bits);
  Weights w;
  w.bits = bits;
  w.pack = std::shared_ptr<const Pack>(Pack::open(dir).release());
  const Pack &p = *w.pack;
  const Json &cfg = p.config();
  NE_CHECK(cfg.find("dflash_config"), "%s: no dflash_config (not a DFlash2 checkpoint)", dir.c_str());
  const Json &dc = cfg["dflash_config"];
  w.mask = (int)dc["mask_token_id"].i64();
  w.block = (int)dc["block_size"].i64();
  {
    const Json &tl = dc["target_layer_ids"];
    NE_CHECK(tl.size() == 5, "dflash: %zu target layers (5 expected)", tl.size());
    for (int i = 0; i < 5; i++) NE_CHECK(tl[i].i64() == kLayers[i], "dflash: target layer %d = %lld", i, (long long)tl[i].i64());
    NE_CHECK(cfg["hidden_size"].i64() == H && cfg["num_hidden_layers"].i64() == NL &&
                 cfg["num_attention_heads"].i64() == NQ && cfg["num_key_value_heads"].i64() == NKV &&
                 cfg["head_dim"].i64() == HD && cfg["intermediate_size"].i64() == FF &&
                 dc["conv_kernel_size"].i64() == 2 && dc["conv_group_size"].i64() == 16 &&
                 dc["selector_rank"].i64() == SELR && dc["selector_top_k"].i64() == NC &&
                 cfg["sliding_window"].i64() == W,
             "dflash: unexpected drafter geometry");
  }
  const double theta = cfg["rope_parameters"]["rope_theta"].num();
  w.invf = alloc(DType::F32, {64});
  for (int i = 0; i < 64; i++) ((float *)w.invf.host())[i] = (float)(1.0 / std::pow(theta, (2.0 * i) / HD));
  auto T = [&](const std::string &n) -> const Tensor & { return p.aligned(n); };
  const int big = bits;  // q / o / gate / up / down at `bits`; the rest 8
  for (int l = 0; l < NL; l++) {
    const std::string pr = "layers." + std::to_string(l) + ".";
    LayerW L;
    L.in_ln = T(pr + "input_layernorm.weight");
    L.post_ln = T(pr + "post_attention_layernorm.weight");
    L.qn = T(pr + "self_attn.q_norm.weight");
    L.kn = T(pr + "self_attn.k_norm.weight");
    L.base_att = f32_of(T(pr + "attention_conv.base_kernel"), 4 * H);
    L.base_mlp = f32_of(T(pr + "mlp_conv.base_kernel"), 4 * H);
    L.kp_att = quantize(T(pr + "attention_conv.kernel_projection.weight"), DYN, H, 8);
    L.kp_mlp = quantize(T(pr + "mlp_conv.kernel_projection.weight"), DYN, H, 8);
    L.q = quantize(T(pr + "self_attn.q_proj.weight"), NQ * HD, H, big);
    L.k = quantize(T(pr + "self_attn.k_proj.weight"), NKV * HD, H, 8);
    L.v = quantize(T(pr + "self_attn.v_proj.weight"), NKV * HD, H, 8);
    L.o = quantize(T(pr + "self_attn.o_proj.weight"), H, NQ * HD, big);
    L.gate = quantize(T(pr + "mlp.gate_proj.weight"), FF, H, big);
    L.up = quantize(T(pr + "mlp.up_proj.weight"), FF, H, big);
    L.down = quantize(T(pr + "mlp.down_proj.weight"), H, FF, big);
    w.L.push_back(std::move(L));
  }
  w.fc = quantize(T("fc.weight"), H, FEAT, 8);
  w.sel = quantize(T("candidate_selector.hidden_projection.weight"), SELR, H, 8);
  w.hnorm = T("hidden_norm.weight");
  w.norm = T("norm.weight");
  w.pcb = T("candidate_selector.predecessor_codebook");
  w.scb = T("candidate_selector.successor_codebook");
  return w;
}

Ctx alloc_ctx() {
  Ctx c;
  c.cr = alloc(DType::BF16, {W, H});
  for (int l = 0; l < NL; l++) {
    c.kr[l] = alloc(DType::BF16, {NKV, W, HD});
    c.vr[l] = alloc(DType::BF16, {NKV, W, HD});
  }
  return c;
}

Scratch alloc_scratch(int R) {
  NE_CHECK(R >= 2 && R <= RMAX, "dflash block rows %d", R);
  Scratch s;
  s.R = R;
  s.tok = alloc(DType::U32, {RMAX});
  for (Tensor *t : {&s.h, &s.nrm, &s.x, &s.o, &s.hn}) *t = alloc(DType::F32, {RMAX, H});
  s.dyn = alloc(DType::F32, {RMAX, DYN});
  s.q = alloc(DType::F32, {RMAX, NQ * HD});
  s.att = alloc(DType::F32, {RMAX, NQ * HD});
  s.k = alloc(DType::F32, {RMAX, NKV * HD});
  s.v = alloc(DType::F32, {RMAX, NKV * HD});
  s.kb = alloc(DType::F32, {NKV, RMAX, HD});
  s.vb = alloc(DType::F32, {NKV, RMAX, HD});
  s.part = alloc(DType::F32, {NKV * SPLITS * 32 * 130});
  s.g = alloc(DType::F32, {RMAX, FF});
  s.u = alloc(DType::F32, {RMAX, FF});
  s.a = alloc(DType::F32, {RMAX, FF});
  s.sh = alloc(DType::F32, {RMAX, SELR});
  s.coarse = alloc(DType::F32, {RMAX, 248320});
  s.cand = alloc(DType::F32, {RMAX, 128, NC, 2});
  s.sl = alloc(DType::F32, {RMAX, NC, 2});
  s.amax = alloc(DType::U32, {16});
  s.qo = alloc(DType::F32, {RMAX, NC, 2});
  s.dr = alloc(DType::U32, {RMAX});
  s.P = alloc(DType::U32, {8});
  s.SP = alloc(DType::U32, {12});
  s.cx = alloc(DType::F32, {CROWS, H});
  s.cc = alloc(DType::F32, {CROWS, H});
  s.ck = alloc(DType::F32, {CROWS, NKV * HD});
  s.cv = alloc(DType::F32, {CROWS, NKV * HD});
  return s;
}

void record_ctx_c(Recorder &r, const Weights &w, Ctx &c, Scratch &s, const Tensor &feat, int rows, uint32_t r0) {
  NE_CHECK(rows >= 1 && rows <= CROWS, "dflash context rows %d", rows);
  nax::record_q8(r, rows, feat, {&w.fc}, {s.cx});
  r.barrier();
  r.dispatch(pso("dfl_rms"), {s.cx, w.hnorm, s.cc, Arg::value(std::array<uint32_t, 2>{(uint32_t)H, 1u})}, sz(rows), sz(256));
  r.barrier();
  r.dispatch(pso("dfl_ctx_c"), {s.cc, c.cr, s.P, Arg::value(std::array<uint32_t, 2>{(uint32_t)rows, r0})},
             sz((rows * H + 255) / 256), sz(256));
}
void record_ctx_kv(Recorder &r, const Weights &w, Ctx &c, Scratch &s, int rows, uint32_t r0) {
  for (int l = 0; l < NL; l++) {
    const LayerW &L = w.L[l];
    nax::record_q8(r, rows, s.cc, {&L.k, &L.v}, {s.ck, s.cv});
    r.barrier();
    r.dispatch(pso("dfl_ctx_kv", 128), {s.ck, s.cv, L.kn, w.invf, c.kr[l], c.vr[l], s.P, Arg::value(r0)}, sz(rows * NKV),
               sz(128));
    r.barrier();
  }
}
void record_ctx_rows(Recorder &r, const Weights &w, Ctx &c, Scratch &s, const Tensor &feat, int rows, uint32_t r0) {
  record_ctx_c(r, w, c, s, feat, rows, r0);
  record_ctx_kv(r, w, c, s, rows, r0);
}
void record_rebuild(Recorder &r, const Weights &w, Ctx &c, Scratch &s) {
  for (int i = 0; i < W / CROWS; i++) {
    r.dispatch(pso("dfl_cr_gather"), {c.cr, s.cc, s.P, Arg::value((uint32_t)(CROWS * i))}, sz(CROWS * H / 256), sz(256));
    r.barrier();
    record_ctx_kv(r, w, c, s, CROWS, (uint32_t)(CROWS * i));
  }
}

void record_block(Recorder &r, const Weights &w, Ctx &c, Scratch &s, const mtp::MtpW &mw, const fn::Q8 &embed,
                  const Tensor &vtok) {
  const int R = s.R, P = R - 1;
  const uint32_t uR = (uint32_t)R;
  r.dispatch(pso("dfl_embed"), {s.tok, embed.w, embed.s, embed.b, s.h, Arg::value((uint32_t)embed.group)}, sz(R), sz(256));
  r.barrier();
  auto rms = [&](const Tensor &x, const Tensor &wt, const Tensor &y, int rows) {
    r.dispatch(pso("dfl_rms"), {x, wt, y, Arg::value(std::array<uint32_t, 2>{(uint32_t)H, 0u})}, sz(rows), sz(256));
    r.barrier();
  };
  auto conv = [&](const Tensor &x, const Tensor &base, uint32_t stage, const Tensor &res, const Tensor &out, bool fin) {
    r.dispatch(pso("dfl_conv"), {x, s.dyn, base, res, out, Arg::value(U4{uR, stage, fin ? 1u : 0u, 0})},
               sz((R * H + 255) / 256), sz(256));
    r.barrier();
  };
  for (int l = 0; l < NL; l++) {
    const LayerW &L = w.L[l];
    rms(s.h, L.in_ln, s.nrm, R);
    nax::record_q8(r, R, s.nrm, {&L.kp_att}, {s.dyn});
    r.barrier();
    conv(s.nrm, L.base_att, 0, s.nrm, s.x, false);
    if (w.bits == 8) {
      nax::record_q8(r, R, s.x, {&L.q, &L.k, &L.v}, {s.q, s.k, s.v});
    } else {
      lin(r, R, s.x, L.q, s.q, w.bits);
      nax::record_q8(r, R, s.x, {&L.k, &L.v}, {s.k, s.v});
    }
    r.barrier();
    r.dispatch(pso("dfl_qk_prep", 128), {s.q, s.k, s.v, L.qn, L.kn, w.invf, s.kb, s.vb, s.P,
                                         Arg::value(std::array<uint32_t, 2>{uR, (uint32_t)RMAX})},
               sz(R * (NQ + NKV)), sz(128));
    r.barrier();
    r.dispatch(pso("dfl_attn"), {s.q, c.kr[l], c.vr[l], s.kb, s.vb, s.part, s.P,
                                 Arg::value(std::array<uint32_t, 2>{uR, (uint32_t)RMAX})},
               MTLSizeMake(NKV, SPLITS, 1), sz(256));
    r.barrier();
    r.dispatch(pso("dfl_attn_reduce", 128), {s.part, s.att, Arg::value(std::array<uint32_t, 2>{uR, (uint32_t)SPLITS})},
               sz(NKV * 32), sz(128));
    r.barrier();
    lin(r, R, s.att, L.o, s.o, w.bits);
    r.barrier();
    conv(s.o, L.base_att, 1, s.h, s.h, true);
    rms(s.h, L.post_ln, s.nrm, R);
    nax::record_q8(r, R, s.nrm, {&L.kp_mlp}, {s.dyn});
    r.barrier();
    conv(s.nrm, L.base_mlp, 0, s.nrm, s.x, false);
    if (w.bits == 8) {
      nax::record_q8(r, R, s.x, {&L.gate, &L.up}, {s.g, s.u});
    } else {
      lin(r, R, s.x, L.gate, s.g, w.bits);
      lin(r, R, s.x, L.up, s.u, w.bits);
    }
    r.barrier();
    r.dispatch(pso("dfl_silu_mul"), {s.g, s.u, s.a, Arg::value((uint32_t)(R * FF))}, sz((R * FF + 255) / 256), sz(256));
    r.barrier();
    lin(r, R, s.a, L.down, s.o, w.bits);
    r.barrier();
    conv(s.o, L.base_mlp, 1, s.h, s.h, true);
  }
  // readout of rows 1 .. R - 1
  const Tensor hn = s.hn.rows(0, P);
  r.dispatch(pso("dfl_rms"), {s.h.rows(1, P), w.norm, hn, Arg::value(std::array<uint32_t, 2>{(uint32_t)H, 0u})}, sz(P), sz(256));
  r.barrier();
  nax::record_q8(r, P, hn, {&w.sel}, {s.sh});
  const bool mapped = mw.draft_ids.buf != nil;
  if (mw.draft_bits) {
    const uint32_t V = (uint32_t)mw.draft_head.rows;
    nax::record_qn(r, P, hn, mw.draft_head, mw.draft_bits, s.coarse);
    r.barrier();
    r.dispatch(pso("mtp_topk_local"), {s.coarse, s.cand, Arg::value(V), Arg::value((uint32_t)NC)}, MTLSizeMake(128, P, 1),
               sz(256));
    r.barrier();
    r.dispatch(pso("mtp_shortlist", 1024), {s.cand, hn, mw.lm_head.w, mw.lm_head.s, mw.lm_head.b, s.amax, s.sl, Arg::value(V),
                                            Arg::value(0u), Arg::value((uint32_t)NC), mapped ? mw.draft_ids : s.amax,
                                            Arg::value(mapped ? 1u : 0u)},
               sz(P), sz(1024));
  } else {  // the full 8-bit head: exact top 16
    nax::record_q8(r, P, hn, {&mw.lm_head}, {s.coarse});
    r.barrier();
    r.dispatch(pso("mtp_topk_local"), {s.coarse, s.cand, Arg::value((uint32_t)mw.vocab), Arg::value((uint32_t)NC)},
               MTLSizeMake(128, P, 1), sz(256));
    r.barrier();
    r.dispatch(pso("mtp_topk_merge", 1024), {s.cand, s.sl, s.amax, Arg::value((uint32_t)NC)}, sz(P), sz(1024));
  }
  r.barrier();
  r.dispatch(pso("dfl_select"), {s.sl, s.sh, w.pcb, w.scb, s.SP, vtok, s.qo, s.dr}, sz(1), sz(256));
  r.barrier();
}

}  // namespace ne::dfl
