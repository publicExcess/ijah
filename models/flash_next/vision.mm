// Flash-Next vision tower (lab-vision): weights -> padded bf16 GPU copies, one recording per encode. See vision.h.
#include "models/flash_next/vision.h"

#include <array>
#include <cmath>
#include <cstring>
#include <string>

namespace ne::vis {

namespace {
struct VitGemm {  // = vision.metal VitGemm
  int K, N, M, ldx, ldy, ldr;
};
struct NaxAttnParams {  // MLX AttnParams (kernels/prefill/mlx/attn_params.h)
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
// MLX's VisionRotaryEmbedding(36) inverse frequencies 1 / 10000^(2k/36) as its GPU computes them (7 of 18 differ by
// one ulp from a correctly rounded f32 table; tests/vision checks the rope against the reference)
const uint32_t kInvFreqBits[18] = {0x3f800000, 0x3f1977cc, 0x3eb800d5, 0x3e5c9d35, 0x3e044133, 0x3d9e91b6,
                                   0x3d3e1e95, 0x3ce3f27e, 0x3c88a69b, 0x3c23d70a, 0x3bc4705e, 0x3b6b8630,
                                   0x3b0d3168, 0x3aa94938, 0x3a4af7f2, 0x39f35a5c, 0x3991e2e1, 0x392ee9be};

int cdiv(int a, int b) { return (a + b - 1) / b; }
MTLSize sz(int x, int y = 1) { return MTLSizeMake(x, y, 1); }
id<MTLComputePipelineState> pso(const std::string &n) { return Pipelines::get(n); }

const uint16_t *src(const Pack &p, const std::string &name, std::initializer_list<int64_t> shape) {
  const TensorInfo *ti = p.info(name);
  NE_CHECK(ti, "vision tower: missing %s", name.c_str());
  NE_CHECK(ti->t.dtype == DType::BF16, "vision tower: %s is not bf16", name.c_str());
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  NE_CHECK((int64_t)ti->nbytes == 2 * n, "vision tower: %s has %llu bytes, expected %lld", name.c_str(),
           (unsigned long long)ti->nbytes, (long long)(2 * n));
  return (const uint16_t *)ti->host;
}
// dst [rows][cols] (zero-filled) <- src [srows][scols] with row / col maps (-1 = drop)
Tensor padded(const uint16_t *s, int srows, int scols, int rows, int cols, int (*rmap)(int), int (*cmap)(int)) {
  Tensor t = alloc(DType::BF16, {rows, cols});
  uint16_t *d = (uint16_t *)t.host();
  for (int r = 0; r < srows; r++) {
    const int dr = rmap ? rmap(r) : r;
    if (!cmap) {
      memcpy(d + (size_t)dr * cols, s + (size_t)r * scols, (size_t)scols * 2);
      continue;
    }
    for (int c = 0; c < scols; c++) d[(size_t)dr * cols + cmap(c)] = s[(size_t)r * scols + c];
  }
  return t;
}
Tensor copy1(const Pack &p, const std::string &name, int n) {
  Tensor t = alloc(DType::BF16, {n});
  memcpy(t.host(), src(p, name, {n}), (size_t)n * 2);
  return t;
}
int head_pad(int i) { return (i / HD) * HDP + i % HD; }          // 72-wide head dims -> 96-wide slots
int qkv_pad(int r) { return (r / HID) * OP + head_pad(r % HID); }  // q | k | v rows -> [3][16][96]
uint16_t f2bf(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
}  // namespace

std::unique_ptr<Tower> load_tower(const Pack &p) {
  const Json &vc = p.config()["vision_config"];
  NE_CHECK(vc.is_object(), "pack config has no vision_config");
  NE_CHECK(vc["hidden_size"].num() == HID && vc["num_heads"].num() == NH && vc["depth"].num() == DEPTH &&
               vc["intermediate_size"].num() == MLP && vc["out_hidden_size"].num() == OUT &&
               vc["patch_size"].num() == 16 && vc["spatial_merge_size"].num() == 2 &&
               vc["temporal_patch_size"].num() == 2 && vc["num_position_embeddings"].num() == GRID * GRID,
           "vision_config differs from the Qwen3.8 tower this engine implements");
  auto t = std::make_unique<Tower>();
  const std::string P = "vision_tower.";
  for (int l = 0; l < DEPTH; l++) {
    const std::string b = P + "blocks." + std::to_string(l) + ".";
    Block k;
    k.n1w = copy1(p, b + "norm1.weight", HID);
    k.n1b = copy1(p, b + "norm1.bias", HID);
    k.n2w = copy1(p, b + "norm2.weight", HID);
    k.n2b = copy1(p, b + "norm2.bias", HID);
    k.qkv_w = padded(src(p, b + "attn.qkv.weight", {3 * HID, HID}), 3 * HID, HID, QKVP, HID, qkv_pad, nullptr);
    k.qkv_b = padded(src(p, b + "attn.qkv.bias", {3 * HID}), 3 * HID, 1, QKVP, 1, qkv_pad, nullptr);
    k.proj_w = padded(src(p, b + "attn.proj.weight", {HID, HID}), HID, HID, HID, OP, nullptr, head_pad);
    k.proj_b = copy1(p, b + "attn.proj.bias", HID);
    k.fc1_w = padded(src(p, b + "mlp.linear_fc1.weight", {MLP, HID}), MLP, HID, MLPP, HID, nullptr, nullptr);
    k.fc1_b = padded(src(p, b + "mlp.linear_fc1.bias", {MLP}), MLP, 1, MLPP, 1, nullptr, nullptr);
    k.fc2_w = padded(src(p, b + "mlp.linear_fc2.weight", {HID, MLP}), HID, MLP, HID, MLPP, nullptr, nullptr);
    k.fc2_b = copy1(p, b + "mlp.linear_fc2.bias", HID);
    t->blk.push_back(k);
  }
  // Conv3d weight in the pack's MLX layout [1152][t 2][16][16][c 3] = a [1152][1536] GEMM weight over (t, h, w, c)
  t->pe_w = padded(src(p, P + "patch_embed.proj.weight", {HID, PIN}), HID, PIN, HID, PIN, nullptr, nullptr);
  t->pe_b = copy1(p, P + "patch_embed.proj.bias", HID);
  t->pos_tab = padded(src(p, P + "pos_embed.weight", {GRID * GRID, HID}), GRID * GRID, HID, GRID * GRID, HID, nullptr,
                      nullptr);
  t->mn_w = copy1(p, P + "merger.norm.weight", HID);
  t->mn_b = copy1(p, P + "merger.norm.bias", HID);
  t->m1_w = padded(src(p, P + "merger.linear_fc1.weight", {MRG, MRG}), MRG, MRG, MRG, MRG, nullptr, nullptr);
  t->m1_b = copy1(p, P + "merger.linear_fc1.bias", MRG);
  t->m2_w = padded(src(p, P + "merger.linear_fc2.weight", {OUT, MRG}), OUT, MRG, OUT, MRG, nullptr, nullptr);
  t->m2_b = copy1(p, P + "merger.linear_fc2.bias", OUT);
  t->invf = alloc(DType::F32, {18});
  memcpy(t->invf.host(), kInvFreqBits, sizeof kInvFreqBits);
  for (const Block &k : t->blk)
    for (const Tensor *x : {&k.n1w, &k.n1b, &k.n2w, &k.n2b, &k.qkv_w, &k.qkv_b, &k.proj_w, &k.proj_b, &k.fc1_w,
                            &k.fc1_b, &k.fc2_w, &k.fc2_b})
      t->weight_bytes += x->nbytes();
  for (const Tensor *x : {&t->pe_w, &t->pe_b, &t->pos_tab, &t->mn_w, &t->mn_b, &t->m1_w, &t->m1_b, &t->m2_w, &t->m2_b})
    t->weight_bytes += x->nbytes();
  if (const char *e = getenv("NE_VIT_LAYERS")) t->depth = std::max(0, std::min(DEPTH, atoi(e)));
  return t;
}

static void grow(Tower &t, int n) {
  if (n <= t.cap) return;
  const int c = cdiv(n, 256) * 256;
  t.xin = alloc(DType::BF16, {c, PIN});
  t.pos = alloc(DType::BF16, {c, HID});
  t.h = alloc(DType::BF16, {c, HID});
  t.ln = alloc(DType::BF16, {c, HID});
  t.qkv = alloc(DType::BF16, {c, QKVP});
  t.o = alloc(DType::BF16, {c, OP});
  t.mlp = alloc(DType::BF16, {c, MLPP});
  t.m1 = alloc(DType::BF16, {c / 4, MRG});
  t.out = alloc(DType::BF16, {c / 4, OUT});
  t.rpos = alloc(DType::I32, {c, 2});
  t.pidx = alloc(DType::I32, {4, c});
  t.pwt = alloc(DType::F32, {4, c});
  t.cap = c;
}

static void gemm(Recorder &r, const char *epi, const Tensor &w, const Tensor &x, const Tensor &b, const Tensor *res,
                 const Tensor &y, int M, int N, int K, int ldx, int ldy, int ldr) {
  NE_CHECK(N % 64 == 0 && K % 32 == 0, "vit gemm %d x %d", N, K);
  VitGemm p{K, N, M, ldx, ldy, ldr};
  r.dispatch(pso(std::string("vit_gemm_") + epi), {w, x, b, res ? *res : b, y, Arg::value(p)}, sz(N / 64, cdiv(M, 64)),
             sz(128));
}

void encode(Tower &t, const float *patches, int n, const std::vector<std::array<int, 3>> &grids, uint16_t *out,
            uint16_t *hidden) {
  NE_CHECK(n > 0 && !grids.empty(), "vision encode: no patches");
  int tot = 0;
  for (auto &g : grids) {
    NE_CHECK(g[0] == 1 && g[1] >= 2 && g[2] >= 2 && g[1] % 2 == 0 && g[2] % 2 == 0,
             "vision encode: grid (%d, %d, %d) unsupported (images only: t 1, even h / w)", g[0], g[1], g[2]);
    tot += g[1] * g[2];
  }
  NE_CHECK(tot == n, "vision encode: %d patches but the grids hold %d", n, tot);
  grow(t, n);
  // inputs: patches -> bf16; per image: rope positions and the position-table interpolation (merge-block order)
  uint16_t *xb = (uint16_t *)t.xin.host();
  for (size_t i = 0; i < (size_t)n * PIN; i++) xb[i] = f2bf(patches[i]);
  int32_t *rp = (int32_t *)t.rpos.host(), *pi = (int32_t *)t.pidx.host();
  float *pw = (float *)t.pwt.host();
  const int cap = t.cap;
  int base = 0;
  for (auto &g : grids) {
    const int gh = g[1], gw = g[2];
    // = fast_pos_embed_interpolate: mx.linspace(0, 47, k) = t * 47 with t = i / (k - 1) (f32), floor / ceil corners
    auto axis = [](int k, std::vector<int> &fl, std::vector<int> &ce, std::vector<float> &fr) {
      fl.resize(k); ce.resize(k); fr.resize(k);
      for (int i = 0; i < k; i++) {
        const float tt = k > 1 ? (float)i / (float)(k - 1) : 0.0f;
        const float v = (1.0f - tt) * 0.0f + tt * (float)(GRID - 1);
        fl[i] = (int)v;
        ce[i] = std::min(fl[i] + 1, GRID - 1);
        fr[i] = v - (float)fl[i];
      }
    };
    std::vector<int> hf, hc, wf, wc;
    std::vector<float> dh, dw;
    axis(gh, hf, hc, dh);
    axis(gw, wf, wc, dw);
    int k = base;
    for (int bh = 0; bh < gh / 2; bh++)
      for (int bw = 0; bw < gw / 2; bw++)
        for (int ih = 0; ih < 2; ih++)
          for (int iw = 0; iw < 2; iw++, k++) {
            const int row = bh * 2 + ih, col = bw * 2 + iw;
            rp[2 * k] = row;
            rp[2 * k + 1] = col;
            const int ids[4] = {hf[row] * GRID + wf[col], hf[row] * GRID + wc[col], hc[row] * GRID + wf[col],
                                hc[row] * GRID + wc[col]};
            const float wts[4] = {(1.0f - dh[row]) * (1.0f - dw[col]), (1.0f - dh[row]) * dw[col],
                                  dh[row] * (1.0f - dw[col]), dh[row] * dw[col]};
            for (int c = 0; c < 4; c++) {
              pi[c * cap + k] = ids[c];
              uint32_t u = (uint32_t)f2bf(wts[c]) << 16;  // the weights as bf16 (weight_tensor's dtype)
              memcpy(&pw[c * cap + k], &u, 4);
            }
          }
    base += gh * gw;
  }
  Recorder r;
  r.begin();
  const int capi = cap;
  // pos_interp reads idx / wt with stride N = cap (rows [0, n))
  r.dispatch(pso("vit_pos_interp"), {t.pos_tab, t.pidx, t.pwt, t.pos, Arg::value(capi)}, sz(n), sz(256));
  r.barrier();
  gemm(r, "conv", t.pe_w, t.xin, t.pe_b, &t.pos, t.h, n, HID, PIN, PIN, HID, HID);
  r.barrier();
  const float scale = 1.0f / std::sqrt((float)HD);
  for (int l = 0; l < t.depth; l++) {
    const Block &k = t.blk[l];
    r.dispatch(pso("vit_layernorm"), {t.h, k.n1w, k.n1b, t.ln, Arg::value(n)}, sz(n), sz(256));
    r.barrier();
    gemm(r, "bias", k.qkv_w, t.ln, k.qkv_b, nullptr, t.qkv, n, QKVP, HID, HID, QKVP, 0);
    r.barrier();
    r.dispatch(pso("vit_rope"), {t.qkv, t.rpos, t.invf, Arg::value(n)}, sz(n, 2 * NH), sz(64));
    r.barrier();
    int b0 = 0;
    for (auto &g : grids) {  // attention within each image (cu_seqlens)
      const int L = g[1] * g[2];
      NaxAttnParams ap{};
      ap.B = 1; ap.H = NH; ap.D = 80; ap.qL = L; ap.kL = L; ap.gqa_factor = 1; ap.scale = scale;
      ap.NQ = cdiv(L, 64); ap.NK = cdiv(L, 32); ap.NQ_aligned = L / 64; ap.NK_aligned = L / 32;
      ap.qL_rem = L - ap.NQ_aligned * 64; ap.kL_rem = L - ap.NK_aligned * 32; ap.qL_off = 0;
      const int64_t qs[3] = {(int64_t)L * QKVP, HDP, QKVP}, os[3] = {(int64_t)L * OP, HDP, OP};
      for (int i = 0; i < 3; i++) { ap.Q_strides[i] = ap.K_strides[i] = ap.V_strides[i] = qs[i]; ap.O_strides[i] = os[i]; }
      auto ps = Pipelines::get("vit_attn", {FC::b(200, L % 64 == 0), FC::b(201, L % 32 == 0), FC::b(300, false),
                                            FC::b(301, false), FC::b(302, false)});
      const size_t q0 = t.qkv.offset + (size_t)b0 * QKVP * 2;
      r.dispatch(ps, {Arg(t.qkv.buf, q0), Arg(t.qkv.buf, q0 + OP * 2), Arg(t.qkv.buf, q0 + 2 * OP * 2),
                      Arg(t.o.buf, t.o.offset + (size_t)b0 * OP * 2), Arg::value(ap)},
                 sz(ap.NQ, NH), sz(128));
      b0 += L;
    }
    r.barrier();
    gemm(r, "res", k.proj_w, t.o, k.proj_b, &t.h, t.h, n, HID, OP, OP, HID, HID);
    r.barrier();
    if (l == t.depth - 1 && getenv("NE_VIT_HALF")) break;  // debug: the last block stops after its attention
    r.dispatch(pso("vit_layernorm"), {t.h, k.n2w, k.n2b, t.ln, Arg::value(n)}, sz(n), sz(256));
    r.barrier();
    gemm(r, "gelu_t", k.fc1_w, t.ln, k.fc1_b, nullptr, t.mlp, n, MLPP, HID, HID, MLPP, 0);
    r.barrier();
    gemm(r, "res", k.fc2_w, t.mlp, k.fc2_b, &t.h, t.h, n, HID, MLPP, MLPP, HID, HID);
    r.barrier();
  }
  // merger: LayerNorm(1152) per patch, 4 consecutive patches (one merge block) -> 4608, fc1 + GELU, fc2 -> 2560
  r.dispatch(pso("vit_layernorm"), {t.h, t.mn_w, t.mn_b, t.ln, Arg::value(n)}, sz(n), sz(256));
  r.barrier();
  const int m = n / 4;
  gemm(r, "gelu_e", t.m1_w, t.ln, t.m1_b, nullptr, t.m1, m, MRG, MRG, MRG, MRG, 0);
  r.barrier();
  gemm(r, "bias", t.m2_w, t.m1, t.m2_b, nullptr, t.out, m, OUT, MRG, MRG, OUT, 0);
  r.end();
  r.run();
  t.last_gpu_ms = r.gpu_ms();
  memcpy(out, t.out.host(), (size_t)m * OUT * 2);
  if (hidden) memcpy(hidden, t.h.host(), (size_t)n * HID * 2);
}

}  // namespace ne::vis

struct ne_vision {
  std::unique_ptr<ne::vis::Tower> t;
};

extern "C" {
NE_API ne_vision *ne_vision_create(ne_pack *pack) {
  ne_vision *v = nullptr;
  int rc = ne::c_call([&] {
    const ne::Pack *p = ne::pack_from_c(pack);
    NE_CHECK(p, "ne_vision_create: no pack");
    auto h = std::make_unique<ne_vision>();
    h->t = ne::vis::load_tower(*p);
    v = h.release();
  });
  return rc == 0 ? v : nullptr;
}
NE_API void ne_vision_free(ne_vision *v) { delete v; }
NE_API int ne_vision_encode(ne_vision *v, const float *patches, int n_patches, const int *grids, int n_img,
                            uint16_t *out, uint16_t *hidden, double *gpu_ms) {
  return ne::c_call([&] {
    NE_CHECK(v && patches && grids && out && n_img > 0, "ne_vision_encode: bad arguments");
    std::vector<std::array<int, 3>> g(n_img);
    for (int i = 0; i < n_img; i++) g[i] = {grids[3 * i], grids[3 * i + 1], grids[3 * i + 2]};
    ne::vis::encode(*v->t, patches, n_patches, g, out, hidden);
    if (gpu_ms) *gpu_ms = v->t->last_gpu_ms;
  });
}
NE_API int64_t ne_vision_weight_bytes(ne_vision *v) { return v ? (int64_t)v->t->weight_bytes : -1; }
}
