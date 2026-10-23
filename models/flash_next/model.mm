// Flash-Next whole model: one recording per decode step (embedding, 48 layers, mixer, lm_head, sampler). Owner:
// lab-forward. Layout and numbers: docs/forward.md; layer graph: layer.mm; kernels: kernels/fn_model.metal.
#include "models/flash_next/model.h"

#include "models/flash_next/prefill.h"

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace ne::fn {

static MTLSize sz1(int n) { return MTLSizeMake(n, 1, 1); }

// mixer + lm_head + sampler on hf. The 5 dispatches take their threadgroup counts from m.ind (indirect), so one
// recording serves decode steps (counts = kHeadTGs) and prefill tokens (counts = 0: the head is skipped) -- one
// recorder = one residency set: alternating two recorders over the 81 GB pack costs ~300 ms per switch.
static const uint32_t kHeadTGs[5] = {(LR / 8 + 1) * HC, HID / 8, (VOCAB + 7) / 8, (VOCAB + 1023) / 1024, 1};
static void set_head(Model &m, bool on) {
  uint32_t *ind = (uint32_t *)m.ind.host();
  for (int i = 0; i < 5; i++) {
    ind[i * 4] = on ? kHeadTGs[i] : 0;
    ind[i * 4 + 1] = ind[i * 4 + 2] = 1;
  }
}
static void record_head(Model &m, Recorder &r, const Tensor &hf) {
  const int R = m.rows;
  const std::string rs = "_r" + std::to_string(R);
  const HcW &x = m.mixer;
  Scratch &s = m.s;
  auto ind = [&](int i) { return m.ind.offset + (size_t)i * 16; };
  if (x.down.t6) {  // lab-t6wire: bf16 mixer (T6 pack): direct dispatches (also run for prefill tokens; ~30 us)
    t6_hc_read(r, x, R, hf, false, s, s.mix_a, s.inj_a, m.x);
  } else if (hc_v2(R)) {  // HC read v2 (fn_hc_pre3 -> partials, fn_hc_mix2), same threadgroup counts
    r.dispatch_indirect(pso("fn_hc_pre3_g" + std::to_string(x.down.group) + rs),
                        {hf, x.norm, x.down.w, x.down.s, x.down.b, m.dummy_inj, s.inj_a, s.hc_part}, m.ind.buf, ind(0),
                        sz1(256));
    r.barrier();
    r.dispatch_indirect(pso("fn_hc_mix2_g" + std::to_string(x.up.group) + rs),
                        {hf, x.norm, x.up.w, x.up.s, x.up.b, s.mix_a, s.inj_a, m.x, s.hc_part}, m.ind.buf, ind(1),
                        sz1(256));
    r.barrier();
  } else {
    r.dispatch_indirect(pso("fn_hc_pre_g" + std::to_string(x.down.group) + rs),
                        {hf, x.norm, x.down.w, x.down.s, x.down.b, m.dummy_inj, s.mix_a, s.inj_a, s.hc_part, s.hc_ctr},
                        m.ind.buf, ind(0), sz1(256));
    r.barrier();
    r.dispatch_indirect(pso("fn_hc_mix_g" + std::to_string(x.up.group) + rs),
                        {hf, x.norm, x.up.w, x.up.s, x.up.b, s.mix_a, s.inj_a, m.x}, m.ind.buf, ind(1), sz1(256));
    r.barrier();
  }
  const uint32_t nv = VOCAB;
  r.dispatch_indirect(pso("fn_lm_head_g" + std::to_string(m.head.group) + rs),
                      {m.x, m.head.w, m.head.s, m.head.b, m.logits, Arg::value(nv)}, m.ind.buf, ind(2), sz1(256));
  r.barrier();
  // sampler (row 0 only: the decode step)
  r.dispatch_indirect(pso("fn_sample_p1", 1024), {m.logits, m.sp, m.cand, m.lse, m.mask, m.bias, m.cnt}, m.ind.buf, ind(3), sz1(1024));
  r.barrier();
  r.dispatch_indirect(pso("fn_sample_p2", 1024), {m.cand, m.lse, m.sp, m.res, m.tok, m.top, m.cnt}, m.ind.buf, ind(4), sz1(1024));
  r.barrier();
}

static void record_a(Model &m, Recorder &r, Scratch &s, const Tensor &tok, const Tensor *h) {
  r.begin();
  r.dispatch(pso("fn_embed_g" + std::to_string(m.embed.group)), {tok, m.embed.w, m.embed.s, m.embed.b, h[0], m.vfeat},
             sz1(s.rows), sz1(256));
  r.barrier();
  record_gdn_layer(r, m.L[0], m.st[0], s, h[0], h[1]);
  r.end();
}
static void record_b(Model &m, Recorder &r, Scratch &s, const Tensor *h, bool head) {
  r.begin();
  for (size_t l = 1; l < m.L.size(); l++) {
    record_gdn_layer(r, m.L[l], m.st[l], s, h[l % 2], h[(l + 1) % 2]);
    if (m.dfl_cap && dfl_slot((int)l) >= 0) {  // lab-dflash: the drafter's features of this step's rows
      struct { uint32_t x, y, z, w; } a{(uint32_t)s.rows, (uint32_t)dfl_slot((int)l), 0, 0};
      r.dispatch(pso("dfl_hc_mean"), {h[(l + 1) % 2], m.dfl_f, Arg::value(a)}, sz1(s.rows * 10), sz1(256));
    }
  }
  if (head) record_head(m, r, h[m.L.size() % 2]);
  r.end();
}
void set_dfl_capture(Model &m, bool on) {
  if (m.dfl_cap == on) return;
  if (on && !m.dfl_f.buf) m.dfl_f = alloc(DType::F32, {4, 5 * HID});
  m.dfl_cap = on;
  record_b(m, *m.rec_b, m.s, m.h, true);
  if (m.rec_b4) record_b(m, *m.rec_b4, m.s4, m.h4, false);
  m.n_full = m.rec_a->size() + m.rec_b->size();
  m.n_body = m.n_full - 5;
}

Model::~Model() {
  if (ngram_fd >= 0) close(ngram_fd);
  for (auto &e : extra_rs) {
    [Device::get().queue4 removeResidencySet:e.second];
    [e.second endResidency];
  }
  if (queue_rs) {
    [Device::get().queue4 removeResidencySet:queue_rs];
    [queue_rs endResidency];
  }
}

static void gather_row(Model &m, int64_t p2, int64_t p1, int token, float *emb);
void gather_ple(Model &m, int token) { gather_row(m, m.hist[0], m.hist[1], token, (float *)m.s.ple_emb.host()); }
static void gather_row(Model &m, int64_t p2, int64_t p1, int token, float *emb) {
  int64_t ids[PLE_NG];
  ngram_ids(m.L[1].ple, p2, p1, token, m.eos, ids);
  static thread_local uint16_t raw[PLE_NG * PLE_W];
  __block int bad = 0;
  uint16_t *rp = raw;
  const int fd = m.ngram_fd;
  const uint64_t off = m.ngram_off;
  const int64_t *idp = ids;
  dispatch_apply(PLE_NG, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
    const ssize_t n = pread(fd, rp + i * PLE_W, PLE_W * 2, (off_t)(off + (uint64_t)idp[i] * PLE_W * 2));
    if (n != PLE_W * 2) bad = 1;
  });
  NE_CHECK(!bad, "n-gram row pread failed");
  for (int i = 0; i < PLE_NG * PLE_W; i++) {
    const uint32_t u = (uint32_t)raw[i] << 16;
    memcpy(&emb[i], &u, 4);
  }
}

void gather_ple_rows(Model &m, int64_t p2, int64_t p1, const int *tokens, int n, float *emb) {
  NE_CHECK(n >= 1 && n <= 16, "gather_ple_rows: %d rows", n);
  int64_t ids[16 * PLE_NG];
  for (int r = 0; r < n; r++) {
    ngram_ids(m.L[1].ple, p2, p1, tokens[r], m.eos, ids + r * PLE_NG);
    p2 = p1;
    p1 = tokens[r];
  }
  static thread_local std::vector<uint16_t> raw;
  raw.resize((size_t)n * PLE_NG * PLE_W);
  __block int bad = 0;
  uint16_t *rp = raw.data();
  const int fd = m.ngram_fd;
  const uint64_t off = m.ngram_off;
  const int64_t *idp = ids;
  dispatch_apply((size_t)n * PLE_NG, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
    const ssize_t k = pread(fd, rp + i * PLE_W, PLE_W * 2, (off_t)(off + (uint64_t)idp[i] * PLE_W * 2));
    if (k != PLE_W * 2) bad = 1;
  });
  NE_CHECK(!bad, "n-gram row pread failed");
  for (size_t i = 0; i < (size_t)n * PLE_NG * PLE_W; i++) {
    const uint32_t u = (uint32_t)raw[i] << 16;
    memcpy(&emb[i], &u, 4);
  }
}

void add_queue_residency(Model &m, const std::string &prefix) {
  Device &d = Device::get();
  if (!d.has_metal4 || !d.queue4 || !m.queue_rs) return;
  for (auto &e : m.extra_rs)
    if (e.first == prefix) return;
  MTLResidencySetDescriptor *rsd = [MTLResidencySetDescriptor new];
  rsd.initialCapacity = 1024;
  NSError *err = nil;
  id<MTLResidencySet> rs = [d.dev newResidencySetWithDescriptor:rsd error:&err];
  NE_CHECK(rs, "newResidencySet: %s", err ? err.localizedDescription.UTF8String : "?");
  for (const std::string &n : m.pack->names()) {
    if (n.rfind(prefix, 0) != 0) continue;
    const TensorInfo *ti = m.pack->info(n);
    if (ti && ti->t.buf) [rs addAllocation:ti->t.buf];
  }
  [rs commit];
  [rs requestResidency];
  [d.queue4 addResidencySet:rs];
  m.extra_rs.emplace_back(prefix, rs);
}

std::unique_ptr<Model> load_model(const Pack &pack, int tmax, const Model *state_from) {
  NE_CHECK(tmax >= 32 && tmax % 32 == 0, "tmax %d must be a positive multiple of 32", tmax);
  if (g_attn_qsa) tmax = (tmax + 63) / 64 * 64;  // lab-attn's caches
  const Model *sf = state_from;
  NE_CHECK(!sf || sf->tmax == tmax, "state_from: tmax %d != %d", sf ? sf->tmax : 0, tmax);
  auto m = std::make_unique<Model>();
  m->pack_hold = pack.weak_from_this().lock();
  m->pack = &pack;
  m->tmax = tmax;
  const Json &tc = pack.text_config();
  const int nl = (int)tc["num_hidden_layers"].i64();
  NE_CHECK(nl == 48, "expected 48 layers, config says %d", nl);
  // debug / CPU-RAM-bounded tests: NE_FN_LAYERS=k loads only layers 0..k-1 (k >= 2: PLE lives on layer 1; ~1.7 GB
  // per layer). The outputs are meaningless, the state machinery (caches, snapshots, the MTP loop) is the real one.
  int nload = nl;
  if (const char *e = getenv("NE_FN_LAYERS")) nload = std::max(2, std::min(nl, atoi(e)));
  m->eos = tc.get_int("eos_token_id", 248044);
  NE_CHECK(tc["vocab_size"].i64() == VOCAB, "vocab %lld", (long long)tc["vocab_size"].i64());
  m->pos = sf ? sf->pos : alloc(DType::U32, {4});
  m->st.reserve(nl);
  int first_attn = -1;
  for (int l = 0; l < nload; l++) {
    m->L.push_back(load_layer(pack, l));
    if (sf) {  // lab-srvperf: the same state buffers (KV caches, GDN / PLE conv + ssm, pending rows) as state_from
      NE_CHECK((size_t)l < sf->st.size() && sf->L[l].linear == m->L.back().linear && sf->L[l].ple.on == m->L.back().ple.on,
               "state_from: layer %d differs", l);
      m->st.push_back(sf->st[l]);
    } else
      m->st.push_back(alloc_state(m->L.back(), tmax, 4, first_attn >= 0 ? &m->st[first_attn] : nullptr));
    if (!m->L.back().linear && first_attn < 0) first_attn = l;
    if (!m->L.back().linear) {  // one position for every attention layer
      m->st.back().pos = m->pos;
      if (m->st.back().qc.tmax > 0) m->st.back().qc.pos = m->pos;
    }
  }
  {  // lab-vision: one invf + MRoPE table for every attention layer; image embedding rows
    m->mrope = sf ? sf->mrope : alloc(DType::F32, {36 + 4 * (int64_t)tmax});
    NE_CHECK(first_attn >= 0, "no attention layer");
    if (!sf) memcpy(m->mrope.host(), m->st[first_attn].qc.invf.host(), 32 * 4);
    for (auto &st : m->st)
      if (st.qc.tmax > 0) st.qc.invf = m->mrope;
    const char *vr = getenv("NE_VISION_ROWS");
    m->vcap = vr ? std::max(1, atoi(vr)) : 65536;   // ~28 full-size images; 16384 filled up in long pi sessions
    m->vfeat = sf && sf->vcap == m->vcap ? sf->vfeat : alloc(DType::BF16, {m->vcap, HID});
  }
  int nple = 0;
  for (auto &w : m->L) nple += w.ple.on;
  NE_CHECK(nple == 1 && m->L[1].ple.on, "expected PLE on layer 1 only");
  const std::string mp = "language_model.model.";
  m->embed = load_q8(pack, mp + "embed_tokens");
  NE_CHECK(m->embed.rows == VOCAB && m->embed.in == HID, "embed_tokens shape");
  m->head = load_q8(pack, "language_model.lm_head");
  NE_CHECK(m->head.rows == VOCAB && m->head.in == HID, "lm_head shape");
  const std::string hm = mp + "hyper_connection_mixer.";
  m->mixer.norm = load_tensor(pack, hm + "hc_norm.weight");
  m->mixer.down = load_q8(pack, hm + "input_mix_weight_down");
  m->mixer.up = load_q8(pack, hm + "input_mix_weight_up");
  NE_CHECK(m->mixer.down.rows == LR && m->mixer.down.in == HCD && m->mixer.up.rows == HCD && m->mixer.up.in == LR,
           "mixer shapes");
  m->dummy_inj = alloc(DType::BF16, {HC, HCD});  // the mixer has no inject: fn_hc_pre's inject block reads zeros
  m->s = alloc_scratch(m->rows);
  m->tok = alloc(DType::U32, {4});
  m->h[0] = alloc(DType::F32, {m->rows, HC, HID});
  m->h[1] = alloc(DType::F32, {m->rows, HC, HID});
  m->x = alloc(DType::F32, {m->rows, HID});
  m->logits = alloc(DType::F32, {m->rows, VOCAB});
  m->cand = alloc(DType::F32, {(VOCAB + 1023) / 1024, kSampleMaxK, 2});
  m->lse = alloc(DType::F32, {(VOCAB + 1023) / 1024, 2});
  m->sp = alloc(DType::U32, {16});
  static_assert(sizeof(SampleParams) <= 64, "SampleParams must fit Model::sp");
  m->mask = alloc(DType::U32, {(VOCAB + 31) / 32});
  m->bias = alloc(DType::F32, {VOCAB});
  m->cnt = alloc(DType::U32, {VOCAB});
  m->top = alloc(DType::U32, {kSampleMaxK, 2});
  m->res = alloc(DType::U32, {4});
  {  // n-gram table rows by pread (an mmap fault reads a whole cluster; 16 rows per token in parallel instead)
    const TensorInfo *ti = pack.info("ngram_table.weight");
    NE_CHECK(ti && ti->host, "pack has no ngram_table.weight");
    m->ngram_fd = open(ti->file.c_str(), O_RDONLY);
    NE_CHECK(m->ngram_fd >= 0, "open %s failed", ti->file.c_str());
    fcntl(m->ngram_fd, F_RDAHEAD, 0);
    m->ngram_off = ti->file_offset;
  }
  m->ind = alloc(DType::U32, {5, 4});
  m->rec_a = std::make_unique<Recorder>();
  m->rec_b = std::make_unique<Recorder>(m->rec_a->backend());
  NE_CHECK(m->rec_a->backend() != Backend::ICB, "the ICB backend has no indirect dispatch (NE_BACKEND=mtl4)");
  record_a(*m, *m->rec_a, m->s, m->tok, m->h);
  record_b(*m, *m->rec_b, m->s, m->h, true);
  m->n_full = m->rec_a->size() + m->rec_b->size();
  m->n_body = m->n_full - 5;
  const char *p4 = getenv("NE_FN_PREFILL4");
  if (!(p4 && std::string(p4) == "0")) {
    m->s4 = alloc_scratch(4);
    m->tok4 = alloc(DType::U32, {4});
    m->h4[0] = alloc(DType::F32, {4, HC, HID});
    m->h4[1] = alloc(DType::F32, {4, HC, HID});
    m->rec_a4 = std::make_unique<Recorder>(m->rec_a->backend());
    m->rec_b4 = std::make_unique<Recorder>(m->rec_a->backend());
    record_a(*m, *m->rec_a4, m->s4, m->tok4, m->h4);
    record_b(*m, *m->rec_b4, m->s4, m->h4, false);
    m->n_pre4 = m->rec_a4->size() + m->rec_b4->size();
  }
  // Every text-model pack tensor resident on the MTL4 queue: each recorder's own residency set then only adds
  // references (switching recorders over the 81 GB otherwise re-makes it resident: ~300 ms per switch).
  const char *qr = getenv("NE_FN_QUEUE_RESIDENCY");
  Device &d = Device::get();
  if (d.has_metal4 && d.queue4 && !(qr && std::string(qr) == "0")) {
    MTLResidencySetDescriptor *rsd = [MTLResidencySetDescriptor new];
    rsd.initialCapacity = pack.names().size();
    NSError *err = nil;
    m->queue_rs = [d.dev newResidencySetWithDescriptor:rsd error:&err];
    NE_CHECK(m->queue_rs, "newResidencySet: %s", err ? err.localizedDescription.UTF8String : "?");
    for (const std::string &n : pack.names()) {
      if (n.rfind("language_model.model.", 0) != 0 && n.rfind("language_model.lm_head", 0) != 0) continue;
      if (nload < nl && n.rfind("language_model.model.layers.", 0) == 0 &&
          atoi(n.c_str() + strlen("language_model.model.layers.")) >= nload)
        continue;  // NE_FN_LAYERS: never make the unloaded layers resident
      const TensorInfo *ti = pack.info(n);
      if (ti && ti->t.buf) [m->queue_rs addAllocation:ti->t.buf];
    }
    [m->queue_rs commit];
    [m->queue_rs requestResidency];
    [d.queue4 addResidencySet:m->queue_rs];
  }
  if (sf) {  // shared state buffers: the donor's conv / ssm / pos stay as they are; only this model's own counters
    m->T = 0;
    m->dirty = false;
    m->gen++;
    m->hist[0] = m->hist[1] = m->eos;
    m->P.step = 0;
  } else {
    reset(*m);
  }
  return m;
}

static void zero(const Tensor &t) {
  if (t.buf) memset(t.host(), 0, t.nbytes());
}

void reset(Model &m) {
  for (auto &s : m.st) {
    zero(s.conv);
    zero(s.ssm);
    gdn_clear_pending(s);
    zero(s.ple_conv);
    // KV caches need no clearing: positions >= T are never read
  }
  m.T = 0;
  m.dirty = false;
  m.gen++;
  m.hist[0] = m.hist[1] = m.eos;
  ((uint32_t *)m.pos.host())[0] = 0;
  m.P.step = 0;
}

// fn_sample_p2 flags a step with nothing to sample (every candidate -inf) by res[0] = ~0u
static bool sample_none(const Model &m) { return ((const uint32_t *)m.res.host())[0] == 0xFFFFFFFFu; }
static void check_sampled(const Model &m) {
  NE_CHECK(!sample_none(m), "sampler: no token can be sampled (every candidate is -inf: the token mask allows only "
                            "tokens that logit_bias bans (<= -100), or the logits are -inf / NaN)");
}

static StepOut read_res(Model &m) {
  check_sampled(m);
  StepOut o;
  const uint32_t *res = (const uint32_t *)m.res.host();
  o.token = (int)res[0];
  memcpy(&o.logprob, &res[1], 4);
  o.argmax = (int)res[2];
  return o;
}

// B half of a step for `token` (A already committed or done): gather, params, commit B; T / history advance.
static void commit_b(Model &m, int token, bool head) {
  const auto t0 = std::chrono::steady_clock::now();
  gather_ple(m, token);
  m.gather_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  ((uint32_t *)m.pos.host())[0] = (uint32_t)m.T;
  if (head) m.P.step++;  // the RNG counter counts sampled tokens only (lab-sampler: the draws do not depend on how the prompt was fed)
  memcpy(m.sp.host(), &m.P, sizeof(SampleParams));
  set_head(m, head);
  m.rec_b->commit();
  m.hist[0] = m.hist[1];
  m.hist[1] = token;
  m.T += 1;
  m.gen++;
}

void step4(Model &m, const int *tokens) {
  NE_CHECK(m.rec_a4, "4-row prefill disabled (NE_FN_PREFILL4=0)");
  NE_CHECK(m.T + 4 <= m.tmax, "context full (%d + 4 > %d)", m.T, m.tmax);
  NE_CHECK(!m.dirty, "model state is one token ahead after generate(): reset() first");
  uint32_t *tk = (uint32_t *)m.tok4.host();
  for (int r = 0; r < 4; r++) {
    NE_CHECK(tokens[r] >= 0 && tokens[r] < VOCAB, "token %d out of range", tokens[r]);
    tk[r] = embed_id(m, m.T + r, tokens[r]);
  }
  const auto t0 = std::chrono::steady_clock::now();
  m.rec_a4->commit();
  float *emb = (float *)m.s4.ple_emb.host();
  try {
    for (int r = 0; r < 4; r++) {
      gather_row(m, m.hist[0], m.hist[1], tokens[r], emb + (size_t)r * HID);
      m.hist[0] = m.hist[1];
      m.hist[1] = tokens[r];
    }
  } catch (...) {
    abort_inflight(m, {m.rec_a4.get()});
    throw;
  }
  ((uint32_t *)m.pos.host())[0] = (uint32_t)m.T;
  m.rec_b4->commit();
  m.rec_a4->wait();
  m.rec_b4->wait();
  m.T += 4;
  m.gen++;
  m.last_gpu_ms = m.rec_a4->gpu_ms() + m.rec_b4->gpu_ms();
  m.last_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void set_mrope(Model &m, const int32_t *pos3, int n, int delta) {
  NE_CHECK(n >= 0 && n <= m.tmax, "mrope rows %d (0..%d)", n, m.tmax);
  NE_CHECK(n == 0 || pos3, "mrope: no positions");
  int32_t *hd = (int32_t *)m.mrope.host() + 32;
  int32_t *tab = hd + 4;
  for (int i = 0; i < n; i++) {
    tab[4 * i] = pos3[3 * i];
    tab[4 * i + 1] = pos3[3 * i + 1];
    tab[4 * i + 2] = pos3[3 * i + 2];
    tab[4 * i + 3] = 0;
  }
  hd[0] = n;
  hd[1] = delta;
}

void set_image_rows(Model &m, const int32_t *rows, int k, const uint16_t *feats) {
  NE_CHECK(k >= 0 && k <= m.vcap, "%d image rows > NE_VISION_ROWS %d", k, m.vcap);
  m.vmap.clear();
  if (k == 0) return;
  NE_CHECK(rows && feats, "image rows: null input");
  int mx = 0;
  for (int i = 0; i < k; i++) {
    NE_CHECK(rows[i] >= 0 && rows[i] < m.tmax, "image row %d out of range", rows[i]);
    mx = std::max(mx, rows[i]);
  }
  m.vmap.assign(mx + 1, -1);
  for (int i = 0; i < k; i++) m.vmap[rows[i]] = i;
  memcpy(m.vfeat.host(), feats, (size_t)k * HID * 2);
}

bool use_chunked(const Model &m, int n) {
  const char *e = getenv("NE_FN_PREFILL");  // read per call (tests switch it within one process)
  NE_CHECK(!e || !*e || std::string(e) == "steps" || std::string(e) == "chunked", "NE_FN_PREFILL=%s: chunked or steps", e);
  const bool steps = e && std::string(e) == "steps";
  static const bool partial = getenv("NE_FN_LAYERS") != nullptr;  // load_model's partial model: a full Model
  return !steps && g_attn_qsa && n >= kChunkedMin && (m.L.size() == 48 || partial);
}

Prefill &chunked(Model &m) {
  if (!m.pf) m.pf = prefill_create(m, kChunk);
  return *m.pf;
}

void check_feed(const Model &m, const int *toks, int n, int rows, const char *what) {
  NE_CHECK(n >= 1 && toks, "%s: no tokens", what);
  NE_CHECK(rows >= n && (int64_t)m.T + rows <= m.tmax, "%s: context full (%d + %d > %d)", what, m.T, rows, m.tmax);
  NE_CHECK(!m.dirty, "model state is one token ahead after generate(): reset() first");
  for (int i = 0; i < n; i++) NE_CHECK(toks[i] >= 0 && toks[i] < VOCAB, "%s: token %d out of range", what, toks[i]);
}

void abort_inflight(Model &m, std::initializer_list<Recorder *> rs) {
  for (Recorder *r : rs)
    if (r) try {
        r->wait();
      } catch (...) {
      }
  m.dirty = true;
}

StepOut prefill(Model &m, const int *prompt, int n) {
  NE_CHECK(n >= 1, "empty prompt");
  check_feed(m, prompt, n, n - 1 + m.rows, "prefill");  // the last token's decode step takes m.rows rows
  if (use_chunked(m, n)) return prefill_prompt(chunked(m), prompt, n);  // lab-prefill: 2K 13 s -> ~1 s
  int i = 0;
  if (m.rec_a4)
    for (; i + 4 <= n - 1; i += 4) step4(m, prompt + i);
  for (; i < n - 1; i++) step(m, prompt[i], false);
  return step(m, prompt[n - 1], true);
}

StepOut step(Model &m, int token, bool head) {
  NE_CHECK(token >= 0 && token < VOCAB, "token %d out of range", token);
  NE_CHECK(m.T + m.rows <= m.tmax, "context full (%d of %d)", m.T, m.tmax);
  NE_CHECK(!m.dirty, "model state is one token ahead after generate(): reset() first");
  const auto t0 = std::chrono::steady_clock::now();
  ((uint32_t *)m.tok.host())[0] = embed_id(m, m.T, token);
  m.rec_a->commit();
  try {
    commit_b(m, token, head);  // the n-gram gather (pread) runs while A is in flight
  } catch (...) {
    abort_inflight(m, {m.rec_a.get(), m.rec_b.get()});
    throw;
  }
  Recorder &rb = *m.rec_b;
  m.rec_a->wait();
  rb.wait();
  m.last_gpu_ms = m.rec_a->gpu_ms() + rb.gpu_ms();
  m.last_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return head ? read_res(m) : StepOut{};
}

}  // namespace ne::fn

// ------------------------------------------------------------------------------------------------ C entry points
using namespace ne;
using namespace ne::fn;


extern "C" {

NE_API ne_fn_model *ne_fn_model_create(ne_pack *pack, int tmax) {
  ne_fn_model *out = nullptr;
  int rc = c_call([&] {
    const Pack *p = pack_from_c(pack);
    NE_CHECK(p, "null pack");
    auto u = std::make_unique<ne_fn_model>();
    u->m = load_model(*p, tmax);
    out = u.release();
  });
  return rc == 0 ? out : nullptr;
}

NE_API ne_fn_model *ne_fn_model_create_shared(ne_pack *pack, int tmax, ne_fn_model *state_from) {
  ne_fn_model *out = nullptr;
  int rc = c_call([&] {
    const Pack *p = pack_from_c(pack);
    NE_CHECK(p && state_from && state_from->m, "null argument");
    auto u = std::make_unique<ne_fn_model>();
    u->m = load_model(*p, tmax, state_from->m.get());
    out = u.release();
  });
  return rc == 0 ? out : nullptr;
}

NE_API void ne_fn_model_free(ne_fn_model *m) { delete m; }

// lab-pfpath: one trivial command buffer on the queue (wall ms in *ms): the GPU's / residency's wake-up after idle
NE_API int ne_fn_gpu_ping(ne_fn_model *m, double *ms) {
  return c_call([&] {
    NE_CHECK(m && m->m, "null model");
    static std::unique_ptr<Recorder> r;
    static Tensor buf;
    if (!r) {
      buf = alloc(DType::U32, {8});
      r = std::make_unique<Recorder>();
      r->begin();
      r->dispatch(pso("pf_set_pos"), {buf, buf, Arg::value(0u)}, sz1(1), sz1(32));
      r->end();
    }
    const auto t0 = std::chrono::steady_clock::now();
    r->run();
    // and the residency of the model's recordings (NE_PING_TOUCH=0: the trivial buffer only)
    const char *te = getenv("NE_PING_TOUCH");
    Model &md = *m->m;
    if (!(te && std::string(te) == "0")) {
      for (Recorder *x : {md.rec_a.get(), md.rec_b.get(), md.rec_a4.get(), md.rec_b4.get()})
        if (x) x->touch();
      if (md.pf) prefill_touch(*md.pf);
    }
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  });
}

NE_API int ne_fn_model_reset(ne_fn_model *m) {
  return c_call([&] {
    NE_CHECK(m, "null model");
    reset(*m->m);
  });
}

NE_API int ne_fn_model_set_sampler(ne_fn_model *m, float temp, float top_p, int top_k, uint32_t seed) {
  return c_call([&] {
    NE_CHECK(m, "null model");
    NE_CHECK(top_k >= 1 && top_k <= (int)kSampleMaxK, "top_k %d (1..%u)", top_k, kSampleMaxK);
    m->m->P.temp = temp;
    m->m->P.top_p = top_p;
    m->m->P.top_k = (uint32_t)top_k;
    m->m->P.seed = seed;
  });
}

NE_API int ne_fn_model_set_mask(ne_fn_model *m, const uint32_t *bits) {
  return c_call([&] {
    NE_CHECK(m, "null model");
    Model &mm = *m->m;
    if (!bits) {
      mm.P.mask_on = 0;
      return;
    }
    bool any = false;
    for (int i = 0; i < (VOCAB + 31) / 32 && !any; i++) any = bits[i] != 0;
    NE_CHECK(any, "set_mask: no token allowed");
    memcpy(mm.mask.host(), bits, (size_t)((VOCAB + 31) / 32) * 4);
    mm.P.mask_on = 1;
  });
}

NE_API int ne_fn_model_set_sampler_ex(ne_fn_model *m, float min_p, float presence, float frequency, float repetition,
                                      int logprob_k) {
  return c_call([&] {
    NE_CHECK(m, "null model");
    NE_CHECK(min_p >= 0.0f && min_p <= 1.0f, "min_p %g (0..1)", min_p);
    NE_CHECK(repetition > 0.0f, "repetition penalty %g (> 0)", repetition);
    NE_CHECK(logprob_k >= 0 && logprob_k <= (int)kSampleMaxK, "logprob_k %d (0..%u)", logprob_k, kSampleMaxK);
    Model &mm = *m->m;
    const bool was = mm.P.pen_on != 0;
    mm.P.min_p = min_p;
    mm.P.presence = presence;
    mm.P.frequency = frequency;
    mm.P.repetition = repetition;
    mm.P.pen_on = (presence != 0.0f || frequency != 0.0f || repetition != 1.0f) ? 1 : 0;
    mm.P.cand_k = (uint32_t)logprob_k;
    mm.P.lp_on = logprob_k > 0 ? 1 : 0;
    mm.n_top = 0;
    if (mm.P.pen_on || was) memset(mm.cnt.host(), 0, mm.cnt.nbytes());
  });
}

NE_API int ne_fn_model_set_seen(ne_fn_model *m, const int *ids, int n) {
  return c_call([&] {
    NE_CHECK(m && (ids || n == 0), "null argument");
    uint32_t *c = (uint32_t *)m->m->cnt.host();
    for (int i = 0; i < n; i++)
      if (ids[i] >= 0 && ids[i] < VOCAB) c[ids[i]] |= 0x80000000u;
  });
}

NE_API int ne_fn_model_set_logit_bias(ne_fn_model *m, const int *ids, const float *bias, int n) {
  return c_call([&] {
    NE_CHECK(m && n >= 0 && (n == 0 || (ids && bias)), "null argument");
    Model &mm = *m->m;
    for (int i = 0; i < n; i++) NE_CHECK(ids[i] >= 0 && ids[i] < VOCAB, "logit_bias: token %d outside the vocabulary", ids[i]);
    if (mm.P.bias_on || n > 0) memset(mm.bias.host(), 0, mm.bias.nbytes());
    float *b = (float *)mm.bias.host();
    for (int i = 0; i < n; i++) b[ids[i]] += bias[i];
    mm.P.bias_on = n > 0 ? 1 : 0;
  });
}

NE_API int ne_fn_model_set_rng_step(ne_fn_model *m, uint32_t step) {
  return c_call([&] {
    NE_CHECK(m, "null model");
    m->m->P.step = step;
  });
}

NE_API int ne_fn_model_top(ne_fn_model *m, int *ids, float *logprobs, int max) {
  int n = 0;
  const int rc = c_call([&] {
    NE_CHECK(m && ids && logprobs, "null argument");
    Model &mm = *m->m;
    NE_CHECK(mm.P.lp_on, "ne_fn_model_top: set_sampler_ex with logprob_k > 0 first");
    const uint32_t *res = (const uint32_t *)mm.res.host();
    const uint32_t *t = (const uint32_t *)mm.top.host();
    n = std::min<int>(std::min<int>((int)res[3], max), (int)kSampleMaxK);
    for (int i = 0; i < n; i++) {
      ids[i] = (int)t[2 * i];
      memcpy(&logprobs[i], &t[2 * i + 1], 4);
    }
  });
  return rc == 0 ? n : -1;
}

NE_API int ne_fn_model_step(ne_fn_model *m, int token, int head, float *logits, int *next, float *logprob,
                            int *argmax, double *gpu_ms) {
  return c_call([&] {
    NE_CHECK(m, "null model");
    StepOut o = step(*m->m, token, head != 0);
    if (head && logits) memcpy(logits, m->m->logits.host(), (size_t)VOCAB * 4);
    if (next) *next = o.token;
    if (logprob) *logprob = o.logprob;
    if (argmax) *argmax = o.argmax;
    if (gpu_ms) *gpu_ms = m->m->last_gpu_ms;
  });
}

NE_API int ne_fn_model_prefill(ne_fn_model *mm, const int *prompt, int n, float *logits, int *next, double *ms) {
  return c_call([&] {
    NE_CHECK(mm && prompt, "null argument");
    auto t0 = std::chrono::steady_clock::now();
    StepOut o = prefill(*mm->m, prompt, n);
    if (logits) memcpy(logits, mm->m->logits.host(), (size_t)VOCAB * 4);
    if (next) *next = o.token;
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  });
}

NE_API int ne_fn_model_generate(ne_fn_model *mm, const int *prompt, int n, int max_new, int stop, int *out,
                                float *out_logprob, double *stats) {
  int count = -1;
  int rc = c_call([&] {
    NE_CHECK(mm && prompt && n >= 1, "bad arguments");
    Model &m = *mm->m;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto t0 = now();
    StepOut o = prefill(m, prompt, n);
    auto t1 = now();
    double gpu = 0, gat = 0;
    int k = 0;
    // GPU-resident token: the sampler wrote the next token into m.tok, so A(t+1) (embedding + layer 0) is committed
    // behind B(t) before the host has seen the token; the host then gathers the n-gram rows while A(t+1) runs.
    bool a_ahead = false;
    while (k < max_new) {
      out[k] = o.token;
      if (out_logprob) out_logprob[k] = o.logprob;
      k++;
      if (o.token == stop || k == max_new || m.T + 1 > m.tmax) break;
      if (!a_ahead) m.rec_a->commit();
      try {
        commit_b(m, o.token, true);
      } catch (...) {
        abort_inflight(m, {m.rec_a.get(), m.rec_b.get()});
        throw;
      }
      m.rec_a->wait();
      a_ahead = m.T + 1 <= m.tmax && k + 1 < max_new;
      if (a_ahead) m.rec_a->commit();  // speculative: A(t+2) runs right after B(t+1)
      m.rec_b->wait();
      gpu += m.rec_b->gpu_ms() + m.rec_a->gpu_ms();
      gat += m.gather_us;
      if (a_ahead && sample_none(m)) {  // A(t+2) is in flight on a placeholder token: wait for it, then raise
        m.rec_a->wait();
        m.dirty = true;
      }
      o = read_res(m);
    }
    if (a_ahead) {
      m.rec_a->wait();
      m.dirty = true;  // layer 0 consumed one token more than the caches hold
    }
    auto t2 = now();
    count = k;
    if (stats) {
      stats[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
      stats[1] = std::chrono::duration<double, std::milli>(t2 - t1).count();
      stats[2] = k - 1;
      stats[3] = k > 1 ? gpu / (k - 1) : 0;
      stats[4] = k > 1 ? gat / (k - 1) : 0;
      stats[5] = (double)m.n_full;
    }
  });
  return rc == 0 ? count : -1;
}

NE_API int ne_fn_model_time(ne_fn_model *mm, int T, int reps, double *gpu_med, double *gpu_min, double *wall_med) {
  return c_call([&] {
    NE_CHECK(mm && reps >= 1, "bad arguments");
    Model &m = *mm->m;
    NE_CHECK(T >= 0 && T + 1 <= m.tmax, "T %d out of range (tmax %d)", T, m.tmax);
    std::vector<double> g, w;
    for (int i = 0; i < reps + 2; i++) {
      m.T = T;
      step(m, 1000 + i, true);
      if (i >= 2) {  // 2 warm-up replays
        g.push_back(m.last_gpu_ms);
        w.push_back(m.last_wall_ms);
      }
    }
    std::sort(g.begin(), g.end());
    std::sort(w.begin(), w.end());
    if (gpu_med) *gpu_med = g[g.size() / 2];
    if (gpu_min) *gpu_min = g[0];
    if (wall_med) *wall_med = w[w.size() / 2];
    reset(m);
  });
}

// Debug: the head alone (mixer + lm_head + sampler) on a given final hidden state h f32 [4*2560].
NE_API int ne_fn_model_run_head(ne_fn_model *mm, const float *h, float *logits, int *next, float *logprob) {
  return c_call([&] {
    NE_CHECK(mm && h, "null argument");
    Model &m = *mm->m;
    if (!m.rec_head) {
      m.rec_head = std::make_unique<Recorder>(m.rec_b->backend());
      m.rec_head->begin();
      record_head(m, *m.rec_head, m.h[1]);
      m.rec_head->end();
    }
    upload(m.h[1], h, (size_t)HCD * 4);
    set_head(m, true);
    m.P.step++;
    memcpy(m.sp.host(), &m.P, sizeof(SampleParams));
    m.rec_head->run();
    if (logits) memcpy(logits, m.logits.host(), (size_t)VOCAB * 4);
    check_sampled(m);
    const uint32_t *res = (const uint32_t *)m.res.host();
    if (next) *next = (int)res[0];
    if (logprob) memcpy(logprob, &res[1], 4);
  });
}

// The 16 n-gram row ids of token t after (p2, p1) (C++ math) and, if emb != NULL, the gathered row f32 [2560].
NE_API int ne_fn_ngram(ne_pack *pack, int64_t p2, int64_t p1, int64_t t, int64_t *ids, float *emb) {
  return c_call([&] {
    const Pack *p = pack_from_c(pack);
    NE_CHECK(p && ids, "null argument");
    PleW w;
    const std::string pl = "language_model.model.layers.1.ple.ple_embedding.";
    memcpy(w.mult, p->at(pl + "layer_multipliers").host(), 3 * 8);
    memcpy(w.sizes, p->at(pl + "ngram_heads_vocab_sizes").host(), PLE_NG * 8);
    memcpy(w.offsets, p->at(pl + "ngram_heads_offsets").host(), PLE_NG * 8);
    ngram_ids(w, p2, p1, t, p->text_config().get_int("eos_token_id", 248044), ids);
    if (emb) ngram_gather(*p, ids, emb);
  });
}

}  // extern "C"
struct ne_fn_snap {
  int T = 0;
  int64_t hist[2] = {0, 0};
  std::vector<std::vector<uint8_t>> parts;
};
template <class F> static void each_state(Model &m, F f) {
  for (auto &s : m.st)
    for (Tensor *t : {&s.conv, &s.ssm, &s.ple_conv})
      if (t->valid()) f(*t);
}
extern "C" {

NE_API ne_fn_snap *ne_fn_model_snapshot(ne_fn_model *h) {
  ne_fn_snap *out = nullptr;
  c_call([&] {
    NE_CHECK(h && h->m, "null model");
    Model &m = *h->m;
    NE_CHECK(!m.dirty, "snapshot of a dirty model (generate() ran ahead): reset() or restore first");
    gdn_materialise(m.st);  // lab-gdnlazy: deferred GDN rows into ssm (the snapshot holds plain states)
    auto s = std::make_unique<ne_fn_snap>();
    s->T = m.T;
    s->hist[0] = m.hist[0];
    s->hist[1] = m.hist[1];
    each_state(m, [&](const Tensor &t) {
      s->parts.emplace_back(t.nbytes());
      memcpy(s->parts.back().data(), t.host(), t.nbytes());
    });
    out = s.release();
  });
  return out;
}

NE_API int ne_fn_model_restore(ne_fn_model *h, const ne_fn_snap *s) {
  int rc = c_call([&] {
    NE_CHECK(h && h->m && s, "null argument");
    Model &m = *h->m;
    size_t i = 0;
    bool ok = true;
    each_state(m, [&](const Tensor &t) {
      ok = ok && i < s->parts.size() && s->parts[i].size() == t.nbytes();
      i++;
    });
    NE_CHECK(ok && i == s->parts.size() && s->T <= m.tmax, "snapshot does not match this model");
    i = 0;
    each_state(m, [&](const Tensor &t) { memcpy(t.host(), s->parts[i++].data(), t.nbytes()); });
    for (auto &st : m.st) gdn_clear_pending(st);  // lab-gdnlazy
    m.T = s->T;
    m.hist[0] = s->hist[0];
    m.hist[1] = s->hist[1];
    ((uint32_t *)m.pos.host())[0] = (uint32_t)s->T;
    m.dirty = false;
    m.gen++;
  });
  return rc == 0 ? 0 : -2;
}

NE_API int ne_fn_snap_T(const ne_fn_snap *s) { return s ? s->T : -1; }
NE_API int64_t ne_fn_snap_bytes(const ne_fn_snap *s) {
  if (!s) return -1;
  int64_t n = 0;
  for (auto &p : s->parts) n += (int64_t)p.size();
  return n;
}
NE_API void ne_fn_snap_free(ne_fn_snap *s) { delete s; }

NE_API int ne_fn_model_info(ne_fn_model *mm, int *nf, int *nb, int *T) {
  return c_call([&] {
    NE_CHECK(mm, "null model");
    if (nf) *nf = (int)mm->m->n_full;
    if (nb) *nb = (int)mm->m->n_body;
    if (T) *T = mm->m->T;
  });
}

NE_API int ne_fn_model_set_mrope(ne_fn_model *mm, const int32_t *pos3, int n, int delta) {
  return c_call([&] {
    NE_CHECK(mm, "null model");
    set_mrope(*mm->m, pos3, n, delta);
  });
}
NE_API int ne_fn_model_set_image_rows(ne_fn_model *mm, const int32_t *rows, int k, const uint16_t *feats) {
  return c_call([&] {
    NE_CHECK(mm, "null model");
    set_image_rows(*mm->m, rows, k, feats);
  });
}

}  // extern "C"
