// core-test: self-tests of the core (make test). Owner: lab-core.
//   core-test            correctness tests + a quick dispatch-cost table (only if the machine lock is free)
//   core-test --bench    waits for the machine lock, then a longer dispatch-cost table
//   core-test --quick    correctness only (no GPU timing, no lock)
//   env IJAH_PACK=<dir>  pack to test (default ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah; skipped if absent;
//                        NE_PACK is still read)
//   env NE_NO_LOCK=1     do not take the optional machine lock (IJAH_LOCK_SCRIPT) even when it is set
#import <Foundation/Foundation.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>

#include "core/bench.h"
#include "core/engine.hpp"

using namespace ne;

static int g_fail = 0, g_pass = 0;
#define EXPECT(cond, ...)                                        \
  do {                                                           \
    if (cond) g_pass++;                                          \
    else {                                                       \
      g_fail++;                                                  \
      printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      printf(__VA_ARGS__);                                       \
      printf("\n");                                              \
    }                                                            \
  } while (0)

template <class F> static void section(const char *name, F &&f) {
  int f0 = g_fail;
  double t0 = now_us();
  try {
    f();
  } catch (const std::exception &e) {
    g_fail++;
    printf("  FAIL %s: exception: %s\n", name, e.what());
  }
  printf("[%s] %s (%.1f ms)\n", g_fail == f0 ? " ok " : "FAIL", name, (now_us() - t0) / 1e3);
  fflush(stdout);
}

// ---------------------------------------------------------------- JSON
static void test_json() {
  Json j = Json::parse(R"({"a": 1, "b": [true, false, null, -2.5e3, "xé😀\n"], "big": 102400491520,
                          "neg": -9007199254740993, "o": {"k": "v"}, "f": 1.0})");
  EXPECT(j["a"].i64() == 1, "a");
  EXPECT(j["b"].size() == 5 && j["b"][0].boolean() && !j["b"][1].boolean() && j["b"][2].is_null(), "b");
  EXPECT(j["b"][3].num() == -2500.0, "num %g", j["b"][3].num());
  EXPECT(j["b"][4].str() == "x\xc3\xa9\xf0\x9f\x98\x80\n", "unicode '%s'", j["b"][4].str().c_str());
  EXPECT(j["big"].i64() == 102400491520LL, "big");
  EXPECT(j["neg"].i64() == -9007199254740993LL, "exact int64 %lld", (long long)j["neg"].i64());
  EXPECT(j["o"]["k"].str() == "v" && j.find("zz") == nullptr && j.get_int("zz", 7) == 7, "obj");
  EXPECT(j["f"].is_int() && j["f"].i64() == 1, "1.0 as int");
  EXPECT(Json::parse(j.dump()).dump() == j.dump(), "dump roundtrip");
  for (const char *bad : {"{", "[1,]", "{\"a\" 1}", "tru", "\"abc", "1 2", "{\"a\":1,}", "-", "01x"}) {
    bool threw = false;
    try { Json::parse(bad); } catch (const Error &) { threw = true; }
    EXPECT(threw, "accepted bad json %s", bad);
  }
}

// ---------------------------------------------------------------- device / buffers / pipelines
static void test_device() {
  Device &d = Device::get();
  printf("  device %s  has_nax=%d has_metal4=%d maxBuffer=%.2f GiB  default backend %s\n", d.name.c_str(), d.has_nax,
         d.has_metal4, d.max_buffer_length / 1073741824.0, backend_name(resolve_backend(Backend::Auto)));
  EXPECT(d.dev != nil && d.queue != nil, "device");
  EXPECT(d.max_buffer_length > (1ull << 32), "max buffer");
  Tensor t = alloc(DType::BF16, {3, 5});
  EXPECT(t.nbytes() == 30 && t.buf.length >= 32 && t.row_bytes() == 10, "alloc sizes");
  uint16_t src[15], dst[15] = {};
  for (int i = 0; i < 15; i++) src[i] = (uint16_t)(i * 1000 + 7);
  upload(t, src, sizeof src);
  download(t, dst, sizeof dst);
  EXPECT(memcmp(src, dst, sizeof src) == 0, "upload/download");
  Tensor r = t.rows(1, 2);
  EXPECT(r.offset == 10 && r.shape[0] == 2 && ((uint16_t *)r.host())[0] == src[5], "rows view");
  EXPECT(dtype_from_name("BF16") == DType::BF16 && dtype_size(DType::I64) == 8 && dtype_from_name("Q4") == DType::Invalid,
         "dtypes");
  EXPECT(Pipelines::has("ne_st_step") && !Pipelines::has("no_such_kernel"), "metallib functions");
  auto p1 = Pipelines::get("ne_st_step");
  auto p2 = Pipelines::get("ne_st_step");
  auto p3 = Pipelines::get("ne_st_step", {FC::u32(0, 5)});
  EXPECT(p1 == p2 && p1 != p3, "pipeline cache keyed by constants");
  bool threw = false;
  try { Pipelines::get("no_such_kernel"); } catch (const Error &) { threw = true; }
  EXPECT(threw, "missing kernel throws");
}

// ---------------------------------------------------------------- recorder correctness
// CPU model of ne_st_step.
static void cpu_step(const std::vector<uint32_t> &src, std::vector<uint32_t> &dst, uint32_t mul, uint32_t add) {
  const uint32_t n = (uint32_t)src.size();
  for (uint32_t i = 0; i < n; i++) {
    uint32_t v = src[n - 1 - i] * mul + add;
    for (int k = 0; k < 32; k++) v = v * 1664525u + 1013904223u;
    dst[i] = v;
  }
}

// Chain of `steps` ping-pong dispatches A->B->A..., replayed `replays` times (each replay continues from the buffers'
// state). Returns mismatching words over all replays.
static size_t chain_check(Backend b, int steps, int replays, bool barriers, uint32_t mul, bool patch,
                          uint32_t n = 1u << 16) {
  Tensor A = alloc(DType::U32, {n}), B = alloc(DType::U32, {n});
  std::vector<uint32_t> ca(n), cb(n);
  std::mt19937 rng(123);
  for (auto &v : ca) v = rng();
  upload(A, ca.data(), n * 4);
  auto pso = mul == 3 ? Pipelines::get("ne_st_step") : Pipelines::get("ne_st_step", {FC::u32(0, mul)});
  Recorder r(b);
  for (int s = 0; s < steps; s++) {
    if (s && barriers) r.barrier();
    uint32_t add = (uint32_t)s * 2654435761u;
    r.dispatch(pso, {s % 2 ? B : A, s % 2 ? A : B, Arg::value(n), Arg::value(add)}, MTLSizeMake(n / 256, 1, 1),
               MTLSizeMake(256, 1, 1));
  }
  if (barriers) r.barrier();  // a trailing barrier is harmless
  r.end();
  size_t bad = 0;
  for (int rep = 0; rep < replays; rep++) {
    uint32_t extra = 0;
    if (patch && rep > 0) {  // patch the add constant of step 0 between replays
      extra = 1000u + rep;
      memcpy(r.bytes_at(0, 3), &extra, 4);
    }
    r.run();
    for (int s = 0; s < steps; s++) {  // mirrors the GPU: ca = A, cb = B
      uint32_t add = (patch && rep > 0 && s == 0) ? extra : (uint32_t)s * 2654435761u;
      if (s % 2 == 0) cpu_step(ca, cb, mul, add);
      else cpu_step(cb, ca, mul, add);
    }
    const uint32_t *ga = (const uint32_t *)A.host(), *gb = (const uint32_t *)B.host();
    for (uint32_t i = 0; i < n; i++) bad += (ga[i] != ca[i]) + (gb[i] != cb[i]);
    memcpy(ca.data(), ga, n * 4);  // resync so one race does not cascade into every later replay's count
    memcpy(cb.data(), gb, n * 4);
  }
  return bad;
}

static void test_recorder() {
  std::vector<Backend> bs = {Backend::ICB, Backend::Classic, Backend::Serial};
  if (Device::get().has_metal4) bs.insert(bs.begin(), Backend::MTL4);
  for (Backend b : bs) {
    size_t bad = chain_check(b, 64, 3, true, 3, false);
    EXPECT(bad == 0, "%s: chain of 64 x 3 replays: %zu mismatches", backend_name(b), bad);
    bad = chain_check(b, 64, 1, true, 3, false, 1u << 20);
    EXPECT(bad == 0, "%s: chain of 64, n=2^20: %zu mismatches", backend_name(b), bad);
    bad = chain_check(b, 9, 3, true, 7, true);
    EXPECT(bad == 0, "%s: function constant MUL=7 + patched bytes: %zu mismatches", backend_name(b), bad);
    // threadgroup memory
    Tensor s = alloc(DType::F32, {1024}), d = alloc(DType::F32, {1024});
    for (int i = 0; i < 1024; i++) ((float *)s.host())[i] = (float)i;
    Recorder r(b);
    r.dispatch(Pipelines::get("ne_st_tgmem"), {s, d}, MTLSizeMake(4, 1, 1), MTLSizeMake(256, 1, 1), 256 * 4);
    r.end();
    r.run();
    int tgbad = 0;
    for (int i = 0; i < 1024; i++) tgbad += ((float *)d.host())[i] != (float)((i / 256) * 256 + 255 - i % 256);
    EXPECT(tgbad == 0, "%s: threadgroup memory: %d mismatches", backend_name(b), tgbad);
    EXPECT(r.gpu_ms() > 0 && r.gpu_ms() < 1000, "%s: gpu_ms %.4f", backend_name(b), r.gpu_ms());
    // indirect dispatch (not ICB)
    if (b != Backend::ICB) {
      Tensor ind = alloc(DType::U32, {3});
      uint32_t tgs[3] = {4, 1, 1};
      upload(ind, tgs, 12);
      memset(d.host(), 0, 4096);
      Recorder ri(b);
      ri.dispatch_indirect(Pipelines::get("ne_st_tgmem"), {s, d}, ind.buf, 0, MTLSizeMake(256, 1, 1), 1024);
      ri.end();
      ri.run();
      tgbad = 0;
      for (int i = 0; i < 1024; i++) tgbad += ((float *)d.host())[i] != (float)((i / 256) * 256 + 255 - i % 256);
      EXPECT(tgbad == 0, "%s: indirect dispatch: %d mismatches", backend_name(b), tgbad);
    }
  }
  // Several replays of one recording in flight (commit x3, then wait): equals three sequential replays.
  for (Backend b : bs) {
    const uint32_t n = 1u << 16;
    Tensor A = alloc(DType::U32, {n}), B = alloc(DType::U32, {n});
    std::vector<uint32_t> ca(n), cb(n);
    for (uint32_t i = 0; i < n; i++) ca[i] = i ^ 0x5a5a5a5au;
    upload(A, ca.data(), n * 4);
    auto pso = Pipelines::get("ne_st_step");
    Recorder r(b);
    r.dispatch(pso, {A, B, Arg::value(n), Arg::value(1u)}, MTLSizeMake(n / 256, 1, 1), MTLSizeMake(256, 1, 1));
    r.barrier();
    r.dispatch(pso, {B, A, Arg::value(n), Arg::value(2u)}, MTLSizeMake(n / 256, 1, 1), MTLSizeMake(256, 1, 1));
    r.end();
    for (int k = 0; k < Recorder::kMaxInFlight; k++) r.commit();
    bool threw = false;
    try { r.commit(); } catch (const Error &) { threw = true; }
    EXPECT(threw && r.in_flight() == (size_t)Recorder::kMaxInFlight, "%s: commit beyond max in flight", backend_name(b));
    r.wait_oldest();
    EXPECT(r.in_flight() == (size_t)Recorder::kMaxInFlight - 1, "%s: wait_oldest", backend_name(b));
    r.wait();
    for (int k = 0; k < Recorder::kMaxInFlight; k++) {
      cpu_step(ca, cb, 3, 1);
      cpu_step(cb, ca, 3, 2);
    }
    size_t bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += ((const uint32_t *)A.host())[i] != ca[i];
    EXPECT(bad == 0 && r.in_flight() == 0, "%s: %d replays in flight: %zu mismatches", backend_name(b),
           Recorder::kMaxInFlight, bad);
  }
  // Non-uniform grid: n not a multiple of the threadgroup size; the kernel's gid < n guard is the only bound.
  for (Backend b : bs) {
    const uint32_t n = 1000, cap = 1024;
    Tensor A = alloc(DType::U32, {cap}), B = alloc(DType::U32, {cap});
    std::vector<uint32_t> ca(n), cb(n);
    for (uint32_t i = 0; i < n; i++) ((uint32_t *)A.host())[i] = ca[i] = i * 77u;
    for (uint32_t i = n; i < cap; i++) ((uint32_t *)B.host())[i] = 0xdeadbeefu;
    Recorder r(b);
    r.dispatch_threads(Pipelines::get("ne_st_step"), {A, B, Arg::value(n), Arg::value(5u)}, MTLSizeMake(n, 1, 1),
                       MTLSizeMake(256, 1, 1));
    r.end();
    r.run();
    cpu_step(ca, cb, 3, 5);
    size_t bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += ((uint32_t *)B.host())[i] != cb[i];
    for (uint32_t i = n; i < cap; i++) bad += ((uint32_t *)B.host())[i] != 0xdeadbeefu;
    EXPECT(bad == 0, "%s: dispatch_threads: %zu mismatches", backend_name(b), bad);
  }
  // Pointer table: buffers reached only through GPU addresses, made resident with use().
  for (Backend b : bs) {
    const uint32_t n = 4096;
    Tensor srcs[4], tab = alloc(DType::U64, {4}), out = alloc(DType::U32, {n});
    for (int k = 0; k < 4; k++) {
      srcs[k] = alloc(DType::U32, {n});
      for (uint32_t i = 0; i < n; i++) ((uint32_t *)srcs[k].host())[i] = i * 10 + k;
      ((uint64_t *)tab.host())[k] = srcs[k].gpu_address();
    }
    Recorder r(b);
    for (auto &t : srcs) r.use(t);
    r.dispatch(Pipelines::get("ne_st_ptr"), {tab, out, Arg::value(n)}, MTLSizeMake(n / 256, 1, 1), MTLSizeMake(256, 1, 1));
    r.end();
    r.run();
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += ((uint32_t *)out.host())[i] != i * 10 + i % 4;
    EXPECT(bad == 0, "%s: pointer table + use(): %d mismatches", backend_name(b), bad);
  }
  // Two recorders committed back to back without a host wait: B's chain reads A's output (queue ordering).
  for (Backend b : bs) {
    const uint32_t n = 1u << 20;
    Tensor X = alloc(DType::U32, {n}), Y = alloc(DType::U32, {n}), Z = alloc(DType::U32, {n});
    std::vector<uint32_t> cx(n), cy(n), cz(n);
    for (uint32_t i = 0; i < n; i++) cx[i] = i * 2654435761u;
    upload(X, cx.data(), n * 4);
    auto pso = Pipelines::get("ne_st_step");
    Recorder ra(b), rb(b);
    for (int s = 0; s < 8; s++) {
      if (s) ra.barrier();
      ra.dispatch(pso, {s % 2 ? Y : X, s % 2 ? X : Y, Arg::value(n), Arg::value((uint32_t)s)}, MTLSizeMake(n / 256, 1, 1),
                  MTLSizeMake(256, 1, 1));
    }
    rb.dispatch(pso, {X, Z, Arg::value(n), Arg::value(99u)}, MTLSizeMake(n / 256, 1, 1), MTLSizeMake(256, 1, 1));
    ra.end();
    rb.end();
    ra.commit();
    rb.commit();
    rb.wait();
    ra.wait();
    for (int s = 0; s < 8; s++) {
      if (s % 2 == 0) cpu_step(cx, cy, 3, s);
      else cpu_step(cy, cx, 3, s);
    }
    cpu_step(cx, cz, 3, 99);
    size_t bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += ((const uint32_t *)Z.host())[i] != cz[i];
    EXPECT(bad == 0, "%s: back-to-back recorders: %zu mismatches", backend_name(b), bad);
  }
  // Sensitivity control: the same chain without barriers on concurrent backends should go wrong (not asserted).
  for (Backend b : bs) {
    if (b == Backend::Serial) continue;
    for (uint32_t n : {1u << 16, 1u << 20}) {
      size_t bad = chain_check(b, 64, 1, false, 3, false, n);
      printf("  control (%s, n=%u): no barriers -> %zu mismatching words (nonzero = the check sees races)\n",
             backend_name(b), n, bad);
    }
  }
}

// ---------------------------------------------------------------- pack
static std::string home() { return getenv("HOME") ? getenv("HOME") : ""; }
static std::vector<uint8_t> pread_bytes(const std::string &file, uint64_t off, size_t n) {
  std::vector<uint8_t> v(n);
  int fd = open(file.c_str(), O_RDONLY);
  NE_CHECK(fd >= 0, "open %s", file.c_str());
  size_t got = 0;
  while (got < n) {
    ssize_t k = pread(fd, v.data() + got, n - got, (off_t)(off + got));
    NE_CHECK(k > 0, "pread %s", file.c_str());
    got += (size_t)k;
  }
  close(fd);
  return v;
}
static std::string slurp(const std::string &path) {
  FILE *f = fopen(path.c_str(), "rb");
  NE_CHECK(f, "open %s", path.c_str());
  std::string s;
  char buf[65536];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k);
  fclose(f);
  return s;
}
// GPU-copies a view's bytes into a fresh buffer and compares with pread from the file.
static bool gpu_bytes_match(const Tensor &t, const std::string &file, uint64_t file_off, size_t n) {
  Tensor out = alloc(DType::U8, {(int64_t)n});
  Recorder r;
  uint64_t nn = n;
  r.dispatch(Pipelines::get("ne_st_copy_bytes"), {t, out, Arg::value(nn)}, MTLSizeMake(256, 1, 1),
             MTLSizeMake(256, 1, 1));
  r.end();
  r.run();
  std::vector<uint8_t> want = pread_bytes(file, file_off, n);
  return memcmp(out.host(), want.data(), n) == 0;
}

static void test_pack() {
  const char *ep = getenv("IJAH_PACK") && *getenv("IJAH_PACK") ? getenv("IJAH_PACK") : getenv("NE_PACK");
  const char *eps = getenv("IJAH_PACKS");
  std::string dir = ep && *ep ? std::string(ep)
                              : (eps && *eps ? std::string(eps) : home() + "/.cache/ijah/packs") +
                                    "/Qwen3.8-Flash-Next-EXL3-Ijah";
  struct stat st;
  if (stat(dir.c_str(), &st) != 0) {
    printf("  SKIP: no pack at %s\n", dir.c_str());
    return;
  }
  double t0 = now_us();
  auto p = Pack::open(dir);
  printf("  %s: %zu tensors, %zu files, %.1f GB mapped, open %.0f ms\n", dir.c_str(), p->names().size(), p->num_files(),
         p->mapped_bytes() / 1e9, (now_us() - t0) / 1e3);
  const Json &tc = p->text_config();
  const int64_t H = tc["hidden_size"].i64(), E = tc["num_experts"].i64(), I = tc["moe_intermediate_size"].i64();
  const int64_t L = tc["num_hidden_layers"].i64(), HC = tc["hc_count"].i64(), V = tc["vocab_size"].i64();
  const int64_t NV = tc["linear_num_value_heads"].i64();
  EXPECT(H == 2560 && E == 512 && I == 640 && L == 48 && HC == 4 && V == 248320, "config values");
  std::string ix = dir + "/model.safetensors.index.json";
  if (stat(ix.c_str(), &st) == 0) {
    size_t nidx = Json::parse(slurp(ix))["weight_map"].size();
    EXPECT(p->names().size() == nidx + (p->has("ngram_table.weight") ? 1 : 0), "%zu tensors vs %zu in index",
           p->names().size(), nidx);
  }
  auto shape_is = [&](const std::string &name, DType dt, std::vector<int64_t> shp) {
    const TensorInfo *ti = p->info(name);
    if (!ti) { EXPECT(false, "missing %s", name.c_str()); return; }
    bool ok = ti->t.dtype == dt && ti->t.ndim == (int)shp.size();
    for (size_t i = 0; ok && i < shp.size(); i++) ok = ti->t.shape[i] == shp[i];
    EXPECT(ok, "%s is %s", name.c_str(), ti->t.desc().c_str());
  };
  const std::string m = "language_model.model.", l0 = m + "layers.0.";
  const Json *eq = p->config().find("expert_quant");
  const int64_t K = eq ? (*eq)["layer_k"][0].i64() : 5, Kmtp = eq ? eq->get_int("mtp_k", 4) : 4;
  shape_is(m + "embed_tokens.weight", DType::U32, {V, H * 8 / 32});
  shape_is(l0 + "mlp.gate.weight", DType::BF16, {E, H});
  shape_is(l0 + "mlp.switch_mlp.down_proj.trellis", DType::U16, {E, I / 16, H / 16, 16 * K});
  shape_is(l0 + "mlp.switch_mlp.gate_proj.trellis", DType::U16, {E, H / 16, I / 16, 16 * K});
  shape_is(l0 + "mlp.switch_mlp.down_proj.suh", DType::F16, {E, I});
  shape_is(l0 + "mlp.switch_mlp.down_proj.svh", DType::F16, {E, H});
  shape_is("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.trellis", DType::U16, {E, H / 16, I / 16, 16 * Kmtp});
  shape_is(l0 + "attn_hyper_connection.hc_norm.weight", DType::BF16, {HC * H});
  shape_is(l0 + "linear_attn.A_log", DType::BF16, {NV});
  shape_is(l0 + "linear_attn.conv1d.weight", DType::BF16, {10240, tc["linear_conv_kernel_dim"].i64(), 1});
  shape_is(m + "layers.47.mlp.switch_mlp.up_proj.trellis", DType::U16, {E, H / 16, I / 16, 16 * K});

  // Byte-exact reads through the no-copy buffers (GPU copy) vs pread.
  for (const std::string &name : {l0 + "linear_attn.A_log", l0 + "linear_attn.dt_bias", l0 + "linear_attn.conv1d.weight",
                                  l0 + "mlp.switch_mlp.down_proj.svh", m + "layers.47.mlp.shared_expert_gate.weight"}) {
    const TensorInfo *ti = p->info(name);
    if (!ti) { EXPECT(false, "missing %s", name.c_str()); continue; }
    EXPECT(ti->t.buf && (const char *)ti->t.buf.contents + ti->t.offset == (const char *)ti->host,
           "%s: no-copy (contents == mapping)", name.c_str());
    EXPECT(gpu_bytes_match(ti->t, ti->file, ti->file_offset, ti->nbytes), "%s: GPU bytes != pread", name.c_str());
    EXPECT(memcmp(ti->host, pread_bytes(ti->file, ti->file_offset, ti->nbytes).data(), ti->nbytes) == 0,
           "%s: host bytes != pread", name.c_str());
  }
  // aligned(): an odd-aligned EXL3 tensor gets an aligned copy with identical bytes; an aligned one is returned as is.
  {
    const std::string name = l0 + "mlp.switch_mlp.down_proj.svh";
    const Tensor &a = p->aligned(name, 16);
    const TensorInfo *ti = p->info(name);
    EXPECT(a.gpu_address() % 16 == 0 && memcmp(a.host(), ti->host, ti->nbytes) == 0, "aligned() copy");
    EXPECT(&p->aligned(name, 16) == &a, "aligned() cached");
    if (ti->t.gpu_address() % 16 != 0) EXPECT(p->aligned_copy_bytes() == ti->nbytes, "aligned bytes");
    else EXPECT(&a == &ti->t, "aligned() of an aligned tensor is the view");
  }
  // A row view of a big expert tensor (experts 300..301 of layer 47 up trellis) without binding all 512 experts.
  {
    const std::string name = m + "layers.47.mlp.switch_mlp.up_proj.trellis";
    const TensorInfo *ti = p->info(name);
    Tensor v = p->view_rows(name, 300, 2);
    EXPECT(v.buf.length < (8u << 20), "row view window %lu bytes", (unsigned long)v.buf.length);
    EXPECT(gpu_bytes_match(v, ti->file, ti->file_offset + 300 * ti->t.row_bytes(), v.nbytes()),
           "trellis rows: GPU != pread");
  }
  // The n-gram table: no single GPU view (> maxBufferLength); host pointer + row windows work.
  if (const TensorInfo *ng = p->info("ngram_table.weight")) {
    EXPECT(ng->t.dtype == DType::BF16 && ng->t.ndim == 2 && ng->t.shape[1] == 160, "ngram %s", ng->t.desc().c_str());
    EXPECT(ng->nbytes <= Device::get().max_buffer_length || p->get("ngram_table.weight") == nullptr, "ngram view policy");
    EXPECT(ng->file_offset == 8 + 176, "ngram data offset %llu", (unsigned long long)ng->file_offset);
    for (int64_t row : {(int64_t)0, (int64_t)123456789, ng->t.shape[0] - 3}) {
      Tensor v = p->view_rows("ngram_table.weight", row, 3);
      uint64_t off = ng->file_offset + (uint64_t)row * ng->t.row_bytes();
      EXPECT(gpu_bytes_match(v, ng->file, off, v.nbytes()), "ngram rows %lld: GPU != pread", (long long)row);
      EXPECT(memcmp((const char *)ng->host + (uint64_t)row * ng->t.row_bytes(), pread_bytes(ng->file, off, 960).data(),
                    960) == 0,
             "ngram rows %lld: host != pread", (long long)row);
    }
  }
  // C API over the same pack.
  ne_pack *cp = ne_pack_open(dir.c_str());
  EXPECT(cp != nullptr, "ne_pack_open: %s", ne_last_error());
  if (cp) {
    ne_tensor_desc d;
    EXPECT(ne_tensor_info(cp, (l0 + "mlp.gate.weight").c_str(), &d) == 0 && d.dtype == NE_BF16 && d.ndim == 2 &&
               d.shape[0] == E && d.shape[1] == H && d.buf && d.nbytes == (uint64_t)(E * H * 2),
           "ne_tensor_info");
    EXPECT(ne_upload(d.buf, 0, "x", 1) == -1, "upload into pack refused");
    EXPECT(ne_tensor_info(cp, "nope", &d) == -1 && strstr(ne_last_error(), "nope"), "missing tensor error");
    EXPECT(ne_pack_num_tensors(cp) == (int64_t)p->names().size(), "count");
    uint64_t off = 0;
    ne_buf *vb = ne_pack_view_rows(cp, "ngram_table.weight", 5, 2, &off);
    EXPECT(vb && ne_buf_contents(vb) && memcmp((char *)ne_buf_contents(vb) + off,
                                               (const char *)p->info("ngram_table.weight")->host + 5 * 320, 640) == 0,
           "ne_pack_view_rows");
    ne_free(vb);
    ne_pack_close(cp);
  }
}

// ---------------------------------------------------------------- C API kernel run
static void test_capi() {
  ne_device_info di;
  EXPECT(ne_device_query(&di) == 0 && di.max_buffer_length > 0, "device query");
  const uint32_t n = 4096;
  ne_buf *a = ne_alloc(n * 4), *b = ne_alloc(n * 4);
  std::vector<uint32_t> h(n), want(n);
  for (uint32_t i = 0; i < n; i++) h[i] = i * 7 + 1;
  EXPECT(ne_upload(a, 0, h.data(), n * 4) == 0, "upload");
  uint32_t add = 11;
  ne_arg args[4] = {{a, 0, nullptr, 0}, {b, 0, nullptr, 0}, {nullptr, 0, &n, 4}, {nullptr, 0, &add, 4}};
  ne_fconst fc = {0, 0, 5};
  uint32_t grid[3] = {n / 256, 1, 1}, tg[3] = {256, 1, 1};
  for (const char *be : {"", "icb", "classic"}) {
    double ms = -1;
    EXPECT(ne_run_kernel("ne_st_step", &fc, 1, args, 4, grid, tg, 0, 1, be, &ms) == 0, "run: %s", ne_last_error());
    std::vector<uint32_t> got(n);
    ne_download(b, 0, got.data(), n * 4);
    for (uint32_t i = 0; i < n; i++) {
      uint32_t v = h[n - 1 - i] * 5 + 11;
      for (int k = 0; k < 32; k++) v = v * 1664525u + 1013904223u;
      want[i] = v;
    }
    EXPECT(got == want && ms > 0, "ne_run_kernel result (%s)", be);
  }
  EXPECT(ne_run_kernel("no_such", nullptr, 0, args, 4, grid, tg, 0, 1, "", nullptr) == -1, "missing kernel -> -1");
  EXPECT(ne_kernel_exists("ne_st_tiny") == 1 && ne_kernel_exists("zz") == 0, "kernel_exists");
  ne_free(a);
  ne_free(b);
}

// ---------------------------------------------------------------- dispatch cost (machine lock)
static void test_timing(bool full) {
  MachineLock lock("lab-core", "core-test dispatch timing", 15, full);
  if (!lock.held()) {
    printf("  SKIP: machine lock held (see $IJAH_LOCK_SCRIPT status); run `make bench` later\n");
    return;
  }
  std::vector<Backend> bs = {Backend::ICB, Backend::Classic, Backend::Serial};
  if (Device::get().has_metal4) bs.insert(bs.begin(), Backend::MTL4);
  const int reps = full ? 9 : 5;
  // Warm-up: the first backend measured right after start-up read 1.37-1.55 us (tiny, mtl4) instead of 0.93-0.97.
  for (int i = 0; i < 3; i++) bench_dispatch(bs[0], "read256", 256, 2048, 3, false);
  printf("  per dispatch, differential (T(2048) - T(256)) / 1792, median of %d; dep = barrier between dispatches\n",
         reps);
  printf("  %-8s %-8s %9s %9s %9s %9s\n", "backend", "kernel", "dep_us", "naive_us", "enc_us", "indep_us");
  for (Backend b : bs)
    for (const char *k : {"tiny", "read256"}) {
      DispatchCost c = bench_dispatch(b, k, 256, 2048, reps, false);
      DispatchCost ci = bench_dispatch(b, k, 256, 2048, reps, true);
      printf("  %-8s %-8s %9.3f %9.3f %9.3f %9.3f\n", backend_name(b), k, c.gpu_us, c.naive_us, c.encode_us,
             ci.gpu_us);
      EXPECT(c.gpu_us > 0 && c.gpu_us < 20, "%s %s: %.3f us per dispatch", backend_name(b), k, c.gpu_us);
    }
  // Round trip of a one-dispatch replay (commit -> GPU -> wait returns): the fixed cost per replay / per token.
  for (Backend b : bs) {
    Tensor c1 = alloc(DType::F32, {4});
    Recorder r(b);
    r.dispatch(Pipelines::get("ne_st_tiny"), {c1}, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
    r.end();
    std::vector<double> wall, gpu;
    for (int i = 0; i < 200; i++) {
      r.run();
      if (i >= 20) { wall.push_back(r.wall_us()); gpu.push_back(r.gpu_ms() * 1e3); }
    }
    std::sort(wall.begin(), wall.end());
    std::sort(gpu.begin(), gpu.end());
    // Pipelined: keep 2 replays in flight (commit k+1, then wait for k).
    double t0 = now_us();
    const int N = 200;
    r.commit();
    for (int i = 1; i < N; i++) {
      r.commit();
      r.wait_oldest();
    }
    r.wait();
    double per = (now_us() - t0) / N;
    printf("  %-8s one-dispatch replay: wall median %.1f us (p10 %.1f, p90 %.1f), GPU %.1f us; pipelined x2: %.1f us "
           "per replay\n", backend_name(b), wall[wall.size() / 2], wall[wall.size() / 10], wall[wall.size() * 9 / 10],
           gpu[gpu.size() / 2], per);
  }
  // time_per_unit on the same probe agrees with bench_dispatch
  auto pso = Pipelines::get("ne_st_tiny");
  Tensor cnt = alloc(DType::F32, {4});
  double us = time_per_unit(
      [&](Recorder &r, int n) {
        for (int i = 0; i < n; i++) {
          if (i) r.barrier();
          r.dispatch(pso, {cnt}, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
        }
      },
      256, 2048, reps);
  printf("  time_per_unit(tiny chain, default backend) = %.3f us\n", us);
  EXPECT(us > 0 && us < 20, "time_per_unit %.3f", us);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    bool bench = argc > 1 && !strcmp(argv[1], "--bench");
    bool quick = argc > 1 && !strcmp(argv[1], "--quick");  // correctness only (no timing, no lock)
    if (!bench) {
      section("json", test_json);
      section("device, buffers, pipelines", test_device);
      section("recorder replay (all backends)", test_recorder);
      section("pack loader", test_pack);
      section("C API", test_capi);
    }
    if (!quick) section("dispatch cost per backend", [&] { test_timing(bench); });
    printf("%s: %d checks passed, %d failed\n", g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
  }
}
