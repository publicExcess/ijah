// lab-mtp test helpers: npy reader, tensor read/write, error metrics (same conventions as
// models/flash_next/fn_layer_check.mm, duplicated so tests/mtp builds on its own).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/engine.hpp"
#include "models/flash_next/layer.h"

namespace mtpt {
using namespace ne;

struct Npy {
  std::vector<float> f;
  std::vector<int32_t> i;
  std::vector<int64_t> l;
  std::vector<int64_t> shape;
  bool isint = false;
  size_t n = 0;
};
inline Npy npy(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  NE_CHECK(in.good(), "cannot open %s", path.c_str());
  char magic[8];
  in.read(magic, 8);
  NE_CHECK(memcmp(magic, "\x93NUMPY", 6) == 0, "%s: not npy", path.c_str());
  uint32_t hl = 0;
  if (magic[6] == 1) { uint16_t h; in.read((char *)&h, 2); hl = h; }
  else in.read((char *)&hl, 4);
  std::string hdr(hl, ' ');
  in.read(hdr.data(), hl);
  Npy r;
  const bool f4 = hdr.find("'<f4'") != std::string::npos, i4 = hdr.find("'<i4'") != std::string::npos,
             i8 = hdr.find("'<i8'") != std::string::npos;
  NE_CHECK(f4 || i4 || i8, "%s: dtype %s", path.c_str(), hdr.c_str());
  NE_CHECK(hdr.find("'fortran_order': False") != std::string::npos, "%s: fortran order", path.c_str());
  auto a = hdr.find('('), b = hdr.find(')');
  std::stringstream ss(hdr.substr(a + 1, b - a - 1));
  size_t n = 1;
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (tok.find_first_not_of(" ") != std::string::npos) { r.shape.push_back(std::stoll(tok)); n *= std::stoull(tok); }
  r.n = n;
  r.isint = i4 || i8;
  if (f4) { r.f.resize(n); in.read((char *)r.f.data(), n * 4); }
  else if (i4) { r.i.resize(n); in.read((char *)r.i.data(), n * 4); }
  else {
    r.l.resize(n);
    in.read((char *)r.l.data(), n * 8);
    r.i.assign(r.l.begin(), r.l.end());
  }
  NE_CHECK(in.good(), "%s: short read", path.c_str());
  return r;
}
inline std::vector<float> rd(const Tensor &t, size_t n = 0, size_t off = 0) {
  if (!n) n = t.numel() - off;
  std::vector<float> v(t.numel());
  download(t, v.data(), t.numel() * 4);
  return std::vector<float>(v.begin() + off, v.begin() + off + n);
}
inline void wr(const Tensor &t, const std::vector<float> &v, size_t off = 0) {
  std::vector<float> all(t.numel());
  download(t, all.data(), all.size() * 4);
  std::copy(v.begin(), v.end(), all.begin() + off);
  upload(t, all.data(), all.size() * 4);
}
inline double rel(const std::vector<float> &a, const std::vector<float> &b) {
  NE_CHECK(a.size() == b.size(), "size %zu vs %zu", a.size(), b.size());
  double e = 0, n = 0;
  for (size_t i = 0; i < a.size(); i++) { const double d = (double)a[i] - b[i]; e += d * d; n += (double)b[i] * b[i]; }
  return std::sqrt(e / std::max(n, 1e-30));
}
inline double maxabs(const std::vector<float> &a, const std::vector<float> &b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs((double)a[i] - b[i]));
  return m;
}
inline bool finite(const std::vector<float> &a) {
  for (float x : a) if (!std::isfinite(x)) return false;
  return true;
}
// prints and returns rel L2
inline double row(const char *mode, const char *ours, const char *golden, const std::vector<float> &a,
                  const std::vector<float> &b) {
  const double r = rel(a, b);
  printf("%-6s %-22s %-24s n=%7zu rel_l2 %.3e  max_abs %.3e%s\n", mode, ours, golden, a.size(), r, maxabs(a, b),
         finite(a) ? "" : "  NONFINITE");
  return r;
}
// Attention-layer KV state from f32 k, v [2][T][256] (roped keys), raw indexer keys [T][128]: lab-attn's QSA caches
// when the layer uses them (st.qc, the default), else the dense path's buffers; sets pos = T.
inline void import_kv(ne::fn::GdnState &st, const ne::fn::LayerW &w, const std::vector<float> &k,
                      const std::vector<float> &v, const std::vector<float> &ik, int T) {
  if (st.qc.tmax > 0) {
    ne::attn::import_cache(st.qc, w.qsa, k.data(), v.data(), ik.data(), T);
    return;
  }
  for (int g = 0; g < 2; g++) {
    memcpy((float *)st.kc.host() + (size_t)g * st.tmax * 256, k.data() + (size_t)g * T * 256, (size_t)T * 1024);
    memcpy((float *)st.vc.host() + (size_t)g * st.tmax * 256, v.data() + (size_t)g * T * 256, (size_t)T * 1024);
  }
  memcpy(st.ikeys.host(), ik.data(), (size_t)T * 512);
  ((uint32_t *)st.pos.host())[0] = (uint32_t)T;
}
// KV rows [0, T) as f32 [2][T][256] each
inline void export_kv(ne::fn::GdnState &st, int T, std::vector<float> &k, std::vector<float> &v) {
  k.assign((size_t)2 * T * 256, 0.0f);
  v.assign((size_t)2 * T * 256, 0.0f);
  if (st.qc.tmax > 0) { ne::attn::export_cache(st.qc, T, k.data(), v.data(), nullptr, nullptr); return; }
  for (int g = 0; g < 2; g++) {
    memcpy(k.data() + (size_t)g * T * 256, (float *)st.kc.host() + (size_t)g * st.tmax * 256, (size_t)T * 1024);
    memcpy(v.data() + (size_t)g * T * 256, (float *)st.vc.host() + (size_t)g * st.tmax * 256, (size_t)T * 1024);
  }
}
inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

}  // namespace mtpt
