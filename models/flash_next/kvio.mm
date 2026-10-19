// KV-row and recurrent-state copies between the model and caller buffers (lab-srvcache: the server's multi-
// conversation prefix cache, docs/server.md section 3b). Host memcpy on shared-storage buffers, parallel over
// (layer, part) chunks; call only with no GPU work in flight.
//
// KV rows [a, b) of the 12 QSA attention layers <-> a caller buffer laid out for `rows` rows (ne_fn_kv_bytes):
//   per attention layer (in layer order): kc of head 0, head 1 (rows x 256 x esz each), vc likewise, kv8 only: ksb
//   and vsb per head (rows x 16 B), ikeys (rows x 256 B), then bkeys (rows / 4 + 1 blocks x 256 B).
// Rows [a, b) occupy the first b - a rows of each part; block keys: blocks [a / 4, b / 4) (block j is complete once
// row 4j + 3 is in, so a segment holds the blocks it completes; consecutive segments [a, b) [b, c) tile them).
// Recurrent state <-> a caller buffer of ne_fn_state_bytes(): GDN conv + ssm of the 36 GDN layers and the PLE conv
// (what ne_fn_model_snapshot copies), with meta = {T, hist0, hist1}. state_io to the model = ne_fn_model_restore:
// T / pos / history set, lazy GDN rows dropped, dirty cleared; KV rows are not touched (load them with kv_io).
#include "models/flash_next/model.h"

#include <cstring>
#include <dispatch/dispatch.h>
#include <vector>

using namespace ne;
using namespace ne::fn;

namespace {

struct Cp {
  char *dst;
  const char *src;
  size_t n;
};

void run_copies(std::vector<Cp> &cps) {
  // split into <= 8 MB pieces, then copy them in parallel
  constexpr size_t kPiece = 8u << 20;
  std::vector<Cp> pieces;
  for (const Cp &c : cps)
    for (size_t o = 0; o < c.n; o += kPiece) pieces.push_back({c.dst + o, c.src + o, std::min(kPiece, c.n - o)});
  if (pieces.empty()) return;
  Cp *p = pieces.data();
  dispatch_apply(pieces.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t i) { memcpy(p[i].dst, p[i].src, p[i].n); });
}

struct AttnPart {
  const Tensor *t;
  size_t row_bytes;   // bytes of one row in this part
  size_t head_off;    // byte offset of the head's [tmax] rows in the tensor (0 for ikeys / bkeys)
  bool blocks;        // bkeys: indexed by block
};

// the parts of these caches, in buffer order
std::vector<AttnPart> attn_parts(const std::vector<attn::QsaCache *> &cs) {
  std::vector<AttnPart> out;
  for (attn::QsaCache *cp : cs) {
    attn::QsaCache &c = *cp;
    NE_CHECK(c.tmax > 0, "not a QSA cache (NE_FN_ATTN=dense has no KV export)");
    const size_t T = (size_t)c.tmax;
    const size_t esz = c.kc.nbytes() / ((size_t)attn::NKV * T * attn::HD);
    for (const Tensor *t : {&c.kc, &c.vc})
      for (int h = 0; h < attn::NKV; h++) out.push_back({t, attn::HD * esz, h * T * attn::HD * esz, false});
    if (c.fmt == attn::KvFmt::KV8)
      for (const Tensor *t : {&c.ksb, &c.vsb})
        for (int h = 0; h < attn::NKV; h++) out.push_back({t, 16, h * T * 16, false});
    out.push_back({&c.ikeys, attn::IHD * 2, 0, false});
    out.push_back({&c.bkeys, attn::IHD * 2, 0, true});
  }
  return out;
}

size_t part_cap(const AttnPart &p, int rows) { return p.row_bytes * (p.blocks ? (size_t)rows / 4 + 1 : (size_t)rows); }

std::vector<attn::QsaCache *> model_caches(Model &m) {
  std::vector<attn::QsaCache *> cs;
  for (size_t l = 0; l < m.st.size(); l++)
    if (!m.L[l].linear) cs.push_back(&m.st[l].qc);
  return cs;
}

template <class F> void each_state(Model &m, F f) {
  for (auto &s : m.st)
    for (Tensor *t : {&s.conv, &s.ssm, &s.ple_conv})
      if (t->valid()) f(*t);
}

}  // namespace

namespace ne::fn {
int64_t kv_bytes(const std::vector<attn::QsaCache *> &cs, int rows) {
  NE_CHECK(rows >= 0, "rows %d", rows);
  int64_t n = 0;
  for (const AttnPart &p : attn_parts(cs)) n += (int64_t)part_cap(p, rows);
  return n;
}

void kv_io(const std::vector<attn::QsaCache *> &cs, bool to_model, int a, int b, void *buf, int rows) {
  NE_CHECK(buf, "null buffer");
  for (attn::QsaCache *c : cs)
    NE_CHECK(0 <= a && a <= b && b <= c->tmax && b - a <= rows, "rows [%d, %d) vs buffer %d / tmax %d", a, b, rows,
             c->tmax);
  std::vector<Cp> cps;
  char *base = (char *)buf;
  for (const AttnPart &p : attn_parts(cs)) {
    size_t first = p.blocks ? (size_t)a / 4 : (size_t)a, cnt = p.blocks ? (size_t)(b / 4 - a / 4) : (size_t)(b - a);
    char *mp = (char *)p.t->host() + p.head_off + first * p.row_bytes;
    NE_CHECK(p.head_off + (first + cnt) * p.row_bytes <= p.t->nbytes(), "kv part out of range");
    if (cnt) cps.push_back(to_model ? Cp{mp, base, cnt * p.row_bytes} : Cp{base, mp, cnt * p.row_bytes});
    base += part_cap(p, rows);
  }
  run_copies(cps);
}
}  // namespace ne::fn

extern "C" {

NE_API int64_t ne_fn_kv_bytes(ne_fn_model *h, int rows) {
  int64_t n = -1;
  c_call([&] {
    NE_CHECK(h && h->m && rows >= 0, "bad argument");
    n = kv_bytes(model_caches(*h->m), rows);
  });
  return n;
}

// to_model = 1: buffer -> model rows [a, b); 0: model -> buffer. buf holds `rows` >= b - a rows (ne_fn_kv_bytes).
NE_API int ne_fn_kv_io(ne_fn_model *h, int to_model, int a, int b, void *buf, int rows) {
  return c_call([&] {
    NE_CHECK(h && h->m && buf, "null argument");
    Model &m = *h->m;
    NE_CHECK(to_model || !m.dirty, "KV export of a dirty model (generate() ran ahead)");
    NE_CHECK(b <= m.tmax, "rows [%d, %d) vs tmax %d", a, b, m.tmax);
    kv_io(model_caches(m), to_model != 0, a, b, buf, rows);
  });
}

NE_API int64_t ne_fn_state_bytes(ne_fn_model *h) {
  if (!h || !h->m) return -1;
  int64_t n = 0;
  each_state(*h->m, [&](const Tensor &t) { n += (int64_t)t.nbytes(); });
  return n;
}

// meta: int64 [3] = T, hist[0], hist[1] (written on export, read on import)
NE_API int ne_fn_state_io(ne_fn_model *h, int to_model, void *buf, int64_t nbytes, int64_t *meta) {
  return c_call([&] {
    NE_CHECK(h && h->m && buf && meta, "null argument");
    Model &m = *h->m;
    NE_CHECK(nbytes == ne_fn_state_bytes(h), "state buffer %lld B, want %lld", (long long)nbytes,
             (long long)ne_fn_state_bytes(h));
    std::vector<Cp> cps;
    char *p = (char *)buf;
    if (to_model) {
      NE_CHECK(meta[0] >= 0 && meta[0] <= m.tmax, "T %lld > tmax", (long long)meta[0]);
      each_state(m, [&](const Tensor &t) { cps.push_back({(char *)t.host(), p, t.nbytes()}); p += t.nbytes(); });
      run_copies(cps);
      for (auto &st : m.st) gdn_clear_pending(st);  // lab-gdnlazy
      m.T = (int)meta[0];
      m.hist[0] = meta[1];
      m.hist[1] = meta[2];
      ((uint32_t *)m.pos.host())[0] = (uint32_t)m.T;
      m.dirty = false;
      m.gen++;  // lab-mtpcont: a model-only restore leaves the MTP head stale (ne_fn_mtp_state_io restores both)
    } else {
      NE_CHECK(!m.dirty, "state export of a dirty model (generate() ran ahead)");
      gdn_materialise(m.st);  // deferred GDN rows into ssm first
      each_state(m, [&](const Tensor &t) { cps.push_back({p, (const char *)t.host(), t.nbytes()}); p += t.nbytes(); });
      run_copies(cps);
      meta[0] = m.T;
      meta[1] = m.hist[0];
      meta[2] = m.hist[1];
    }
  });
}

}  // extern "C"
