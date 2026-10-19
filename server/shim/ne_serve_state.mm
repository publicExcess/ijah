// Server-side state snapshot / restore of the whole Flash-Next model (lab-server), for prefix reuse across requests.
//
// A snapshot = host copies of every recurrent state (GDN conv + ssm of the 36 GDN layers, the PLE conv state) + T + the
// n-gram history. Attention KV / QSA caches are NOT copied: restoring sets pos = T (the same truncation the MTP
// rollback uses), so a snapshot is valid only while KV rows [0, T) still hold its tokens (the server tracks that).
// ~113 MB per snapshot. Call only with no GPU work in flight (after ne_fn_model_step / prefill / mtp_generate returned).
//
// This reads ne::fn::Model's fields, so it must be compiled against the same headers as the libengine.dylib it is
// loaded next to: tools/serve.sh builds it per library directory and refuses when the library is older than the
// model headers; ne_srv_header lets the server cross-check T / tmax against the C API before anything is copied.
// Request to the model owner (docs/kernels.md Requests): an official ne_fn_model_snapshot / restore, then this goes.
#include "models/flash_next/model.h"

#include <cstring>
#include <vector>

#ifdef NE_SRV_DEFINE_HANDLE  // older model.h without the shared handle definition
struct ne_fn_model {
  std::unique_ptr<ne::fn::Model> m;
};
#endif

using namespace ne;
using namespace ne::fn;

struct ne_srv_snap {
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

// Plain fields only (no pointer chasing): T, tmax, number of layers and of layer states, sizeof(Model).
NE_API int ne_srv_header(ne_fn_model *h, int *T, int *tmax, int *n_layers, int *n_states, int *dirty, int *model_size) {
  if (!h || !h->m) return -1;
  Model &m = *h->m;
  *T = m.T;
  *tmax = m.tmax;
  *n_layers = (int)m.L.size();
  *n_states = (int)m.st.size();
  *dirty = m.dirty ? 1 : 0;
  *model_size = (int)sizeof(Model);
  return 0;
}

// Bytes a snapshot copies (call after ne_srv_header matched the C API).
NE_API int64_t ne_srv_state_bytes(ne_fn_model *h) {
  if (!h || !h->m) return -1;
  int64_t n = 0;
  each_state(*h->m, [&](const Tensor &t) { n += (int64_t)t.nbytes(); });
  return n;
}

NE_API ne_srv_snap *ne_srv_snapshot(ne_fn_model *h) {
  if (!h || !h->m || h->m->dirty) return nullptr;
  Model &m = *h->m;
  gdn_materialise(m.st);  // lab-gdnlazy: deferred GDN rows into ssm first (docs/kernels.md 5d)
  auto s = new ne_srv_snap;
  s->T = m.T;
  s->hist[0] = m.hist[0];
  s->hist[1] = m.hist[1];
  each_state(m, [&](const Tensor &t) {
    s->parts.emplace_back(t.nbytes());
    memcpy(s->parts.back().data(), t.host(), t.nbytes());
  });
  return s;
}

NE_API int ne_srv_restore(ne_fn_model *h, const ne_srv_snap *s) {
  if (!h || !h->m || !s) return -1;
  Model &m = *h->m;
  size_t i = 0;
  bool ok = true;
  each_state(m, [&](const Tensor &t) {
    if (i >= s->parts.size() || s->parts[i].size() != t.nbytes()) { ok = false; return; }
    i++;
  });
  if (!ok || i != s->parts.size() || s->T > m.tmax) return -2;
  i = 0;
  each_state(m, [&](const Tensor &t) { memcpy(t.host(), s->parts[i].data(), t.nbytes()); i++; });
  for (auto &st : m.st) gdn_clear_pending(st);  // lab-gdnlazy
  m.T = s->T;
  m.hist[0] = s->hist[0];
  m.hist[1] = s->hist[1];
  ((uint32_t *)m.pos.host())[0] = (uint32_t)s->T;
  m.dirty = false;
  return 0;
}

NE_API int ne_srv_snap_T(const ne_srv_snap *s) { return s ? s->T : -1; }
NE_API void ne_srv_snap_free(ne_srv_snap *s) { delete s; }

}  // extern "C"
