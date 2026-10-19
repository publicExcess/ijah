// C API (include/engine.h) over the C++ core. Owner: lab-core.
#import <Foundation/Foundation.h>

#include "core/engine.hpp"

struct ne_pack {
  std::shared_ptr<ne::Pack> pack;  // shared: models keep it alive after ne_pack_close (Pack::weak_from_this)
  std::map<std::string, ne_buf *, std::less<>> bufs;  // pack-owned handles, one per tensor with a GPU view
};

namespace ne {
static thread_local std::string g_last_error;
void set_last_error(const std::string &msg) { g_last_error = msg; }
ne_buf *to_c(id<MTLBuffer> b) {
  ne_buf *h = new ne_buf();
  h->buf = b;
  h->owned = true;
  return h;
}
Tensor tensor_from_c(const ne_buf *b, uint64_t offset, DType dt, std::initializer_list<int64_t> shape) {
  NE_CHECK(b && b->buf, "null buffer handle");
  return make_tensor(b->buf, offset, dt, shape);
}
const Pack *pack_from_c(const ne_pack *p) { return p ? p->pack.get() : nullptr; }
}  // namespace ne

using namespace ne;

extern "C" {

NE_API int ne_abi_version(void) { return NE_ABI_VERSION; }
NE_API const char *ne_last_error(void) { return g_last_error.c_str(); }
NE_API const char *ne_dtype_name(int t) { return dtype_name((DType)t); }
NE_API int ne_dtype_size(int t) { return (int)dtype_size((DType)t); }

NE_API int ne_device_query(ne_device_info *out) {
  return c_call([&] {
    NE_CHECK(out, "null out");
    Device &d = Device::get();
    memset(out, 0, sizeof *out);
    snprintf(out->name, sizeof out->name, "%s", d.name.c_str());
    out->has_nax = d.has_nax;
    out->has_metal4 = d.has_metal4;
    out->max_buffer_length = d.max_buffer_length;
    out->recommended_working_set = d.dev.recommendedMaxWorkingSetSize;
    out->max_threadgroup_memory = d.dev.maxThreadgroupMemoryLength;
    snprintf(out->backend, sizeof out->backend, "%s", backend_name(resolve_backend(Backend::Auto)));
  });
}

NE_API ne_buf *ne_alloc(uint64_t nbytes) {
  ne_buf *h = nullptr;
  c_call([&] { h = to_c(alloc_bytes(nbytes)); });
  return h;
}
NE_API void ne_free(ne_buf *b) {
  if (b && b->owned) delete b;
}
NE_API uint64_t ne_buf_size(const ne_buf *b) { return b && b->buf ? b->buf.length : 0; }
NE_API void *ne_buf_contents(const ne_buf *b) { return b && b->buf ? b->buf.contents : nullptr; }
NE_API uint64_t ne_buf_gpu_address(const ne_buf *b) { return b && b->buf ? b->buf.gpuAddress : 0; }
NE_API int ne_upload(ne_buf *b, uint64_t offset, const void *src, uint64_t n) {
  return c_call([&] {
    NE_CHECK(b && b->buf && offset <= b->buf.length && n <= b->buf.length - offset, "upload out of range");
    NE_CHECK(!b->readonly, "upload into a pack tensor (read-only mapping)");
    memcpy((char *)b->buf.contents + offset, src, n);
  });
}
NE_API int ne_download(const ne_buf *b, uint64_t offset, void *dst, uint64_t n) {
  return c_call([&] {
    NE_CHECK(b && b->buf && offset <= b->buf.length && n <= b->buf.length - offset, "download out of range");
    memcpy(dst, (const char *)b->buf.contents + offset, n);
  });
}

NE_API ne_pack *ne_pack_open(const char *dir) {
  ne_pack *p = nullptr;
  c_call([&] {
    NE_CHECK(dir, "null dir");
    auto pk = Pack::open(dir);
    // the handle is built in a unique_ptr (its buffer handles too): a throw in the loop leaves nothing half-built
    struct Closer {
      void operator()(ne_pack *q) const {
        for (auto &kv : q->bufs) delete kv.second;
        delete q;
      }
    };
    std::unique_ptr<ne_pack, Closer> u(new ne_pack());
    for (const std::string &n : pk->names()) {
      const TensorInfo *ti = pk->info(n);
      if (ti->t.buf) {
        auto h = std::make_unique<ne_buf>();
        h->buf = ti->t.buf;
        h->owned = false;
        h->readonly = true;
        u->bufs.emplace(n, h.get());
        h.release();
      }
    }
    u->pack = std::shared_ptr<Pack>(std::move(pk));
    p = u.release();
  });
  return p;
}
// The handle and its tensor handles go now; the Pack itself lives on while a model made from it does.
NE_API void ne_pack_close(ne_pack *p) {
  if (!p) return;
  for (auto &kv : p->bufs) delete kv.second;
  delete p;
}
NE_API const char *ne_pack_config_json(const ne_pack *p) { return p ? p->pack->config_text().c_str() : nullptr; }
NE_API int64_t ne_pack_num_tensors(const ne_pack *p) { return p ? (int64_t)p->pack->names().size() : -1; }
NE_API const char *ne_pack_tensor_name(const ne_pack *p, int64_t i) {
  if (!p || i < 0 || i >= (int64_t)p->pack->names().size()) {
    set_last_error("tensor index out of range");
    return nullptr;
  }
  return p->pack->names()[i].c_str();
}
NE_API int ne_tensor_info(const ne_pack *p, const char *name, ne_tensor_desc *out) {
  return c_call([&] {
    NE_CHECK(p && name && out, "null argument");
    const TensorInfo *ti = p->pack->info(name);
    NE_CHECK(ti, "no tensor %s", name);
    memset(out, 0, sizeof *out);
    out->dtype = (int)ti->t.dtype;
    out->ndim = ti->t.ndim;
    for (int i = 0; i < ti->t.ndim; i++) out->shape[i] = ti->t.shape[i];
    out->nbytes = ti->nbytes;
    out->file = ti->file.c_str();
    out->file_offset = ti->file_offset;
    out->host = ti->host;
    if (auto it = p->bufs.find(name); it != p->bufs.end()) {
      out->buf = it->second;
      out->buf_offset = ti->t.offset;
    }
  });
}
NE_API ne_buf *ne_pack_view_rows(const ne_pack *p, const char *name, int64_t row0, int64_t nrows, uint64_t *offset) {
  ne_buf *h = nullptr;
  c_call([&] {
    NE_CHECK(p && name, "null argument");
    Tensor v = p->pack->view_rows(name, row0, nrows);
    if (offset) *offset = v.offset;
    h = to_c(v.buf);
    h->readonly = true;  // caller frees the handle; the mapping is read-only
  });
  return h;
}

NE_API int ne_kernel_exists(const char *fn) {
  int r = -1;
  c_call([&] { r = Pipelines::has(fn ? fn : "") ? 1 : 0; });
  return r;
}

NE_API int ne_run_kernel(const char *fn, const ne_fconst *fc, int nfc, const ne_arg *args, int nargs,
                         const uint32_t grid[3], const uint32_t tg[3], uint32_t tg_mem, int reps, const char *backend,
                         double *gpu_ms) {
  return c_call([&] {
    NE_CHECK(fn && grid && tg && (nargs == 0 || args) && (nfc == 0 || fc), "null argument");
    FunctionConstants fcs;
    for (int i = 0; i < nfc; i++) fcs.push_back(FC{fc[i].index, (FC::Type)fc[i].type, fc[i].bits});
    id<MTLComputePipelineState> pso = Pipelines::get(fn, fcs);
    std::vector<Arg> av;
    for (int i = 0; i < nargs; i++) {
      if (args[i].bytes) av.push_back(Arg::bytes(args[i].bytes, args[i].nbytes));
      else {
        NE_CHECK(args[i].buf && args[i].buf->buf, "argument %d: null buffer", i);
        av.push_back(Arg(args[i].buf->buf, args[i].offset));
      }
    }
    Recorder r(backend_from_name(backend ? backend : ""));
    for (int i = 0; i < std::max(reps, 1); i++) {
      if (i) r.barrier();
      r.dispatch(pso, av, MTLSizeMake(grid[0], grid[1], grid[2]), MTLSizeMake(tg[0], tg[1], tg[2]), tg_mem);
    }
    r.end();
    r.run();
    if (gpu_ms) *gpu_ms = r.gpu_ms();
  });
}

}  // extern "C"
