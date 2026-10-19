/* Ijah C API (plain C, for ctypes and any host language). Owner: lab-core.
 *
 * Conventions
 * - Functions returning int: 0 = ok, -1 = error (message in ne_last_error(), per thread).
 * - Functions returning a pointer: NULL = error.
 * - Buffers are shared-storage MTLBuffers (unified memory): ne_buf_contents() is a CPU pointer to the same bytes the
 *   GPU sees. Synchronise yourself: read results only after the GPU work that wrote them has completed.
 * - Pack tensors are no-copy views into the mmapped shard files. Their ne_buf handles belong to the pack (do not
 *   ne_free them; they die with ne_pack_close). Binding one to a GPU dispatch pages in its whole window (= the
 *   tensor, rounded out to 16 KiB pages), see docs/core.md "Traps".
 * - Other directories add C entry points by defining NE_API functions in their own .mm files, which link into
 *   libengine.dylib (see docs/core.md "Adding a C entry point"). Declare them in your own header, not here.
 */
#ifndef NATIVE_ENGINE_H
#define NATIVE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NE_API __attribute__((visibility("default")))
#define NE_ABI_VERSION 1

/* ---- dtypes (values are stable; the C++ enum ne::DType uses the same numbers) ---- */
typedef enum {
  NE_F32 = 0, NE_F16 = 1, NE_BF16 = 2, NE_U8 = 3, NE_U16 = 4, NE_U32 = 5, NE_I32 = 6,
  NE_I64 = 7, NE_I8 = 8, NE_I16 = 9, NE_U64 = 10, NE_F64 = 11, NE_BOOL = 12
} ne_dtype;
#define NE_MAX_DIMS 8

NE_API int ne_abi_version(void);
NE_API const char *ne_last_error(void);
NE_API const char *ne_dtype_name(int dtype);  /* "BF16", ... (safetensors spelling); "?" if unknown */
NE_API int ne_dtype_size(int dtype);          /* bytes per element; 0 if unknown */

/* ---- device ---- */
typedef struct {
  char name[64];
  int has_nax;                   /* Apple10 family (M5): neural accelerators behind MPP tensor ops */
  int has_metal4;                /* MTL4 command queue / encoders available */
  uint64_t max_buffer_length;    /* bytes; larger tensors have no single GPU view */
  uint64_t recommended_working_set;
  uint64_t max_threadgroup_memory;
  char backend[16];              /* recorder backend that Auto resolves to here: "mtl4", "icb", "classic", "serial" */
} ne_device_info;
NE_API int ne_device_query(ne_device_info *out);

/* ---- buffers ---- */
typedef struct ne_buf ne_buf;
NE_API ne_buf *ne_alloc(uint64_t nbytes);  /* shared storage, zero-filled */
NE_API void ne_free(ne_buf *b);            /* no-op on NULL and on pack-owned buffers */
NE_API uint64_t ne_buf_size(const ne_buf *b);
NE_API void *ne_buf_contents(const ne_buf *b);
NE_API uint64_t ne_buf_gpu_address(const ne_buf *b);
NE_API int ne_upload(ne_buf *b, uint64_t offset, const void *src, uint64_t nbytes);
NE_API int ne_download(const ne_buf *b, uint64_t offset, void *dst, uint64_t nbytes);

/* ---- packs (a directory of safetensors shards + config.json [+ ngram_table.bin]) ---- */
typedef struct ne_pack ne_pack;
typedef struct {
  int dtype;                     /* ne_dtype */
  int ndim;
  int64_t shape[NE_MAX_DIMS];
  uint64_t nbytes;
  const char *file;              /* resolved (symlink-free) path of the shard; valid while the pack is open */
  uint64_t file_offset;          /* absolute byte offset of the tensor's data in that file */
  const void *host;              /* CPU pointer into the read-only mapping (always set) */
  ne_buf *buf;                   /* GPU view (no-copy window), NULL if nbytes > max_buffer_length or nbytes == 0 */
  uint64_t buf_offset;           /* byte offset of the tensor inside buf */
} ne_tensor_desc;

NE_API ne_pack *ne_pack_open(const char *dir);
NE_API void ne_pack_close(ne_pack *p);
NE_API const char *ne_pack_config_json(const ne_pack *p);   /* config.json text ("{}" if absent) */
NE_API int64_t ne_pack_num_tensors(const ne_pack *p);
NE_API const char *ne_pack_tensor_name(const ne_pack *p, int64_t i);  /* sorted by name */
NE_API int ne_tensor_info(const ne_pack *p, const char *name, ne_tensor_desc *out);  /* -1 if missing */
/* A GPU view of rows [row0, row0 + nrows) of a pack tensor (dim 0), as a new caller-owned buffer handle (ne_free it;
 * the mapping stays alive until then). For tensors too large for one buffer, e.g. the 95 GiB n-gram table. */
NE_API ne_buf *ne_pack_view_rows(const ne_pack *p, const char *name, int64_t row0, int64_t nrows, uint64_t *offset);

/* ---- kernels ---- */
/* Function-constant value: type 0 = uint32, 1 = int32, 2 = float32, 3 = bool, 4 = uint16, 5 = int16; bits holds the
 * value's bit pattern (float: its IEEE bits; bool: 0/1). */
typedef struct { uint32_t index; uint32_t type; uint32_t bits; } ne_fconst;
/* A kernel argument at buffer index = its position: a buffer (+ byte offset), or inline bytes when bytes != NULL. */
typedef struct { ne_buf *buf; uint64_t offset; const void *bytes; uint32_t nbytes; } ne_arg;

NE_API int ne_kernel_exists(const char *fn);  /* 1 if the metallib has it, 0 if not, -1 on error */
/* Run one kernel `reps` times (dependent chain with barriers, one command buffer) and wait. grid = threadgroups per
 * grid, tg = threads per threadgroup, tg_mem = threadgroup memory bytes at index 0. backend: NULL/"" = default
 * (NE_BACKEND env or auto), else "mtl4" | "icb" | "classic" | "serial". gpu_ms (may be NULL) = whole command buffer. */
NE_API int ne_run_kernel(const char *fn, const ne_fconst *fc, int nfc, const ne_arg *args, int nargs,
                         const uint32_t grid[3], const uint32_t tg[3], uint32_t tg_mem, int reps,
                         const char *backend, double *gpu_ms);

#ifdef __cplusplus
}
#endif
#endif /* NATIVE_ENGINE_H */
