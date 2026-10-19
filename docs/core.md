# Core (lab-core): device, tensors, pack loader, pipelines, recorder, C API

Status: 15 Oct 2026. Headers are stable (`core/READY` exists); additions are backwards compatible at source level
(rebuild after pulling: `make` tracks header dependencies).

`make test` (core part): **PASSED, 141 checks** + the dispatch-cost table below (skipped when the machine lock is
held; `make bench` waits for it). `build/core-test --quick` = correctness only (no GPU timing, no lock).

## Build

```
make            # build/libengine.dylib, build/engine.metallib, build/engine-cli, build/core-test
make test       # build/core-test, then TEST_CMDS, then TEST_TARGETS (tests/tests.mk etc.)
make bench      # dispatch cost per backend (takes the machine lock itself)
```

- Host: `clang++ -std=c++20 -fobjc-arc -O2 -g`, includes `-I. -Iinclude` (so `#include "core/engine.hpp"`,
  `#include "engine.h"`). Header dependencies tracked (`-MMD`).
- Metal: every `*.metal` under `core/`, `kernels/`, `models/` is compiled (`xcrun -sdk macosx metal -std=metal4.0
  -I. -Iinclude -c`) and linked into one `build/engine.metallib`. **`-std=metal4.0` is required** for
  `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` (with metal3.2 the `mpp::tensor_ops`
  namespace and `dextents` are undeclared). No .mk entry needed for .metal files.
- `kernels/kernels.mk`, `models/*/model.mk`, `tests/tests.mk` are included if present. Variables they may append to:
  - `ENGINE_SRCS += kernels/foo.mm` — host sources linked into libengine.dylib (.mm or .cpp).
  - `TEST_BINS += build/foo-test` plus a rule; link with `$(LINK_ENGINE)`, e.g.
    `build/foo-test: $(BUILD)/obj/tests/foo_test.mm.o $(LIB) ; $(CXX) $< $(LINK_ENGINE) -o $@`
    (the generic rule `$(BUILD)/obj/%.mm.o: %.mm` compiles the object).
  - `TEST_CMDS += build/foo-test` — run by `make test` after the core test. **One word per entry** (make splits on
    spaces: `TEST_CMDS += python3 tests/x.py` would become two "commands"); wrap commands with arguments in a script.
  - `TEST_TARGETS += test-foo` with your own phony rule `test-foo: ...` (any recipe) — made after TEST_CMDS.
  - `EXTRA_CXXFLAGS`, `EXTRA_METALFLAGS` (e.g. `-DFOO=1`), `METAL_STD`, `OPT`.
- Files named `core/main_*.mm` are executables, not library sources.
- `.DEFAULT_GOAL := all` is set before the includes (a rule in an included .mk would otherwise become the default),
  and `BUILD LIB METALLIB LINK_ENGINE CXX CXXFLAGS` are defined before them, so .mk rules can use them.

## Device (`core/device.h`)

`ne::Device::get()` — singleton: `dev`, `queue` (classic), `queue4` (MTL4), `has_nax` (= `supportsFamily:Apple10`,
the M5 GPU family with neural accelerators behind MPP tensor ops), `has_metal4` (Metal4 family and the MTL4 queue
was created), `max_buffer_length` (80.64 GiB on this M5 Max 128 GB), `page_size` (16 KiB). M5 Max: maxThreadgroupMemory
32 KiB, recommendedMaxWorkingSetSize 116 GiB.

## Tensors (`core/tensor.h`)

- `enum class DType { F32, F16, BF16, U8, U16, U32, I32, I64, I8, I16, U64, F64, BOOL }` — numbering = `ne_dtype` in
  `engine.h`. (Deviation: I64/I8/I16/U64/F64/BOOL added; the pack has I64 tensors, e.g. `ngram_heads_offsets`.)
- `struct Tensor { id<MTLBuffer> buf; size_t offset; DType dtype; int ndim; int64_t shape[8]; }` — contiguous
  row-major view (deviation: up to 8 dims; the trellis tensors are 4-D, conv1d 3-D). Helpers: `numel()`, `nbytes()`,
  `row_bytes()`, `gpu_address()`, `host()` (CPU pointer, shared storage), `rows(r0, n)`, `view(off, dtype, shape)`,
  `desc()`.
- `alloc(dtype, {shape})` (shared storage, zero-filled, size rounded up to 16 B), `alloc_bytes(n)`,
  `upload/download(t, ptr, n)` (memcpy; only when no GPU work on it is in flight).

## Pack loader (`core/pack.h`)

`auto p = ne::Pack::open("<packs>/Qwen3.8-Flash-Next-EXL3-K5");` (deviation: returns `std::unique_ptr<Pack>`).

- Files: the `weight_map` values of `model.safetensors.index.json` (every index entry must be found in its file), else
  every `*.safetensors`; plus `config["ngram_table"]["file"]` (default `ngram_table.bin`, a safetensors-format file).
  Tensors from non-`.safetensors` files are prefixed with the file stem: **`ngram_table.weight`** BF16
  [320001536, 160].
- Symlinks are resolved (`realpath`); each distinct real file is mmapped once (`PROT_READ`, `MAP_SHARED`, whole file).
- Each tensor gets its own no-copy `MTLBuffer` window over the mapping (page-aligned start, rounded-out end) and
  `Tensor.offset` = position inside that window. Buffers keep the mapping alive (deallocator holds a `shared_ptr`),
  so a Tensor copied out of the pack stays valid after the pack is destroyed.
- Tensors larger than `max_buffer_length` (only the n-gram table, 95.4 GiB) get no buffer: `get()` returns nullptr,
  `info()->host` is the CPU pointer, `view_rows(name, row0, n)` makes a window over a row range.
- `get(name)` -> `const Tensor*` (nullptr if missing/no view), `at(name)` (throws), `info(name)` (file, file offset,
  host pointer, nbytes), `names()` (sorted), `config()` (whole config.json), `text_config()`, `config_text()`.
- `aligned(name, 16)` -> the view if its GPU address is 16-aligned, else a cached aligned copy (anonymous RAM =
  tensor size; `aligned_copy_bytes()`). For big tensors use an aligned pack instead (below).
- **Alignment**: the K5 pack is badly aligned — tensor data sits at `8 + header_len + offset`, and the EXL3 shards
  have odd header lengths (L00-down: 805 -> data at byte 813). 1126 tensors (37 GB) are not even 4-aligned, 1521
  (56 GB, excl. the n-gram table) not 16-aligned. A no-copy view cannot fix this (mmap offsets are page multiples, so
  address mod 16 = file offset mod 16). `engine-cli align-pack SRC DST --align 16` writes a copy where every tensor
  is aligned: misaligned shards are rewritten (header padded with spaces, padding between tensors, metadata kept +
  `aligned_by`; copied with uncached pread/pwrite (`F_NOCACHE`, so tens of GB do not flush everyone's page cache)
  and each tensor's CRC32C re-checked from the written file before the rename), everything else (aligned
  shards, config, tokenizer, the n-gram table) becomes a symlink to the original's real path. Resumable. For the K5
  pack: 130 files, 56.3 GB to write at align 16 (202 files, 82.8 GB at 64). Not run on the full pack yet (heavy I/O:
  schedule it under the machine lock); tested with `--only L00-down` / `--only L01-up` (0.53 GB each: verified,
  bytes and alignment re-checked independently in Python, `NE_PACK=<aligned> core-test --quick` passes).
- Flash-Next K5 pack: 3071 tensors, 204 files (186 GB mapped incl. the n-gram table), opens in 25-80 ms.
- dtypes: F32 F16 BF16 U8 U16 U32 I32 I64 I8 I16 U64 F64 BOOL. Header checks: length sane, `end - begin ==
  numel * size`, data inside the file, no duplicate names.

JSON: `ne::Json` (`core/json.h`), a small parser written for this repo: `parse`, `find`, `operator[]`, `i64()`,
`num()`, `str()`, `arr()`, `obj()`, `get_int/get_num/get_str(key, default)`, `dump()`. Integers are exact int64.

## Pipelines (`core/pipelines.h`)

`ne::Pipelines::get("fn_name", {ne::FC::u32(0, 128), ne::FC::b(1, true)})` -> cached
`id<MTLComputePipelineState>`, keyed by (name, sorted constants); thread-safe; every pipeline has
`supportIndirectCommandBuffers = YES`. Library = `$NE_METALLIB`, else `engine.metallib` next to libengine.dylib.
`Pipelines::add_source(msl)` adds a runtime-compiled library (experiments). `has(name)`, `function_names()`.

## Recorder (`core/recorder.h`)

```cpp
ne::Recorder r;                                   // backend: NE_BACKEND env, else auto (mtl4 here)
r.dispatch(pso, {x, w, ne::Arg::value(n)}, MTLSizeMake(tgs, 1, 1), MTLSizeMake(256, 1, 1), tg_mem_bytes);
r.barrier();                                      // next dispatch waits for all earlier ones
r.dispatch(...);
r.end();                                          // finalise: constant arena, ICB / residency set
r.run(); double ms = r.gpu_ms();                  // replay (commit + wait); repeat at will
```

- Arguments bind at buffer index = position. `Arg(tensor)` / `Arg(buffer, offset)` / `Arg::bytes(p, n)` /
  `Arg::value(v)`. Inline bytes live in a recorder-owned shared "constant arena" (no setBytes: ICB and MTL4 have
  none), so all backends behave the same; patch them between replays with `bytes_at(cmd, slot)`.
- Threadgroup memory: one allocation at `[[threadgroup(0)]]` (`tg_mem` bytes, rounded up to 16).
- `dispatch_threads(pso, args, threads, tg)` — non-uniform grid (total threads; partial edge threadgroups).
- `dispatch_indirect(pso, args, ind_buf, offset, tg)` — threadgroup counts from a GPU buffer (mtl4/classic/serial;
  the ICB backend rejects it).
- `use(buffer)` / `use(tensor)`: buffers a kernel reaches only through GPU addresses stored in another buffer (pointer
  tables, e.g. per-expert weight addresses) must be declared, or they are not resident: measured, every backend then
  reads **zeros, silently, with no GPU error** (4095 of 4096 wrong in `ne_st_ptr`). Directly bound buffers need nothing.
- Without `barrier()`, consecutive dispatches may run concurrently (all backends except `serial`).
- Timing: `gpu_ms()` = command-buffer GPUEndTime - GPUStartTime of the last waited replay (MTL4: from the commit
  feedback), `encode_us()` (last commit), `wall_us()` (commit -> wait return).
- `commit()` / `wait()` / `wait_oldest()` separately for overlap: up to `Recorder::kMaxInFlight` (3) replays of one
  recorder may be in flight (MTL4 keeps one command allocator per slot); `in_flight()`. Patching `bytes_at()` while a
  replay is in flight races with it. Different recorders may also be committed back to back without host waits:
  each replay waits for the queue's earlier work (classic queues do this themselves; MTL4 replays start with
  `barrierAfterQueueStages:Dispatch beforeStages:Dispatch`).

Backends (`NE_BACKEND=mtl4|icb|classic|serial`, or `Recorder(ne::Backend::ICB)`):

| backend | how a replay is encoded | resources |
|---|---|---|
| `mtl4` (default when available) | MTL4 command buffer + compute encoder, one argument table (`setAddress` per arg; the encoder snapshots it per dispatch), `barrierAfterEncoderStages:Dispatch beforeEncoderStages:Dispatch visibility:Device` per barrier; re-encoded each replay | residency set of every buffer the recording references, built at end() |
| `icb` | indirect command buffer built once at end(); each replay = one classic concurrent encoder + `useResources` + `executeCommandsInBuffer`; barrier = `setBarrier` on the command after the barrier | `useResources` per replay |
| `classic` | classic concurrent encoder, `memoryBarrierWithScope:Buffers` per barrier | implicit |
| `serial` | classic serial encoder (every dispatch waits for the previous; barrier() is implied) | implicit |

Measured (`make test`, M5 Max, machine lock held; per dispatch, differential (T(2048) - T(256)) / 1792, median of 5):

| backend | tiny, dependent | read256, dependent | CPU encode / dispatch | tiny / read256 independent |
|---|---|---|---|---|
| mtl4 | 0.96 us | 1.50 us | 0.12-0.13 us | 0.13 / 0.51 us |
| icb | 0.93 | 1.67 | 0.003 (built once) | 0.18 / 0.36 |
| classic | 1.41 | 2.02 | 0.27-0.34 | 0.13 / 0.52 |
| serial | 0.96 | 1.48 | 0.10-0.15 | (always serial) |

This reproduces lab lead 4 (MTL4 0.93 / 1.51, ICB 1.53 / 1.64, concurrent+barrier 2.53 / 1.96). Default: **mtl4**
(lowest dependent cost on both probes, supports indirect dispatch). `icb` wins on CPU cost (a 500-dispatch forward
encodes in ~1.5 us instead of ~65 us), worth it if the host ever becomes the bottleneck. Since the chains are long
(2048), "naive" T/N equals the differential number here; for short chains it does not (clock ramp, see Traps).

Synchronous round trip of a one-dispatch replay (`run()` = commit -> GPU -> wait returns; 180 replays, median):
~135-150 us wall on every backend (p10 ~95-130, p90 ~190), of which the GPU part is 2-10 us. A raw classic command
buffer splits roughly into ~100 us from commit to GPUStartTime and ~70 us from GPUEndTime to the waiter waking
(host and GPU timestamps, so approximate); spinning on `cb.status` instead of `waitUntilCompleted` does not help.
With two replays kept in flight (commit k+1, then wait for k) the cost per one-dispatch replay drops to ~73-80 us.
So a host sync per token costs ~0.14 ms (~1% of a 12 ms token) plus GPU idle: keep the token loop GPU-resident and
commit token t+1 before waiting for token t (back-to-back commits are ordered, see above).

Machine lock from C++: `ne::MachineLock lock("lab-x", "job", minutes, /*wait=*/true)` (core/bench.h) takes
the optional lock script (`IJAH_LOCK_SCRIPT`, e.g. `tools/machine-lock.sh`; none = no lock) with this process's PID and releases it in the destructor; `core-test` and
`engine-cli bench` use it (core-test skips timing if the lock is held, `--bench` waits).

## C API (`include/engine.h`)

Plain C for ctypes: `ne_last_error`, `ne_device_query`, `ne_alloc/free/upload/download`, `ne_buf_contents`,
`ne_buf_gpu_address`, `ne_pack_open/close`, `ne_pack_config_json`, `ne_pack_num_tensors/tensor_name`,
`ne_tensor_info(pack, name, &desc)` (dtype, shape, nbytes, file, file_offset, host pointer, `ne_buf*` + offset),
`ne_pack_view_rows`, `ne_kernel_exists`, `ne_run_kernel(fn, fconsts, args, grid, tg, tg_mem, reps, backend, &ms)`
(one kernel, any function in the metallib, `reps` dependent repeats in one command buffer).
Errors: int functions return 0 / -1, pointer functions NULL; message in `ne_last_error()`.

## Adding a kernel

1. Put `kernels/foo.metal` (or `models/flash_next/foo.metal`) in the tree; `make` compiles it into the metallib.
   Shared MSL headers: include them relative to the repo root (`#include "kernels/ne_common.h"`) or relative to the file.
2. Specialise shapes with function constants (`constant uint K [[function_constant(0)]];`) and fetch with
   `Pipelines::get("foo", {FC::u32(0, K)})`.
3. Quick test without host code: `ne_run_kernel("foo", ...)` from Python (tools/engine.py).

## Adding a C entry point (from your own directory)

```objc
// kernels/foo.mm  (listed in kernels/kernels.mk: ENGINE_SRCS += kernels/foo.mm)
#include "core/engine.hpp"
extern "C" NE_API int ne_foo_forward(ne_buf *x, ne_buf *y, uint32_t n, double *gpu_ms) {
  return ne::c_call([&] {
    auto pso = ne::Pipelines::get("foo");
    ne::Recorder r;
    r.dispatch(pso, {ne::Arg(ne::from_c(x)), ne::Arg(ne::from_c(y)), ne::Arg::value(n)},
               MTLSizeMake((n + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1));
    r.end();
    r.run();
    if (gpu_ms) *gpu_ms = r.gpu_ms();
  });
}
```

Declare it in your own header (e.g. `kernels/kernels.h`), not in `include/engine.h`. Helpers in `core/engine.hpp`:
`ne::from_c(ne_buf*)` -> `id<MTLBuffer>`, `ne::tensor_from_c(buf, offset, dtype, {shape})`, `ne::to_c(buffer)`
(new caller-owned handle), `ne::pack_from_c(ne_pack*)` -> `const ne::Pack*`, `ne::c_call(lambda)` (exceptions ->
-1 + `ne_last_error`). For a model/layer that lives across calls, keep a C++ object behind your own opaque handle.

## Traps

- **Binding a no-copy mmapped buffer pages in all of it.** Measured: a 1 GiB window of `ngram_table.bin` with 7%
  resident went to 100% resident when one dispatch read one uint from it (classic encoder; the GPU time was 4 us, so
  the page-in is done at submission). `MTLResidencySet` commit + `requestResidency` does the same. Creating the
  buffer pages in nothing. Hence: one window per tensor (binding a tensor wires only that tensor), and **never bind
  the n-gram table whole** (95 GiB; it is also above maxBufferLength). Gather n-gram rows on the CPU from
  `info("ngram_table.weight")->host` (the host knows the token ids), or `view_rows()` small ranges.
- **MTL4 command buffers on one queue are not ordered by themselves.** Two recorders committed back to back (B reads
  A's output) gave 100% wrong results in 1 of 3 runs without a queue barrier; the recorder now puts
  `barrierAfterQueueStages` at the start of every MTL4 replay.
- **Kernels with function constants cannot be fetched with plain `newFunctionWithName`** (Metal validation asserts,
  even when every constant has a default): `Pipelines::get` always specialises through `MTLFunctionConstantValues`.
- ICB `setBarrier` on command i = "i waits for all earlier commands" (verified by the race-sensitive chain test: with
  barriers 0 mismatches on all backends; without, 10^3-10^4 mismatching words on mtl4, icb and classic).
- A race test needs a race-prone kernel: `dst[i] = src[(i + 257) % n]` showed 0 races on mtl4 without barriers; the
  reversed index `src[n - 1 - i]` + a 32-step LCG per thread (`ne_st_step`) shows them reliably.
- Pack buffers are read-only memory (`PROT_READ`): `upload()` into one crashes (the C API refuses it); a GPU write into
  one is undefined (not tried).
- GPU/DRAM clocks ramp inside short command buffers (lab lead 4): time differentially, (T(n2) - T(n1)) / (n2 - n1)
  — `ne::time_per_unit(record_fn, n1, n2, reps, backend)` (core/bench.h) does it for any recording. Also warm up
  first: the first measurement after process start read 1.37-1.55 us for the MTL4 tiny chain instead of 0.87-0.97
  (3 runs out of 5 before the warm-up was added; 0 of 3 after).
- MTL4 command allocators may only be reset once their command buffer has completed: the recorder keeps one per
  in-flight slot (hence the kMaxInFlight limit, checked in `commit()`).

## Known gaps

- No per-dispatch GPU timestamps (lead 4 found Apple's stage-boundary counters unreliable); time sub-blocks
  differentially with `time_per_unit`.
- ICB backend: no `dispatch_indirect`; MTL4 re-encodes the whole list every replay (~0.13 us CPU per dispatch).
- One threadgroup-memory slot per dispatch (`[[threadgroup(0)]]`).
- The full `align-pack` of the K5 pack (56 GB) has not been run; until then kernels need `Pack::aligned()` copies or
  misalignment-tolerant loads for the EXL3 shards.
- The n-gram table cannot be gathered on the GPU without wiring it (see Traps); CPU gather per token.

## Attribution

- Recorder encodings (MTL4 argument table + `barrierAfterEncoderStages`, ICB with `setBarrier`, concurrent encoder +
  `memoryBarrierWithScope`), the dispatch-cost probe kernels `ne_st_tiny` / `ne_st_read256` and the differential
  timing method come from lab lead 4, `the lab tree's engine/fused/disp.mm`, `disp.metal`,
  `NOTES.md` (15 Oct 2026). No Sushi / exllamav3 code is used in core/.
- The JSON parser, safetensors loader, pack aligner and the rest of core/ were written for this repo.
