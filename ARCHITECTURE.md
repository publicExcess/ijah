# Ijah (formerly "native-engine", working name) — architecture

A ground-up LLM decode engine for Apple Silicon (M5 Max first), Objective-C++ core on raw Metal. First model:
Qwen3.8-Flash-Next; designed so a second architecture (dense 27B, GLM-5.3-Flash) is one more model file.
Evidence for every choice below: the lab's lead notes (leads 1-6; not published).

## Decisions (15 Oct 2026)
- **Host**: Objective-C++ (ARC) + C++20 on raw Metal. No MLX in the engine (MLX costs ~10 us per dependent dispatch;
  raw Metal 0.9-1.5 us). MLX/Python stay as the *reference* and test tooling only.
- **Recording**: a forward is recorded once (Metal 4 encoder or indirect command buffer, classic encoder as fallback)
  and replayed; dynamic data (top-k ids, positions) lives in GPU buffers / indirect arguments, not in host branches.
- **Weights**: EXL3 trellis (MCG, window 15) for experts and trunk, 8-bit affine g64 as the fallback; packs mmapped
  and wrapped as no-copy MTLBuffers. First target: the existing packs (`<packs>/Qwen3.8-Flash-Next-EXL3-K5`:
  K5 experts + Sushi 8-bit trunk); new trunk formats (K6 + K8 HC-up/PLE + bf16 final mixer) come later.
- **Kernels**: 1 row = SIMD GEMV; >= 2 rows (MTP verify) and prefill = NAX matmul2d bodies (M5); fused MoE block
  in 3 dispatches (lead 4).

## Layout and ownership (lab agents: touch only your own directories)
| path | what | owner |
|---|---|---|
| `include/engine.h` | the C API (for Python tests and any future host language) | lab-core |
| `core/` | device, buffers, dtypes/tensor views, pipeline cache (kernels compiled into one metallib at build, function constants for shapes), recorder (Metal 4 / ICB / classic), safetensors + pack loader (mmap, no-copy), GPU timers | lab-core |
| `Makefile` | build: `make` -> `build/libengine.dylib`, `build/engine.metallib`, `build/engine-cli`; `make test` | lab-core (others add targets via `kernels/kernels.mk`, `models/*/model.mk`, `tests/tests.mk`) |
| `kernels/` | `.metal` kernels + thin `.mm` wrappers that record dispatches through the core API | lab-layer |
| `models/flash_next/` | the Flash-Next layer graph (one layer first, then 48 + head) | lab-layer |
| `tools/`, `tests/` | Python reference + golden generation (from fn_exl3 / mlx-vlm on the same pack), ctypes binding `tools/engine.py`, comparison tests | lab-ref |
| `docs/` | notes per area: `docs/core.md`, `docs/kernels.md`, `docs/reference.md` | each owner its own file |

## Core API sketch (C++ side, namespace `ne`; lab-core may refine, and must document changes in docs/core.md)
- `Device& Device::get()` — MTLDevice, queue, `has_nax`, `has_metal4`.
- `enum class DType { F32, F16, BF16, U8, U16, U32, I32 }`; `struct Tensor { id<MTLBuffer> buf; size_t offset;
  DType dtype; int ndim; int64_t shape[4]; }` (contiguous row-major; views by offset).
- `Tensor alloc(DType, shape)`; `upload/download` helpers (shared storage; unified memory).
- `class Pack { static Pack open(path); const Tensor* get(name) const; const Json& config() const; }` — every
  safetensors shard mmapped once, tensors are no-copy views (page-aligned buffer + offset).
- `Pipelines::get(name, FunctionConstants)` -> cached `id<MTLComputePipelineState>`.
- `class Recorder { begin(); dispatch(pso, args, grid, threadgroup, tg_mem_bytes); barrier(); commit(); wait();
  double gpu_ms(); }` with args = buffers (Tensor) and small constant blobs; recordings replayable.
- C API (`engine.h`): `ne_pack_open/close`, `ne_tensor_info`, `ne_alloc/free/upload/download`,
  `ne_layer_forward(model, layer, ...)` style entry points that tests call through ctypes.

## Correctness ladder
1. Kernel vs numpy/MLX on the same weights (rel ~1e-3).
2. Engine layer vs the reference layer (fn_exl3 / mlx-vlm on the same pack, same inputs): sub-block goldens.
3. Engine forward vs reference forward (logits KLD ~0), then vs the bf16 teacher (the quantization story).
