# Image input (lab-vision)

Owner: lab-vision (`server/vision.py`, `models/flash_next/vision.{h,mm}`, `kernels/vision/`, `tests/vision/`, this
file; plus small hooks in `kernels/attn/qsa.*`, `kernels/fn_model.metal`, `models/flash_next/model.*`,
`prefill.mm`, `server/{chat,backend,ne_server}.py`). Status 16 Oct 2026: working end to end (server, whole model, T6 and K5); numbers in section 6.

## 1. Pipeline

```
OpenAI image_url part (data: URL | file path | file://) --server/chat.py--> template item {"type": "image"}
   -> '<|vision_start|><|image_pad|><|vision_end|>'  (the pack's chat template)
   -> tokens; each <|image_pad|> expanded to gh*gw/4 copies (server/vision.py expand_prompt)
image -> server/vision.py preprocess (CPU, = mlx-vlm Qwen3VLImageProcessor bit-exact):
   RGB uint8 -> smart resize (multiples of 32; min 65536 px, max NE_VISION_MAX_PIXELS = 1536^2) -> PIL bicubic ->
   /255, (x-0.5)/0.5 -> frame x2 along T -> patches [gh*gw][3*2*16*16] in merge-block order
   -> engine order (t, ph, pw, c) = the Conv3d weight layout of the pack
vision tower (engine, NAX; once per image content, LRU by digest in the backend):
   Conv3d as GEMM + bias + bilinear position table (48x48) -> 27 x [LN, qkv, 2-D rope, flash attention (per image),
   proj + residual, LN, fc1 + GELU(tanh), fc2 + residual] -> merger LN -> 4 patches -> fc1 + GELU(erf) -> fc2
   -> bf16 [gh*gw/4][2560]
language model: image rows' input embedding = the feature row (fn_embed: token id VOCAB + feature row), PLE / n-gram
   hashing keep the real <|image_pad|> id (as the reference: it hashes input_ids); rope positions from the MRoPE table
```

## 2. Positions (MRoPE, interleaved, mrope_section [11, 11, 10])

`get_rope_index` (mlx-vlm qwen3_5): text before an image counts on (t = h = w = i); the image block at offset s
(= next free position) gets (s + t, s + h, s + w) for its llm grid (1, gh/2, gw/2); the next text starts at max + 1.
After the prompt every position is row + delta (delta = max + 1 - n, negative: an image of k tokens advances the
positions by max(gh, gw)/2 only). The rotary frequency d (32 of 64 rotated dims) takes axis h if d % 3 == 1, w if
d % 3 == 2 and d < 30, else t. The QSA indexer uses the same rope: its query at the row's position and each
compressed block key at the position of the block's first token (`full_position_ids[..., 4b]`).

Engine: the 12 attention layers share one buffer `Model::mrope` = their `QsaCache::invf` (f32 [32] inverse
frequencies) + i32 header {mlen, delta} + i32 [mlen][4] table; `qsa.metal mrope_pos(row)` reads the table for rows
< mlen, else row + delta. A zero header is the old text rope bit for bit (same float(row) * invf). Set by
`ne_fn_model_set_mrope(m, pos3, n, delta)` before a prefill (host-synchronous); image rows by
`ne_fn_model_set_image_rows(m, rows, k, feats)` (absolute cache rows, bf16 features; `NE_VISION_ROWS` = capacity,
default 65536 rows = 335 MB, about 28 full-size images; 16384 until v0.2.1). Both survive `reset()`; the server sets them for every request.

## 3. Rounding points of the tower (vs MLX 0.32.3, found on all 65536 bf16 inputs / real activations)

| op | reference | engine |
|---|---|---|
| Linear | addmm: one rounding of acc + bias | vit_gemm epilogue bf16(acc + b) |
| Conv3d patch embed | conv (bf16) then `+ bias` (bf16) | bf16(bf16(bf16(acc) + b) + pos) |
| LayerNorm | f32 stats, bf16(w * bf16(xhat) + b) | same (vit_layernorm) |
| GELU tanh / erf | compiled graph, every op in bf16 | op by op with bf16 rounding (MLX's erf / expm1f ported) |
| vision rope | f32 (x cos + rotate_half(x) sin), one rounding | same, MLX's inv_freq bits |
| attention | fused SDPA, D 72 padded to 80 | MLX's NAX flash attention, D 80 (q, k) / 96 (v) zero pad |
| pos table | bf16 products, left-to-right bf16 sum | same (vit_pos_interp) |

GELU kernels: 0 mismatches over every finite bf16 input (denormals flush to zero on both).

## 4. Server

`chat.Template.render(req, images)` collects the image sources in placeholder order; system images are refused,
remote http(s) URLs are refused (loopback server; no fetching). `ne_server.make_job` preprocesses on the request
thread, the backend's `prepare()` runs the tower on the engine thread and sets the positions of the whole prompt.
Prefix reuse: the session's prompt is a key list; image tokens are `(1 << 62) | (sha256 digest[40 bits] << 20) |
index`, so a different image (even of the same size) never matches a cached prefix, the same image does. Image
prompts decode with MTP like text (lab-mtpimg, section 7; before: the step path at ~70 tok/s).
Slots (lab-srvcache): every slot is its own model instance, so its MRoPE table and image rows are its own; the worker
calls `prepare()` on the slot pick() chose; the tower and the feature cache (LRU, 65536 rows) are shared. Prefix
keys of parked conversations are int64 (image keys need 63 bits).

Open: video (grid t > 1);
remote URLs; the feature cache could evict an image another slot is still prefilling if > 65536 image rows are hot.

## 5. Tests

- `tests/vision/test_units.py` (CPU): template placeholders, expansion, positions, refusals, reuse with images.
- `tests/vision/vision_ref.py IMAGE OUT` (MLX tower only, ~1 GB): preprocessing bit-exact vs mlx-vlm, positions ==
  get_rope_index, reference embeddings (bf16 and f32 towers).
- `tests/vision/test_tower.py OUT` (GPU, ~2 GB): engine tower vs those embeddings.
- `tests/vision/lm_image_ref.py` + `lm_image_engine.py` (whole model, one after the other): first-token and
  teacher-forced decode logits on an image prompt vs fn_exl3 on the K5 pack.

## 6. Numbers (16 Oct, M5 Max; kld-chart.png 2800x1800 -> grid 76x118 = 8968 patches, 2242 tokens)

Tower (engine vs mlx-vlm VisionModel in bf16, as get_input_embeddings runs it; tests/vision/test_tower.py, tower_floor.py):
- GPU 465-605 ms for 8968 patches (MLX: 560 ms warm in bf16), 101 ms for 3344 (flow.png); weights 0.95 GB of padded
  bf16 copies, loaded in 0.1-0.8 s.
- Bit level: patch embedding identical in 99.997 % of elements; LN, fc1 + GELU, fc2, merger kernels bit-exact (100 /
  99.99 %) on the reference's own inputs; GELU tanh / erf exact on all 65536 bf16 values.
- Output rel L2 3.3e-2 (chart; per-token cosine mean 0.9992, min 0.974) / 2.9e-2 (flow) vs the reference. NOT a
  bug: the bf16 tower is chaotic - the reference itself, run from the engine's patch embedding (1 ulp in 0.003 % of
  elements), ends 3.5e-2 from itself. Per layer: 1.0e-3 after block 1, 3.8e-3 after 2, ~1.5e-2 by 16. The reference
  in f32 is 8.4e-2 from the reference in bf16. The ~1e-3 target is only reachable bit for bit.
- Gotcha found: an f32 pixel_values input makes mlx-vlm run the whole tower with f32 activations (8e-2 away from
  what it serves); the goldens cast to bf16 first as get_input_embeddings does.

Language model on the image prompt (K5-a16, fn_exl3 reference vs engine, 1 prefill + 16 teacher-forced decode
positions; tests/vision/lm_image_{ref,engine}.py):
- positions == get_rope_index (delta -2183).
- reference features: first-token KLD 1.0e-2, mean 5.0e-3, top-1 100 %; engine's own tower: 1.5e-2 / 4.8e-3 / 100 %;
  engine greedy == reference greedy 16 / 16. Control with 1-D positions: mean 3.3e-2, max 0.34, top-1 88 % (the
  reference's own 1-D-vs-MRoPE distance: 3.6e-2) - the positions are carried. Text-path floor for comparison: mean
  4.35e-3 (make test-logits arm).
- Text unchanged: compare_logits engine logits bit-identical to master's build (dbce604) on all 289 positions.
- Prefill of the 2264-token image prompt 1.1-1.3 s (+ tower ~0.5 s GPU / 1.0 s wall on first sight of the image).

Server end to end (T6, 2 slots, tests/vision/e2e_job.sh; thinking on, effort low):
- "List every bar with its exact value." -> all 9 labelled points with exact values (0.208, 0.136, 0.104, 0.059,
  0.199, 0.082, 0.137, 0.063, 0.207), noting the chart is a scatter / line plot, not bars.
- diagram -> Mermaid (tests/vision/assets/flow.png, data: URL) -> a correct `flowchart LR` with all 6 nodes, both
  ellipse shapes and the 5 labelled edges.
- follow-up turns reuse 2287 of 2292 prompt tokens (image included) also after another image conversation ran on the
  other slot; TTFT ~3.1 s cold for the chart (tower 1.0 s + prefill 2.1 s), 0.46 s on the follow-up; decode 70 tok/s.

## 7. MTP after image prompts (lab-mtpimg, 16 Oct)

What the MTP head needs on an image prompt, from the reference (mlx-vlm 0.7.7 `speculative/drafters/mtp_base.py`,
`qwen4_exp_mtp`, `speculative/mtp.py`): `prefill_from_target_hidden(input_ids, hidden, ...)` embeds the **token ids**
(`embed_tokens(<|image_pad|>)` for image rows, not the vision features: `prompt_tokens = input_ids`) and runs the head
at **1-D positions** `_position_ids` = its cache length from 0 (`set_shared_kv` only moves it while the head cache is
empty); 2-D ids put the same position on all three MRoPE axes. The engine already did exactly that: the loop's feeds
(`feed_body` -> `prefill_tokens` / `step4` / `step`) give the target its image rows through `embed_id`, the head gets
the real token ids, and its QSA cache has its own text rope table. The target's verify rows read `Model::mrope` (rows
past the prompt: row + delta). So nothing in the engine was missing; the server excluded image jobs from MTP.

- `server/ne_server.py`: image jobs may take MTP on MTP-continuation engines (`mtp_from` sets the image rows; the
  older library's fresh `mtp()` has none: step there).
- `server/backend.py` fix: `prepare()` of a text-only request now clears the previous image request's rows
  (`set_image_rows(k = 0)`). Before, a text request on a slot after an image request kept `vmap`, so its rows at the
  old image's row indices were embedded as image features (found by `tests/vision/mtp_image.py`: 2 garbage tokens).
- `NE_MTP_HEAD_POS=mrope` (`mtp_create`): the head's cache shares `Model::mrope` (head row j at the target's (t, h, w)
  of row j). Checked effective (head KV rows of image / later text rows differ from `text`, row 5 = (5, 5, 5) and the
  text before the image identical), but the greedy acceptance is **identical** to the reference's positions (same
  (depth, accepted) histogram over 154 / 120 rounds), so the default stays `text` (= the reference).

`tests/vision/mtp_image.py` (whole model, in-process RealBackend, greedy d2, K5-T6, one run each):

| prompt | step tok/s | MTP d2 tok/s | tokens / round | rounds by accepted 0 / 1 / 2 | first mismatch (step-path logit gap) | answer |
|---|---|---|---|---|---|---|
| chart (2292, 2242 image tokens) | 73.6 | **117.7** | 2.73 | 13 / 14 / 127 | token 75 (0.000, exact tie) | 9 / 9 values (both paths) |
| diagram -> Mermaid (896, 836 image) | 73.3 | **103.0** | 2.38 | 27 / 21 / 72 | token 20 (0.000) | |
| text: code_lru (60, effort medium) | 73.9 | 104.1 | 2.46 | 36 / 41 / 131 | token 140 (0.250) | |

Server end to end (`tests/vision/e2e_job.sh`, sampled T 1, 1 slot): chart 118.0 tok/s (before 71.3), follow-up from
the cache 107.2 (70.7), Mermaid 100.1 (72.3), chart again from the parked conversation 107.8 (69.9); all on the MTP
path; the chart answer lists all 9 values, the follow-ups and the Mermaid diagram are correct.

## Attribution

GEMM / attention bodies: MLX 0.32.3 NAX steel kernels vendored under kernels/prefill/mlx (MIT, Apple). erf /
expm1f: MLX (MIT, Apple; expm1f after Norbert Juffa, BSD-2-Clause). Preprocessing and positions: ports of mlx-vlm
0.7.7 (MIT). Sushi's native Qwen tower (src/qwen_vision.zig, MIT, (c) 2026 Theinruj Toranavikrai, David Dalcu) was
read for the pixel cap (ENGINE_MAX_PIXELS 1536^2) and its layout checks; no code copied.
