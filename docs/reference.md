# Reference, goldens and tests (lab-ref)

Owner: lab-ref (`tools/`, `tests/`, this file). Everything here is measured against ONE reference: **fn_exl3** =
mlx-vlm 0.7.7 `qwen4_exp` (pip, requirements-ref.txt) with the routed experts on our EXL3 Metal kernels,
loaded from the pack **`<packs>/Qwen3.8-Flash-Next-EXL3-K5`** (K5 MCG w15 experts + Sushi's 8-bit trunk),
native rope. Python for every tool: `.venv/bin/python -I -B`
(`tools/refenv.py` puts the EXL3 reference modules on `sys.path`; `$PY` below). fn_exl3, the EXL3 codec and the KLD
scorer are lab code that is not part of this repository (`IJAH_REF_DIR`); without them the reference tools SKIP.

## 1. Files

| file | what | needs |
|---|---|---|
| `tools/gen_goldens.py` | builds layers 0..3 + embed from the pack, runs the scenario through an instrumented copy of `Qwen4ExpDecoderLayer.__call__`, checks it bit-exact against the stock layer, writes the goldens | MLX, GPU, machine lock |
| `tools/with_lock.sh` | `with_lock.sh "JOB" HH:MM -- cmd...`: runs one heavy job under the RAM watchdog and the optional machine lock (`IJAH_LOCK_SCRIPT`) with the job's PID | |
| `tools/golden.py` | load a case: `golden.load("L00_decode1")` -> `Case` (`.m` manifest, `c["gdn.qkv"]` arrays, `.list()`) | numpy |
| `tools/opref.py` | numpy op references (section 5) + `PackNP` (numpy safetensors reader of the pack) | numpy |
| `tools/check_opref.py` | every op reference vs the goldens (= vs the MLX reference); prints the errors table | numpy, pack |
| `tools/compare.py` | engine output vs golden: rel L2, max abs / max ref, cosine, pass/fail per tensor kind | numpy |
| `tools/engine.py` | ctypes binding of `include/engine.h` (`build/libengine.dylib`) | core |
| `tests/test_fn_layer.py` | lab-layer's `ne_fn_layer_*` vs a golden: chain + per stage (section 6) | core, GPU |
| `tests/tests.mk` | `make test-ref` (op refs vs goldens + compare self-test + binding check, CPU, ~30 s), `make test-layer-ref`, `make goldens` | |
| `tests/golden/data/` | goldens (gitignored, ~190 MB); `tests/golden/READY` exists once they do | |

Regenerate goldens (~1 min, 7-8 GiB):
`tools/with_lock.sh "goldens" HH:MM -- $PY tools/gen_goldens.py --layers 0,1,3 --long 2304 --head` (or `make goldens`).

## 2. Goldens

Scenario (all cases B = 1, unpadded): a 52-token English prompt (`gen_goldens.PROMPT`, tokenizer of the pack, no
special tokens). The first **T0 = 48** tokens are prefilled (`L{LL}_prefill`, one chunk, positions 0..47); then, both
from the SAME post-prefill state:
- `L{LL}_decode1`: 1 row, token 48 at position 48 (cache length in = 48);
- `L{LL}_verify4`: 4 rows (MTP-verify shape), tokens 48..51 at positions 48..51.

Long scenario (QSA indexer actually selecting): 2304 tokens of mlx-vlm's `qwen4_exp/language.py` source prefilled
in chunks 2048 + 256 (as the model), then `L03_long_decode1` (position 2304) and `L03_long_verify4` (2304..2307).
At 2304 tokens there are 576 complete 4-token blocks > 512 = budget/ratio, so the selection is sparse.

Cases (layer 0 = GDN, layer 1 = GDN + PLE, layer 3 = full attention (QSA)):

| case | rows | tensors | MiB |
|---|---|---|---|
| L00_prefill / decode1 / verify4 | 48 / 1 / 4 | 46 (+1) | 32.5 / 6.8 / 20.4 |
| L01_prefill / decode1 / verify4 | 48 / 1 / 4 | 60 (+1) | 46.8 / 7.8 / 22.3 |
| L03_prefill / decode1 / verify4 | 48 / 1 / 4 | 48-52 | 25.5 / 1.0 / 2.6 |
| L03_long_decode1 / long_verify4 | 1 / 4 | 52 | 21.3 / 22.9 |
| HEAD_decode1 / HEAD_verify4 | 1 / 4 | 7 | 1.1 / 4.3 |

`HEAD_*` = final `hyper_connection_mixer` (HC read without inject) + 8-bit g128 `lm_head` applied to layer 3's
`hc_out` of the same step (a synthetic stand-in for layer 47's output: the head math is what is tested):
`head.hc_in`, `mixer.normed/down/mix/up/block_in`, `head.logits` [1,S,248320] (bf16 in the reference).

Per case directory `tests/golden/data/<case>/`: one `<name>.npy` per tensor + `manifest.json`:
```
{"case", "layer", "layer_type", "has_ple", "step": prefill|decode1|verify4, "rows", "position_start",
 "token_ids" (this step), "cache_len_in", "attn_path": gdn|dense|qsa, "mask", "pack", "bytes",
 "tensors": {name: {"file", "shape", "saved_dtype", "ref_dtype", "desc", "pack_weights": [pack keys]}}}
```
Floats are saved as float32; `ref_dtype` is the dtype in the reference (bf16 activations, f32 GDN state / decay g /
indexer scores). bf16 -> f32 is exact, so `x.astype(bfloat16)` recovers the reference bits. Ints: int32 / int64.
Tensor order in the manifest = execution order. Shapes keep the reference's batch axis ([1, S, ...]).

Tensor names (`[S]` = rows of this step, T = cache length before the step):

- **layer**: `hc_in` [1,S,10240] (4 streams x 2560, stream s = columns s*2560..), `hc_out` (= next layer's `hc_in`),
  `embed` [1,S,2560] (only in layer-0 cases: the 8-bit embedding before the x4 tile).
- **PLE (layer 1)**: `ple.token_history_in/out` [1,2] i64, `ple.ngram_ids` [1,S,16] i64, `ple.embeddings`
  [1,S,2560], `ple.keys` / `ple.queries` [1,S,4,2560], `ple.values` [1,S,2560], `ple.gate` [1,S,4,1],
  `ple.gated_values`, `ple.normed`, `ple.conv_state_in/out` [1,9,10240], `ple.conv_out`, `ple.out`, `hc_after_ple`.
- **HC read/write** (`attn_hc.*` before the mixer branch, `mlp_hc.*` before the MoE): `.normed` [1,S,10240],
  `.down` [1,S,320], `.mix` [1,S,320], `.up` [1,S,10240], `.block_in` [1,S,2560] (the branch input),
  `.inject_logits` / `.inject_w` [1,S,4]; `attn_hc.out` = streams after the attention write.
- **GDN**: `gdn.qkv` [1,S,10240], `gdn.z` [1,S,6144], `gdn.a`, `gdn.b` [1,S,48], `gdn.conv_state_in/out`
  [1,3,10240], `gdn.conv_out` [1,S,10240] (after SiLU), `gdn.q` / `gdn.k` [1,S,16,128] (normalised), `gdn.v`
  [1,S,48,128], `gdn.g` (f32) / `gdn.beta` [1,S,48], `gdn.ssm_state_in/out` [1,48,128,128] f32 (index [h, dv, dk]),
  `gdn.core_out` [1,S,48,128], `gdn.gated_norm` [1,S,48,128], `gdn.out` [1,S,2560]; verify4 only:
  `gdn.ssm_state_rows` [1,4,48,128,128] f32 = the state after each row (MTP rollback points; mlx-vlm's
  speculative with-states kernel, last row bit-identical to `gdn.ssm_state_out`). Conv state after row t =
  rows t+1..t+3 of [conv_state_in; qkv].
- **attention**: indexer `attn.idx_qk` [1,S,640], `attn.idx_q` [1,4,S,128] (normed + roped), `attn.idx_raw_keys`
  [1,S,128], `attn.index_keys_in/out` [1,T(+S),128], `attn.index_block_keys_in/out` [1,1,nb,128],
  `attn.idx_scores` [1,S,nb] f32 (no causal mask applied), `attn.selected_blocks` [1,S,512] i32 (-1 = none; absent
  when the indexer returns None, i.e. decode with <= 512 complete blocks); projections `attn.q_proj` [1,S,12288],
  `attn.k_proj`, `attn.v_proj` [1,S,512]; `attn.q_normed` [1,S,24,256], `attn.k_normed` [1,S,2,256] (before rope);
  `attn.q_rope` [1,24,S,256], `attn.k_rope` [1,2,S,256]; KV cache `attn.k_cache_in/out`, `attn.v_cache_in/out`
  [1,2,T(+S),256] (keys stored roped); `attn.core` [1,S,6144], `attn.gate` [1,S,6144], `attn.gated`, `attn.out`
  [1,S,2560].
- **MoE**: `moe.router_logits`, `moe.router_probs` [1,S,512], `moe.topk_ids` [1,S,10] i32 (a set: argpartition
  order), `moe.topk_weights` [1,S,10] (same order), `moe.routed_out` [1,S,2560], `moe.shared_gate_proj`,
  `moe.shared_up_proj`, `moe.shared_act` [1,S,640], `moe.shared_out` [1,S,2560], `moe.shared_scale` [1,S,1],
  `moe.out` [1,S,2560].

To resume a case: caches in = `*_in` tensors (GDN: `gdn.conv_state_in`, `gdn.ssm_state_in`; PLE:
`ple.conv_state_in`, `ple.token_history_in`; attention: `attn.k_cache_in`, `attn.v_cache_in`,
`attn.index_keys_in`, `attn.index_block_keys_in`; indexer positions are simply 0..T-1). The generator rebuilds the
reference caches from exactly these (through numpy) before running each step, so they are sufficient.

Loading:
```python
import sys; sys.path.insert(0, "tools"); import golden
c = golden.load("L00_decode1")      # c.m = manifest; c["gdn.qkv"] -> np.float32 [1,1,10240]; c.bf16("gdn.qkv")
for name in c.list(): print(name, c.shape(name), c.ref_dtype(name), c.weights(name))
```

## 3. Flash-Next decoder layer, as the reference computes it

Config (text): hidden D = 2560, hc_count = 4 (streams), hc_lowrank = 320, 48 layers (full attention at 3,7,..,47,
GDN elsewhere), PLE only on layer 1 (`ple_layer_ids = [2]`, one-indexed), rms eps = 1e-6 everywhere, 512 experts,
top-10, expert / shared intermediate 640, vocab 248320, EOS 248044. Activations are **bf16** between ops; the
listed f32 steps are f32 in the reference.

**Norm convention (important).** Sushi's pack stores the qwen4 norms ones-centred (1 + w folded): `hc_norm`,
`q_norm`, `k_norm`, indexer `q_layernorm` / `k_layernorm`, `ple.norm_{key,query,conv}` (fn_exl3 loads them as
pack - 1 and mlx-vlm adds the 1 back, exactly). So **every RMSNorm multiplies by the pack weight as stored**:
`rmsnorm(x, w) = x * rsqrt(mean(x^2) + 1e-6) * w` computed in f32, result cast to bf16. "Grouped" = applied
independently per 2560-column stream with w sliced accordingly. `linear_attn.norm.weight` (GDN gated norm) is not
in the fold list and is also used as stored. There is NO separate input_layernorm / post_attention_layernorm: the
HC read's `hc_norm` is the only pre-norm.

**Quantized linears.** Trunk = MLX affine 8-bit: `weight` U32 [out, in/4] (4 codes per word, little-endian:
code j of a word = `(w >> 8j) & 0xff`), `scales` / `biases` bf16 [out, in/gs]; `W[o, i] = scales[o, i/gs] * q +
biases[o, i/gs]`; `y = x @ W^T` (bf16 in/out, f32 accumulation). gs = in / scales.shape[-1], read it from the
shapes: **g128** for GDN `in_proj_a/b/z` and `out_proj`, `in_proj_qkv` on 7 layers (layer 0 is one; layer 1 is
g64), the shared experts and `lm_head`; **g64** for everything else (HC down/up, q/k/v/o, indexer, PLE key/value,
embedding). The 8-bit embedding is dequantized to bf16.
Dense bf16 (no `.scales`): `mlp.gate` (router), `mlp.shared_expert_gate`, `*_hyper_connection.block_inject_weight`.
Routed experts = EXL3 (sections 3.5, 5): `W[in,out] = diag(suh) H W' H diag(svh)`, `y = x @ W`.

### 3.1 Layer
```
h = hc_in                                    # [S, 4*2560]
if layer has PLE:  h = h + PLE(h, token ids)                       -> hc_after_ple
x, inj = HC_READ(attn_hyper_connection, h)                         -> attn_hc.*
b = GDN(x) or ATTN(x)
h = h + concat_s(inj[s] * b)                 # HC write            -> attn_hc.out
x, inj = HC_READ(mlp_hyper_connection, h)                          -> mlp_hc.*
h = h + concat_s(inj[s] * MOE(x))                                  -> hc_out
```
Layer 0's input: `hc_in = tile(embed(ids), 4)` (all 4 streams = the 8-bit embedding row). After layer 47:
`hyper_connection_mixer` = HC_READ without inject (block_in only), then lm_head.

### 3.2 Hyper-connection read / write (hc_count n = 4)
```
normed = grouped_rmsnorm(h, hc_norm.weight)              # per stream, eps 1e-6, f32 -> bf16      [S, 10240]
down   = input_mix_weight_down(normed)                   # 8-bit, 10240 -> 320                   [S, 320]
mix    = silu(down / 4)                                  # bf16
up     = input_mix_weight_up(mix)                        # 8-bit, 320 -> 10240                   [S, 10240]
block_in = mean_{s=0..3}( sigmoid(up_s) * normed_s )     # up_s, normed_s = stream s slices      [S, 2560]
inject_logits = normed @ block_inject_weight^T           # dense bf16 [4, 10240]                [S, 4]
inject_w = 2 * sigmoid(inject_logits / 4)
write:  h'_s = h_s + inject_w[s] * branch                # h = the un-normalised input streams
```
(the pointwise parts are one `mx.compile` each in the reference: `silu(d/4)`, `mean(sigmoid(up)*normed)`,
`2*sigmoid(p/4)`, all in bf16 with f32 internal reductions.)

### 3.3 GDN (gated delta net; layers with `linear_attention`)
Heads: Hk = 16 key heads x Dk = 128, Hv = 48 value heads x Dv = 128; v-head h uses k-head `h // 3`.
```
qkv = in_proj_qkv(x)  [S, 10240] = q [0:2048) | k [2048:4096) | v [4096:10240)
z   = in_proj_z(x)    [S, 6144] -> [S, 48, 128];   a = in_proj_a(x), b = in_proj_b(x)   [S, 48]
conv:  inp = concat(conv_state [3, 10240], qkv [S, 10240])     # oldest first
       out[t, c] = sum_{j=0..3} inp[t + j, c] * w[c, j]        # w = conv1d.weight[:, :, 0] (pack [10240, 4, 1])
       conv_state' = inp[-3:];  conv_out = silu(out)
       (decode S=1: f32 sum then bf16; S>1: MLX conv1d in bf16)
q, k, v = split(conv_out) -> [S,16,128], [S,16,128], [S,48,128]
q = q * rsqrt(sum(q^2) + 1e-6) * 128^-0.5 ;  k = k * rsqrt(sum(k^2) + 1e-6)      # L2 norm per head, bf16
g    = exp(-exp(A_log) * softplus(a + dt_bias))     # f32; a + dt_bias in bf16, softplus in bf16 as
                                                    # max(z,0) + log1p(exp(-|z|)) each step rounded (opref.softplus_bf)
beta = sigmoid(b)                                   # bf16
recurrence, per row t in order, per v-head h (state S_h [Dv, Dk] f32, kh = h // 3):
       S_h   = S_h * g[t, h]
       kv    = S_h @ k[t, kh]                       # [Dv]
       delta = (v[t, h] - kv) * beta[t, h]
       S_h   = S_h + outer(delta, k[t, kh])
       y[t, h] = S_h @ q[t, kh]                     # f32 -> bf16
gated_norm = (rms_norm(y_h, norm.weight, 1e-6) (bf16) as f32) * sigmoid(z_h as f32)  -> bf16   # per v-head of 128
out = out_proj(gated_norm.reshape(S, 6144))         # -> [S, 2560]
```
State layout: `gdn.ssm_state` [1, Hv, Dv, Dk] (row dv, column dk), f32.

### 3.4 Full attention (QSA; layers 3, 7, ...)
24 q heads, 2 kv heads (q head h uses kv head h // 12), head_dim 256, scale 1/16.
```
q_proj(x) [S, 12288] -> [S, 24, 512]: per head [0:256) = query, [256:512) = output gate
k = k_proj(x) [S, 2, 256], v = v_proj(x) [S, 2, 256]
q = rmsnorm_head(q, q_norm.weight);  k = rmsnorm_head(k, k_norm.weight)     # per head over 256
rope (partial 0.25): first 64 dims of each head rotate, the other 192 pass through. Half-split pairs
       (d, d+32), d < 32: inv_freq_d = 1e7^(-2d/64), angle = pos * inv_freq_d,
       x'_d = x_d cos - x_{d+32} sin,  x'_{d+32} = x_{d+32} cos + x_d sin   (f32 angles; bf16 out)
       NATIVE rope: the pack config's YaRN x4 is NOT applied (fn_exl3 convention).
k, v appended to the cache (keys stored roped);  attn = softmax(q k^T / 16 + mask) v   (dense causal, or QSA below)
out = o_proj( attn [S, 6144] * sigmoid(gate [S, 6144]) )
```
**QSA indexer** (runs every step, keeps its own cache; changes the attention only when sparse):
```
qk = index_qk_proj(x) [S, 640] -> [S, 5, 128]: heads 0-3 = queries, head 4 = raw key
iq = rope(q_layernorm(iq))                      # same partial rope (64 of 128 dims), same positions
index_keys += raw key (un-normalised, un-roped)
blocks of 4 tokens (ratio 4): block key b = rope( k_layernorm( mean_f32(raw keys 4b..4b+3) -> bf16 ), pos 4b )
       (complete blocks only; cached in index_block_keys, new blocks appended)
score[t, b] = sum_{h=0..3} relu( iq[h, t] . K_b ) / sqrt(128)       # f32
complete_t = (T + t + 1) // 4     (T = tokens before the step);  blocks b >= complete_t -> -inf
select top 512 (= budget 2048 / 4) blocks per row (ids >= complete_t -> -1)
if complete_t > 512:  row t attends to the tokens of its selected blocks + the tail [4*complete_t, T+t]
else:                 plain causal attention over [0, T+t]
```
Decode (S = 1) with <= 512 complete blocks short-circuits (indexer returns None after updating `index_keys`; no
block keys are updated in that call). Goldens: `L03_decode1` (dense), `L03_verify4` (indexer runs, all blocks
selected = causal), `L03_long_*` (576 blocks, sparse).

### 3.5 MoE
```
logits = x @ gate.weight^T                     # dense bf16 [512, 2560]                 -> moe.router_logits
p = softmax(logits, precise)                   # f32 inside, bf16 out                    -> moe.router_probs
ids = top-10 of p;  w = p[ids] / sum(p[ids])   # bf16                                     -> moe.topk_ids / topk_weights
routed = sum_k w_k * down_k( silu(gate_k(x)) * up_k(x) )      # EXL3 experts             -> moe.routed_out
shared = sigmoid(x @ shared_expert_gate^T) * down( silu(gate(x)) * up(x) )   # 8-bit shared expert
out = routed + shared
```
EXL3 decode kernel numerics (reference `exl3_mlx.Exl3Moe`, rows <= 16): gate/up GEMV f32 out, SwiGLU, down inner
f16, final H128 * svh + weighted top-k sum in f32, bf16 out. Prefill (> 16 rows): NAX `Exl3Prefill.fused`.

### 3.6 PLE (layer 1 only)
```
history = [last 2 token ids before the step (EOS at start)] + ids        # cache[3] keeps the last 2
shift_j(t) = token j positions back (EOS if that crosses the start / an EOS boundary)
for n = 2, 3:  mixed = (t0 * m0) XOR (t1 * m1) [XOR (t2 * m2)]            # int64 wrapping, m = layer_multipliers
               head i of the n-gram: id = mixed mod vocab_size_i + offset_i   (8 heads per n-gram -> 16 ids)
emb = concat_i ngram_table.bin[id_i]           # bf16 rows of 160 -> [S, 2560] (weight_scale 1)
keys    = grouped_rmsnorm(key_proj(emb), norm_key)        [S, 4, 2560]
values  = value_proj(emb)                                 [S, 2560]
queries = grouped_rmsnorm(h, norm_query)                  [S, 4, 2560]
gate_s  = (keys_s . queries_s) / sqrt(2560);  gate = sign(gate) * sqrt(max(|gate|, 1e-6))
gv      = concat_s sigmoid(gate_s) * values               [S, 10240]
normed  = grouped_rmsnorm(gv, norm_conv)
conv    = silu( depthwise conv K=4, dilation 3 over concat(conv_state [9, 10240], normed) )
          out[t, c] = sum_j inp[t + 3j, c] * w[c, j];  conv_state' = inp[-9:]
PLE(h)  = gv + conv                                        # added to h before the HC read
```
`tools/opref.py: ngram_ids()` reproduces the id math exactly (checked against `ple.ngram_ids`).

## 4. Tolerances (`tools/compare.py`)

Metrics per tensor: rel L2 = ||e - r|| / ||r||, max |e - r| / max |r|, cosine. Default thresholds by kind (first
match of the tensor name; override with `--tol name=rel`):

| kind | names | rel L2 | cosine >= |
|---|---|---|---|
| exact | `*_ids`, `selected_blocks`, `token_history*` | 0 (equality; top-k / selection compared as sets) | |
| state | `gdn.ssm_state*` | 1e-4 | 0.999999 |
| norm / pointwise | `*.normed`, `*.mix`, `*.inject_w`, `gdn.g`, `gdn.q/k/v`, `gdn.beta`, `gdn.conv_out`, `ple.gate`, `*_rope` | 5e-3 | 0.99999 |
| projection (8-bit / bf16 GEMV) | `*.down`, `*.up`, `*_proj`, `*.qkv`, `gdn.z/a/b`, `router_logits`, `inject_logits`, `idx_qk` | 5e-3 | 0.99999 |
| expert (EXL3) | `moe.routed_out` | 5e-3 | 0.99998 |
| block output | `*.out`, `*.block_in`, `attn.core`, `gdn.core_out`, `gdn.gated_norm`, `hc_*`, `moe.*` | 1e-2 | 0.9999 |

Top-k ids pass when the sets are equal or differ only among ids whose reference probability equals the k-th
largest (bf16 router probabilities tie: 1 row in L03_verify4, 3 in L03_long_verify4). `gdn.ssm_state` at 1e-4
assumes the golden q/k/v/g/beta as inputs; from an f32 g (see section 7) expect ~3e-3. `tests/test_fn_layer.py`
multiplies the tolerances by 2 in chain mode.

These are for a kernel fed the golden's own inputs (rung 1-2 of the ladder: no error accumulation). bf16 output
rounding alone gives rel L2 ~1.5e-3. Errors that compound across sub-blocks (whole layer from `hc_in`) are expected
near the block-output row; whole-layer `hc_out` from `hc_in` should still be < 1e-2 (layer 1 PLE and MoE routing
near-ties can flip one expert: then compare `moe.topk_ids` first).

## 5. Op references (`tools/opref.py`) and their cross-check vs the MLX reference

numpy, f32 math; `bf=True` (default) rounds to bf16 where the reference stores bf16, so a kernel's output can be
compared with the function or with the golden. Weights from the pack through `PackNP` (no MLX).

| function | math |
|---|---|
| `PackNP().linear(prefix)` / `affine_dequant(wq, scales, biases)` | 8-bit affine (any group size) -> dense f32 [out, in]; plain bf16 weights as is |
| `PackNP().linear_rows(prefix, rows)` | 8-bit embedding rows (bf16 out, as MLX) |
| `PackNP().expert(layer, proj, e)` | EXL3 trellis -> dense f32 [in, out] via `exl3_codec.decode_inner` + `reconstruct` (MCG, window 15, K from the shard) |
| `linear(x, W)` | x @ W^T, f32 accumulate, bf16 out |
| `rmsnorm(x, w, group=2560)` | per-group RMSNorm, w = pack weight as stored |
| `hc_read(h, P, prefix)` / `hc_write(h, branch, inject_w)` | section 3.2 |
| `causal_conv(state, x, w, dilation)` | depthwise causal conv (GDN K4 d1; PLE K4 d3), returns pre-activation + new state |
| `gdn_split`, `gdn_qk_norm`, `gdn_gates`, `gdn_recurrence`, `gated_rmsnorm` | section 3.3 |
| `rope(x, pos, 64, 1e7)`, `attention(q, K, V, q_pos, allowed)` | section 3.4 (GQA, causal or explicit mask) |
| `qsa_block_keys`, `qsa_scores`, `qsa_select` | QSA pooling, scores, top-512 + allowed-token mask |
| `moe_route`, `moe_routed`, `swiglu_expert`, `shared_expert` | section 3.5 (`moe_routed` decodes experts densely) |
| `ngram_ids`, `NgramTable(P).rows`, `ple_block` | section 3.6 |
| `bf16`, `silu_bf`, `sigmoid_bf`, `softplus_bf` | MLX's bf16 evaluation order (bit-exact on most goldens) |

Cross-check (`$PY tools/check_opref.py`, 8 cases: L00/L01/L03 decode1 + verify4, L03_long_*; ~30 s CPU). Each op
fed the golden's inputs; worst rel L2 over the cases (all within the section-4 tolerances; "0" = bit-exact):

| op | worst rel L2 | | op | worst rel L2 |
|---|---|---|---|---|
| 8-bit linears (hc down/up, qkv, z, a, b, out, q/k/v/o, indexer, shared) | 0 - 1.9e-4 | | router logits / softmax | 1.2e-4 / 3.6e-4 |
| embedding rows, x4 tile, hc_norm, inject linear, `2 sigmoid(p/4)` | 0 | | top-10 set (tie-aware) | 0 |
| `silu(d/4)` | 6.6e-4 | | renormalised top-10 weights | 4.4e-3 |
| HC mix_streams / whole HC read from h | 3.1e-3 | | **EXL3 routed experts (dense f32 decode)** | **1.2e-3** |
| GDN causal conv + silu, conv state | 3.9e-3, 0 | | shared swiglu / sigmoid(shared gate) | 4.1e-3 / 2.4e-3 |
| GDN q / k L2 norm, beta | 3.3e-3, 3.1e-3, 2.9e-3 | | hc writes, MoE combine | 0 |
| GDN decay g | 4.9e-8 | | PLE n-gram ids, table gather, history | 0 (exact) |
| **GDN recurrence y / state** | **1.5e-5 / 5e-8** | | PLE keys/values/queries | 4.5e-5 / 3e-7 / 0 |
| gated rmsnorm | 2.9e-3 | | PLE gate / gated values / conv / out | 2.3e-3 / 4.9e-3 / 4.0e-3 / 4.0e-3 |
| indexer q norm+rope, raw keys | 1.4e-6, 0 | | q_norm / k_norm, rope q / k | 2e-5 / 0, 2.2e-5 / 0 |
| QSA block keys / scores / top-512 set | 4.2e-5 / 6.7e-8 / 0 | | KV append, attention core (causal / QSA) | 0, 0 / 1.7e-3 |
| core * sigmoid(gate) | 3.4e-3 | | **whole layer chained from hc_in (op refs only)** | **1.2e-3 - 6.9e-3** |

The remaining 1e-3 - 5e-3 entries are bf16 output rounding of pointwise ops that MLX fuses differently (one ulp
on some elements), not math differences. The chained row runs the whole layer from `hc_in` with only op
references (own routing, no golden intermediates): 1.2e-3 (L03_decode1) to 6.9e-3 (L03_long_verify4); own top-10
equals the reference's (up to exact ties) in every case.

## 6. Engine binding and layer test

`tools/engine.py` binds all of `include/engine.h` (ABI 1): device query, buffers (`alloc`, `from_numpy`, `upload`,
`download`, zero-copy `numpy()` view), packs (`pack_open`, `names`, `info` -> `TensorDesc` with a read-only host
view and the no-copy `buf` + offset, `view_rows`), `kernel_exists`, `run_kernel(fn, args, grid, tg, fconsts,
tg_mem, reps, backend)` (args: Buf, (Buf, offset), TensorDesc or inline numpy scalars/arrays), and lab-layer's
`ne_fn_layer_*` (`eng.fn_layer(pack, layer)` -> `set_state/get_state/step/run_stage/read/write/time`); any other
`NE_API` symbol through `eng.fn(name, restype, argtypes)`. `$PY tools/engine.py` = self-check.

`tests/test_fn_layer.py [CASE]` (`make test-layer-ref`, machine lock): engine layer vs golden, (1) chain = one
`ne_fn_layer_step` from `hc_in` with the golden cache, every exposed intermediate + new state compared; (2) each
stage alone with golden inputs written first (and a check that the stage did not overwrite its inputs); (3) every
kernel in isolation through the engine's OWN inputs: after the step, each stage's input buffers are read back and
the stage is recomputed with the f32 op references (`bf=False`, same pack weights), including the EXL3 gate/up
GEMV per selected expert (`experts_gu` = the inner product ((x * suh) H) W' before H * svh). The
engine-name -> golden-name map is `MAP` / `STAGES` at the top of the file (follows `models/flash_next/layer.h`).

## 7. Findings for lab-layer (15 Oct, engine at 9f8f423 + later, L00_decode1)

- Chain: `hc_out` rel 7.2e-3 (cos 0.999988), routing identical, GDN state 2.9e-3. Every gap traces back to one
  rounding point: the reference's **inject logits are bf16** (`block_inject_weight` output; values ~ -20, ulp 0.125,
  so `2 sigmoid(p/4)` moves ~1%). The engine keeps them f32: `attn_inj` 9.27e-3 = exactly the f32 op reference's
  error; rounding the logits to bf16 before the gate takes it to 4.5e-3. That then explains `h_mid` (stream 2,
  inject 0.107, 1.2e-2) and the MLP-side 9e-3 errors.
- `gdn.g`: reference = `exp(-exp(A_log) * softplus_bf16(bf16(a + dt_bias)))`; an f32 softplus differs by ~1.5e-3.
- Kernels in isolation (section 6 mode 3) are f32-exact: every stage 2e-8 - 7e-7 rel vs the f32 op references,
  GDN state 4e-8, top-10 identical; EXL3 gate/up GEMV worst of 20 slots 2.2e-4 (f16 codebook values / f16 inner),
  routed + shared MoE output 5.6e-5. So all chain-mode gaps vs the goldens are the reference's bf16 rounding
  points, not kernel bugs. Whether to match them (round inject logits to bf16, bf16 softplus) is a decision: the
  reference's rounding is not "more right", but matching it is what makes rung 2-3 comparisons tight.
- `ne_fn_layer_run_stage(k)` is not isolated yet: after writing golden inputs, the stage's inputs
  (`attn_in`, `qkv`, `z`, `core`, `h_mid`) read back changed, and its outputs equal the chain run's (the
  recording appears to recompute earlier stages); so per-stage numbers currently equal the chain numbers.

## 7b. Whole-model reference logits (lab-forward, rung 3)

`tools/ref_logits.py` (`make ref-logits`, ~76 GB MLX, 1-2 min after load; never next to the engine) builds the
whole fn_exl3 model on the pack and writes `tests/golden/data/WHOLE/`: for 4 prompts (`chat` with the chat template
and thinking off, `code`, `prose`, `facts`) the prompt is fed one token per call (the decode path, as the engine
runs it), then greedy to 64 positions: `<name>.tokens.npy` [65], `<name>.decode.npy` uint16 [64, 248320] (bf16 bits),
and the same 64 tokens as ONE prefill call, `<name>.prefill.npy` (the reference's prefill-vs-decode spread = the
noise floor). `--long 2600`: 2600 tokens of mlx-vlm's qwen4_exp source, prefilled in chunks of 2048 (QSA-sparse,
650 blocks), then 32 greedy decode steps: `long.tokens.npy`, `long.decode.npy` [33, V] (the last prompt position +
32). `tools/compare_logits.py` (`make test-logits`) teacher-forces the engine on the same tokens: KL(ref || engine),
top-1, top-5 overlap, per position in `engine_<attn>[_f32].json`. Results: docs/forward.md section 2.

Test changes (lab-forward): `tests/test_fn_layer.py CASE...` runs every case in chain mode (rows 1/4, PLE, attention
with the golden KV via `ne_fn_layer_set_kv`), stage + mode-3 checks for 1-row GDN cases; mode 3 runs on a layer
created with the reference rounding OFF (f32 kernels vs f32 op references); `compare.check(..., cos_scale)` scales
the cosine slack (chain: scale^2); rows whose top-10 differs from the golden only by an exact tie are left out of
the MoE-output comparisons. `tests/test_fn_forward.py`: n-gram ids + gather vs `ple.ngram_ids` / `ple.embeddings`,
the PLE layer, HEAD_decode1 through the head kernels, the sampler. `tools/with_lock.sh`: `LOCK_OWNER`, `LOCK_WAIT`,
`MIN_FREE_PCT`, `MAX_SWAPOUT_MB` (kills the job if system swapouts grow by more).

## 8. Status / gaps
- Goldens: layers 0, 1, 3 (prefill 48, decode1, verify4) + L03 long (2304, sparse QSA) + head (mixer + lm_head
  on a synthetic input; lm_head op ref 5e-5, argmax equal) + whole-model logits (section 7b). Not yet: a later
  attention layer (7, 11, ..), MTP head, chunked prefill > 2048 of GDN layers as goldens, whole-model logits at 8K+.
- `gdn.g`/`beta`, conv, mix_streams op refs are within 1 bf16 ulp, not bit-exact (MLX fusion order).
- Rope: native (no YaRN); if the engine adopts the pack config's YaRN, the reference must change with it.
