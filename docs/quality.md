# Quality vs the bf16 teacher (lab-quality)

Owner: lab-quality (`tools/kld_*`, `tests/quality/`, this file). The real quality bar is the **bf16 teacher**
(Qwen3.8-Flash-Next as released, every module the HF bf16 tensor), not the MLX reference on the same pack:
docs/forward.md measures the engine against fn_exl3 (KLD 2.8e-3), this file measures both against the teacher.

```
make test-quality                       # engine (f32, bf16 KV, QSA) vs teacher, sets A,B,C (~81 GB, lock, ~50 min) + limits
make test-quality QUALITY_SETS=A,B      # ~10 min
make test-quality QUALITY_NAME=kv8 QUALITY_ARGS="--kv kv8"   # the kv8 arm A,B,C (~45 min; the night driver is not shipped)
make test-quality QUALITY_NAME=x QUALITY_ARGS="--round ref"     # an arm: --round ref, --kv kv8, --attn dense,
                                                                #   --windows "C:0,2", --max-pos 8192
make quality-report                     # every tests/quality/results/kld-*.json paired with fn_exl3 K5 (CPU)
$PY tools/kld_teacher.py --selftest --sets A --max-windows 1    # CPU-only scoring check (no engine)
```
(`make test-quality` needs the one-line hook in tests/tests.mk requested in docs/kernels.md; until then
`make -f Makefile -f tests/quality/quality.mk test-quality`.)

## 1. Method

- **Teacher**: the lab tree's `exl3/pack/teacher.py` (lab-exl3-pack, 14 Oct): the bf16 model run
  layer-sequentially over the sets, final HC mixer output `hm` stored per position (`RUN/teacher/ref-{A,B,C}-L48`),
  teacher logits = the bf16 `lm_head(hm)` (MLX `nn.Linear`, bit-identical to the teacher run: the scorer checks the
  rebuilt log-probs against the stored top-64, max |diff| must be 0). Exactness of the teacher itself: NOTES.md
  ("teacher check": A, B bit-identical to a resident forward, C KLD 5e-8). RUN = a scratch directory
  `exl3-pack/run` (read-only here; `NE_TEACHER_RUN` overrides).
- **Sets** (`RUN/teacher/windows.json`): A = 16 x 512 wikitext-2 test windows (8,176 scored positions); B = 16
  reasoning traces (Swift-1.5 Flash-Next traces, chat template, 1,045-2,048 tokens, 28,393 positions); C = 4 x 32,768
  (2 code, 2 wikitext; 131,068 positions; also reported by position bin 0-2K / 2K-8K / 8K-16K / 16K+).
- **Scoring** = `kld_core.score_model` (what whole_kld.py uses for fn_exl3), imported read-only: per window a fresh
  model; at every position except the window's last, KL(teacher || engine) over the full 248,320-token vocabulary
  from f32 log-softmax of both logit rows (engine logits are bf16 values), in the same 512-row blocks; top-1 =
  argmax agreement; NLL of the real next token for both. Per-set means are means over windows; paired differences
  and 95% CIs are a bootstrap over windows (windows = independent texts). Only the lm_head (1.3 GB) is in MLX, in the
  engine's process; the MLX reference model is never loaded next to the engine.
- **Engine path**: `ne_fn_model_step` with the head on every token (teacher forcing through the decode path: one
  1-row step per position, which is also how generation runs; the 4-row prefill has no per-position logits).
  fn_exl3's numbers come from its chunked prefill (chunk 512 / 2048 / 2048): same tokens and positions, a different
  code path (forward.md: fn_exl3's own prefill-vs-decode difference is KLD 2.4e-3).
- **Controls**: rerun (64 positions of the first window twice: logits must be equal bit for bit); teacher rebuild
  (above); `--selftest` (CPU, no engine: the teacher's own logits through the scorer: KLD 6e-5 / 1.5e-4 on one A / B
  window, i.e. only the CPU head's bf16 ulp flips; teacher NLL equal to whole_kld's to 2e-4).
- **Engine build**: a frozen copy of HEAD (built in a scratch directory, `NE_LIB`) so other labs' rebuilds can't
  change the library mid-run; commit given per table.

## 2. Engine vs the bf16 teacher (generation default: f32, bf16 KV, QSA), 15 Oct

Engine build 932ff47 (frozen copy), aligned pack `Qwen3.8-Flash-Next-EXL3-K5-a16` (= the K5 pack fn_exl3 was scored
on: same imatrix sha256, same expert shards). All three sets in full. "diff" = paired per window vs fn_exl3 K5
(`RUN/kld/kld-L48-ours-K5.json`), 95% bootstrap CI over windows (A, B 16 windows; C 4).

| | A wikitext (8,176 pos) | B reasoning (28,393) | C 32K (131,068) | top-1 A / B / C |
|---|---|---|---|---|
| **engine f32, bf16 KV** | **0.0361** | **0.00995** | **0.0163** | 92.8 / 97.4 / 96.8% |
| fn_exl3 K5 (MLX, same pack) | 0.0357 | 0.0105 | 0.0163 | 92.9 / 97.2 / 96.8% |
| engine - fn_exl3 [95% CI] | +0.0004 [-0.0013, +0.0019] | -0.0006 [-0.0010, -0.0002] | -0.0001 [-0.0003, +0.0003] | |
| for scale: fn_exl3 K5hc / K6hc / Sushi-4bpw / port | 0.0325 / 0.0284 / 0.0550 / 0.1416 | 0.0095 / 0.0099 / 0.0180 / 0.0528 | 0.0153 / 0.0124 / 0.0248 / 0.0841 | |

- **The engine is as close to the teacher as the MLX reference on the same pack**: A and C equal within noise, B
  slightly lower (-6%, CI excludes 0; plausibly its decode path vs fn_exl3's chunked prefill). The engine's own
  error vs fn_exl3 (forward.md: KLD 2.8e-3) is ~8% of the pack's distance to the teacher and does not add to it:
  whole-model KLDs are floor + error (NOTES.md: the whole-model floor of any bit-level nudge is ~0.019 on A), and the
  engine is one more nudge of that kind. Pack choice moves the number (K5hc -9%, K6hc -21% on A); the engine does not.
- NLL of the real next token: engine 1.8335 / 1.0611 / 0.7081 vs teacher 1.8267 / 1.0505 / 0.7067 (fn_exl3
  1.8394 / 1.0630 / 0.7086).
- C by position bin (0-2K / 2K-8K / 8K-16K / 16K+): engine 0.0232 / 0.0177 / 0.0148 / 0.0156, fn_exl3 0.0233 /
  0.0183 / 0.0153 / 0.0152. No growth with context: QSA selects 512 blocks of 4 tokens, so it is sparse past 2,048
  tokens, i.e. in all but C's first bin, and holds to 32K like fn_exl3's (the same selection in MLX). Per window the
  16K+ bin is within +-0.0013 of fn_exl3's (code0 0.0074 vs 0.0061 is the largest; 4 windows, so not significant).
- Per position (A / B / C): median 1.1e-2 / 1.2e-3 / 6.8e-4, p99 0.38 / 0.12 / 0.23, max 3.0 / 8.3 / 3.6; the mean
  is carried by a few high-entropy positions, as for every pack.
- Controls: rerun 0 (bit-identical logits), teacher rebuild 0 on all three sets. GPU 14.2-15.4 ms per step
  (A..C; ~16 ms per position with the logits copy and scoring); A 2 min, B 7.4 min, C 33.5 min.

## 3. Rounding emulation: f32 stays the generation default

Same build, same sets, bf16 KV; rounding ON = the reference bf16 rounding points (`NE_FN_ROUND`, the tests' default).
Paired per window vs the f32 run:

| | A | B | C |
|---|---|---|---|
| engine f32 (default) | 0.0361 | 0.00995 | 0.0163 |
| engine rounding ON | 0.0395 (+0.0035 [+0.0018, +0.0051]) | 0.0104 (+0.0005 [-0.0006, +0.0014]) | not run (deferred, then dropped: the decision does not need it) |
| fn_exl3 (MLX) | 0.0357 (-0.0004 [-0.0019, +0.0013]) | 0.0105 (+0.0006 [+0.0002, +0.0010]) | 0.0163 (+0.0001 [-0.0003, +0.0003]) |

top-1 rounding ON 92.4 / 97.0% (f32 92.8 / 97.4%); NLL 1.8340 / 1.0575 (f32 1.8335 / 1.0611). **Recommendation:
keep f32 for generation** (tools/generate.py, bench_forward.py already default to it): against the teacher, the
emulated bf16 points cost +10% on A (CI excludes 0) and nothing measurable on B; they never help. This is the same
direction as against fn_exl3 (forward.md: 3.8e-3 vs 2.8e-3). Keep rounding ON only for the per-layer goldens, where
it makes the comparison tight. Speed is not a factor (GPU per step in these runs: rounding ON 14.2, f32 14.6-15.0 ms;
forward.md's controlled timing has them equal within 0.1 ms, so the gap here is machine load between runs).

## 4. KV cache: bf16 vs kv8 (deferred to the night run)

Not measured yet: the GPU went to the development labs for the rest of 15 Oct. A night driver (not shipped) ran
the kv8 arm (f32, `NE_KV=kv8`, A, B, C in full, ~45 min, the same frozen build as section 2 so it pairs exactly),
then the paired report vs bf16 KV and markdown rows for this table (`tests/quality/results/night-kv8.txt`); C's
position bins (2K-8K / 8K-16K / 16K+) are the long-context answer.

| | A | B | C | C bins 0-2K / 2K-8K / 8K-16K / 16K+ |
|---|---|---|---|---|
| bf16 KV (default) | 0.0361 | 0.00995 | 0.0163 | 0.0232 / 0.0177 / 0.0148 / 0.0156 |
| kv8 | (night) | (night) | (night) | (night) |

Decision rule for the default: kv8 is acceptable if its paired difference vs bf16 KV has a CI upper bound below
+0.001 on B and C, with no bin trend at 16K+ (the size of a pack-level change is ~0.001-0.003, section 2).

## 5. `make test-quality` and limits

`tests/quality/quality.mk` (hooked from tests/tests.mk: requested in docs/kernels.md): `make test-quality` runs
`tools/kld_teacher.py` under `tools/with_lock.sh` (LOCK_OWNER=lab-quality, NEED_GB=85, LOCK_WAIT=30) and then `tests/quality/check.py`, which prints the report paired with fn_exl3 K5 and
applies the limits (full sets only; subsets are reported, not judged), plus rerun = 0 and teacher rebuild = 0:

| set | KLD mean <= | top-1 >= | measured 15 Oct |
|---|---|---|---|
| A | 0.040 | 0.920 | 0.0361 / 0.928 |
| B | 0.0115 | 0.968 | 0.00995 / 0.974 |
| C | 0.0180 | 0.962 | 0.0163 / 0.968 |

The engine's rerun is bit-exact, so the margin (+11-15%) only has to absorb deliberate numeric changes of the size
of one bit-level nudge (rounding ON would sit at A 0.0395: still a pass, by design; a broken kernel lands far
above). Limits are for the K5 pack; a K5hc / K6hc pack should get its own (fn_exl3: section 2's "for scale" row).

## 6. Traps and notes

- **Shared git index**: another lab's commit (254b17b, lab-prefill) swept my files staged in the shared index
  (`tools/kld_teacher.py`, `tests/quality/*`, the first version of this file) into its own commit; the content is
  right, only the commit message is wrong. Use `git commit -- <paths>` (only mode: commits those paths' working-tree
  content, leaves others' staged entries alone). A private index (`GIT_INDEX_FILE`) also works but leaves the shared
  index with the old blobs of your paths (staged reverts someone else's commit would sweep in): `git reset -q --
  <your paths>` right after.
- `with_lock.sh` with NEED_GB refused (exit 76) while another whole-model job ran instead of queueing on LOCK_WAIT,
  so other jobs took every gap (fixed in eb2ce59: queue first, check NEED_GB once the lock is held);
  the night driver still retried on 76 every 30 s for up to WAIT_MIN (240) minutes.
- Teacher logits must come from the same head call shape as kld_core's (the full 512-row block, then sliced): with
  n = 511 rows the GPU matmul can differ by a bf16 ulp; with 512 the rebuild is exactly 0.
- The CPU selftest shows how small the scoring noise is: the teacher's logits re-made on the CPU (ulp flips only)
  already give KLD 6e-5 - 1.5e-4, i.e. the measurement resolution is ~1e-4, far below the 1e-3 differences above.
- Nothing is ported from Sushi or exllamav3 here; `kld_core.py` (the lab tree, lab-exl3-pack) is imported
  read-only for the scoring and the bootstrap.
