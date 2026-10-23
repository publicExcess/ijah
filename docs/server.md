# OpenAI-compatible server (lab-server)

Owner: lab-server (`server/`, `tools/serve.sh`, `tests/server/`, this file). Status 16 Oct 2026: built and tested
(CPU mock + three whole-model end-to-end jobs, section 5).
Python stdlib HTTP over `tools/engine.py` (ctypes) + a small C++ snapshot shim; the project venv (`make venv`,
requirements.txt) for `tokenizers` / `jinja2` / `xgrammar`. Numbers: M5 Max 128 GB, aligned K5 pack, machine lock held.

## 1. Start / stop

```
tools/serve.sh keygen                   # ~/.config/ijah/api-key (CSPRNG, chmod 600 in a 700 dir); never printed
tools/serve.sh start [HH:MM] [-- args]  # builds the shim, runs server/ne_server.py under tools/with_lock.sh
                                        # (NEED_GB=85, RAM / swap watchdog; the optional machine lock until HH:MM
                                        # when IJAH_LOCK_SCRIPT is set) inside the
                                        # supervisor (section 3d: restarted after a crash); waits for /health
tools/serve.sh status | log [N] | stop  # stop = supervisor told first, then SIGTERM to ~/.cache/ijah/server.pid
tools/serve.sh state                    # one line (status bars, scripts): running | starting | crashed ... | waiting ... | failed ...
NE_LIB=/abs/build-x/libengine.dylib tools/serve.sh start   # another build (the shim is built per library directory)
```
Library: `build/libengine.dylib` (`make`). `tools/serve.sh build` refuses the shim when a header the shim includes is
newer than the library (the server then reuses the live state only): rebuild with `make` first.
Server arguments (defaults): `--port 8140` (127.0.0.1 only; refuses other hosts), `--ctx 262144` (= KV size =
the context window = the reply cap), `--mtp 2` (draft depth; 0 = off; above 2 greedy requests still draft 2 deep, and
`NE_MTP_QGATE=1` gates the verify of sampled ones: docs/mtp.md 11), `--mtp-policy auto|always|never`,
`--effort xhigh` (reasoning effort when a request names none),
`--ckpts 4` (checkpoints per conversation), `--slots 1` (model instances per model id = its requests at once), `--slot-ctx`
(= `--ctx`: every slot holds the full window, no reply caps), `--cache-ram 8GB`, `--cache-disk 0` (e.g. `20GB`, `on` = 20GB:
SSD tier on; both budgets split evenly between the model
ids, each id's SSD entries under `--cache-dir/<id>/` and keyed by its model id), `--cache-dir ~/.cache/ijah/prefix-cache`
(kept across restarts, section 3b), `--model-id Qwen3.8-Flash-Next-native` (or `ID1,ID2`, or `--models all`: several
models in one process, section 1b), `--key-file`, `--lib`,
`--mock [--mock-delay S]` (`--shim` is accepted and ignored: the library's `ne_fn_state_io` replaced it).
Memory: 81.2 GB of weights + ~28 KB per context token (12 attention layers: bf16 K/V + indexer keys, one f32 key
sink): measured ~86 GB used at 262144, ~79 GB at 32768 (section 5); + 0.1 GB per checkpoint. Never next to another model server with a model
loaded (with_lock's NEED_GB check refuses).

### 1a. Optional: faster prefill with int8 activations (18 Oct)
Off by default. `--prefill-int8` (= `NE_T6_A8=128`) runs the T6 trunk Linears with in >= 1024 as int8 activations x
int8 weights on the M5 tensor units in prefill: about +3-4% cold prefill through the server (same client: 2,248 / 2,118
vs 2,179-2,256 / 2,047-2,051 tok/s at 10K / 32K). Quality, prefill path, INT8 minus default with a 95% CI: set A
-0.00099 [-0.00418, +0.00121] (n.s.), B +0.00005 [-0.00000, +0.00011] (leans ~2% worse), C -0.00041 [-0.00069,
-0.00020] (better); docs/prefill.md 4f. Kept off to protect quality. Turn it on with `tools/serve.sh start -- --prefill-int8`. The server logs
"prefill int8 trunk activations ON" when it is on.

## 1b. Which model (lab-swift, 16 Oct)

`server/models.json` maps model ids to packs; `--model-id ID` serves the first **complete** pack of ID's list (a T6
pack counts only with all 48 layers converted and the bf16 mixer; a partial T6 pack is a valid but mixed model),
`--pack DIR` overrides the registry (the id is then only the name `/v1/models` reports). No `--model-id`: the
registry's default, `Qwen3.8-Flash-Next-native`. Packs live in `IJAH_PACKS` (default `~/.cache/ijah/packs`), one
directory per Hugging Face repo (`hf download SiliconSpecies/<repo> --local-dir <IJAH_PACKS>/<repo>`).

| model id | pack |
|---|---|
| `Qwen3.8-Flash-Next-native` (default) | `<IJAH_PACKS>/Qwen3.8-Flash-Next-EXL3-Ijah` (the K5-T6 pack of these notes) |
| `Swift-1.5-Flash-Next-native` | `<IJAH_PACKS>/Swift-1.5-Flash-Next-EXL3-Ijah` (Swift's trunk in T6, docs/trunk6.md section 9) |

```
NE_MODEL=Swift-1.5-Flash-Next-native tools/serve.sh start      # = tools/serve.sh start -- --model-id Swift-1.5-Flash-Next-native
tools/serve.sh models                                          # each id -> the pack it resolves to now
E2E_MODEL=Swift-1.5-Flash-Next-native tests/server/e2e_job.sh  # (inside one with_lock job) the e2e test on that id
```
The server logs `model ID: pack DIR (registry (first choice | fallback ...) | --pack)` at start. Swift changed only
trunk Linears: its expert files are the base's (identical bytes; shared in memory only when they are the same files,
e.g. hard links, see below).

**Several models in one process** (lab-srvperf, 16 Oct): `--model-id Qwen3.8-Flash-Next-native,Swift-1.5-Flash-Next-native`
(or `--models all`; `NE_MODEL=all tools/serve.sh start`). Each id gets its own model
instance(s) (`--slots` per id: own trunk, KV, recurrent state, MTP head, prefix cache) on ONE engine and one engine
thread; the packs' common files are one mapping and one set of buffers (`core/pack.mm`: files are shared per process
by inode, so the 77.3 GB of experts / embeddings / lm_head / MTP head that both packs list are mapped and made resident
once; only Swift's 2.9 GB of T6 trunk tensors are its own), and every instance's chunked prefill uses one set of chunk
buffers (`ne_fn_model_share_prefill`, ~3.5 GB at a 262K window; the engine thread runs one prefill at a time;
`NE_SRV_SHARE_PF=0` turns it off). Requests are routed by `model` (exact id, else case-insensitive; none = the first id;
an unknown id = 400 listing the ids; with ONE model the field is ignored, as before); `/v1/models` lists every id; the
responses carry the model's id. The prefix cache is per model (a conversation never continues on the other model's
weights); `--cache-ram` / `--cache-disk` are split evenly (disk: `--cache-dir/<id>/`). Requests to different models run
interleaved like two slots (section 3c): an MTP reply yields when the other model's request arrives. A model switch
costs nothing measurable (both models stay resident: time to first token 175-183 ms after a switch vs 176-177 ms on the
same model, section 10). Memory at 262K: +12.4 GB for the second model; with `--share-state` (`NE_SHARE_STATE=1
tools/serve.sh start`) slot j of every model uses ONE KV + recurrent state (`ne_fn_model_create_shared`): +5.9 GB, but
requests to the two models then run one after another and a switch parks the other model's conversation (section 10).

## 2. API

- `POST /v1/chat/completions` (structured output: `response_format` and `tool_choice` required / a named function,
  section 9), `stream` true (SSE, `data: {chunk}` ... `data: [DONE]`) or false. Also `POST /v1/completions` (raw prompt)
  and Anthropic's `POST /v1/messages` + `/v1/messages/count_tokens` (section 11). Request fields:
  `messages` (system / developer / user / assistant / tool; text content or text parts; assistant
  `tool_calls` with JSON-string arguments; assistant `reasoning_content` (or `reasoning`) is rendered back, the
  template's `preserve_thinking` default), `tools` (OpenAI function tools), `tool_choice` (`none` drops the tools;
  `required` and `{"type": "function", "function": {"name": ...}}` constrain the output, section 9; `auto` and unset =
  the model chooses), `response_format` (`text`, `json_object`, `json_schema`, section 9), `temperature` (1.0; < 0.01 = greedy), `top_p` (0.95), `top_k` (20; 0 or > 64 -> 64, the
  engine sampler's limit), `min_p`, `seed`, `logit_bias`, `presence_penalty`, `frequency_penalty`,
  `repetition_penalty`, `logprobs` / `top_logprobs`, `n` (section 11), `max_tokens` / `max_completion_tokens` (default and cap: context window minus the
  prompt; a client value is honoured but never raised above that), `stop` (string or up to 16 strings),
  `reasoning_effort` (`none` = thinking off; `low`; `medium`; `high` / `xhigh` = xhigh), also `reasoning.effort`,
  `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, any other template variable), `stream_options.
  include_usage`. Extension `ne_path`: `step` | `mtp` forces the decode path (benchmarks).
- Response: `message.content`, `message.reasoning_content` (the text between the prompt's `<think>\n` and
  `</think>`, newlines around the tags dropped), `message.tool_calls` (`call_<24 hex>` ids, arguments a JSON string),
  `finish_reason` `stop` | `length` | `tool_calls`; `usage` with `prompt_tokens_details.cached_tokens` (the reused
  prefix) and `completion_tokens_details.reasoning_tokens`; extension `timings` (path, prefill / decode / queue ms,
  engine decode tok/s). Streaming deltas: `reasoning_content`, `content`, `tool_calls` (first the id + name with
  empty arguments, then argument pieces, one per finished parameter, that concatenate to the JSON object).
- Tool calls: the template's format `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n...
  </function>\n</tool_call>`; values are converted by the tool's JSON schema (`string` stays text, integer / number /
  boolean / object / array parsed as JSON, falling back to the text); a JSON body `{"name", "arguments"}` inside
  `<tool_call>` is accepted too; markup that does not parse goes back out as content.
- `GET /v1/models`: one entry per served model id (section 1b), `context_length` / `max_model_len` = the context window, `capabilities` (Sushi's names:
  `chat`, `tool_use`, `streaming`, `reasoning`, `json_schema` while the grammar kit is loaded, `vision` once
  `App.vision` is set by lab-vision's image input), `input_modalities` (`text`, + `image`), `reasoning_efforts`
  (`none`, `low`, `medium`, `high`, `xhigh`) and `default_reasoning_effort`. `GET /health` (no key):
  `{"status": "ok", "busy", "queued"}`.
- Auth: every `/v1` route needs `Authorization: Bearer <key>` (or Anthropic's `x-api-key: <key>`; constant-time compare); a refusal is 401 and the log
  shows the client address and sha256(presented key)[:8] only.
- Up to `--slots` generations at once (section 3c): one engine thread interleaves them token by token; further
  requests queue in arrival order (`/health`: `busy`, `active`, `queued`).

## 3. How a request runs

1. Template (`chat_template.jinja` of the pack, jinja2 sandbox, HF's trim / lstrip) -> tokens (pack tokenizer).
2. **Path choice** (`--mtp-policy auto`). With a library that has `ne_fn_mtp_generate_from` (lab-mtpcont, 16 Oct;
   docs/mtp.md 10.9) **MTP continues from the reused state**: every feed / step goes through `ne_fn_mtp_feed` /
   `ne_fn_mtp_step` (the MTP head kept in step), checkpoints carry the head (`ne_fn_mtp_state_io`: model state + the head's
   bookkeeping; parked conversations also copy the head's KV rows, section 3b), and an MTP turn = the same reuse + feed as the step path, then `ne_fn_mtp_generate_from` over the
   last prompt token; it leaves state = prompt + output[:-1] (live, checkpoints kept). Both paths then prefill the
   same n - k tokens, so the cost model reduces to the decode rates: MTP if `(n - k) / pf + E / dec_mtp < (n - k) / pf
   + E / dec` (always, at ~100 vs ~70 tok/s). Measured (whole model, ~8K-token prompts, turns 2-3 reusing 94-99%):
   100-102 tok/s on MTP vs 70.5 on the step path, prefill 0.68-0.70 s on both (keeping the head in step costs
   ~160 us per prefilled token, +30% over a plain prefill: docs/mtp.md 10.9). A stale head (state changed outside
   the MTP calls: only after an engine error) makes that turn take the step path with the reuse. With other
   requests running or waiting the step path is taken (an MTP reply holds the engine thread, section 3c). Not MTP: top_k > 32
   (the loop's shortlist); `logprobs` and a bare `seed` (section 11); stop strings only if the loop is abortable (`ne_fn_mtp_generate_cb`: the cancel ends the
   run after the round). Older library: the old rule -- MTP if `n / pf_mtp + E / dec_mtp + n / pf < (n - k) / pf +
   E / dec` (a full prefill now and a re-prefill next turn), rates EMAs of measured values (start: pf 1600, pf_mtp
   1000, dec 68, dec_mtp 95).
3. **Prefix reuse** (both paths; section 3b): the longest reusable prefix among a slot's live state (its KV rows'
   tokens are a prefix of the prompt) and every conversation's checkpoints, resident in a slot, parked in RAM or on
   disk. Checkpoints are taken at the end of the system prompt (>= 256 tokens; shared by conversations) and at the
   start of each generation prompt (the last `<|im_start|>`: the next turn re-renders `<think>\n` + reasoning and the
   `\n` merges with what follows, so the end of the prompt itself is rarely a prefix); since lab-snap also
   **shared-prefix snapshots** across sessions (section 3f), which replace the system prompt's checkpoint when one lies
   below it. GDN states cannot be truncated, so nothing between checkpoints is reusable. Each segment of the suffix (up to a checkpoint, then the
   rest) is prefilled in full chunks of 4097 tokens (`ne_fn_model_prefill`; with the MTP head in step `ne_fn_mtp_feed`,
   4096) and **the rest as one piece** (< 16 tokens = the engine's step path); since lab-ttft (section 3e), before it
   binary pieces 2049 ... 17 (`NE_SRV_PIECES=binary`).
4. Decode: one `ne_fn_model_step` per token in the engine thread (interleaved with the other slots' steps); the HTTP thread detokenises (tokenizers
   `DecodeStream`), applies stop strings, splits reasoning / content / tool calls and writes SSE, in parallel with
   the next GPU step. A stop string or a client disconnect sets the cancel flag (the engine thread stops after the
   current step). The MTP path streams through the loop's per-round callback (older libraries: polling `out`), and a
   cancel ends it after the current round.

### 3b. Multi-conversation prefix cache (lab-srvcache, `server/prefix_cache.py`, `models/flash_next/kvio.mm`)

- A **conversation** = the tokens its KV rows hold + up to `--ckpts` checkpoints (the whole recurrent state at a
  position: GDN conv + ssm of the 36 GDN layers, PLE conv, T, n-gram history; 113 MB) + host copies of its KV rows.
  Each slot has one **resident** conversation (its KV buffer holds the rows); the others are **parked**: KV rows
  [0, newest checkpoint) copied to host memory, the tokens past that point forgotten.
- KV is O(T) (27.7 KB per token: 12 QSA layers x (bf16 K + V of 2 kv heads, bf16 indexer keys) + block keys),
  the recurrent state O(1). Parking **copies the KV rows** rather than re-prefilling them: 28,672 tokens = 0.76 GB
  copied out in 40 ms (19 GB/s incl. first-touch page faults of the new buffer) and back in 16 ms (48 GB/s), against
  18.0 s to re-prefill them (section 5b): ~1000x cheaper, so re-prefill is never the better choice. A conversation parked before is extended incrementally: only rows past its
  last copy are copied again (segments; shared, read-only, between conversations that fork from one another).
- **Lookup** (per request): the longest common prefix of the prompt with each conversation's tokens; the largest
  checkpoint position <= it (parked: <= the copied rows) is the candidate; a slot's live state counts when all of its
  tokens are a prefix. Slot choice: the longest candidate, then a slot already holding it, then the slot whose
  resident conversation is worth least (none, else least recently used).
- **Restore**: same slot: load the checkpoint (`ne_fn_state_io`), truncate (rows past p are overwritten later); if
  the resident conversation has checkpoints past p it is parked first and a fork continues (an edited earlier
  message keeps the original conversation). Another conversation: park the resident one, copy rows [0, p) in
  (`ne_fn_kv_io`; from host segments, or straight from another slot's KV buffer when it is resident there, also
  while that slot is generating: rows below its position are stable), load the checkpoint.
- **Budget** `--cache-ram` (host copies + checkpoints, buffers shared by forks counted once): least recently used
  parked conversations go first (to the SSD tier with `--cache-disk`, else dropped); if only resident ones are left,
  their host copies, then their older checkpoints (the system prompt's and the newest kept). `--ckpts` per
  conversation: the first (system prompt) and the newest.
- **SSD tier** (`--cache-disk 20GB`; lab-persist, 16 Oct): a durable mirror of the conversations that **survives
  restarts and crashes** (Sushi's `--prefix-cache-disk` idea). Write-through: at the end of every request the slot's
  conversation is written up to its newest checkpoint m: only the KV rows past what its entry already holds (one
  `ne_fn_kv_io` copy of the new rows + the MTP head's, ~1 ms on the engine thread) and the new checkpoints; a
  background thread writes the files and frees the copies. A RAM eviction completes the mirror (usually nothing left
  to write) and keeps nothing in RAM; a hit reads the files back (`take`: the conversation is parked in RAM again,
  still mirrored, the entry stays on disk). A conversation truncated below its mirror (an edit, a regenerate) keeps
  only the mirror's rows below the cut (`Conv.dvalid`): rows past it are another prefill history (other piece
  boundaries, other bytes) and are rewritten, so a restore is always one history, bit-exact.
  - Layout: `<dir>/<fingerprint[:16]>/e<ns>-<pid>-<conv>/` = `toks<k>.bin` (int64 tokens [0, m)), `snap<k>.bin` per
    checkpoint, `seg<k>.bin` per KV segment (`ne_fn_kv_io` layout, head rows behind), `manifest.json` (magic, format
    1, fingerprint, n, per file: name, bytes, crc32; checkpoint meta; segment rows / head rows / capacities).
    Crash-safe: every file is written as `<name>.tmp` + rename, the manifest last (the commit point), replaced files
    are deleted after it; nothing is fsynced (a process crash keeps the last manifest's record whole; a power loss
    could leave a manifest naming bytes the disk never kept: the checksums then reject it).
  - **Fingerprint** = sha256 of: the format, the engine library's and metallib's contents (sha256: any rebuild),
    the `NE_*` numerics env (not lock / port / profiling knobs), the pack's files (relative name, size, mtime_ns,
    through symlinks), the model id, the state / KV layout (`cache_layout()`: snapshot bytes, KV bytes per 4096 rows,
    head KV, head_io). `ident.json` in the fingerprint directory says what it was. Other fingerprints' directories
    (another build, the Swift pack next to the base) are kept and ignored; they are the first to go when the
    budget is exceeded.
  - **Start**: the fingerprint directory is indexed, not wiped: each entry's manifest is parsed, every file's size
    checked, the tokens file read and its crc32 checked (the lookup needs the tokens); entries that fail are deleted,
    stray files (tmp, never named by a manifest) removed; the old tier's `c<N>` directories and dead `private-<pid>`
    ones go. One log line: `prefix cache: SSD tier DIR (20 GB budget): N conversations, X GB, indexed in T ms`.
  - **Hit**: every checkpoint / segment file read is checked (size + crc32, 31 GB/s on the M5): a mismatch logs
    `SSD entry ... rejected (seg1.bin: checksum mismatch); prefilling cold`, deletes the entry and the request
    prefills cold (`timings.cache` = `none`). `timings.cache`: `live` | `ram` | `ssd` | `none`; `timings.restore_ms`,
    `timings.persist_ms` (the write-through's engine-thread time).
  - **Budget**: other fingerprints first, then entries least recently used (in-process use; across restarts the
    manifest's mtime, touched on every hit). A conversation larger than the whole budget is not kept (one log line).
    No writes while the volume has < 8 GB free. RAM: a long cold first turn copies all its rows at once (~28 KB per
    token, transient until written: 1 GB per 38K tokens).
  - **One live tier per directory**: an flock on `<dir>/.lock`; a second server on the same directory (a test server
    on 8141 next to the main one) gets `<dir>/private-<pid>`, not kept, deleted at its exit.
  - Shutdown (SIGTERM): the queued writes land first (up to 30 s). A crash loses only the writes still queued (the
    last turn's, typically < 1 s of SSD time); the entry keeps its previous record.
- **The MTP head travels with the conversation** (library with `ne_fn_mtp_state_io`): a checkpoint = the model's
  recurrent state + the head's bookkeeping (in step or not, its KV length Lh, the <= 9 pending hidden rows: 113 MB +
  0.66 MB); a segment also copies the head's KV rows (one attention layer, ~2.3 KB per token: head row j pairs
  position j with token j + 1, so a prefix p holds rows < p - 1; parking copies up to the largest Lh of the
  conversation's checkpoints, rows below a checkpoint's Lh are never rewritten later). A restored conversation
  continues with MTP (`ne_fn_mtp_generate_from`); if the head rows a checkpoint needs are missing, the model alone is
  restored (head stale: that turn takes the step path). Older libraries: MTP runs from position 0, park the resident
  conversation first and keep no checkpoint.
- Exactness: a park + restore is a byte copy of the KV rows and the state, so decoding after a restore is
  bit-identical to decoding from the never-parked state (section 5b). Reuse vs a fresh prefill of the whole prompt
  differs only by the prefill piece boundaries (as before).

### 3c. Concurrency (`--slots`)

- Each slot is its own model instance (`ne_fn_model_create` on the same open pack: weights shared, own KV buffer, GDN
  state, scratch, sampler, MTP head; the chunked-prefill buffers are shared by all instances of the process since
  lab-srvperf, section 10). Memory per extra slot: below the noise of the free-RAM reading at
  `--slot-ctx` 32768 (server with 2 slots + MTP: 79 GB used, as the 1-slot server) + 27.7 KB per context token
  (KV, allocated on first use): ~3.6 GB at the default `--slot-ctx` 131072.
- The engine thread runs every active request's generator in turn: one prefill piece or one decode token each (the
  server worker, `server/ne_server.py`); time-slicing, not batching, so the aggregate equals one stream's rate
  (section 5b) and one request alone runs exactly as before. A request whose prompt needs more context than a free
  slot has waits for slot 0 (the full `--ctx`). On a smaller slot the reply cap is that slot's window (`finish_reason`
  `length` when it is reached), as `--ctx` is for slot 0.
- Prefill pieces stay canonical while another request decodes (its decode then stalls up to one 4097-token piece,
  ~2.7 s); `--pf-cap 1025` shortens the stall, but other piece boundaries round differently and the output then
  depends on what else ran (measured: greedy diverged at token 45 with the cap, identical without).
- Structured output (lab-api's token mask, `ne_fn_model_set_mask`) is per model instance, so per slot: a constrained
  request sets its slot's mask before each of its samples and clears it at its end (also on an error); the other
  slot never sees it (mock check: 103 masked samples on one slot, 0 on the other, its output = the request alone).
- MTP (also on reused turns) whenever no OTHER request needs the engine thread (lab-sched): idle conversations
  resident on other slots do not count, nor does a queued request when no other slot is free (so with one slot MTP
  is never preempted). A request arriving during an MTP reply (or one running on another slot) makes the MTP run stop
  at its next round boundary (`preempt` in the per-round callback of `ne_fn_mtp_generate_from`); the reply continues
  on the step path, interleaved token by token, and resumes MTP (`mtp_from` over the pending token) as soon as the
  engine is free again. The state is fed + output[:-1] at every switch, so the output is the request's alone
  (mock: exact; `timings.path` = `mtp`, `step` or `mtp+step`, `timings.slot`). Older libraries (MTP from an empty
  state, not resumable): MTP only when nothing else runs or waits, as before.
- Slot choice (`Session.pick`): the longest reusable prefix; then the slot holding this conversation's own earlier
  state (a prefix beyond the shared system prompt, or the live state); else the slot with the smallest resident
  conversation (empty first, then least recently used, then the lowest slot). A shared system checkpoint resident next
  to another conversation is copied into a free slot instead of displacing that conversation (before: a subagent's
  request displaced the main conversation, whose next turn then moved to slot 1).
- Batching two conversations as rows of one forward would give ~1.5x aggregate (a 2-row step costs 1.28-1.32x a
  1-row step, docs/mtp.md 10.5; independent tokens pick experts as differently as consecutive ones), but the
  stateful kernels assume one sequence per step: `fn_gdn_step(_lz)` recurses the R rows through ONE state per layer,
  the GDN / PLE conv states are single, the QSA step appends R rows at one shared `pos`. Multi-sequence needs per-row
  state / cache / position arguments in those kernels (the GEMMs and the R-row MoE already treat rows
  independently): a kernel lab's job, not the server's.

### 3e. Time to first token of small reused turns (lab-ttft, 17 Oct)

An agent turn adds ~50-500 tokens to a 30-160K-token conversation (the previous reply's end, a tool result, the
generation prompt) on the live state. Measured with `tools/bench_turn.py` (the server's Session + RealBackend
in-process, every engine call timed; T6 pack, greedy, ctx 262144) and `tests/server/ttft_http.py` (a test server,
request sent -> first streamed delta), what a turn cost on master f35658e and why:

| cost | before | after |
|---|---|---|
| prefill pieces per turn: the binary split (150 new tokens = 128 + 16 + 1 + checkpoint + 4 + last; 500 = 256 + 128 + 64 + 32 + 15 + ...) | 2-5 chunk passes, each a whole pass over the weights: ~105 ms floor at 16-32 rows (trunk GEMMs ~58 ms, experts ~30-85 ms, glue ~15 ms), +0.45 ms per row; a < 16 remainder as 4-row + 1-row decode steps (13-23 ms each) | one chunk per segment (+ the 4-token generation prompt after the checkpoint as one 4-row step, the last token in the MTP start); a chunk length is recorded once (2.5 ms, LRU of 32) |
| T6 trunk GEMM at <= 32 rows: lab-prefill's NAX GEMM runs (NP / 64) x 1 threadgroups over the whole K (HC down 10240 -> 320: 5; GDN a / b: 1), 73% of the trunk's time at 16 rows | ~58 ms per chunk | split-K over the dense W' (`t6_pf_sk*` + fixed-order `t6_pf_skr`, docs/prefill.md 4c): -20 ms at 20 rows, -9 ms at 32 rows; from 33 rows the NAX GEMM (simdgroup MMA is compute-bound there) |
| prompt tokenisation on the HTTP thread: the whole rendered conversation every request | 93 ms at 160K tokens, 55 at 100K, 18 at 32K | `chat.Template.encode` reuses an earlier text's ids up to the last `<|im_start|>` inside the common prefix (the tokenizer splits on added special tokens before the BPE: exact by construction; `tests/server/test_tokcache.py`, 42 agent-like requests bit-identical): ~3 ms. `NE_TOK_CACHE=0` = the whole text |
| checkpoint export (113 MB host buffer per turn) | 2-35 ms (page faults of a fresh buffer) | buffers of dropped checkpoints are reused (refcount-checked pool of 3) |
| image-key scan of the whole prompt in `prepare` (engine thread) | ~5 ms at 160K | skipped for text prompts (`max(prompt)`) |
| the MTP loop's first token | after the head's final catch-up | streamed before it (the drafts need it, the token does not) |

**TTFT, request sent -> first streamed delta** (`tests/server/ttft_http.py` on a test server, ctx 262144, one conversation
grown to each level, 2 reps each, median; both arms in one with_lock job, 17 Oct ~07:15; new = prompt - cached tokens:
42 / 142 / 492):

| context | 50 new: master | lab/ttft | 150 new: master | lab/ttft | 500 new: master | lab/ttft |
|---|---|---|---|---|---|---|
| 32K | 239 ms | **213** | 332 | **298** | 929 | **514** |
| 100K | 304 | **208** | 409 | **294** | 1099 | **579** |
| 160K | 354 | **206** | 448 | **308** | 1150 | **581** |

In-engine (`tools/bench_turn.py`, same job, Session.generate -> first token): 210-233 / 280-318 / 522-607 ms; master's
first run that morning (06:29) 259-279 / 377-397 / 909-967. The HTTP - engine gap was 22 / 67 / 100 ms at 32K / 100K /
160K before the incremental encode, 0-8 ms after. One indicative A/B per change (06:35-07:15): single piece (in-engine
50 / 150 / 500: 259-292 -> 189-214, 378-432 -> 257-288, 908-1056 -> 469-512 ms), split-K (20-row chunk 125-131 -> 104-113
ms), incremental encode (HTTP gap above).

Not reducible here: the experts (a 64+-row chunk of real text touches nearly every expert: ~70-150 ms of expert GEMMs,
bandwidth), the T6 trunk's dense W' decode + read (~2.5 ms per 4 layers at any row count), attention over the cache
(5-6 ms at 16 rows; 38 -> 69 ms at 512 rows from 32K to 160K), the generation prompt's 4-row step (23 ms) + the last
token's step with the head (16 ms). Leads: one 5-row verify pass for the generation prompt + last token (~-15 ms), a
NAX split-K GEMM for 33-128 rows (needs the MLX tile headers), the n-gram gather (5-16 ms cold) overlapped.

Greedy outputs: pieces of >= 33 rows through the chunk path give bit-identical states whatever the split (the
chunked prefill is row-independent apart from the GDN scan's fixed order); the old split's < 16-token remainders ran
the decode kernels and its 16-32-row pieces now split-K, so a reused turn's logits differ from before at the decode
family's own floor: whole model, 8K context, 12 suffixes of 25-1000 tokens, KLD(old || new) 1.2e-5 .. 3.9e-2, top-1
equal in 11/12 (the 12th: a near-tie that another valid split also resolves the new way, KLD(old || that) 9.2e-2);
split-K vs NAX on 16-32-row pieces: KLD <= 2.0e-2, top-1 equal in all 24 (docs/forward.md 2c: any two valid
arithmetics differ by 2e-2 .. 9e-2 on teacher-forced repository text).

### 3d. Crash recovery (lab-persist, `tools/serve.sh`)

`tools/serve.sh start` runs the server inside a small supervisor
(`serve.sh _supervise`, a detached zsh loop; files `~/.cache/ijah/supervisor-8140.{pid,state,stop}`):

- The server runs under `tools/with_lock.sh` as before (optional machine lock, `NEED_GB=85`, RAM / swap watchdog). When it exits
  without a stop request (crash, jetsam / OOM SIGKILL, the watchdog's SIGTERM, an unhandled error) the supervisor logs
  `supervisor: server exited (rc R after S s); restart n/5 in D s` and starts it again after a backoff of 10, 20, 40,
  80, 160 s; a run of >= 10 min resets the count; after 5 restarts in a row it gives up (`failed`). Never a tight loop.
- Each restart goes through with_lock again: if the lock is held (another job took it after the crash) or
  NEED_GB is not free (another model server loaded a model), it waits (`waiting ...`, 60 s steps) up to 30 min, then gives up.
  The first start does not wait: lock held / no RAM = `failed: not started: ...` and `start` reports it, as before.
- The restarted server indexes the SSD tier (section 3b): the user sees one slower turn (the restore from SSD, not a
  cold prefill) after the restart's load time. A turn in flight during the crash is lost (the client sees the
  connection drop; agent clients retry).
- `tools/serve.sh stop` touches the stop file first, then SIGTERMs the server: no restart. `kill` on the server pid
  alone counts as a crash (it is restarted): use `stop`. Killing the supervisor leaves the server running unsupervised.
- Nothing starts at login (a launchd agent running `tools/serve.sh start` once, without KeepAlive, would: the
  supervisor handles restarts).
- `tools/serve.sh state`: `running`, `starting (attempt n)`, `crashed (rc 137 after 812 s;
  restart 1/5 in 10 s)`, `waiting (machine lock held (owner); retry in 60 s)`, `failed (...)`, `died (...)` (no
  server and no supervisor: e.g. both killed), `stopped`; one line for a status-bar plugin.
- Test: `tests/server/test_supervisor.sh` (CPU mock, a with_lock stand-in, port 8147, scratch `NE_HOME`; part of
  `run_mock.sh`): kill -9 -> crashed -> restarted, the next turn restored from the SSD tier; a refused restart -> waiting
  -> up when free; a server that keeps dying -> failed after 5 restarts (31 s with a 1 s base backoff); stop -> nothing
  left running. Test knobs: `NE_HOME`, `NE_SERVE_WRAP`, `NE_SKIP_BUILD`, `NE_SUP_BACKOFF_S`, `NE_SUP_WAIT_S`,
  `NE_NO_SUPERVISE=1` (the old unsupervised start).

### 3f. Shared-prefix snapshots (lab-snap, 18 Oct; `server/backend.py` Session, `server/prefix_cache.py`)

**Why new agent sessions prefilled cold.** 20 sessions of a coding-agent client on 17 Oct: every first turn (~7.9K tokens) logged
`cached 0` (median TTFT 4.8 s), the second turn reused only its own generation-prompt checkpoint. The server already
took a checkpoint at the end of the system message (section 3b, shared by conversations; `test_session` checks that a
new conversation with the same system prompt reuses it), so the system message itself must differ per session. The
pack's template renders `<|im_start|>system\n` + the reasoning-effort line + the **tools block** (`# Tools ... <tools>`
25 schemas `</tools>` ... `</IMPORTANT>`) + **then** the client's system text, so a per-session tail of that text (a
date, a directory, a session id: typical agent prompts; the client's contents were not read) moves the only shared
checkpoint out of reach although ~95% of the prompt is byte-identical. GDN states cannot be truncated: a resume point
needs a state snapshot taken exactly there, during a prefill.

**What a shared-prefix snapshot is.** A `Conv` with `shared=True`: toks = the token prefix [0, x), one checkpoint at x
(the whole recurrent state + the MTP head's bookkeeping, `state_save`, as any checkpoint), host copies of KV rows
[0, x) + the head's KV rows that checkpoint needs. It is found like any conversation (the lookup = the longest prefix
of the prompt among all stored token sequences: the key is the token prefix itself, the SSD tier's fingerprint adds
the build / pack / model id / layout), never resident, never extended: a request that restores it continues on a
**fork** (`Conv.fork`, the buffers shared), so the snapshot stays the same for every later session. Taken at a piece
boundary of a prefill (`state_save` there, ~5-20 ms), made after the reply (the KV copy, ~10 ms per 7K rows, off the
time to first token), written through to the SSD tier at once (crash-safe like every entry: tmp + rename, manifest
last; `"shared": true` in the manifest, format unchanged). A later snapshot reuses an earlier one's KV copy (rows
[0, x1) once in RAM), and the slot's conversation references the copies too (parking it copies only the rows past
them).

**Where (the boundaries).**
1. **The tools block's end** (structural; `server/ne_server.py` `SHARED_MARKERS = ("</IMPORTANT>",)`, the last
   occurrence inside the first message): taken when it is >= `NE_SHARED_MIN` (512) tokens past the restored prefix and
   the system text behind it is >= 512 tokens. Serves every session with the same tools + effort whatever its system
   text: the second session of a new tool set.
2. **Learned** (the shared prefix of two sessions): at a request's start the cache's longest common prefix L with the
   prompt, over every stored sequence (conversations resident / parked / on SSD, shared snapshots), held by another
   conversation than the one this request restores from; x = (L - 8) rounded down to a multiple of 64
   (`NE_SHARED_MARGIN`, `NE_SHARED_ALIGN`: two sessions' tails differ at slightly different tokens; one block serves
   them all); taken when x saves >= 512 tokens over the restored prefix, x <= 32768 (`NE_SHARED_MAX_POS`), and the
   stored sequence does not reach this prompt's generation prompt (that is the same first turn again, e.g. a retry with
   reuse off, not a shared prefix). The second session of a system text captures it, every later one restores it.
   The scan is skipped when the restored prefix is already past `NE_SHARED_MAX_POS` - 512 (a long conversation's turns).
3. **Shared boundaries supersede the system checkpoint**: with one below the system message's end (captured now or
   restored) its checkpoint is not taken. That keeps a cold prompt at master's number of prefill passes (each pass
   costs ~0.1-0.2 s at this size: the first run with both boundaries cost +3-6% cold) and saves a pass on restored
   ones; a client whose system text is identical across sessions gets the learned boundary <= 71 tokens before it
   from its second session on (before: the system checkpoint from the second session on).

**Bounds.** RAM: <= `NE_SHARED_RAM` (8) shared snapshots, inside `--cache-ram` like every buffer, least recently used
first (a hit touches it) to the SSD tier; SSD: <= `NE_SHARED_DISK` (32) entries inside `--cache-disk`, least recently
used first. Size at ~7.9K tokens: the tools block's ~0.29 GB (113 MB state + 6.4K rows x 27.7 KB), a learned one
behind it ~0.15 GB more (in RAM; on SSD each entry holds all its rows). Off: `NE_SHARED_SNAPS=0` (the earlier
behaviour); per request `ne_shared: false` (no captures: A/B). `timings.shared_prefix` = restored from one,
`timings.shared_made` = captured; log `prefix cache: shared prefix cN at X tokens`.

**Measured** (`tests/server/shared_job.sh` = `tests/server/shared_http.py`, one with_lock job, test server on 8144, ctx
65536, `--cache-disk 8GB` in a scratch directory, worktree build, 18 Oct 12:49; synthetic agent-like first turn: 25
tool schemas + 16 rules of agent prompt + a 5-line per-session environment tail in the system message + a ~55-token
user question, 7,883-7,895 tokens; greedy, thinking xhigh, MTP unless "step"; TTFT = request sent -> first streamed
delta; results `tests/server/results/shared-1018-1249*`):

| request | cached | TTFT |
|---|---|---|
| new session B, before (`ne_shared: false`: no captures, the earlier server) | 0 | 3,583 ms |
| new session B (restores the tools block snapshot, captures the learned boundary) | 6,369 | **1,105 ms** |
| new session C (restores the learned boundary; MTP decode) | 7,744 | **308 ms** |
| session C's next turn (its own checkpoint: the unchanged path) | 7,983 | 163 ms |
| new session E, step path | 7,744 | 311 ms |
| new session F: other system text, same tools (step path) | 6,369 | 903 ms |
| after a server restart, new session D (SSD tier; restore 61 ms) | 7,744 | **361 ms** |

Cold path: the same prompt cold back to back, earlier schedule vs this server: 3,763 -> 3,819 ms (+1.5%), 3,965 ->
3,977 ms (+0.3%); means over the job: 3,729 ms (7 cold prompts before) vs 3,849 (4 with captures), the cold TTFTs spread
3,471-3,977 ms (the first requests of a run are the fastest or slowest by a few %); two earlier runs of the job:
+1.8% / +1.3% (means). A first version that kept the system checkpoint next to the tools block's cost +3-6% (one more
prefill pass): hence rule 3. The KV copy (~10 ms) runs after the reply.

**Exactness.**
- Through the server (step path, greedy, 96 tokens with logprobs): a request restored from the learned snapshot (E), the
  tools block snapshot (F) and the learned one read back from the SSD tier after a restart (G) vs the same prompt
  prefilled cold on this server (`ne_cache: false`): **all 96 tokens equal, every logprob bit-identical** (|d| = 0).
  MTP-path replies (B, C) differ from cold after 333-351 characters: the drafts' history differs (another verify
  batching), as for any reuse.
- Against the earlier server's cold schedule the same requests differ by the engine's piece-schedule effect, not by
  the restore: E 96/96 tokens equal, |dlogprob| <= 0.27; F 63 tokens, then a tie; **two cold prefills without any
  restore** (this server vs the earlier schedule) differ exactly as much (0.27; 63 tokens, the same tie). The earlier
  schedule's system checkpoint leaves a ~25-55-row piece before the generation prompt, and such small pieces round
  differently from the long ones (16-32 rows run split-K, section 3e; at ~55 rows still not bit-identical; a first run
  with 25-token questions: first difference at token 2-42, the same between the two cold schedules). Long pieces (the
  >= ~130-row ones here) are split-invariant (below).
- In-process (`tools/check_shared.py`: one RealBackend, the server's feeds with the MTP head in step, then 48 greedy steps
  with logprobs): restoring a snapshot (state + KV rows + head rows) vs the same feeds without one: bit-identical at
  x = 6,360 / 6,361 / 6,362 / 6,363 / 7,680 / 7,744 / 7,745 (inside and at the 4-row index blocks), with the slot's KV
  overwritten by 9K other tokens first, without a reset, and with the rows copied only after a step-path or an MTP
  reply (as the server does); a piece boundary at x vs none: bit-identical. `tools/check_shared_sess.py diag`: the
  snapshot the server made = a plain export at that position (state and KV rows byte-identical; the head's KV rows
  differ in 1,078 of 15 MB bytes: they depend on the chunking, docs/prefill.md 4e).
- CPU mock (`tests/server/test_units.py` `test_shared_prefix`, step + MTP): outputs = a cold run, MTP continues after
  the restore, after a restart from the SSD tier, snapshots never adopted, the learned one shares the tools one's rows,
  off = cold, the RAM / SSD caps hold.

## 4. Not supported

Audio and video, `/v1/embeddings`, the Responses API, `echo` / `best_of` / `suffix` of `/v1/completions`, Anthropic's
server tools and document blocks, more than `--slots` requests at once (queued). (`n`, `logprobs`, the penalties,
`logit_bias`, `min_p`, `seed`, `/v1/completions` and `/v1/messages` are in section 11; bias, penalties and `min_p` ride on
MTP, `logprobs` and a bare `seed` take the step path.) (Structured output, forced
`tool_choice` and images are supported: sections 9 and docs/vision.md.) An MTP run stops at the end of the round in
which a stop string, a second EOS or a disconnect cancelled it. Context: agent sessions ran to ~160K tokens on 16 Oct
(QSA checked to 140K synthetic,
prefill to 32K).

## 5. Measurements

**Tests** (all pass): `tests/server/run_mock.sh` (CPU, no lock: 47 unit checks of the template, parsers, stop strings,
detokeniser and the session's reuse bookkeeping against a mock engine that records every fed token; then the HTTP
client test on the mock); `tests/server/e2e_job.sh` (one with_lock job, NEED_GB=85: starts the real server, runs
`tests/server/test_client.py`, stops it; then `tools/generate.py` direct). Three whole-model jobs, 16 Oct 05:23-05:29,
lab-integrate's build (`build-integ`, chunked prefill routed), 23 client checks each: auth refusals (no key / wrong
key / chat without key -> 401), /v1/models, /health, 400 on bad input, streamed text ("Paris"), reasoning separated
(17 x 23: 58 reasoning tokens, content "391"), a streamed tool call (`get_weather {"city": "Paris", "days": 3}`, the
integer typed by the schema) and the tool result back (answer from the forecast), stop strings streamed and not
(`1 2 3 4 5 6` for stop "7"), two concurrent requests queued, usage counts.

**Prefix reuse** (run 1, ctx 32768): tool turn 2 (452 tokens) reused 409 (the turn-1 generation-prompt checkpoint +
the reply re-rendered): prefill 201 ms vs 1009 ms fresh for the same prompt; a regenerate reused 447 (42 ms).
Greedy decode from the restored checkpoint vs from an empty state: the first 191 of 191 characters identical.

**Speed through HTTP vs the engine** (1024 tokens, temperature 1 / top-p 0.95 / top-k 20, seed 7, one run each;
"engine" = the worker's own decode timing around the C calls; "direct" = `tools/generate.py`, no server):

| prompt | path | HTTP tok/s | engine tok/s | direct tok/s |
|---|---|---|---|---|
| essay, thinking off, ctx 32768 (run 1) | step (1-row `ne_fn_model_step`) | 65.0 | 65.05 | |
| | MTP d2 | 77.2 | 77.26 | |
| essay, thinking off, ctx 262144 (run 2) | step | 72.1 | 72.07 | 70.0 (`generate`, pipelined) |
| | MTP d2 | 86.9 | 86.92 | 83.3 (`--mtp 2`) |
| code_lru, thinking medium, ctx 262144 (run 3) | step | 64.7 | 64.71 | (docs/mtp.md 10.3: 69.7) |
| | MTP d2 | 84.0 | 84.01 | (docs/mtp.md 10.3: 101.2, lab-loop's build) |

HTTP equals the engine's own rate to 0.1%: the server costs nothing measurable on the decode path. Per-token cost
on the CPU: the worker's ctypes step call ~3.5 us (8-argument call measured on the error path), one `queue.put`
0.4 us; detokenise + parse + SSE runs in the HTTP thread while the GPU computes the next token, ~6 us per token
(mock engine at 0 s/token: 167K tok/s through HTTP; worker loop alone 1.6M tok/s). Run-to-run spread on the engine
is larger than the server's cost (step 64.7-72.1, MTP 77-87 on one sample each); the code_lru MTP run (84) is below
lab-loop's 101.2 on the same prompt: one seed, another build: to recheck on the engine side, not a server effect.
Time to first token at 34 prompt tokens: 0.18-0.34 s (prefill 175-340 ms, incl. the end-of-prompt checkpoint
copies, ~10-20 ms each).

**Memory**: free RAM 116 GB before; 37 GB with the server up at ctx 32768, 30 GB at ctx 262144 (~79 / ~86 GB used).
Engine load 0.7-1.0 s with the pack in the page cache (+ the MTP head's recordings, + 3 shim-check steps).

### 5b. Prefix cache and slots (lab-srvcache, 16 Oct 13:02-13:10, `tests/server/cache_job.py`, ctx 32768, greedy)

**Park / restore vs re-prefill** (slot 0; save = `ne_fn_state_io` + `ne_fn_kv_io` out, restore = in; then 32 greedy
tokens compared with decoding from the never-parked state, after another prompt overwrote rows and state):

| T | re-prefill | park: state + KV | KV GB | restore: KV + state | decode after restore |
|---|---|---|---|---|---|
| 8192 | 5471 ms | 8.6 + 32.7 ms | 0.22 | 6.4 + 2.2 ms | identical |
| 16384 | 10179 ms | 1.8 + 19.0 ms | 0.43 | 6.6 + 0.8 ms | identical |
| 28672 | 18017 ms | 0.9 + 40.0 ms | 0.76 | 15.9 + 0.8 ms | identical |

**Three conversations on ONE slot** (docs/kernels.md, mtp.md, trunk6.md as system prompts, 5.1-6.0K tokens, round
robin, 3 turns each: every turn parks the previous conversation and restores its own): turn 1 cold (3.7-4.4 s);
turns 2-3 reuse 5067-5932 tokens (all but the 34-67 new ones), prefill 300-457 ms (restore 14-17 ms of it) vs
3.97-4.51 s fresh; greedy output = a fresh run (no reuse) of the same prompt in 9 of 9 turns (40 tokens). RAM 4.9 GB
of the 8 GB budget at the end (12 conversations incl. the fresh runs' own). The 300-457 ms are the chunked prefill's
floor, not the cache: one chunk costs ~128 ms from 16 tokens on (17: 128 ms, 33: 135, 65: 161 at T 5888; 1-row
steps 14 ms per token), and a turn feeds 2-3 pieces (split at the generation-prompt checkpoint).

**Two streams at once** (slots 0 and 1, interleaved per token, prompts 2179 / 1961 tokens, 384 greedy tokens each):
aggregate 72.1 tok/s vs 71.8 for one stream alone (35.8 + 36.1); both outputs identical to each run alone. A
subagent's request no longer waits for the whole main turn; the sum is one stream's rate (time-slicing, section 3c).

**After merging lab-mtpcont + lab-api** (15:42-15:52, `cache-1016-1542.json`, `cache-1016-1549.json`): park +
restore with the MTP head (main KV + 8184-28664 head rows, 0.235-0.822 GB, park 15-55 ms, restore 3-15 ms): greedy
MTP output AND its round count identical after the restore (18/18, 23/23, 29/29 rounds at T 8K/16K/28K). Three
conversations rotating on 2 slots with MTP turns: turns 2-3 reuse 5065-5937 tokens, prefill 292-439 ms (restore
4-17 ms) vs 4.5-5.4 s fresh, decode 91-118 tok/s, output = fresh 9/9. Two streams: 63.7 tok/s aggregate vs 63.3
alone, identical. Server e2e (2 slots) and lab-api's structured e2e (mask checks, 5 schemas, forced / required
tools, step / MTP / constrained benches) all pass.

**HTTP e2e** (`tests/server/e2e_job.sh`, 2 slots, 13:03): the 23 client checks pass (step 67.9 tok/s, MTP 84.1 through
HTTP); 79 GB used with the server up (as with 1 slot). **CPU tests** (`tests/server/run_mock.sh`): the mock now keeps
KV rows apart from the recurrent state and asserts on every feed that the rows attention would read hold the state's
tokens; new checks: 3 conversations x 4 turns on 1 and 2 slots (each turn reuses its last checkpoint), budgets of 1 B /
6 KB / 1 GB (RAM within budget), the disk tier (every evicted conversation restored), two interleaved requests
(identical outputs, the subagent reusing the main agent's system checkpoint from the other slot), and a random mix
(5 conversations, edits, regenerates, MTP, 1-3 slots, budgets, disk; 74 requests, all outputs = the request alone;
a mutant that skips the KV copy fails it).

### 5c. SSD tier across restarts, crash recovery (lab-persist, 16 Oct 18:38-18:40)

`tests/server/persist_job.sh` (one with_lock job, NEED_GB=85, port 8141, ctx 65536, main repo `build/` at 7247783, K5-T6
pack, greedy, thinking off, 160 tokens; results `tests/server/results/persist-1016-1838*`). A 52K-token conversation
(docs/mtp.md + server.md + kernels.md as the system prompt), three server processes on one scratch SSD directory:

| run | request | cached | from | prefill | restore | write-through (engine thread) |
|---|---|---|---|---|---|---|
| A | turn 1 | 0 | | 30.8 s | | 77 ms (52K rows + 2 checkpoints = 1.8 GB queued) |
| A | turn 2 | 52,085 | live | 250 ms | | 1.7 ms |
| A | turn 2 regenerated | 52,104 | RAM | 67 ms | 1.0 ms | 0 |
| B (restarted) | turn 2 | 52,104 | **SSD** | **531 ms** | 178 ms (1.8 GB read + crc32) | 0 |
| B | turn 2, `ne_cache` false | 0 | | 30.0 s | | 69 ms |
| C (every seg file 1 byte flipped) | turn 2 | 0 | rejected | 30.7 s | | 57 ms |

- **Output after the restart = the never-restarted server's, bit for bit** (B's SSD turn vs A's RAM regenerate: same
  checkpoint, same 160 tokens; MTP path, 101.9 tok/s). The cold reference differs from both (other prefill piece
  boundaries, section 3), as expected; C's cold turn = B's cold reference.
- Corruption: `SSD entry ... rejected (seg1.bin: checksum mismatch); prefilling cold`, the request answers normally,
  the entry is deleted. Start-up index of 1-2 entries (1.8-3.5 GB): 1 ms; server load 2 s (pack in the page cache).
- The restore read came from the page cache (written seconds before); not measured: a read from the SSD itself, estimated ~0.3 s more for
  1.8 GB (~6 GB/s). The 531 ms are restore + the 7-token suffix's prefill chunk + the MTP feed floor (section 5b).
- An earlier run of the job (18:29) changed its prompt between A and B (this doc was edited meanwhile; the client now
  freezes the system prompt in its state file): B then found no prefix, prefilled cold, as it should.
- CPU: `test_units.py` `test_persist` (restart on a fresh mock engine: every conversation restored from SSD, step and
  MTP with the head; other identity: no reuse, directory kept; a flipped byte -> rejected, cold, output unchanged;
  crash debris and a broken manifest cleaned at start; second tier -> private dir; budget with LRU) and the random
  mix with ~12 restarts per run (seeds 3-7: 77-156 requests, 0 differ); `test_supervisor.sh` (section 3d). Writer
  throughput (CPU microbench, scratch dir): 0.98 GB in 180 ms (page cache), read back + crc32 in 109 ms.

## 6. Requests (docs/kernels.md)

(1) `ne_fn_mtp_generate_cb` (per-round callback, nonzero = stop): streaming, stop strings and aborts on the MTP path
(the server uses it as soon as the symbol exists). (2) MTP from the current state (prefix reuse with MTP): done
(lab-mtpcont, docs/mtp.md 10.9). (3) An official model snapshot / restore: done (`ne_fn_state_io` / `ne_fn_kv_io`,
`ne_fn_mtp_state_io` / `ne_fn_mtp_kv_io`, lab-srvcache; `server/shim` unused). (4) lab-prefill: an LRU cap on the
chunk recordings.

## 7. Traps

- `dlopen` refuses relative library paths in a hardened-runtime Python: the server resolves `--lib` /
  `--shim`.
- The shim reads `ne::fn::Model` fields: it must be compiled against the headers the library was built from.
  `tools/serve.sh build` takes the header list from the compiler (`-MM`) and refuses when any is newer than the
  library (lab-integrate's work tree vs `build/` on 16 Oct); at load the server steps 3 tokens and cross-checks T,
  tmax, the layer count and the state bytes against the C API before trusting it.
- Re-rendered history is not token-identical to what was generated (`<think>\n` + `\n` merges; empty reasoning
  renders as `<think>\n\n</think>`): checkpoint the start of the generation prompt, not its end.
- `sleep`-based mock delays overshoot on macOS (time.sleep(0.014) sleeps 18.5-19.3 ms here, usleep / select / Event.wait /
  mach_wait_until alike: timer coalescing): a 14 ms mock looked like ~3 ms of server overhead per token (57.9 tok/s).
  The mock now waits on a kqueue timer with NOTE_CRITICAL (`backend.precise_sleep`: 14.02 ms).

## 8. Attribution

The OpenAI wire format follows OpenAI's public API reference; the tool-call parsing follows the Qwen chat template's
format (no code copied from vLLM / Sushi / oMLX). The persistent SSD tier (section 3b) follows the design described in
Sushi's `docs/engine-prefix-cache.md` (MIT; itself ported from mlx-serve #680): a fingerprint namespace per build and
pack, files as tmp + rename with the manifest last, a startup scan that drops damaged entries, one live tier per root
(flock) with a private root for a second process. Written from that description; no Sushi code copied.

## 9. Structured output (lab-api, 16 Oct 2026)

`response_format` `{"type": "json_object"}` / `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`
and `tool_choice` `"required"` / `{"type": "function", "function": {"name"}}` are enforced by grammar-constrained
sampling: the output is a word of the grammar, token by token, or the request fails with 400 (unsupported schema).

**Hook (engine).** `ne_fn_model_set_mask(model, bits | NULL)` (models/flash_next/model.h, kernels/fn_model.metal):
`bits` = 7760 u32 words, bit i set = token i may be sampled. The sampler's pass 1 (`fn_sample_p1`) reads one bit per
vocabulary entry and treats a cleared token as logit -inf, so the top-k, top-p, temperature draw, the logsumexp (= the
reported logprob) and greedy all run over the allowed tokens only. It is a field of `SampleParams` (`mask_on`) plus one
buffer bound in the recorded step: no new dispatch, nothing re-recorded, nothing read when the mask is off. Applied by
`step()`, `prefill()` (the sample at the end of the prompt) and the pipelined `generate()` (one fixed mask); persists
across `reset()` / `set_sampler()`; an all-zero mask is refused. The host writes it between steps (the step is synchronous
on the server's path). The MTP loop does not apply it: `ne_fn_mtp_generate` errors while a mask is set.

**Grammar (server/grammar.py).** xgrammar 0.2.8 (Apache-2.0; a pip dependency in requirements.txt,
which brings torch along; nothing copied). A `TokenizerInfo` over the pack's
byte-level vocabulary (248320 entries; the markup tokens `<|im_start|>`, `<|vision_pad|>` ... are made special = never
allowed; `<think>`, `</think>`, `<tool_call>`, `</tool_call>` stay because the grammars spell them; the stop ids
`<|im_end|>` / `<|endoftext|>` are only allowed once the grammar is complete). One `GrammarCompiler` + an LRU of 64
compiled grammars per (format, thinking, tool_choice). The model's whole output is one grammar:

| thinking | output grammar |
|---|---|
| on (prompt ends `<think>\n`) | any text but `<think>` / `</think>`, then `</think>` and `\n\n`, then FORMAT |
| off (prompt ends `</think>\n\n`) | FORMAT |

FORMAT: `json_object` = a JSON object; `json_schema` = the schema (declared property order, strict: no undeclared
properties or items; at most 32 whitespace characters in a row so a blank run cannot trap the decoder); `required` = at
least one call, to any of the tools, in the template's XML (`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>
...\n</function>\n</tool_call>`, arguments typed by the tool's JSON schema, `qwen_3_5` structural tag), more calls unless
`parallel_tool_calls` is false; a named function = exactly that one call, nothing before or after (xgrammar's
`get_model_structural_tag("qwen_3_5", ...)`). While reasoning runs the mask only keeps special tokens (and the stop
token) out; after `</think>` the format begins at the very next token. `tool_choice` auto / none / unset and no
`response_format` = no mask: the request is exactly as before (MTP, speed).

**Request side.** With a `json_schema` / `json_object` the schema (or an instruction to answer with one JSON object) is
appended to the first system message (one is added if there is none), as OpenAI shows the schema to the model;
`ne_format_hint: false` in the request turns that off. `response_format` without the grammar kit (xgrammar missing,
`--no-grammar`, library without `ne_fn_model_set_mask`) -> 400; `tool_choice` required / named then degrades to auto
with a log line. A schema xgrammar cannot compile -> 400 `cannot constrain the output: <reason>`. Compile of a new schema
5-20 ms (200 ms with string `excludes`, not used); the compile runs in the request's HTTP thread.

**Decode path.** Per token the worker advances the matcher with the sampled token, fills the bitmask (xgrammar 10-60 us,
max ~220 us), copies 31 KB into the engine and steps: ~0.1 ms of 14-15 ms. A constrained request takes the **step path**
(`timings.path` "step", `timings.grammar` = `schema` | `object` | `required` | `forced`, `+tools` added beside a response_format with tools): the MTP loop samples and
verifies on the GPU with its own shortlist, and the mask of row j+1 depends on the draft token j, which only the GPU
knows, so a mask there needs a CPU round trip per draft (or the verify kernels taught the grammar). Unconstrained
requests keep the policy of section 3. The mask is cleared when the request ends or fails.
Follow-ups: MTP during the reasoning phase (needs "MTP from the current state", Request 2) and jump-forward decoding
(feed the tokens the grammar forces without sampling them).

**Tests.** `tests/server/test_grammar.py` (CPU, real tokenizer + xgrammar: five schemas in `tests/server/schemas.py`
accepted token by token / refused at the offending token, a seeded random walk over the masks, forced and required tool
calls, the Session plumbing on the mock engine), `tests/server/test_structured.py` (HTTP, mock or real model),
`tests/server/mask_check.py` (the engine's mask on the whole model), `tests/server/e2e_structured_job.sh` (one with_lock
job on port 8141: mask check, baseline vs this build speed A/B with `speed_ab.py` / `step_ab.py`, the HTTP test);
`run_mock.sh` runs the CPU ones. Results: section 9b.

### 9b. Results (16 Oct 2026, 13:04-13:08, M5 Max, one with_lock job, port 8141, ctx 32768, K5-T6 pack, aligned build)

`tests/server/results/structured-1016-1304-*` (client, mask, speed, server logs, json).

- **Engine mask** (`mask_check.py`, 13 checks): an all-ones mask gives the same tokens as no mask (greedy and seeded-sampled);
  a one-token mask forces that token at temperature 0 and 1 with logprob 0.0 (a first run showed NaN there: a 1024-block
  with every logit masked made the block's logsumexp -inf - -inf; fixed in `fn_sample_p1`, 0212ee0); a digits-only mask
  keeps 40 sampled steps inside the digits; the sample at the end of the prefill obeys it; an empty mask and the MTP loop
  with a mask set are refused.
- **Unconstrained speed unchanged.** GPU time of the full decode step at T = 2048 (`ne_fn_model_time`, 5 x 60 replays,
  median of the medians; the baseline is this tree's parent commit built in a scratch worktree): baseline 13.615 ms (min
  13.200; the five rounds span 13.35-13.84), this build with no mask 13.336 ms (min 13.129), with an all-ones mask on
  13.340 ms (min 13.129): the bit read costs +0.004 ms = 0.03 %, below the run-to-run spread. Through HTTP, unconstrained
  1024-token essays, seeds 8-9: step 73.26 -> 73.36 tok/s, MTP d2 90.84 -> 89.29 tok/s (baseline vs this build; two runs
  each, the MTP spread is +-2).
- **Constrained speed** (step path, same server): the five schemas 66-426 tokens at 71.5-73.2 tok/s; `json_object` 768
  tokens 72.3 tok/s against 73.9 tok/s for an unconstrained step run (-2 %: grammar advance + mask copy + the bit read)
  and 94.5 tok/s for an unconstrained MTP run (constrained requests give up the MTP gain, ~-23 %).
- **HTTP test** (`test_structured.py`, 36 checks, all pass): `/v1/models` capabilities `chat tool_use streaming reasoning
  json_schema`; person / ticket / recipe / verdict / trip (nested objects, enums, nullable, bounded arrays, nested number
  arrays) thinking off: schema-valid on the first try and the content answers the prompt (ticket: `assignee` null,
  priority urgent, two tasks; verdict: yes, 1.0; trip: train then plane); ticket and verdict with thinking on at low
  effort (168 / 87 reasoning tokens, then the document); a streamed schema (77 chunks); `json_object`; four 400s (unknown
  type, no schema, unsupported schema type, a forced tool that is not in `tools`); `max_tokens` 12 -> finish_reason
  length; tool_choice forced `get_time` against a weather prompt, thinking off / on / streamed -> exactly one call with
  valid arguments; `required` on "Hello! How are you today?" -> a `get_time` call; `auto` unconstrained; `timings.grammar`
  and `timings.path` as documented, absent on unconstrained requests.
- **CPU tests** (`run_mock.sh`: `test_grammar.py` 106 checks on the real tokenizer, then the mock-server HTTP tests, plus the
  47 earlier unit checks and the client test): all pass. xgrammar mask fill 10-60 us per token (json_object on a 2.4K-token
  document: mean 58 us, max 189 us; a 5-property schema: 30 us); a new schema compiles in 3-20 ms.

### 9c. After the merge with lab/mtpcont (16 Oct 2026, 13:58, same job, no baseline half)

Both behaviours kept: a constrained request still takes the step path (`Worker.run`: `can` is false with a constraint) and
sets the mask before the prefill's last-token sample and before every `mtp_step`/`step` (the step goes through
`ne_fn_mtp_step`, which samples through the same masked `fn_sample_p1` and keeps the MTP head in step); an unconstrained
reused turn continues with MTP (`mtp_from`). `ne_fn_mtp_generate` / `_generate_from` (one `generate_impl`) refuse while a mask
is set. `Model::mask` and `Model::gen` are both in model.h. Whole-model results
(`tests/server/results/structured-1016-1358-*`): mask check 13/13, HTTP test 37/37 including the new check that an unconstrained
follow-up turn after a forced tool call runs on the MTP path with the prefix reused (path `mtp`, 391 cached tokens);
GPU step 13.281 ms with no mask, 13.314 ms with an all-ones mask (+0.03 ms, 0.2 %, inside the round spread 13.20-13.31);
step 69.0 tok/s, MTP 90.9, constrained `json_object` 67.3 (-2.4 % vs the step run). The step path is slower than before the merge
(73 -> 69 tok/s on this sample, 63 in the speed_ab run on a busier machine) because every step now also advances the MTP head
(mtpcont's design, not the mask: the GPU step time is unchanged). lab-headlazy (docs/mtp.md 10.10) made the head lazy: `ne_fn_mtp_step` now
only queues the step's hidden row and the head catches up (KV-only head steps, ~28 us per row) before an MTP run, a
checkpoint or a park; engine-level step path 74.1 tok/s vs 74.2 for the plain step (the eager head measured ~1% there).

## 10. Per-token overhead and several models in one process (lab-srvperf, 16 Oct 2026)

**Per-token host cost.** The "~3 ms/token since the slot scheduler" (mock at 14 ms/token: 57.9 tok/s) was the mock's
`time.sleep` overshoot (18.5-19.3 ms per 14 ms sleep on this Mac, timer coalescing), not the server: with a kqueue
NOTE_CRITICAL timer (`backend.precise_sleep`, 14.02 ms) the mock gives **71.1 tok/s through HTTP = 71.1 in the worker**
(ideal 71.3): ~0.04 ms/token incl. the timer's own error. The step path now reports `timings.host_us_per_token` (the
engine thread's time between step calls: generator, scheduler checks, callbacks, queue put, GIL waits; step path, alone):
**11-20 us per token on the real engine** (mock 13 us). Small cuts kept: no `queue.Empty` exception per token in the
worker loop, the step call's ctypes arguments bound once. Real engine, one with_lock job (`tests/server/srvperf_job.sh`,
results `tests/server/results/srvperf-1016-1820*`; T6 pack, ctx 262144, essay prompt, 1024 tokens, seed 7):

| path | through HTTP | engine (worker) | direct `tools/generate.py` |
|---|---|---|---|
| step (1-row `ne_fn_mtp_step`) | **73.6 tok/s** | 73.6 (host 11.6 us/token) | 72.1 (pipelined `generate`) |
| MTP d2 | **94.6 tok/s** | 94.6 | 90.7 (`--mtp 2`) |

HTTP = the engine's own rate; both are at or above the direct tool (the 69-vs-74 gap of section 9c was mtpcont's eager
head, removed by lab-headlazy). Nothing left to cut on the host side that would show at 13.5 ms GPU per token.

**Two models, one process** (section 1b). Engine changes: `core/pack.mm` maps each file once per process by inode (the
second pack's shared files reuse the first's mapping and buffers: 77.3 GB shared, Swift's own 2.9 GB of T6 trunk),
`ne_fn_model_share_prefill` (every model instance's chunked prefill on one set of chunk buffers), and for
`--share-state` `ne_fn_model_create_shared` (the second model's weights over the first's KV + recurrent state). Greedy
texts on the two-model server = the one-model server's (base) and = `tools/generate.py --pack Swift...` (Swift), also
after a ~9K-token prefill on each model through the shared prefill buffers (identical with and without sharing).

| ctx 262144, after a ~9K prefill + decode on each model | server footprint | wired + anon + compressor (system) |
|---|---|---|
| base alone | 15 GB (+ 80.2 GB weights) | +90.8 GB |
| base + Swift, own states, shared prefill buffers (default) | 24-25 GB (+ 83.1 GB weights) | +103.2 GB (**+12.4**) |
| base + Swift, own states, own prefill buffers (`NE_SRV_SHARE_PF=0`) | 29-30 GB | +107.2 GB (+16.4) |
| base + Swift, `--share-state` (one KV + recurrent state for both) | 18 GB (+ 83.1 GB weights) | +96.7 GB (**+5.9**) |

(`tests/server/results/srvperf-1016-1826-memory.txt`, `-1840-two.txt`, `-1844-share-state.txt`; "system" = the
change of wired + anonymous + compressed pages from before the server started; the footprint excludes the file-backed
weights. At 262K the second model's own state is the KV (7.3 GB) + the MTP head's KV (0.6 GB); a smaller `--ctx`
shrinks it by 27.7 KB + 2.3 KB per token.) Not committing big buffers lazily (no memset in `alloc_bytes`) was tried:
the footprint and the system total stayed the same (Metal accounts / wires the whole allocation), so it was dropped.

- **Own states (default)**: +12.4 GB at 262K (+2.9 trunk, +7.9 KV, ~+1.6 decode scratch / recordings / draft head);
  both models' requests run at once (interleaved like two slots), each keeps its live conversation: a model switch costs
  nothing (TTFT 175-182 ms after a switch vs 177 ms on the same model, 22-token fresh prompts).
- **`--share-state`**: +5.9 GB at 262K (the trunk 2.9 + scratch, recordings, the MTP head and its KV per model). Slot j
  of every model is one state: a request to the other model waits for the running one and takes the buffers over (the
  previous owner's conversation is parked first: KV rows up to its newest checkpoint copied to host, 15-55 ms at
  8-28K tokens, section 5b); a conversation coming back restores from that checkpoint (bit-identical) and re-prefills
  the tokens after it: base turn 2 after a Swift request reused 4356 of 4414 tokens, TTFT 353 ms vs 161 ms with own
  states (4394 reused from the live state); same texts in both configurations (turns, greedy 48-token essays).
  Fresh-prompt switches: TTFT 173-178 ms either way.
- Speed is the one-model speed on both (step 71-75 tok/s, MTP 89-95; greedy, 48-1024 tokens).
- Not done: the MTP draft head's 4-bit lm_head copy (0.33 GB) is still per model instance (`mtp.mm requantize`); the
  vision tower stays per pack (lazy).

## 11. Sampling knobs, logprobs, n, /v1/completions, /v1/messages (lab-sampler, 16 Oct 2026)

Owner: lab-sampler (`server/sampling.py`, `server/anthropic_api.py`, the sampler kernels in `kernels/fn_model.metal`,
`models/flash_next/model.{h,mm}`). Branch `lab/sampler`.

### 11a. Request fields (chat.completions, completions, messages)

| field | meaning | range / default |
|---|---|---|
| `logit_bias` | `{token id: bias}` added to the logits; -100 bans the token (exact -inf), +100 selects it | -100..100, <= 4096 ids; ids are the pack's tokenizer ids |
| `presence_penalty`, `frequency_penalty` | OpenAI: `logit -= frequency * count + presence * (count > 0)`, count = the tokens THIS request generated so far (reasoning included) | -2..2 |
| `repetition_penalty` | HF / vLLM: a seen token's logit is divided by it (> 0) or multiplied (< 0); seen = prompt + generated | (0, 10], 1 = off |
| `min_p` | after the temperature softmax, tokens with p < min_p * p_max are dropped (before top-p) | 0..1, 0 = off |
| `seed` | the draws depend on (seed, index of the sampled token); `n` choices use seed, seed + 1 ... | any integer |
| `logprobs`, `top_logprobs` | chat: `logprobs: true` + `top_logprobs` 0..20; completions: `logprobs` = number of alternatives 0..20 | off |
| `n` | choices per request (completions: per prompt), 1..64, run one after the other | 1 |

Order on the logits: `logit_bias` -> penalties -> grammar / token mask (section 9) -> temperature -> top-k -> `min_p`
-> top-p -> draw. **Logprobs are log-softmax at temperature 1 of the transformed logits** (after bias, penalties and mask;
the raw model distribution when none is set, which is OpenAI's default case), over the whole vocabulary, not renormalised by top-k /
top-p. The sampled token's entry is its value in the candidate list (the list always contains it: it holds
`max(top_k, top_logprobs, 1)` candidates, sorted by transformed logit, ties by lowest id; -inf candidates (banned, masked) are dropped).

### 11b. Engine (pass 1 / pass 2, no new dispatch)

`SampleParams` grew (kernels/fn_model.metal, models/flash_next/model.h): `min_p, presence, frequency, repetition,
bias_on, pen_on, cand_k, lp_on`; `Model` has `bias` (f32 [vocab]), `cnt` (u32 [vocab]: bit 31 = in the prompt, low 28
bits = generated count) and `top` (u32 [64][2]). `fn_sample_p1` adds `bias[i]` (<= -100 -> -inf), applies the penalties to
tokens with `cnt != 0`, then the mask, then the top-K of `max(top_k, cand_k)`; `fn_sample_p2` draws from the first `top_k`
(min_p cut, then top-p, as before), **increments `cnt[token]` of the token it sampled** (GPU-side counts, no host
round trip) and writes the candidates' (id, logprob) pairs to `top`. With every knob off, `bias_on = pen_on = lp_on = 0`
and the kernels read nothing extra (the draw's arithmetic is unchanged: the same (seed, counter) gives the same token as before).
C API (`ne_fn_model_*`): `set_sampler_ex(min_p, presence, frequency, repetition, logprob_k)` (also **zeroes the counts**: a
request's start), `set_seen(ids)` (the prompt, for the repetition penalty), `set_logit_bias(ids, bias, n)` (n = 0 clears),
`set_rng_step(step)`, `top(ids, logprobs, max)` -> count. Python: `FnModel.set_sampler_ex / set_seen / set_logit_bias /
set_rng_step / top` (tools/engine.py). The RNG counter now advances once per **sampled** token (was: per fed token, so
the draws depended on how the prompt was fed); `set_rng_step(0)` at a request's start makes the draws a function of
(seed, token index).

**MTP applies logit_bias, penalties and min_p exactly; logprobs and seeds step.** Speculative sampling stays exact when the
target rows use the same transformed distribution the step path samples from, and the draft's proposal is whatever it
really drew from (its recorded `q`; ignoring the knobs only costs acceptance). So (models/flash_next/mtp.metal
`mtp_logit_xform`, mtp_loop.mm): after the verify's lm_head, one indirect dispatch (0 threadgroups unless `bias_on` /
`pen_on`) transforms the `[R][vocab]` logits in place: verify row r predicts the token after `[t1, d_1 .. d_r]`, so its
counts are `cnt` (everything emitted so far, t1 included) plus the drafts `d_1 .. d_r` of the same round; the top-K shortlist
(sampled) or the argmax (greedy) then read the transformed rows. `min_p` is applied by the host distribution
(`mtp::make_dist`, between top-k and top-p, as pass 2). After each round the host adds the emitted tokens to `cnt`
(the step path counts them in pass 2), so a run that yields to the step path (scheduler) and resumes keeps one consistent
count. `generate_impl` refuses only `logprobs` (`lp_on`) and the token mask. Checks: `tests/server/sampler_check.py` (the verify-row
kernel against float64 with drafts in the counts: max diff 1.6e-7), `tests/mtp/test_mtp_knobs.py` (whole model: sampled
MTP, 700 seeds x 2 knob sets: out[0] (the step path's sample), out[1] and out[2] given the earlier tokens match the exact
filtered distributions computed from the 1-row logits and a float64 transform, P(noise >= TV) 0.08-1.0, the same
comparison against the untransformed distribution is rejected 4 of 4 times; greedy MTP = greedy step except at
near-ties, see the file; sets with a repeating answer so drafts repeat earlier tokens).
**Speed** (essay, 1024 tokens, sampled, seeds 7 / 8, one server, `tests/server/results/sampler-1016-1945-speed.txt`): step 74.5
tok/s plain, 74.5 with presence 0.5 + frequency 0.2 + bias + min_p and 74.7 with logprobs 5 (the knobs cost the step path
nothing); MTP d2 89.9 plain; MTP with one knob: min_p 0.05 88.3 (-2 %), bias 86.2 (-4 %), presence 1.5 84.1 (-6 %),
frequency 0.2 79.2 (-12 %), all of them together 79.3 (-12 %). The draft proposes without the knobs, so acceptance falls where
a penalty moves the target away from it (0.2 x count is a big shift for common words); with every knob MTP is still ~6 %
faster than the step path. `logprobs` (and a bare seed) lose the MTP gain (-17 %). **Unconstrained default speed unchanged**
(baseline = the parent commit built in a scratch checkout; `sampler-1016-1936-speed.txt`, two interleaved A B A B rounds of 5
x 60 replays, GPU step at T 2048): median of medians base 13.225 / 13.433 ms vs this build 13.259 / 13.400 ms, minimum
13.150 / 13.260 vs 13.149 / 13.251 ms (the spread between rounds of one build is 0.1-0.3 ms); HTTP step 71.1 -> 71.3 tok/s,
MTP 83.0 -> 84.0 (this build with the extra indirect dispatch in every verify). Each knob on (GPU step medians, 4 rounds x 60
replays, quiet machine): off 13.205, bias 13.245, penalties 13.193, min_p 13.184, logprob_k 20 13.225, all together 13.197, off
again 13.196 ms: inside the +-0.05 ms spread.
**A seed alone also steps** (`--seed-mtp` or `"ne_path": "mtp"` lets it use MTP, best effort): the step path's draws are a
pure function of (seed, token index), the MTP rounds are not (they depend on when the server switches paths).
Reproducibility holds for the same prefill layout: with the prefix cache hit/miss pieces round differently, so a late
token can differ (measured: 54 of 54 characters identical on a 40-token sentence, 28 of 40 tokens between a one-piece and a two-piece prefill
on the engine); `ne_cache: false` prefills from scratch every time.

### 11c. Server

- `Extras.parse` (server/sampling.py) validates the fields -> 400 with the offending value. `Job.extra` = the engine dict (None
  when nothing but temperature / top-p / top-k applies); `Worker._admit` makes `can` (MTP allowed) false for `logprobs` and
  for a seed (unless `--seed-mtp` or `ne_path: "mtp"`); `ne_path` cannot force MTP over `logprobs`. Bias, penalties and `min_p` follow the
  usual MTP-or-step policy (`timings.sampling` names them either way).
- `Session.run(extra=...)`: `RealBackend.set_extra` sets bias + sampler_ex (+ prompt flags for the repetition penalty) at the
  request's start; **`arm()` zeroes the counts and restarts the RNG counter right before the kept first sample** (a plain
  prefill samples after every piece and discards all but the last: those samples must not count); `clear_extra()` at the end,
  also after an error (like the mask). Nothing is touched for a plain request.
- Responses: `choices[].logprobs.content[] = {token, logprob, bytes, top_logprobs[{token, logprob, bytes}]}` for every generated
  token (reasoning tokens included; the stop token is not), streamed chunks carry the entries of the tokens they spell (a
  partial UTF-8 character's entry goes with the next chunk); `bytes` is exact, `token` a replacement character for a partial
  character. `timings.sampling` lists the active knobs (and `timings.path` is `step`).
- `n`: the choices of one request are submitted one after the other (the prefix cache makes later ones nearly free, the same
  prompt's checkpoint); streamed `index` 0 completes before 1 starts; `usage` counts the prompt once, completion tokens summed.
- **`/v1/completions`**: `prompt` = string | list of strings | token ids | list of token-id lists (indexes `prompt * n + i`);
  raw text, no template, no reasoning / tool parsing (`chat.RawParser`: verbatim); `stop`, `stream` (+ `include_usage`),
  `logprobs` (legacy object: `tokens, token_logprobs, top_logprobs, text_offset`), all the knobs; `echo`, `best_of`, `suffix`,
  `response_format` -> 400. No `max_tokens` = the context window (the server's no-reply-cap rule; OpenAI's default is 16).
- **`/v1/messages`** (`server/anthropic_api.py` converts to the chat request and back): `system` (string | blocks), message
  blocks `text`, `image` (base64 / url), `tool_use`, `tool_result`, `thinking`, `tools` (custom `input_schema` tools),
  `tool_choice` auto / any (-> required) / tool (-> named function, grammar-constrained) / none, `stop_sequences`,
  `temperature` / `top_p` / `top_k`, `thinking {enabled, budget_tokens}`, `stream`. Response: `content` blocks
  (`thinking`, `text`, `tool_use`), `stop_reason` `end_turn | max_tokens | stop_sequence | tool_use`, `usage` with
  `input_tokens` = the uncached part and `cache_read_input_tokens` = the reused prefix. Stream: `message_start`, `ping`,
  `content_block_start / _delta (text_delta, thinking_delta, input_json_delta, signature_delta) / _stop`, `message_delta`
  (stop reason + usage), `message_stop`. Auth `x-api-key` (Bearer works too); errors in Anthropic's body
  `{"type": "error", "error": {"type", "message"}}`. **Differences**: no thinking unless `thinking.type` is `enabled`
  (the API's default; the budget picks the effort: <= 4096 low, <= 16384 medium, else xhigh); thinking blocks have an empty
  signature (any signature is accepted back); server tools and document blocks are refused; `max_tokens` is required and
  honoured. `/v1/messages/count_tokens` renders the template and counts.

### 11d. Tests and results

- `tests/server/sampler_check.py` (GPU, synthetic logits, no model, a few seconds): bias (ban, additive, off), penalties vs a
  float64 reference (greedy token and logprob, negative-logit repetition, counts incremented by pass 2, off = untouched),
  the mask after the penalties, the candidate list (cand_k vs top_k, under bias), draw frequencies with min_p / top-p /
  temperature (chi-square-ish bound): all pass.
- `tests/server/sampler_model_check.py` (whole model, direct engine): against numpy on the model's own logits: candidate list = top-8
  (logprob diff 1.4e-6); -100 ban (the runner-up wins, logprob over the rest), +100 forces, additive bias per step;
  presence / frequency / repetition / all three: 40 steps each = the argmax of the penalised logits, logprob diff ~1e-6; the
  two-piece prefill + re-arm = the one-piece run (28 of 32 tokens, then a rounding fork); min_p keeps 60 of 60 draws above the
  line (control: 9 of 60 fall below it without); seeds; the MTP loop applies a ban and penalties (greedy: the banned token never
  appears), refuses logprobs, and equals the plain greedy run once the knobs are cleared (32 checks, `sampler-1016-1936-model.txt`).
- `tests/mtp/test_mtp_knobs.py` (whole model, `sampler-1016-mtp-knobs.txt`): MTP exactness with the knobs, see 11b.
- `tests/server/test_sampling_units.py` (CPU: parsing, token bytes, logprob entries, the Session plumbing on the mock, the
  Anthropic converter and event builder), `tests/server/test_sampling.py` (HTTP, mock and the whole model: validation
  errors, n, logprobs streamed and not, bans, the path of each knob, seeds, completions, messages), run in
  `tests/server/run_mock.sh` (mock) and `tests/server/e2e_sampler_job.sh` (one with_lock job on port 8142; `E2E_STEPS` picks parts). Whole model
  (19:36 / 19:45, port 8142, ctx 32768, K5-T6 pack, `tests/server/results/sampler-1016-19*`): 84 HTTP checks + the two-slot test
  pass. On the real model: a ban of `The` makes the first token `End`, +100 selects it every time, presence 2 + frequency 1
  take a 60-echo answer to 10 echoes, repetition penalty 1.5 does not move a confident greedy 'echo' loop (10 does), `min_p` 1 at
  temperature 1 = greedy, 3 choices at temperature 1 differ, a seed repeats its text (and 54 of 54 characters match with the
  prefix reused), streamed logprobs = non-streamed (tokens and values), `/v1/messages` tool use / forced tool / thinking /
  streaming events / stop_sequences; `test_sampling_slots.py`: a knob request next to a plain one on two slots, each equal to its
  solo run in 3 rounds, nothing left on the slot afterwards, a 7025-token prompt (several prefill pieces) with penalties and logprobs.
- Regression (after the merge with master, IJAH_GOLDEN_PACK = the K5-a16 pack the layer goldens were made from): `make test-forward`'s
  `tests/forward_gpu.sh` passes (test_fn_forward incl. the sampler section updated for the wider SampleParams, test_fn_layer);
  `tests/mtp/test_verify_forced.py` OK (KLD 1.2-1.9e-4), `build/mtp-accept-test` passes, `test_mtp_knobs.py` OK again after the merge.
- Several models in one process (section 10): the knobs, logprobs, `n`, `/v1/completions` and `/v1/messages` work per model id
  (`model` routes; `/v1/messages` takes the model name too). The sampler buffers (`bias`, `cnt`, `top`, the mask) belong to each
  model instance, so a knob request on one model leaves nothing on the other (also with `--share-state`: the instances share KV
  and recurrent state, not sampler buffers). Logprob token texts come from each model's own tokenizer. `tests/server/test_sampling.py
  --models A,B` runs the whole test once per id and then the cross-model checks (a knob request on A, then a plain one on B and back; both
  at once). Whole model, `--models all` at ctx 32768, port 8142: both ids pass (`sampler-1016-2011-models.txt`); mock suite
  (`run_mock.sh`) runs it on the two-model and the `--share-state` servers.

## 12. Robustness fixes from the review (lab-fixsrv, 17 Oct 2026)

Branch `lab/fixsrv`; tests `tests/server/test_robust_http.py` (HTTP, mock engine; run_mock.sh starts it with 10 ms/token,
prefill 20K tok/s, `NE_SRV_KEEPALIVE_S=0.2`, 1 MB body cap, 2 s socket timeout), `test_worker_robust.py`,
`test_special_text.py`, `test_limits.py`, `test_with_lock.sh`, and cases in `test_units.py`.

- **Disconnects.** A request thread never blocks on its job for good: it polls the job's queue every 0.25 s
  (`POLL_S`) and looks at its client socket (`select` + `MSG_PEEK`: EOF or reset = gone). `_complete` and `_stream`
  cancel their jobs in a `finally` (any exit: reply, error, client gone, an exception). Mock: slot free 0.03 s after the
  client left a non-streamed reply (was 1.5-5.6 s, i.e. the end of the reply); `n > 1` starts no further choice.
  The cancel is also checked **between prefill pieces** (`Session._run`): a client that leaves during a long prefill
  costs one piece (0.17 s on the mock at 4097 tokens / 20K tok/s), not the whole prompt; the pieces fed stay a valid
  prefix for the next request (`stats.cancelled = "prefill"`).
- **Keepalives.** A stream that sent nothing for `NE_SRV_KEEPALIVE_S` (10 s: a long prefill, the arguments of a tool call
  that are held back until each parameter is complete) gets `: keepalive` (an SSE comment; `/v1/messages`: a `ping`
  event); a dead socket then fails the write.
- **Cut-off tool calls.** A call is *closed* when its `</function>` was seen (or `</tool_call>` after complete
  parameters). One that ends unclosed (length, EOS or a stop string mid-parameter; malformed markup) is not a tool call:
  `finish_reason` is `length` / `stop` (Anthropic `max_tokens` / `end_turn`), a non-streamed reply drops it and keeps its
  raw text as `content` (delta `tool_abort`). A streamed reply cannot retract what it sent (the name and the finished
  parameters): its arguments stay unterminated and the finish reason says why. Complete calls before a cut one are kept.
- **Special-token text is text.** `Template.render` swaps every added-token string in the messages, tools and template
  kwargs (`<|im_end|>`, `<|image_pad|>`, `<tool_call>`, `<think>` ...) for one private-use guard character (a block absent from
  the request); `render_ids` encodes the template's own markup as special tokens and the restored literals as plain BPE
  (a copy of the tokenizer without added tokens). A tool result showing `<|im_end|>` no longer ends the turn, a literal
  `<|image_pad|>` is not an image placeholder (it used to be a 400). `expand_prompt` accepts only
  `<|vision_start|><|image_pad|><|vision_end|>`. A request without such text encodes exactly as before (checked against the
  whole-text encode; the incremental encode keys its cache on the guard block). `/v1/completions` raw prompts keep their
  special-token strings as tokens.
- **Engine thread.** Admission is per job (that job gets an error), the loop catches anything else, fails the requests in
  flight, resets their slots and goes on. 20 scheduler errors in a row end the thread: `/health` answers **503**
  `{"status": "error", "reason": ...}`, queued requests get an error, and the process exits 70 so the supervisor
  restarts it. `/health` also reports a pass of the loop stuck longer than `NE_SRV_STALL_S` (900 s). (`Worker._stop`
  shadowed `Thread._stop`: `is_alive()` raised once the thread had ended.)
- **After a finished reply** the SSD write-through (`kv_save` can raise `MemoryError`) and the budget check are guarded:
  the reply stays a success, the failure is logged, the mirror catches up at the next turn.
- **Limits.** `Content-Length` required (411), a non-negative integer (400), at most `NE_SRV_MAX_BODY_MB` = 64 (413, also
  answered before the body for `Expect: 100-continue`); chunked bodies 411; the body is read before the 401 so keep-alive
  stays in step; sockets time out after `NE_SRV_SOCKET_TIMEOUT` = 120 s. Local image paths: only under `$HOME` or
  `NE_VISION_ROOTS` (os.pathsep list; empty = none), symlinks resolved; images over `NE_VISION_MAX_INPUT_PIXELS` = 40
  megapixels are refused from the header. Shutdown closes the prefix cache of every model.
- **`tools/with_lock.sh`.** `SWAP_WINDOW_S=N` makes the swap watchdog count the swapouts of the last N seconds (`tools/serve.sh`
  passes 60, `NE_SWAP_WINDOW_S`; batch jobs keep the total since the start) and `MIN_FREE_SAMPLES=N` needs N low-RAM
  samples in a row (serve.sh: 3). A lock whose holder is gone is cleared before waiting (the lock script's `clear-stale`;
  for our own owner also a pid that is now a process younger than the lock), with a log line.

**Whole-model check** (`tests/server/robust_job.sh`, one with_lock job, test server on 8144, ctx 32768, K5-T6 pack, this
branch's `build/libengine.dylib`, 17 Oct 12:21; results `tests/server/results/robust-1017-1221-*`): non-streamed and
streamed answers, a streamed tool call (`get_weather {"city": "Paris", "days": 3}`, `finish_reason` `tool_calls`), a client
leaving mid-generation: slot free after **0.10 s** (non-streamed) and **0.05 s** (streamed), a client leaving during a
17K-token prefill: free after **0.3 s**, the next identical request reused the 4096 tokens already fed. 11 of 11 checks.

## 13. A request naming a model this server does not serve (lab-dflash, 19 Oct 2026)

`App.model_for(name)` picks the model a request names: exact id, else case-insensitive, no name = the default. Until
today an **unknown** name was an error only when several models were served; with one model served it was answered by
the default model, HTTP 200. That leniency cost a day of measurement credibility:

- `~/.pi/agent/models.json` registered `Swift-1.5-Flash-Next-native-dflash` (a drafter id) against `127.0.0.1:8140`.
  The server on :8140 was the master build serving `Swift-1.5-Flash-Next-native`, and the master lib has **0** `dfl`
  symbols (`nm -gU`) - the drafter was not even compiled in.
- Every pi request naming the drafter id got **200** from the plain MTP path: `{"model": "Swift-1.5-Flash-Next-native",
  "timings": {"path": "mtp"}}`. Nothing - status code, body, pi's `PI_MODEL`, a chat that answered coherently - said so.
  `PI_MODEL` records what the *client asked for*, never what the server used.
- Consequence: a "DFlash" benchmark run on that server would have measured the control arm under the drafter's label.
  The +6% figure itself is safe (it came from `bench_draft`/`session.sh` on a drafter build), but every number produced
  through pi before this fix is void.

**Now:** an unknown name is a 400 whose message lists the served ids, in the single-model case too. Ask for the drafter
id on a plain server and you get `unknown model 'Swift-1.5-Flash-Next-native-dflash'; this server has
Swift-1.5-Flash-Next-native`, and the reverse (a plain id on a drafter server) too - a drafter run cannot quietly be a
control, which is the property the measurement needed. `GET /v1/models` is the authoritative list; clients should read
their own id from it (`tests/server/test_client.py:134`, and `tests/server/robust_real.py`, which sent `"model": "x"`).

`--allow-any-model` restores the old behaviour for a **single-model** server only (a placeholder name gets the default,
warned once per name in serve.log); it exists for the suites that send `"model": "x"` (`run_mock.sh`'s robust-http and
scheduler launches) and for legacy clients. It never applies when several models are served - there the name is the
only thing deciding which model answers. Tests: `tests/server/test_model_id.py` (13 checks, no engine, no GPU, no
port; runs in `run_mock.sh`). Related: docs/mtp.md 13.7.

## 14. DFlash2 by default (release, 23 Oct 2026)

The user's decision of 23 Oct: the plain model ids use the DFlash2 block drafter (docs/mtp.md 13: Inco's 0.6B drafter,
the exact sequential gate `--dflash-gate 1`, block drafts only up to `--dflash-max-ctx 8192` context tokens; the MTP
head drafts the rest).

- `server/models.json`: each plain id has a `drafter` entry (candidate checkpoint dirs, first with `config.json` +
  `model.safetensors` wins; `--drafter-path` overrides). The `-dflash` ids are now `alias_of` entries.
- `--drafter auto` (default; env `NE_DRAFTER`): the drafter when its checkpoint is present; otherwise the MTP head, with
  one log line per model that names the download (`ijah download drafter`, or `hf download incoai/Qwen3.8-Flash-Next-DFlash2
  --revision a22a1744e85c --local-dir ~/.cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2`). `--drafter dflash2`: refuse
  to start without it. `--drafter none` (or `--mtp 0`): the MTP head only (the control arm of a benchmark).
- A `-dflash` id at start (`--model-id X-dflash`) = `X` with the drafter required (missing: SystemExit before any
  load; with `--drafter none`: refused). The process serves the plain id; requests may name either. A request for
  `X-dflash` on a server that runs `X` without the drafter is a 400 that says so and how to get it, also with
  `--allow-any-model`, so the 19 Oct property (section 13: a drafter request is never answered by the control) holds.
  The reverse check (a plain id refused on a drafter server) is gone: the plain id is the drafter run now.
- `--models all` = the plain ids (each with its drafter when present); `--with-drafters` is accepted and does nothing.
- `GET /v1/models` adds `"drafter": "dflash2" | "mtp"` per id; `--list-models` prints the drafter each id would load.
- Memory: each served model holds its own drafter (weights and buffers; the checkpoint is 1.2 GB).

Tests: `tests/server/test_model_id.py` (alias, fallback, force-MTP, refusals; CPU, in `run_mock.sh`).
