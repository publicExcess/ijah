# Changelog

## v0.2.0 (unreleased)

**DFlash2 by default**
- Every model id uses Inco's DFlash2 block drafter when it is downloaded (`ijah download drafter`, into
  `~/.cache/ijah/drafters`); without it the model's MTP head drafts and the server logs once how to get it.
  `--drafter none` (or `NE_DRAFTER=none`) forces the MTP head; `--drafter dflash2` refuses to start without it.
  The `-dflash` model ids are aliases of the plain ids that require the drafter. `GET /v1/models` reports
  `"drafter": "dflash2" | "mtp"` per id.
- Exact verify gate: whether a draft is verified no longer depends on the drafted token (the v0.1 port's gate could
  shift the sampled distribution); checked with a 2-million-sample host test and a GPU distribution check.
- Block drafts stop past 8,192 context tokens (`--dflash-max-ctx`): +7-15% decode at short context, parity at 8-32K.
- Opt-in lookahead gate, `--dflash-gate 2`.

**Engine**
- Grouped routed-expert kernels for multi-row steps (MTP / DFlash2 verify): each expert's weights are read once per
  cycle instead of once per row; bit-identical output.

**Tools**
- `tools/ijah`: `serve start|run|stop|status|log|restart`, `download base|swift|base-4bit|drafter|ngram|all` (pinned
  revisions, disk check, the n-gram table hard-linked into every pack), `link`, `models`, `doctor`.
  `server/serve_config.py` holds the layout the server and the CLI share.
- `tools/serve.sh run`: the server in the foreground, for launchd and `brew services`.
- Homebrew tap: `brew install publicExcess/ijah/ijah`, `brew services start ijah`.
- `tools/gguf_logits.cpp` + `tools/kld_gguf.sh`: the KLD of a GGUF from llama.cpp's full-vocabulary logits, scored by
  `tools/kld_teacher.py --logits-dir` exactly like the engine.
- `tools/with_lock.sh`: the RAM / swap watchdog stops the job's whole process tree.

**Models**
- New model id `Qwen3.8-Flash-Next-native-4bit` for the published 4-bit pack
  (SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah-4bit); the T6 pack stays the preferred one. `--models all` leaves it out.

**Fixes**
- `--models all` serves the base and Swift-1.5 models, each with its drafter; `--with-drafters` is no longer needed.
- A request for an id the server does not serve, or for a `-dflash` id where the drafter is not loaded, is a 400
  that says what to do.

**Docs**: README rewritten (quality and speed comparison with Splash 1.4 and oMLX, the `ijah` quick start);
docs/server.md 14, docs/mtp.md 12.8 and 13.9-13.10.

## v0.1 (19 Oct 2026)
- First public release; DFlash2 drafter ported as an opt-in feature.
