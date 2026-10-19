# lab-ref tests (included by the Makefile). Reference / goldens: docs/reference.md.
#   make test-ref     numpy op references vs the MLX goldens + compare.py self-test + engine.py binding check (CPU)
#   make test-layer-ref   engine Flash-Next layer (ne_fn_layer_*) vs goldens, chain + per stage (GPU, machine lock)
#   make goldens      regenerate tests/golden/data with the MLX reference (GPU, machine lock, ~1 min, ~8 GiB)
REF_PY ?= $(IJAH_PY) -I -B
REF_CASE ?= L00_decode1
REF_UNTIL = $(shell date -v+15M +%H:%M)

.PHONY: test-ref test-layer-ref goldens test-forward test-logits ref-logits
#   make test-forward  lab-forward: n-gram ids, PLE layer, head, sampler, every layer golden case (GPU, lock)
#   make test-logits   whole model (81 GB wired) teacher-forced vs the saved MLX reference logits: KLD, top-1 (lock)
#   make ref-logits    save the MLX whole-model reference logits (fn_exl3, ~76 GB, lock; never next to the engine)

tests/golden/READY:
	@echo "no goldens: run 'make goldens' (MLX reference, machine lock)"; exit 1

test-ref: tests/golden/READY
	$(REF_PY) tools/compare.py --self-test
	$(REF_PY) tools/check_opref.py
	@if [ -f $(BUILD)/libengine.dylib ]; then $(REF_PY) tools/engine.py; else echo "(engine.py check skipped: no libengine.dylib)"; fi

test-layer-ref: all tests/golden/READY
	tools/with_lock.sh "test-layer-ref $(REF_CASE)" $(REF_UNTIL) -- $(REF_PY) tests/test_fn_layer.py $(REF_CASE)

goldens:
	mkdir -p tests/golden/data
	tools/with_lock.sh "goldens (lab-ref)" $(REF_UNTIL) -- $(REF_PY) tools/gen_goldens.py --layers 0,1,3 --long 2304 --head
	touch tests/golden/READY

# `make test` also runs the cheap CPU checks (TEST_CMDS entries must be single words: the Makefile splits on spaces) when goldens exist (op refs ~30 s; no GPU, no lock).
ifneq ($(wildcard tests/golden/READY),)
TEST_CMDS += tests/ref_quick.sh tests/forward_quick.sh
endif

test-forward: all tests/golden/READY
	LOCK_OWNER=lab-forward LOCK_WAIT=30 tools/with_lock.sh "test-forward (small)" $(REF_UNTIL) -- tests/forward_gpu.sh

test-logits: all
	LOCK_OWNER=lab-forward LOCK_WAIT=30 MIN_FREE_PCT=3 MAX_SWAPOUT_MB=6000 tools/with_lock.sh \
	  "test-logits: engine whole model vs MLX reference (81 GB)" $(REF_UNTIL) -- $(REF_PY) tools/compare_logits.py --prefill4

ref-logits:
	LOCK_OWNER=lab-forward LOCK_WAIT=30 MIN_FREE_PCT=3 MAX_SWAPOUT_MB=6000 tools/with_lock.sh \
	  "ref-logits: MLX whole model (~76 GB, up to 30 min)" $(shell date -v+30M +%H:%M) -- $(REF_PY) tools/ref_logits.py
# lab-integrate (whole model, ~83 GB, lock): API checks (snapshot / restore, MTP callback, prefill routing); the
# decode path's floor (1-row vs 4-row vs chunked prefill, rounding ON vs f32; docs/forward.md section 2c); TTFT
.PHONY: test-fn-api forward-floor bench-ttft
test-fn-api: all
	NEED_GB=85 LOCK_OWNER=lab-forward LOCK_WAIT=30 tools/with_lock.sh "test-fn-api (83 GB)" $(REF_UNTIL) -- $(REF_PY) tests/test_fn_api.py
forward-floor: all
	NEED_GB=85 LOCK_OWNER=lab-forward LOCK_WAIT=30 tools/with_lock.sh "forward-floor (81 GB)" $(REF_UNTIL) -- $(REF_PY) tests/forward_floor.py
bench-ttft: all
	NEED_GB=85 LOCK_OWNER=lab-forward LOCK_WAIT=30 tools/with_lock.sh "bench-ttft (83 GB)" $(REF_UNTIL) -- $(REF_PY) tools/bench_ttft.py

-include tests/quality/quality.mk
