# lab-attn tests (included from kernels/attn/attn.mk). Not part of `make test` (run them when touching kernels/attn):
#   make test-attn        tests/attn/test_qsa.py against build/libengine.dylib (~3 s, light GPU work, no machine lock)
#   make bench-attn       tools/attn_bench.py under the machine lock (GPU timing)
ATTN_PY ?= $(IJAH_PY) -I -B
.PHONY: test-attn bench-attn
test-attn: all tests/golden/READY
	$(ATTN_PY) tests/attn/test_qsa.py --lib $(LIB)
bench-attn: all
	LOCK_OWNER=lab-attn LOCK_WAIT=30 tools/with_lock.sh "bench-attn (lab-attn)" $$(date -v+15M +%H:%M) -- $(ATTN_PY) tools/attn_bench.py --lib $(LIB) --dispatches
