# lab-quality (included from tests/tests.mk). Method and numbers: docs/quality.md.
#   make test-quality       engine (generation default: f32, bf16 KV, QSA) vs the bf16 teacher on sets A, B, C:
#                           whole model (81 GB wired), machine lock, ~50 min; then the limits check
#   make test-quality QUALITY_SETS=A,B          (~10 min)    QUALITY_ARGS="--round ref" / "--kv kv8" for the arms
#   make quality-report     every result json in tests/quality/results next to fn_exl3's (CPU, seconds)
QUALITY_PY   ?= $(IJAH_PY) -I -B
QUALITY_SETS ?= A,B,C
QUALITY_ARGS ?=
QUALITY_NAME ?= test-quality
.PHONY: test-quality quality-report

test-quality: $(LIB) $(METALLIB)
	NEED_GB=85 LOCK_OWNER=lab-quality LOCK_WAIT=30 MIN_FREE_PCT=3 MAX_SWAPOUT_MB=6000 tools/with_lock.sh \
	  "test-quality: engine whole model vs bf16 teacher (81 GB)" $(shell date -v+75M +%H:%M) -- \
	  $(QUALITY_PY) tools/kld_teacher.py --name $(QUALITY_NAME) --sets $(QUALITY_SETS) $(QUALITY_ARGS)
	$(QUALITY_PY) tests/quality/check.py tests/quality/results/kld-$(QUALITY_NAME).json

quality-report:
	$(QUALITY_PY) tests/quality/check.py --report-only tests/quality/results/kld-*.json
