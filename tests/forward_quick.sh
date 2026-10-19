#!/bin/zsh
# lab-forward checks for `make test`: the n-gram id/gather check always (CPU); the GPU parts (PLE layer, head,
# sampler, every layer golden case in chain mode) only if the machine lock is free right now (skipped otherwise:
# `make test-forward` waits for it). The whole-model logits check needs ~81 GB and the saved MLX reference:
# `make test-logits`.
set -e
source ${0:A:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
cd ${0:A:h}/..
PACK=$IJAH_GOLDEN_PACK   # the pack the layer goldens were made from
ijah_need_pack forward_quick $PACK
$PY tests/test_fn_forward.py --pack $PACK --only ngram | tail -1
if [ -n "${NE_NO_LOCK:-}" ]; then   # inside a tools/with_lock.sh job: we hold the lock, run the GPU checks now
  exec tests/forward_gpu.sh
fi
if [ -n "${IJAH_LOCK_SCRIPT:-}" ] && [ -x "$IJAH_LOCK_SCRIPT" ] && [ "$($IJAH_LOCK_SCRIPT status | head -1)" != free ]; then
  echo "forward_quick: machine lock held, GPU checks skipped (make test-forward)"; exit 0
fi
LOCK_OWNER=lab-forward tools/with_lock.sh "make test: forward GPU checks (small)" $(date -v+15M +%H:%M) -- \
  tests/forward_gpu.sh
