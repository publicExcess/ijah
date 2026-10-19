#!/bin/zsh
# tests/cpu_tests.sh   (make test-cpu) the CPU tests: no GPU, no model weights, no lock.
# Most of them read the pack's tokenizer and chat template; the small files of the two packs are enough:
#   for r in Qwen3.8-Flash-Next-EXL3-Ijah Swift-1.5-Flash-Next-EXL3-Ijah; do
#     hf download SiliconSpecies/$r --include '*.json' '*.jinja' --local-dir ${IJAH_PACKS:-~/.cache/ijah/packs}/$r; done
# With them: tests/server/run_mock.sh (chat layer, sampling, grammar, prefix cache, the HTTP API on the mock engine,
# two models, SSD tier, supervisor, with_lock) + the vision preprocessing units. Without them: the tests that need no
# tokenizer run and the rest report SKIP.
set -u
ROOT=${0:A:h:h}
source $ROOT/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
cd $ROOT
have() { [ -f $1/tokenizer.json ] && [ -f $1/chat_template.jinja ] && [ -f $1/config.json ] && [ -f $1/model.safetensors.index.json ]; }
rc=0
if have $IJAH_PACKS/Qwen3.8-Flash-Next-EXL3-Ijah && have $IJAH_PACKS/Swift-1.5-Flash-Next-EXL3-Ijah; then
  echo "--- tests/server/run_mock.sh (packs' tokenizer + templates from $IJAH_PACKS)"
  tests/server/run_mock.sh || rc=1
else
  echo "SKIP tests/server/run_mock.sh: the small files of both packs are not in $IJAH_PACKS (see the top of this script)"
  for t in tests/server/test_units.py tests/server/test_sampling_units.py tests/server/test_special_text.py \
           tests/server/test_tokcache.py tests/server/test_grammar.py tests/server/test_worker_robust.py; do
    $PY $t | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1
  done
  echo "--- with_lock.sh: stale locks, swap rate window (fake vm_stat), low-RAM samples"
  tests/server/test_with_lock.sh | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1
fi
echo "--- vision preprocessing / placeholders / prefix reuse with images (mock backend)"
$PY tests/vision/test_units.py | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1
[ $rc = 0 ] && echo "test-cpu: ALL PASS" || echo "test-cpu: FAIL"
exit $rc
