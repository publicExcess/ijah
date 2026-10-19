#!/bin/zsh
# tests/vision/whole_job.sh OUT_DIR   lab-vision whole-model validation, ONE with_lock job (models one after another):
#   1. MLX reference (fn_exl3, K5 pack) on the kld-chart image prompt (+ a 1-D-position control)
#   2. the engine on the same prompt (K5): reference features / own tower / 1-D positions
#   3. text: tools/compare_logits.py with this build; with BASE_LIB=<a master build's libengine.dylib> the same with
#      that library too, and the engine logits compared bit for bit (the image hooks must not change text)
#   4. the tower's sensitivity floor (tests/vision/tower_floor.py)   5. server end to end (T6; tests/vision/e2e_job.sh)
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-vision tools/with_lock.sh "vision whole-model checks (~83 GB, ~30 min)" \
#       HH:MM -- tests/vision/whole_job.sh OUT_DIR
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
OUT=${1:?out dir}
mkdir -p $OUT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
K5=$IJAH_GOLDEN_PACK   # the pack the MLX reference reads
CHART=${IJAH_TEST_CHART:-$PWD/tests/vision/assets/flow.png}   # any bar-chart PNG
if [ "${SKIP_LM:-0}" != 1 ]; then
  [ "${SKIP_REF:-0}" = 1 ] || $PY tests/vision/lm_image_ref.py $CHART "List every bar with its exact value." $OUT/chart --pack $K5 --n 16 2>&1 | tail -20
  $PY tests/vision/lm_image_engine.py $OUT/chart --pack $K5 2>&1 | tail -12
fi
if [ "${SKIP_TEXT:-0}" != 1 ]; then
  $PY tools/compare_logits.py --prefill4 --save $OUT/text 2>&1 | tail -3
  if [ -n "${BASE_LIB:-}" ]; then
    NE_LIB=$BASE_LIB $PY tools/compare_logits.py --prefill4 --save $OUT/text_base 2>&1 | tail -1
    $PY -c "
import sys; sys.path.insert(0, '$ROOT/tools'); import refenv; refenv.add_paths()
import numpy as np, glob, os
for f in sorted(glob.glob('$OUT/text/*.engine.npy')):
    b = f.replace('/text/', '/text_base/')
    a, c = np.load(f), np.load(b)
    print(os.path.basename(f), 'bit-identical to the base build' if np.array_equal(a, c) else f'DIFFERS ({(a != c).mean():.4f})')"
  fi
fi
[ "${SKIP_FLOOR:-0}" = 1 ] || { $PY tests/vision/vision_ref.py $CHART $OUT/tower >/dev/null 2>&1 && $PY tests/vision/tower_floor.py $OUT/tower 2>&1 | tail -3; }
[ "${SKIP_E2E:-0}" = 1 ] || tests/vision/e2e_job.sh
