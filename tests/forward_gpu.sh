#!/bin/zsh
# GPU part of the lab-forward checks (run under the machine lock: tests/forward_quick.sh or make test-forward)
set -e
source ${0:A:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
cd ${0:A:h}/..
PACK=$IJAH_GOLDEN_PACK   # the pack the layer goldens were made from
ijah_need_pack forward_gpu $PACK
$PY tests/test_fn_forward.py --pack $PACK --only ple,head,sampler | tail -1
$PY tests/test_fn_layer.py --pack $PACK L00_decode1 L00_verify4 L01_decode1 L01_verify4 L03_decode1 L03_verify4 \
  L03_long_decode1 L03_long_verify4 | tail -1
