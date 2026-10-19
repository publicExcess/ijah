#!/bin/zsh
# Cheap lab-ref checks for `make test` (CPU, no lock): compare.py self-test + op references on two goldens.
set -e
source ${0:A:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
cd ${0:A:h}/..
$PY tools/compare.py --self-test
$PY tools/check_opref.py --no-experts L00_decode1 L03_verify4 | tail -1
