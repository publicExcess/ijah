#!/bin/zsh
# tools/setup.sh [PYTHON]   create .venv (python3.11 or newer) and install requirements.txt (make venv calls this).
# Then: .venv/bin/python is what every script uses (override with IJAH_PY). Optional MLX reference tools:
#   .venv/bin/python -m pip install -r requirements-ref.txt
set -eu
ROOT=${0:A:h:h}
cd $ROOT
PY=${1:-}
if [ -z "$PY" ] || ! command -v $PY >/dev/null; then
  PY=
  for c in python3.13 python3.12 python3.11 python3; do
    command -v $c >/dev/null && $c -c 'import sys; sys.exit(sys.version_info < (3, 11))' && { PY=$c; break; }
  done
fi
[ -n "$PY" ] || { echo "setup: no python3.11+ found (pass one: tools/setup.sh /path/to/python3.12)"; exit 1; }
$PY -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "setup: $PY is older than 3.11"; exit 1; }
echo "setup: .venv from $PY ($($PY --version))"
$PY -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
touch .venv/bin/python
echo "setup: done (.venv/bin/python)"
