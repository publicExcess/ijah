#!/bin/zsh
# tests/server/e2e_job.sh   the whole-model server test; ONE with_lock job (starts the server and stops it inside):
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-server tools/with_lock.sh "server e2e (~83 GB)" HH:MM -- tests/server/e2e_job.sh
# Env: NE_LIB (default build/libengine.dylib), E2E_CTX (32768), E2E_BENCH (tokens, 1024), E2E_DIRECT=1 (also time
# tools/generate.py on the bench prompt, with and without MTP, after the server is gone), E2E_MODEL (a model id of
# server/models.json, e.g. Swift-1.5-Flash-Next-native: the server resolves its pack, the client checks the id, the
# direct runs use that pack). Results: tests/server/results/.
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${E2E_PORT:-8140}
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
PIDF=$(mktemp -t ne-e2e-pid)
SP=
stop_server() {
  [ -n "$SP" ] || return 0
  kill -TERM $SP 2>/dev/null
  for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill -0 $SP 2>/dev/null && kill -KILL $SP
  wait $SP 2>/dev/null
  echo "server stopped (pid $SP)"
  SP=
}
trap 'stop_server; rm -f $PIDF' EXIT INT TERM
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT busy"; exit 1; }
free_gb() { local t f; t=$(sysctl -n hw.memsize); f=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); echo $(( t / 1073741824 * ${f:-0} / 100 )); }
F0=$(free_gb)
tools/serve.sh keygen
tools/serve.sh build
SHIM=$ROOT/build-server/${NE_LIB:h:t}/libne_serve.dylib
SARGS=() CARGS=() DARGS=()
[ -f $SHIM ] && SARGS=(--shim $SHIM)
if [ -n "${E2E_MODEL:-}" ]; then
  SARGS+=(--model-id $E2E_MODEL); CARGS=(--expect-model $E2E_MODEL)
  DARGS=(--pack "$($PY server/ne_server.py --list-models | sed -n "s|^$E2E_MODEL: \([^ ]*\) .*|\1|p")")
  echo "model $E2E_MODEL -> ${DARGS[2]}"
fi
$PY server/ne_server.py --port $PORT --ctx ${E2E_CTX:-32768} --lib $NE_LIB --pid-file $PIDF $SARGS \
    2> $OUT/e2e-$STAMP-server.log &
SP=$!
for i in {1..150}; do
  curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
  kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $OUT/e2e-$STAMP-server.log; exit 1; }
  sleep 2
done
grep "engine ready" $OUT/e2e-$STAMP-server.log
echo "free RAM: $F0 GB before, $(free_gb) GB with the server up (ctx ${E2E_CTX:-32768})"
$PY tests/server/test_client.py --url http://127.0.0.1:$PORT --bench ${E2E_BENCH:-1024} $CARGS ${=E2E_CLIENT_ARGS:-} --json $OUT/e2e-$STAMP.json \
    2>&1 | tee $OUT/e2e-$STAMP-client.txt
rc=${pipestatus[1]}
stop_server
grep -v "^\s*$" $OUT/e2e-$STAMP-server.log | tail -25
if [ "${E2E_DIRECT:-1}" = 1 ]; then
  Q="Write a detailed, long essay about the history of the printing press, its technology and its effects on Europe."
  for m in "" "--mtp 2"; do
    echo "--- direct tools/generate.py ${m:-no MTP}"
    $PY tools/generate.py "$Q" --max ${E2E_BENCH:-1024} --seed 7 $DARGS ${=m} 2>&1 | grep -- '^--- [0-9]'   # zsh MULTIOS: no 2>&1 >/dev/null
  done 2>&1 | tee $OUT/e2e-$STAMP-direct.txt
fi
exit $rc
