#!/bin/zsh
# tests/server/e2e_structured_job.sh   the whole-model structured-output test; ONE with_lock job (starts and stops the test
# servers inside, on port 8141 -- never 8140):
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-api tools/with_lock.sh "structured output e2e (~83 GB)" HH:MM -- tests/server/e2e_structured_job.sh
# Env: NE_LIB (default build/libengine.dylib; this tree's build), BASE_ROOT (a checkout of the commit before the mask
# patch, built, with the same IJAH_PACK: its build/libengine.dylib and server run the baseline half of the speed A/B;
# unset = skip the A/B),
# E2E_CTX (32768), E2E_PORT (8141). A throw-away key in a temp dir: ~/.config/ijah/api-key is not touched or read.
# Results: tests/server/results/structured-*.{txt,json}.
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${E2E_PORT:-8141}
[ $PORT = 8140 ] && { echo "refusing port 8140"; exit 1; }
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t ne-structured)
( umask 077; $PY -c 'import secrets; print("ne-" + secrets.token_urlsafe(32), end="")' > $T/key ); chmod 600 $T/key
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
trap 'stop_server; rm -rf $T' EXIT INT TERM
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT busy"; exit 1; }
free_gb() { local t f; t=$(sysctl -n hw.memsize); f=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); echo $(( t / 1073741824 * ${f:-0} / 100 )); }

start_server() {   # root lib log
  local r=$1 lib=$2 log=$3 shim=()
  ( cd $r && NE_LIB=$lib tools/serve.sh build ) | tail -2
  local s=$r/build-server/${lib:h:t}/libne_serve.dylib
  [ -f $s ] && shim=(--shim $s)
  ( cd $r && exec $PY server/ne_server.py --port $PORT --ctx ${E2E_CTX:-32768} --key-file $T/key --lib $lib $shim 2> $log ) &
  SP=$!
  for i in {1..150}; do
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
    kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $log; return 1; }
    sleep 2
  done
  grep -E "engine ready|structured output" $log
}

echo "free RAM $(free_gb) GB before; swap: $(sysctl -n vm.swapusage)"
rc=0
echo "=== sampler mask on the engine (no server)"
$PY tests/server/mask_check.py 2>&1 | tee $OUT/structured-$STAMP-mask.txt
[ ${pipestatus[1]} = 0 ] || { echo "mask_check failed; not starting the servers"; exit 1; }
if [ -n "${BASE_ROOT:-}" ]; then
  echo "=== baseline (before the mask patch): $BASE_ROOT"
  start_server $BASE_ROOT $BASE_ROOT/build/libengine.dylib $OUT/structured-$STAMP-base-server.log || exit 1
  $PY tests/server/speed_ab.py --url http://127.0.0.1:$PORT --key-file $T/key --label baseline 2>&1 | tee -a $OUT/structured-$STAMP-speed.txt
  stop_server
  echo "--- GPU step time, baseline library"
  NE_LIB=$BASE_ROOT/build/libengine.dylib $PY tests/server/step_ab.py 2>&1 | grep STEP_AB | tee -a $OUT/structured-$STAMP-speed.txt
fi
echo "=== this build: $NE_LIB"
start_server $ROOT $NE_LIB $OUT/structured-$STAMP-server.log || exit 1
echo "free RAM $(free_gb) GB with the server up"
$PY tests/server/test_structured.py --url http://127.0.0.1:$PORT --key-file $T/key --bench ${E2E_BENCH:-768} \
    --json $OUT/structured-$STAMP.json 2>&1 | tee $OUT/structured-$STAMP-client.txt
rc=${pipestatus[1]}
$PY tests/server/speed_ab.py --url http://127.0.0.1:$PORT --key-file $T/key --label this-build 2>&1 | tee -a $OUT/structured-$STAMP-speed.txt
stop_server
echo "--- GPU step time, this build (mask off / all-ones mask)"
$PY tests/server/step_ab.py 2>&1 | grep STEP_AB | tee -a $OUT/structured-$STAMP-speed.txt
grep -v "^\s*$" $OUT/structured-$STAMP-server.log | tail -30
echo "swap after: $(sysctl -n vm.swapusage)"
exit $rc
