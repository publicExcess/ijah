#!/bin/zsh
# tests/server/robust_job.sh   the whole-model check of the review fixes; ONE with_lock job (starts a test server on
# port 8144 with a throw-away key and stops it inside), < 10 min:
#   LOCK_OWNER=lab-fixsrv NEED_GB=85 LOCK_WAIT=30 tools/with_lock.sh "fixsrv server check (~83 GB)" HH:MM -- tests/server/robust_job.sh
# Env: NE_LIB (default build/libengine.dylib), ROBUST_CTX (32768). Results: tests/server/results/robust-*.
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${ROBUST_PORT:-8144}
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t ne-robust)
( umask 077; $PY -c 'import secrets; print(secrets.token_urlsafe(32), end="")' > $T/key ); chmod 600 $T/key
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
$PY server/ne_server.py --port $PORT --ctx ${ROBUST_CTX:-32768} --lib $NE_LIB --key-file $T/key --cache-dir $T/pc \
    2> $OUT/robust-$STAMP-server.log &
SP=$!
for i in {1..150}; do
  curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
  kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $OUT/robust-$STAMP-server.log; exit 1; }
  sleep 2
done
grep "engine ready" $OUT/robust-$STAMP-server.log
$PY tests/server/robust_real.py --url http://127.0.0.1:$PORT --key-file $T/key 2>&1 | tee $OUT/robust-$STAMP-client.txt
rc=${pipestatus[1]}
stop_server
grep -v "^\s*$" $OUT/robust-$STAMP-server.log | tail -25
exit $rc
