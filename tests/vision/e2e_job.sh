#!/bin/zsh
# tests/vision/e2e_job.sh   lab-vision whole-model end to end: ONE with_lock job that starts a server on port 8148
# (T6 pack, ctx 32768, this worktree's build), runs tests/vision/e2e_client.py, stops the server.
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-vision tools/with_lock.sh "vision e2e (~83 GB)" HH:MM -- tests/vision/e2e_job.sh
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${E2E_PORT:-8148}
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/vision/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t ne-vis-e2e)
( umask 077; $PY -c 'import secrets; print(secrets.token_urlsafe(32), end="")' > $T/key ); chmod 600 $T/key
SP=
stop_server() {
  [ -n "$SP" ] || return 0
  kill -TERM $SP 2>/dev/null
  for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill -0 $SP 2>/dev/null && kill -KILL $SP
  wait $SP 2>/dev/null
  SP=
}
trap 'stop_server; rm -rf $T' EXIT INT TERM
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT busy"; exit 1; }
tools/serve.sh build >/dev/null
SHIM=$ROOT/build-server/${NE_LIB:h:t}/libne_serve.dylib
SARGS=()
[ -f $SHIM ] && SARGS=(--shim $SHIM)
$PY server/ne_server.py --port $PORT --ctx ${E2E_CTX:-32768} --lib $NE_LIB --key-file $T/key $SARGS \
    2> $OUT/e2e-$STAMP-server.log &
SP=$!
for i in {1..150}; do
  curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
  kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $OUT/e2e-$STAMP-server.log; exit 1; }
  sleep 2
done
$PY tests/vision/e2e_client.py --url http://127.0.0.1:$PORT --key-file $T/key --out $OUT/e2e-$STAMP.json \
    ${=E2E_CLIENT_ARGS:-} 2>&1 | tee $OUT/e2e-$STAMP-client.txt
rc=${pipestatus[1]}
stop_server
grep -E "vision|image|engine ready|prompt" $OUT/e2e-$STAMP-server.log | tail -20
exit $rc
