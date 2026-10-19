#!/bin/zsh
# tests/server/shared_job.sh   shared-prefix snapshots on the whole model (lab-snap, docs/server.md 3f); ONE with_lock job:
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=snap tools/with_lock.sh "shared-prefix A/B (~81 GB)" HH:MM -- tests/server/shared_job.sh
# Two server runs on port 8144 (SJ_PORT) with one scratch SSD directory and a throw-away key (umask 077, never printed):
#   1: tests/server/shared_http.py --phase first (arms off / on, cold references); SIGTERM.  2: --phase restart.
# Env: NE_LIB (default: this tree's build/libengine.dylib), SJ_CTX (65536), SJ_ARGS (more server args; SJ_ARGS=--mock:
# the flow on the CPU mock, no lock needed). Results: tests/server/results/shared-<stamp>*.
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${SJ_PORT:-8144}
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t ne-shared-job)
( umask 077; $PY -c 'import secrets; print(secrets.token_urlsafe(32), end="")' > $T/key ); chmod 600 $T/key
SP=
stop_server() {
  [ -n "$SP" ] || return 0
  kill -TERM $SP 2>/dev/null
  for i in {1..90}; do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill -0 $SP 2>/dev/null && kill -KILL $SP
  wait $SP 2>/dev/null
  SP=
}
trap 'stop_server; rm -rf $T' EXIT INT TERM
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT busy"; exit 1; }
start_server() {  # start_server PHASE
  local t0=$(date +%s)
  $PY server/ne_server.py --port $PORT --ctx ${SJ_CTX:-65536} --lib $NE_LIB --key-file $T/key \
      --cache-disk 8GB --cache-dir $T/pc ${=SJ_ARGS:-} 2> $OUT/shared-$STAMP-server-$1.log &
  SP=$!
  for i in {1..400}; do
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
    kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $OUT/shared-$STAMP-server-$1.log; exit 1; }
    sleep 1
  done
  echo "--- server $1 up in $(( $(date +%s) - t0 )) s"
  grep -E "SSD tier" $OUT/shared-$STAMP-server-$1.log
}
rc=0
for ph in first restart; do
  start_server $ph
  $PY tests/server/shared_http.py --url http://127.0.0.1:$PORT --key-file $T/key --phase $ph --state $OUT/shared-$STAMP.json \
      2>&1 | tee -a $OUT/shared-$STAMP-client.txt
  [ ${pipestatus[1]} = 0 ] || rc=1
  stop_server
  grep -E "shared prefix|prompt [0-9]+ \(cached" $OUT/shared-$STAMP-server-$ph.log | sed -E 's/; sampling.*//' | tail -14
done
echo "results: $OUT/shared-$STAMP*"
[ $rc = 0 ] && echo "shared job: all passed" || echo "shared job: FAILED"
exit $rc
