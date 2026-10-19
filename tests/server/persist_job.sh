#!/bin/zsh
# tests/server/persist_job.sh   the SSD tier across server restarts on the whole model (lab-persist); ONE with_lock job:
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-persist tools/with_lock.sh "persist e2e (~83 GB)" HH:MM -- tests/server/persist_job.sh
# Three server runs on port 8141 with one scratch SSD directory (tests/server/persist_client.py, a ~38K-token conversation):
#   A: turn 1 cold, turn 2 (RAM reuse), turn 2 regenerated; SIGTERM.  B: turn 2 restored from SSD (= A's regenerate bit for
#   bit?), a cold reference; SIGTERM.  Then every KV segment file gets one flipped byte.  C: turn 2 -> entry rejected, cold.
# Env: NE_LIB (default this tree's build/libengine.dylib), PJ_CTX (65536), PJ_ARGS (more server args;
# PJ_ARGS=--mock: the flow on the CPU mock, no lock needed). Results: tests/server/results/.
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${PJ_PORT:-8141}
export NE_LIB=${NE_LIB:-${0:A:h:h:h}/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t ne-persist-job)
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
  $PY server/ne_server.py --port $PORT --ctx ${PJ_CTX:-65536} --lib $NE_LIB --key-file $T/key \
      --cache-disk 8GB --cache-dir $T/pc ${=PJ_ARGS:-} 2> $OUT/persist-$STAMP-server-$1.log &
  SP=$!
  for i in {1..300}; do
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
    kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $OUT/persist-$STAMP-server-$1.log; exit 1; }
    sleep 1
  done
  echo "--- server $1 up in $(( $(date +%s) - t0 )) s"
  grep -E "SSD tier|engine ready" $OUT/persist-$STAMP-server-$1.log
}
rc=0
for ph in A B C; do
  if [ $ph = C ]; then  # flip one byte in the middle of every KV segment file
    $PY - $T/pc <<'EOF'
import sys
from pathlib import Path
n = 0
for f in Path(sys.argv[1]).rglob("seg*.bin"):
    with open(f, "r+b") as fh:
        fh.seek(f.stat().st_size // 2)
        b = fh.read(1)
        fh.seek(-1, 1)
        fh.write(bytes([b[0] ^ 0x40]))
    n += 1
print(f"corrupted {n} segment files")
EOF
  fi
  start_server $ph
  $PY tests/server/persist_client.py --url http://127.0.0.1:$PORT --key-file $T/key --phase $ph --state $OUT/persist-$STAMP.json \
      2>&1 | tee -a $OUT/persist-$STAMP-client.txt
  [ ${pipestatus[1]} = 0 ] || rc=1
  stop_server
  grep -E "prefix cache|stopped" $OUT/persist-$STAMP-server-$ph.log | tail -6
  du -sh $T/pc | cut -f1 | sed 's/^/SSD tier on disk: /'
done
echo "results: $OUT/persist-$STAMP*"
[ $rc = 0 ] && echo "persist job: all passed" || echo "persist job: FAILED"
exit $rc
