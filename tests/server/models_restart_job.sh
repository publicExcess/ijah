#!/bin/zsh
# tests/server/models_restart_job.sh   whole model, ONE with_lock job (~3-4 min): `--models all` with the SSD prefix cache,
# both ids answer, then a restart restores base's conversation from disk (tests/server/models_restart.py):
#   NEED_GB=96 LOCK_WAIT=60 LOCK_OWNER=lab-srvperf tools/with_lock.sh "models + SSD restart (~95 GB)" HH:MM -- tests/server/models_restart_job.sh
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${J_PORT:-8142}
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t models-restart)
( umask 077; $PY -c 'import secrets; print(secrets.token_urlsafe(32), end="")' > $T/key ); chmod 600 $T/key
SP=
stop_server() { [ -n "$SP" ] || return 0; kill -TERM $SP 2>/dev/null; for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -0 $SP 2>/dev/null && kill -KILL $SP; wait $SP 2>/dev/null; SP=; }
trap 'stop_server; rm -rf $T' EXIT INT TERM
rc=0
for ph in A B; do
  $PY server/ne_server.py --port $PORT --ctx ${J_CTX:-262144} --lib $NE_LIB --key-file $T/key --models all \
      --cache-disk 20GB --cache-dir $T/pc 2> $OUT/models-restart-$STAMP-$ph-server.log &
  SP=$!
  for i in {1..200}; do curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
    kill -0 $SP 2>/dev/null || { echo "server died"; tail -20 $OUT/models-restart-$STAMP-$ph-server.log; exit 1; }; sleep 1; done
  grep "engine ready\|shared by\|SSD\|disk" $OUT/models-restart-$STAMP-$ph-server.log | cut -c1-240 | head -6
  MM=$(curl -s -H "Authorization: Bearer $(cat $T/key)" http://127.0.0.1:$PORT/v1/models | $PY -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))')
  echo "phase $ph: models $MM"
  $PY tests/server/models_restart.py --url http://127.0.0.1:$PORT --key-file $T/key --models $MM --phase $ph --state $T/st.json || rc=1
  sleep 2
  stop_server
  grep -v "^\s*$" $OUT/models-restart-$STAMP-$ph-server.log | tail -6 | cut -c1-240
  du -sh $T/pc/* 2>/dev/null
done
exit $rc
