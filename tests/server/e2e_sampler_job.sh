#!/bin/zsh
# tests/server/e2e_sampler_job.sh   the whole-model sampling-knobs test (lab-sampler); ONE with_lock job (starts and stops the
# test servers inside, on port 8142 -- never 8140):
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-sampler tools/with_lock.sh "sampler e2e (~83 GB)" HH:MM -- tests/server/e2e_sampler_job.sh
# Steps: kernel checks on synthetic logits; the engine-level model check (numpy on the model's own logits); GPU step time
# A/B baseline vs this build (interleaved A B A B; BASE_ROOT = a built checkout of the commit before the sampler patch with a
# the same IJAH_PACK, unset = skip the A/B); the HTTP test (n, logprobs, bias, penalties, seeds, /v1/completions,
# /v1/messages) on this build's server; unconstrained + knob speed through HTTP (baseline server, then this build's).
# Env: NE_LIB (default build/libengine.dylib), BASE_ROOT, E2E_CTX (32768), E2E_PORT (8142). A throw-away key in a temp dir.
# Results: tests/server/results/sampler-*.
# E2E_STEPS (default all): comma list of kernel,model,ab,http,slots,models to run a part of it (ab = both speed A/Bs;
# models = `--models all` (both registry ids in one process, ~95 GB: NEED_GB=96) and test_sampling.py on each id + the
# cross-model leak checks; not in the default list).
set -u
STEPS=${E2E_STEPS:-kernel,model,ab,http,slots}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${E2E_PORT:-8142}
[ $PORT = 8140 ] && { echo "refusing port 8140"; exit 1; }
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
T=$(mktemp -d -t ne-sampler)
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

start_server() {   # root lib log [server args]
  local r=$1 lib=$2 log=$3; shift 3
  ( cd $r && exec $PY server/ne_server.py --port $PORT --ctx ${E2E_CTX:-32768} --key-file $T/key --lib $lib "$@" 2> $log ) &
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
if has kernel; then
  echo "=== sampler kernels on synthetic logits"
  $PY tests/server/sampler_check.py 2>&1 | tail -3
fi
if has model; then
  echo "=== engine-level model check"
  $PY tests/server/sampler_model_check.py 2>&1 | tee $OUT/sampler-$STAMP-model.txt | grep -v "^ok"
  [ ${pipestatus[1]} = 0 ] || { echo "sampler_model_check failed; not starting the servers"; exit 1; }
fi
if has ab && [ -n "${BASE_ROOT:-}" ]; then
  echo "=== GPU step time A/B (interleaved), baseline = $BASE_ROOT"
  for i in 1 2; do
    NE_LIB=$BASE_ROOT/build/libengine.dylib $PY tests/server/step_ab.py 2>&1 | grep STEP_AB | sed 's/^/base  /' | tee -a $OUT/sampler-$STAMP-speed.txt
    NE_LIB=$NE_LIB $PY tests/server/step_ab.py 2>&1 | grep STEP_AB | sed 's/^/this  /' | tee -a $OUT/sampler-$STAMP-speed.txt
  done
  echo "=== baseline server speed through HTTP"
  start_server $BASE_ROOT $BASE_ROOT/build/libengine.dylib $OUT/sampler-$STAMP-base-server.log || exit 1
  $PY tests/server/speed_ab.py --url http://127.0.0.1:$PORT --key-file $T/key --label baseline 2>&1 | tee -a $OUT/sampler-$STAMP-speed.txt
  stop_server
fi
if has http || has ab; then
  echo "=== this build: $NE_LIB"
  start_server $ROOT $NE_LIB $OUT/sampler-$STAMP-server.log || exit 1
  echo "free RAM $(free_gb) GB with the server up"
  if has http; then
    $PY tests/server/test_sampling.py --url http://127.0.0.1:$PORT --key-file $T/key --json $OUT/sampler-$STAMP.json 2>&1 | tee $OUT/sampler-$STAMP-client.txt | grep -v "^ok"
    rc=${pipestatus[1]}
  fi
  if has ab; then
    $PY tests/server/speed_ab.py --url http://127.0.0.1:$PORT --key-file $T/key --label this-build --knobs 2>&1 | tee -a $OUT/sampler-$STAMP-speed.txt
  fi
  stop_server
fi
if has models; then
  echo "=== --models all: every id answers with logprobs, knobs per model instance"
  start_server $ROOT $NE_LIB $OUT/sampler-$STAMP-server3.log --models all || exit 1
  MM=$(curl -s -H "Authorization: Bearer $(cat $T/key)" http://127.0.0.1:$PORT/v1/models | $PY -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))')
  echo "models: $MM"
  $PY tests/server/test_sampling.py --url http://127.0.0.1:$PORT --key-file $T/key --models $MM 2>&1 | tee $OUT/sampler-$STAMP-models.txt | grep -v "^ok"
  [ ${pipestatus[1]} = 0 ] || rc=1
  stop_server
fi
if has slots; then
  echo "=== two slots: knob requests next to plain ones, a long prompt"
  start_server $ROOT $NE_LIB $OUT/sampler-$STAMP-server2.log --slots 2 || exit 1
  $PY tests/server/test_sampling_slots.py --url http://127.0.0.1:$PORT --key-file $T/key 2>&1 | tee $OUT/sampler-$STAMP-slots.txt | grep -v "^ok"
  [ ${pipestatus[1]} = 0 ] || rc=1
  stop_server
fi
grep -v "^\s*$" $OUT/sampler-$STAMP-server.log | tail -30
echo "swap after: $(sysctl -n vm.swapusage)"
exit $rc
