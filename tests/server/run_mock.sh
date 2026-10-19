#!/bin/zsh
# tests/server/run_mock.sh   CPU-only: unit tests, then the HTTP client test against the server on its mock engine
# (port 8149, a throw-away key), then the per-token server overhead (mock at 0 s/token and at 14 ms/token). No lock.
set -u
ROOT=${0:A:h:h:h}
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${MOCK_PORT:-8149}
T=$(mktemp -d -t ne-serve-mock)
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null; rm -rf $T' EXIT
( umask 077; $PY -c 'import secrets; print(secrets.token_urlsafe(32), end="")' > $T/key ); chmod 600 $T/key
cd $ROOT
$PY tests/server/test_units.py | tail -1 || exit 1
rc=0
$PY tests/server/test_sampling_units.py | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1   # knobs, logprobs, Anthropic converter
$PY tests/server/test_grammar.py 2>&1 | grep -v '^ok \|^\[transformers\]' ; [ ${pipestatus[1]} = 0 ] || rc=1   # grammar layer on the real tokenizer
$PY tests/server/test_tokcache.py | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1   # incremental prompt encode == whole-text encode (lab-ttft)
$PY tests/server/test_worker_robust.py 2>&1 | grep -v '^ok\|^  \|^Traceback\|^RuntimeError'; [ ${pipestatus[1]} = 0 ] || rc=1   # the engine thread survives bugs (lab-fixsrv)
$PY tests/server/test_limits.py | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1   # image path allowlist, pixel cap, close of every cache (lab-fixsrv)
$PY tests/server/test_special_text.py | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1   # literal special-token text in messages stays text (lab-fixsrv)
$PY tests/server/test_model_id.py | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1   # a request naming a model that is not served is a 400, never the default (lab-dflash)
for delay in 0 0.014; do
  $PY server/ne_server.py --mock --mock-delay $delay --port $PORT --key-file $T/key --ctx 32768 2> $T/log &
  SP=$!
  for i in {1..60}; do curl -s -m 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
  echo "--- mock engine, $delay s per token"
  $PY tests/server/test_client.py --mock --url http://127.0.0.1:$PORT --key-file $T/key --bench 400 \
      --json $T/res.json | grep -v '^ok' || rc=1
  echo "--- structured output over HTTP (mock engine)"
  $PY tests/server/test_structured.py --mock --url http://127.0.0.1:$PORT --key-file $T/key | grep -v '^ok' || rc=1
  echo "--- sampling knobs, n, logprobs, /v1/completions, /v1/messages (mock engine)"
  $PY tests/server/test_sampling.py --mock --url http://127.0.0.1:$PORT --key-file $T/key | grep -v '^ok'; [ ${pipestatus[1]} = 0 ] || rc=1
  kill $SP; wait $SP 2>/dev/null
done
echo "--- robustness over HTTP (mock engine, 10 ms per token, prefill 20K tok/s, SSE keepalive 0.2 s, 1 MB body cap, 2 s socket timeout): disconnects, cut-off tool calls, keepalives, limits"
NE_SRV_KEEPALIVE_S=0.2 NE_SRV_MAX_BODY_MB=1 NE_SRV_SOCKET_TIMEOUT=2 $PY server/ne_server.py --mock --mock-delay 0.01 --mock-prefill-rate 20000 --port $PORT --key-file $T/key --ctx 32768 --allow-any-model 2> $T/log6 &
SP=$!
for i in {1..60}; do curl -s -m 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
$PY tests/server/test_robust_http.py --prefill --url http://127.0.0.1:$PORT --key-file $T/key | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
kill $SP; wait $SP 2>/dev/null
echo "--- slot scheduler over HTTP (mock engine, --slots 2)"
$PY server/ne_server.py --mock --mock-delay 0.004 --slots 2 --port $PORT --key-file $T/key --ctx 32768 --allow-any-model 2> $T/log2 &
SP=$!
for i in {1..60}; do curl -s -m 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
$PY tests/server/test_sched_http.py --url http://127.0.0.1:$PORT --key-file $T/key | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
kill $SP; wait $SP 2>/dev/null
echo "--- two models in one server over HTTP (mock engine, --model-id A,B)"
MM=Qwen3.8-Flash-Next-native,Swift-1.5-Flash-Next-native
$PY server/ne_server.py --mock --mock-delay 0.004 --model-id $MM --port $PORT --key-file $T/key --ctx 32768 2> $T/log3 &
SP=$!
for i in {1..60}; do curl -s -m 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
$PY tests/server/test_models_http.py --url http://127.0.0.1:$PORT --key-file $T/key --models $MM | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
echo "--- sampling knobs, logprobs, n, completions, messages on each of the two models (mock engine)"
$PY tests/server/test_sampling.py --mock --url http://127.0.0.1:$PORT --key-file $T/key --models $MM | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
kill $SP; wait $SP 2>/dev/null
echo "--- two models sharing one state (mock engine, --model-id A,B --share-state)"
$PY server/ne_server.py --mock --mock-delay 0.004 --model-id $MM --share-state --port $PORT --key-file $T/key --ctx 32768 2> $T/log4 &
SP=$!
for i in {1..60}; do curl -s -m 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
$PY tests/server/test_models_http.py --url http://127.0.0.1:$PORT --key-file $T/key --models $MM | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
$PY tests/server/test_sampling.py --mock --url http://127.0.0.1:$PORT --key-file $T/key --models $MM | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
kill $SP; wait $SP 2>/dev/null
echo "--- two models + SSD prefix cache across a restart (mock engine, --model-id A,B --cache-disk 1GB)"
for ph in A B; do
  $PY server/ne_server.py --mock --mock-delay 0.002 --model-id $MM --cache-disk 1GB --cache-dir $T/pc --port $PORT --key-file $T/key --ctx 32768 2> $T/log5$ph &
  SP=$!
  for i in {1..60}; do curl -s -m 1 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
  $PY tests/server/models_restart.py --mock --url http://127.0.0.1:$PORT --key-file $T/key --models $MM --phase $ph --state $T/mr.json | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
  sleep 1; kill $SP; wait $SP 2>/dev/null
done
grep -c "401" $T/log | sed 's/^/401 log lines: /'
echo "--- with_lock.sh: stale locks, swap rate window (fake vm_stat), low-RAM samples"
tests/server/test_with_lock.sh | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
echo "--- supervisor + SSD tier across a crash (mock engine, port 8147)"
tests/server/test_supervisor.sh | grep -v '^ok' ; [ ${pipestatus[1]} = 0 ] || rc=1
exit $rc
