#!/bin/zsh
# tests/server/test_supervisor.sh   CPU-only (mock engine, no machine lock): tools/serve.sh's supervisor (lab-persist).
# A scratch NE_HOME, port 8147, a stand-in for with_lock.sh (same interface and exit codes, no lock taken):
#   start -> running; a conversation turn; kill -9 the server -> crashed -> restarted after the backoff -> the next turn
#   restores its prefix from the SSD tier (cached_tokens > 0); a refused restart (lock held: rc 75) -> waiting, then
#   up once the lock is free; a server that keeps dying -> failed after 5 crashes (no tight loop); stop -> stopped,
#   nothing left running.
set -u
ROOT=${0:A:h:h:h}
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
T=$(mktemp -d -t ne-sup)
export NE_HOME=$T NE_SERVE_PORT=8147 NE_SERVE_WRAP=$T/wrap.sh NE_SKIP_BUILD=1 NE_SUP_BACKOFF_S=1 NE_SUP_WAIT_S=1
PORT=8147
S=$ROOT/tools/serve.sh
FAIL=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; FAIL=1; }
cleanup() {
  touch $T/supervisor-$PORT.stop
  [ -f $T/server.pid ] && kill -9 $(cat $T/server.pid) 2>/dev/null
  [ -f $T/supervisor-$PORT.pid ] && kill $(cat $T/supervisor-$PORT.pid) 2>/dev/null
  sleep 1; rm -rf $T
}
trap cleanup EXIT
cat > $T/wrap.sh <<'EOF'
#!/bin/zsh
# with_lock.sh stand-in: JOB UNTIL -- CMD; 75 while $NE_HOME/refuse exists (lock held), 1 at once while crash exists
shift 2; [ "${1:-}" = "--" ] && shift
[ -e $NE_HOME/refuse ] && exit 75
[ -e $NE_HOME/crash ] && exit 1
exec "$@"
EOF
chmod +x $T/wrap.sh
( umask 077; print -n "test-key-$RANDOM$RANDOM$RANDOM$RANDOM" > $T/key )
ARGS=(--mock --key-file $T/key --ctx 8192 --cache-disk 1GB --cache-dir $T/pc)
st() { $S state; }
waitfor() {  # waitfor PATTERN SECONDS
  for i in {1..$(( $2 * 2 ))}; do st | grep -qE "$1" && return 0; sleep 0.5; done; return 1
}
chat() {  # chat TURN -> cached_tokens
  $PY -c '
import json, sys, urllib.request
turn = int(sys.argv[1])
msgs = [{"role": "system", "content": "You are the supervisor test. " * 40}]
for i in range(turn):
    msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": "Hello from the mock engine."}]
msgs.append({"role": "user", "content": f"question {turn}"})
r = urllib.request.Request(f"http://127.0.0.1:{sys.argv[2]}/v1/chat/completions", json.dumps({"messages": msgs,
    "max_tokens": 32, "reasoning_effort": "none"}).encode(), {"Authorization": "Bearer " + open(sys.argv[3]).read(),
    "Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(r, timeout=30))["usage"]["prompt_tokens_details"]["cached_tokens"])' $1 $PORT $T/key
}

$S start -- $ARGS >/dev/null && [ "$(st)" = running ] && ok "start -> running (supervised)" || bad "start: $(st)"
c0=$(chat 0); c1=$(chat 1)
[ "$c1" -gt 0 ] 2>/dev/null && ok "turn 2 reuses its prefix in RAM ($c1 cached)" || bad "turn 2 cached $c1"
sleep 1  # the SSD writer lands the turn
p1=$(cat $T/server.pid); kill -9 $p1
waitfor '^(crashed|starting)' 10 && ok "kill -9 -> $(st | cut -c1-60)" || bad "after kill -9: $(st)"
waitfor '^running' 30 && p2=$(cat $T/server.pid) && [ "$p2" != "$p1" ] && ok "restarted (pid $p1 -> $p2)" \
  || bad "not restarted: $(st)"
c2=$(chat 2)
[ "$c2" -ge "$c1" ] 2>/dev/null && ok "first turn after the crash restores from the SSD tier ($c2 cached)" || bad "after the crash cached $c2"
grep -q "SSD tier .* 1 conversations" $T/serve.log && ok "the restarted server indexed the conversation" || bad "no SSD index line"
touch $T/refuse; kill -9 $(cat $T/server.pid)
waitfor '^waiting' 15 && ok "lock held at restart -> $(st | cut -c1-50)" || bad "refused restart: $(st)"
rm -f $T/refuse
waitfor '^running' 30 && ok "lock free again -> running" || bad "after the lock came free: $(st)"
touch $T/crash; kill -9 $(cat $T/server.pid)
t0=$(date +%s)
waitfor '^failed' 60 && ok "a server that keeps dying -> $(st | cut -c1-40) after $(( $(date +%s) - t0 )) s (backoff, no tight loop)" \
  || bad "crash loop: $(st)"
n=$(grep -c "supervisor: server exited" $T/serve.log)
[ $n -le 8 ] && ok "$n restarts logged in total" || bad "$n restarts"
rm -f $T/crash
$S start -- $ARGS >/dev/null && $S stop >/dev/null && sleep 1 && [ "$(st)" = stopped ] && ! lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 \
  && [ ! -f $T/supervisor-$PORT.pid ] && ok "stop -> stopped, no server, no supervisor" || bad "stop: $(st)"
[ $FAIL = 0 ] && echo "all passed" || echo "FAILED"
exit $FAIL
