#!/bin/zsh
# tools/serve.sh keygen | build | start [UNTIL] [-- server args] | stop | status | log      (docs/server.md)
# Ijah's OpenAI- / Anthropic-compatible server on 127.0.0.1:8140 (whole model, ~85 GB at --ctx 262144).
#   keygen   create ~/.config/ijah/api-key (CSPRNG, chmod 600) if missing; never prints it (IJAH_KEY_FILE overrides)
#   build    the snapshot shim for the library in use (NE_LIB, default build/libengine.dylib) -> build-server/<dir>/
#   start    build + run server/ne_server.py under tools/with_lock.sh (NEED_GB=85, RAM watchdog; the machine lock only
#            when IJAH_LOCK_SCRIPT is set) until
#            UNTIL (HH:MM, default 23:59), inside a supervisor (lab-persist): if the server dies (crash, OOM kill, the
#            watchdog) it is started again after a backoff (10, 20, 40, 80, 160 s; gives up after 5 crashes that each
#            ran < 10 min), through with_lock again (lock + NEED_GB: while either refuses it waits, 60 s steps, up to
#            30 min). Waits until /health answers. Extra args go to the server (e.g. --ctx 65536)
#   stop     stop the supervisor, SIGTERM the server (pid file), wait, release (with_lock releases the lock)
#   state    one line (for a status-bar plugin or scripts): running | starting | crashed ... | waiting ... | failed ... | stopped
#   models   the model ids of server/models.json and the pack each resolves to
# Model: NE_MODEL=<id> (or `-- --model-id <id>`; `-- --pack DIR` overrides), e.g.
#   NE_MODEL=Swift-1.5-Flash-Next-native tools/serve.sh start      (default id: Qwen3.8-Flash-Next-native; packs are
#   looked up in $IJAH_PACKS, default ~/.cache/ijah/packs: see server/models.json)
#   NE_MODEL=Qwen3.8-Flash-Next-native,Swift-1.5-Flash-Next-native tools/serve.sh start   (both in one process: the
#   experts shared, requests routed by `model`; NE_MODEL=all = every registry id; NEED_GB defaults to 96 then;
#   NE_SHARE_STATE=1: the models time-share one KV + recurrent state per slot, --share-state: ~6 GB less at 262K)
# Never starts next to another model server: with_lock's NEED_GB check refuses when the RAM is taken.
# Files: the key in ~/.config/ijah (IJAH_KEY_FILE), log / pid / supervisor state in ~/.cache/ijah (NE_HOME: both).
set -u
ROOT=${0:A:h:h}
source $ROOT/tools/env.sh
PY=$IJAH_PY
HOME_NE=${NE_HOME:-$HOME/.config/ijah}   # (tests: a scratch directory)
RUN_NE=${NE_HOME:-$HOME/.cache/ijah}
KEY=${IJAH_KEY_FILE:-$HOME_NE/api-key}
LOG=$RUN_NE/serve.log
PIDF=$RUN_NE/server.pid
PORT=${NE_SERVE_PORT:-8140}
LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
LIB=${LIB:A}

SUP=$RUN_NE/supervisor-$PORT   # .pid (the supervisor's), .state (one line), .stop (asked to stop)
NO_SUPERVISE=${NE_NO_SUPERVISE:-0}
WRAP=${NE_SERVE_WRAP:-tools/with_lock.sh}       # (tests: a stand-in with with_lock's interface and exit codes)
BACKOFF=${NE_SUP_BACKOFF_S:-10}; WAITSTEP=${NE_SUP_WAIT_S:-60}

listening() { lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; }
sup_alive() { [ -f $SUP.pid ] && kill -0 $(cat $SUP.pid) 2>/dev/null && ps -p $(cat $SUP.pid) -o command= | grep -q 'serve.sh _supervise'; }
setstate() { print -r -- "$1|$(date '+%F %T')|${2:-}" > $SUP.state.tmp && mv -f $SUP.state.tmp $SUP.state; }

keygen() {
  mkdir -p ${KEY:h} && chmod 700 ${KEY:h}
  if [ -s $KEY ]; then chmod 600 $KEY; echo "key exists: $KEY"; return 0; fi
  local kpy=$PY; [ -x $kpy ] || kpy=/usr/bin/python3
  ( umask 077; $kpy -I -B -c 'import secrets; print("ij-" + secrets.token_urlsafe(32), end="")' > $KEY )
  chmod 600 $KEY
  echo "new key written to $KEY (not shown)"
}

shim_path() { echo $ROOT/build-server/${LIB:h:t}/libne_serve.dylib; }

build() {   # the shim reads ne::fn::Model fields: compile it against the headers the library was built from
  local out=$(shim_path) newest
  [ -f $LIB ] || { echo "no library at $LIB"; return 1; }
  # the headers the shim actually includes (the compiler's dependency list), newest first
  newest=$(cd $ROOT && clang++ -std=c++20 -I. -Iinclude -MM server/shim/ne_serve_state.mm | tr -d '\\' | tr ' ' '\n' \
           | grep '\.h$' | sed "s|^|$ROOT/|" | xargs ls -t | head -1)
  if [ $newest -nt $LIB ]; then
    echo "build: ${newest#$ROOT/} is newer than ${LIB#$ROOT/}: the library may not match the headers; no shim (prefix reuse = live state only)"
    rm -f $out; return 0
  fi
  [ -f $out ] && [ $out -nt $LIB ] && [ $out -nt $ROOT/server/shim/ne_serve_state.mm ] && { echo "shim up to date: ${out#$ROOT/}"; return 0; }
  local def=""
  grep -q '^struct ne_fn_model {' $ROOT/models/flash_next/model.h || def=-DNE_SRV_DEFINE_HANDLE
  mkdir -p ${out:h}
  ( cd $ROOT && clang++ -std=c++20 -fobjc-arc -O2 -g -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
      -I. -Iinclude $def -dynamiclib server/shim/ne_serve_state.mm -L${LIB:h} -lengine -Wl,-rpath,${LIB:h} \
      -framework Metal -framework Foundation -install_name @rpath/libne_serve.dylib -o $out ) && echo "built ${out#$ROOT/}"
}

run_once() {  # one server run under with_lock, in the foreground; -> its exit code (75 lock held, 76 RAM)
  local until=$1; shift
  # (a server runs for days: the swap watchdog counts swapouts per minute, not since the start; 3 low-RAM samples in a row)
  ( cd $ROOT && NE_LIB=$LIB NEED_GB=${NEED_GB:-85} LOCK_OWNER=ne-serve LOCK_WAIT=${LOCK_WAIT:-0} \
      SWAP_WINDOW_S=${NE_SWAP_WINDOW_S:-60} MIN_FREE_SAMPLES=${NE_MIN_FREE_SAMPLES:-3} $WRAP \
      "ijah server :$PORT (~85 GB, whole model)" $until -- \
      $PY -I -B server/ne_server.py --port $PORT --lib $LIB --pid-file $PIDF --key-file $KEY "$@" )
}

nap() { local i; for i in {1..$1}; do [ -e $SUP.stop ] && return 1; sleep 1; done; return 0; }   # 1 = stop asked

supervise() {  # serve.sh _supervise UNTIL -- server args   (started detached by start; never at login by itself)
  local until=$1; shift; [ "${1:-}" = "--" ] && shift
  echo $$ > $SUP.pid
  local n=0 waited=0 runs=0 rc t0 ran delay
  while [ ! -e $SUP.stop ]; do
    setstate starting "attempt $((runs + 1))"
    t0=$(date +%s)
    run_once $until "$@" >> $LOG 2>&1
    rc=$?
    [ -e $SUP.stop ] && break
    ran=$(( $(date +%s) - t0 ))
    if [ $rc = 75 ] || [ $rc = 76 ]; then  # with_lock refused: the machine lock is held / not NEED_GB free
      local why="machine lock held ($(${IJAH_LOCK_SCRIPT:-false} status 2>/dev/null | sed -n 's/^owner: //p'))"
      [ $rc = 76 ] && why="not ${NEED_GB:-85} GB free"
      if [ $runs = 0 ]; then setstate failed "not started: $why"; echo "$(date '+%F %T') supervisor: not started: $why" >> $LOG; break; fi
      if [ $waited -ge 1800 ]; then setstate failed "gave up after 30 min: $why"; echo "$(date '+%F %T') supervisor: gave up: $why" >> $LOG; break; fi
      setstate waiting "$why; retry in $WAITSTEP s"
      nap $WAITSTEP || break
      waited=$((waited + WAITSTEP))
      continue
    fi
    runs=$((runs + 1)); waited=0
    [ $ran -ge 600 ] && n=0
    n=$((n + 1))
    if [ $n -gt 5 ]; then setstate failed "5 crashes in a row (last rc $rc)"; echo "$(date '+%F %T') supervisor: 5 crashes in a row; giving up" >> $LOG; break; fi
    delay=$(( BACKOFF << (n - 1) ))
    setstate crashed "rc $rc after ${ran} s; restart $n/5 in ${delay} s"
    echo "$(date '+%F %T') supervisor: server exited (rc $rc after ${ran} s); restart $n/5 in ${delay} s" >> $LOG
    rm -f $PIDF
    nap $delay || break
  done
  [ -e $SUP.stop ] && setstate stopped
  rm -f $SUP.pid $SUP.stop
}

start() {
  local until=23:59
  [ $# -gt 0 ] && [ "$1" != "--" ] && { until=$1; shift; }
  [ "${1:-}" = "--" ] && shift
  if listening; then echo "port $PORT is in use"; return 1; fi
  if sup_alive; then echo "already starting / restarting (supervisor pid $(cat $SUP.pid)): $(state)"; return 1; fi
  ijah_need_py
  [ -s $KEY ] || keygen
  [ "${NE_SKIP_BUILD:-0}" = 1 ] || build || return 1
  local shim=$(shim_path) sargs=()
  [ -f $shim ] && sargs=(--shim $shim)
  local need=${NEED_GB:-85}
  if [ "${NE_MODEL:-}" = all ]; then sargs+=(--models all); need=${NEED_GB:-96}
  elif [ -n "${NE_MODEL:-}" ]; then sargs+=(--model-id $NE_MODEL); [[ $NE_MODEL == *,* ]] && need=${NEED_GB:-96}; fi
  [ "${NE_SHARE_STATE:-0}" = 1 ] && sargs+=(--share-state)
  mkdir -p $RUN_NE
  rm -f $SUP.stop
  echo "$(date '+%F %T') start ($LIB) $*" >> $LOG
  if [ $NO_SUPERVISE = 1 ]; then
    ( cd $ROOT && NE_LIB=$LIB NEED_GB=$need LOCK_OWNER=ne-serve LOCK_WAIT=${LOCK_WAIT:-0} SWAP_WINDOW_S=${NE_SWAP_WINDOW_S:-60} \
        MIN_FREE_SAMPLES=${NE_MIN_FREE_SAMPLES:-3} nohup tools/with_lock.sh \
        "ijah server :$PORT (~85 GB, whole model)" $until -- \
        $PY -I -B server/ne_server.py --port $PORT --lib $LIB --pid-file $PIDF --key-file $KEY $sargs "$@" >> $LOG 2>&1 & )
  else
    setstate starting "attempt 1"
    export NEED_GB=$need   # the supervisor's with_lock (two models: 96 GB)
    ( nohup /bin/zsh $ROOT/tools/serve.sh _supervise $until -- $sargs "$@" >> $LOG 2>&1 < /dev/null & )
  fi
  for i in {1..180}; do
    sleep 2
    if curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"'; then
      [ $NO_SUPERVISE = 1 ] || setstate running "pid $(cat $PIDF 2>/dev/null)"
      echo "up on 127.0.0.1:$PORT (pid $(cat $PIDF 2>/dev/null)$( [ $NO_SUPERVISE = 1 ] || echo ", supervised"))"; return 0
    fi
    if [ $NO_SUPERVISE = 1 ]; then
      if [ $i -gt 3 ] && [ ! -f $PIDF ] && ! pgrep -f "server/ne_server.py --port $PORT" >/dev/null; then
        echo "server did not start:"; tail -5 $LOG; return 1
      fi
    elif { [ $i -gt 3 ] && ! sup_alive; } || grep -qE '^(crashed|failed)' $SUP.state 2>/dev/null; then
      sup_alive && touch $SUP.stop   # the first run failed: no restarts for a start that never came up
      echo "server did not start ($(state)):"; tail -5 $LOG; return 1
    fi
  done
  echo "not up after 6 min; see $LOG"; return 1
}

state() {  # one line: running | starting | crashed ... | waiting ... | failed ... | stopped
  if listening; then
    if sup_alive || [ $NO_SUPERVISE = 1 ]; then echo "running"; else echo "running (unsupervised)"; fi
    sup_alive && [ "$(cut -d'|' -f1 $SUP.state 2>/dev/null)" != running ] && setstate running "pid $(cat $PIDF 2>/dev/null)"
    return 0
  fi
  [ -f $SUP.state ] || { echo stopped; return 0; }
  local st=$(cut -d'|' -f1 $SUP.state) at=$(cut -d'|' -f2 $SUP.state) why=$(cut -d'|' -f3- $SUP.state)
  if ! sup_alive && [ "$st" != failed ]; then
    [ "$st" = stopped ] && { echo stopped; return 0; }
    st="died"; why="no server, no supervisor (last: $(cut -d'|' -f1 $SUP.state) $why)"
  fi
  echo "$st${why:+ ($why)} since ${at#* }"
}

stop() {
  local sup=0
  if sup_alive; then sup=1; touch $SUP.stop; fi   # first: the supervisor must not restart what we stop
  if [ $sup = 1 ] && [ ! -f $PIDF ]; then  # between runs (it exits within 1 s) or a server just starting (pid file soon)
    for i in {1..30}; do { sup_alive && [ ! -f $PIDF ]; } || break; sleep 1; done
  fi
  if [ ! -f $PIDF ]; then
    if [ $sup = 1 ]; then
      for i in {1..15}; do sup_alive || break; sleep 1; done
      setstate stopped; echo "stopped the supervisor (no server was running)"; return 0
    fi
    echo "not running (no pid file)"; return 0
  fi
  local p=$(cat $PIDF)
  if ! ps -p $p -o command= 2>/dev/null | grep -q 'server/ne_server.py'; then echo "pid $p is not the server; removing stale pid file"; rm -f $PIDF; return 0; fi
  kill -TERM $p
  for i in {1..60}; do kill -0 $p 2>/dev/null || break; sleep 1; done
  kill -0 $p 2>/dev/null && { echo "still running after 60 s: SIGKILL $p"; kill -KILL $p; sleep 2; }
  rm -f $PIDF
  if [ $sup = 1 ]; then for i in {1..15}; do sup_alive || break; sleep 1; done; setstate stopped; fi
  echo "stopped (pid $p)"
}

case ${1:-status} in
  keygen) keygen ;;
  build) build ;;
  start) shift; start "$@" ;;
  stop) stop ;;
  state) state ;;
  _supervise) shift; supervise "$@" ;;
  models) $PY -I -B $ROOT/server/ne_server.py --list-models ;;
  status) if listening; then echo "listening on $PORT: $(curl -s -m 2 http://127.0.0.1:$PORT/health); $(state)"; else echo "not running: $(state)"; fi ;;
  log) tail -n ${2:-40} $LOG ;;
  *) echo "usage: [NE_MODEL=id] tools/serve.sh keygen | build | start [UNTIL] [-- args] | stop | status | state | log [N] | models"; exit 64 ;;
esac
