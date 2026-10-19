#!/bin/zsh
# with_lock.sh "JOB" UNTIL(HH:MM) -- CMD...   run one heavy (GPU / model-loading) job under the RAM watchdog and, when
# IJAH_LOCK_SCRIPT names an executable lock script, under a machine-wide lock (for machines shared by several heavy
# jobs). Without IJAH_LOCK_SCRIPT the job runs directly (no lock; NEED_GB and the watchdog still apply).
# Lock script interface: `S acquire OWNER JOB UNTIL PID` (0 = taken), `S release OWNER`, `S status` (first line
# `free`, or `owner: X` / `pid: N` / `since: YYYY-MM-DD HH:MM` lines; `(stale ...` when the holder is gone),
# `S clear-stale`.
# The lock is taken with the job's own PID (the command is exec'd once the lock is held), released at the end.
# RAM watchdog: kills the job (and only it) if system free memory drops under MIN_FREE_PCT (default 10) or the
# system's swapouts grow by more than MAX_SWAPOUT_MB (default 2048) during the job.
# Env: LOCK_OWNER (default lab-ref), LOCK_WAIT=N (retry acquiring every 20 s for up to N minutes; default 0 = fail
# at once with exit 75 if the lock is held).
# A lock whose holder is gone is cleared before waiting for it (`clear-stale`: its pid is dead; and, for
# our own LOCK_OWNER, a pid that now belongs to a process started after the lock was taken = a reused pid), with a log line.
# Watchdog for long-running jobs (servers): SWAP_WINDOW_S=N counts the swapouts of the last N seconds (kill above
# MAX_SWAPOUT_MB within the window) instead of everything since the job started (N=0, the default: batch jobs, as
# before); MIN_FREE_SAMPLES=N needs N consecutive 5 s samples under MIN_FREE_PCT before it kills (default 1).
# Test knobs: NE_LOCK_SCRIPT (the lock script when IJAH_LOCK_SCRIPT is unset), WL_VM_STAT / WL_MEMPRESSURE (commands standing in for vm_stat /
# memory_pressure), WL_POLL_S (watchdog beat, default 5).
set -u
JOB=$1; UNTIL=$2; shift 2; [ "${1:-}" = "--" ] && shift
LOCK=${IJAH_LOCK_SCRIPT:-${NE_LOCK_SCRIPT:-}}
nolock() { case $1 in status) echo free ;; *) return 0 ;; esac; }   # no lock script: every lock call succeeds
if [ -z "$LOCK" ] || [ ! -x "${LOCK/#\~/$HOME}" ]; then
  [ -n "$LOCK" ] && echo "with_lock: lock script $LOCK not found: running without the machine lock" >&2
  LOCK=nolock
else
  LOCK=${LOCK/#\~/$HOME}
fi
OWNER=${LOCK_OWNER:-lab-ref}
MINF=${MIN_FREE_PCT:-10}
MAXSW=${MAX_SWAPOUT_MB:-2048}
SWWIN=${SWAP_WINDOW_S:-0}
MINSAMP=${MIN_FREE_SAMPLES:-1}
WAITM=${LOCK_WAIT:-0}
VMSTAT=${WL_VM_STAT:-vm_stat}
MEMP=${WL_MEMPRESSURE:-memory_pressure}
POLL=${WL_POLL_S:-5}
swapouts() { ${=VMSTAT} | sed -n 's/^Swapouts: *\([0-9]*\).*/\1/p'; }
etime_s() {   # ps etime ([[dd-]hh:]mm:ss) -> seconds
  local e=$1 d=0 h=0 m=0 sec=0 p
  if [[ $e == *-* ]]; then d=${e%%-*}; e=${e#*-}; fi
  p=(${(s.:.)e})
  if (( ${#p} == 3 )); then h=$p[1]; m=$p[2]; sec=$p[3]; else m=$p[1]; sec=$p[2]; fi
  echo $(( ((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$sec ))
}
# the lock's holder is gone? -> clear it (and say so). 0 = cleared, 1 = held by a live job (or free)
clear_stale() {
  local info pid since started
  info=$($LOCK status 2>/dev/null)
  if print -r -- "$info" | grep -q '^(stale'; then
    echo "with_lock: clearing a stale lock (holder pid gone): $(print -r -- "$info" | tr '\n' ' ')" >&2
    $LOCK clear-stale >/dev/null 2>&1 && return 0
    return 1
  fi
  # our own owner's lock whose pid was reused: the process now running as that pid started after the lock was taken
  if print -r -- "$info" | grep -q "^owner: $OWNER\$"; then
    pid=$(print -r -- "$info" | sed -n 's/^pid: //p'); since=$(print -r -- "$info" | sed -n 's/^since: //p')
    if [ -n "$pid" ] && [ "$pid" != none ] && [ -n "$since" ]; then
      local et age t_lock
      et=$(ps -p $pid -o etime= 2>/dev/null | tr -d ' ')   # [[dd-]hh:]mm:ss since the process started
      if [ -n "$et" ]; then
        age=$(etime_s $et)
        t_lock=$(date -j -f "%Y-%m-%d %H:%M" "$since" +%s 2>/dev/null) || return 1
        if [ $(( age + 120 )) -lt $(( $(date +%s) - t_lock )) ]; then   # younger than the lock: not the job that took it
          echo "with_lock: clearing our own stale lock (pid $pid is a process started after the lock was taken, a reused pid): $(print -r -- "$info" | tr '\n' ' ')" >&2
          $LOCK release $OWNER >/dev/null 2>&1 && return 0
        fi
      fi
    fi
  fi
  return 1
}
# NEED_GB=N: refuse to start unless N GB are free now (set it for whole-model jobs, e.g. NEED_GB=85). A model server
# can take ~78 GB at any time; two such residents overflow a 128 GB machine.
free_gb() { local t f; t=$(sysctl -n hw.memsize); f=$(${=MEMP} 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); echo $(( t / 1073741824 * ${f:-0} / 100 )); }
if [ -n "${NEED_GB:-}" ] && [ "$WAITM" = 0 ] && [ $(free_gb) -lt $NEED_GB ]; then   # with LOCK_WAIT, queue first and check RAM once the lock is ours
  echo "with_lock: only $(free_gb) GB free, job needs NEED_GB=$NEED_GB; is another model server loaded? not starting" >&2; exit 76
fi
GO=$(mktemp -t with_lock)
rm -f $GO
# the job waits for the go file, so the lock records its own PID before it starts
( while [ ! -e $GO ]; do sleep 0.2; done; [ "$(cat $GO)" = go ] || exit 75; export NE_NO_LOCK=1; exec "$@" ) &   # the job inherits the held lock: core MachineLock becomes a no-op (no deadlock)
P=$!
deadline=$(( $(date +%s) + WAITM * 60 ))
cleared=0
until $LOCK acquire $OWNER "$JOB" $UNTIL $P >/dev/null 2>&1; do
  if [ $cleared -lt 3 ] && clear_stale; then cleared=$((cleared + 1)); continue; fi   # (the holder is gone: take the lock now)
  if [ $(date +%s) -ge $deadline ]; then echo "with_lock: lock held:" >&2; $LOCK status >&2; echo no > $GO; wait $P; rm -f $GO; exit 75; fi
  sleep 20
done
if [ -n "${NEED_GB:-}" ]; then   # the previous holder's memory can take a moment to free
  for i in {1..12}; do [ $(free_gb) -ge $NEED_GB ] && break; sleep 10; done
  if [ $(free_gb) -lt $NEED_GB ]; then
    echo "with_lock: only $(free_gb) GB free after taking the lock, job needs NEED_GB=$NEED_GB; is a model server loaded? not starting" >&2
    echo no > $GO; wait $P; rm -f $GO; $LOCK release $OWNER >/dev/null 2>&1; exit 76
  fi
fi
echo go > $GO
SW0=$(swapouts)
(
  low=0; hist=()
  while kill -0 $P 2>/dev/null; do
    f=$(${=MEMP} 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
    if [ -n "$f" ] && [ "$f" -lt $MINF ]; then
      low=$((low + 1))
      if [ $low -ge $MINSAMP ]; then echo "with_lock: free RAM $f% < $MINF%, killing $P" >&2; kill $P; fi
    else
      low=0
    fi
    sw=$(swapouts)
    if [ -n "$sw" ]; then
      base=$SW0; over="since the job started"
      if [ "$SWWIN" -gt 0 ]; then   # a rate: the swapouts of the last SWWIN seconds (a server runs for days)
        now=$(date +%s); hist+=("$now:$sw")
        while [ ${#hist} -gt 1 ] && [ $(( now - ${hist[2]%%:*} )) -ge $SWWIN ]; do hist[1]=(); done
        base=${hist[1]#*:}; over="within ~${SWWIN} s"
      fi
      if [ $(( (sw - base) * 16 / 1024 )) -gt $MAXSW ]; then
        echo "with_lock: swapouts +$(( (sw - base) * 16 / 1024 )) MB $over > $MAXSW MB, killing $P" >&2; kill $P
      fi
    fi
    sleep $POLL
  done
) &
W=$!
wait $P; rc=$?
kill $W 2>/dev/null
rm -f $GO
$LOCK release $OWNER >/dev/null
exit $rc
