#!/bin/zsh
# tools/machine-lock.sh -- an optional machine-wide lock: one heavy job (a model load, a GPU benchmark, a server) at a
# time on a Mac shared by several such jobs. tools/with_lock.sh uses it when IJAH_LOCK_SCRIPT points at it (or at any
# script with this interface):
#   machine-lock.sh acquire OWNER "JOB" [UNTIL] [PID]   take the lock (atomic mkdir); exit 1 if held
#   machine-lock.sh wait    OWNER "JOB" [UNTIL] [PID]   wait until free, then take it
#   machine-lock.sh release OWNER                        release (only the owner)
#   machine-lock.sh status                               `free`, or the holder's info; `(stale ...` if its PID is gone
#   machine-lock.sh clear-stale                          remove a lock whose PID is gone
# The lock is a directory: IJAH_LOCK_DIR (default ~/.cache/ijah/machine-lock).
set -u
L=${IJAH_LOCK_DIR:-$HOME/.cache/ijah/machine-lock}
cmd=${1:-status}

show() { [ -d $L ] && cat $L/info 2>/dev/null || echo "free"; }
stale() { [ -d $L ] || return 1; local p=$(sed -n 's/^pid: //p' $L/info); [ -n "$p" ] && [ "$p" != none ] && ! kill -0 $p 2>/dev/null; }
take() {
  mkdir -p ${L:h}
  mkdir $L 2>/dev/null || return 1
  printf "owner: %s\njob: %s\nsince: %s\nuntil: %s\npid: %s\n" "$1" "$2" "$(date '+%Y-%m-%d %H:%M')" "${3:-?}" "${4:-none}" > $L/info
}

case $cmd in
  acquire) take "$2" "$3" "${4:-}" "${5:-}" && { echo "acquired"; show; } || { echo "HELD:"; show; exit 1; } ;;
  wait)
    until take "$2" "$3" "${4:-}" "${5:-}"; do
      if stale; then echo "the holder's process is gone:"; show; echo "(clear-stale to take it)"; exit 2; fi
      sleep 30
    done
    echo "acquired"; show ;;
  release)
    [ -d $L ] || { echo "not held"; exit 0; }
    grep -q "^owner: $2\$" $L/info || { echo "held by someone else:"; show; exit 1; }
    rm -rf $L && echo "released" ;;
  status) show; stale && echo "(stale: its process is gone)"; exit 0 ;;
  clear-stale) stale && { show; rm -rf $L; echo "stale lock cleared"; } || { echo "not stale:"; show; exit 1; } ;;
  *) echo "usage: machine-lock.sh acquire|wait|release|status|clear-stale"; exit 64 ;;
esac
