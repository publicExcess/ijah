#!/bin/zsh
# tests/server/test_with_lock.sh   tools/with_lock.sh without the GPU (lab-fixsrv): stale locks are cleared (dead pid, and our
# own owner's reused pid), a live holder is respected, the swap watchdog counts a rate when SWAP_WINDOW_S is set (a
# long-running server) and a total since the start otherwise (batch jobs), MIN_FREE_SAMPLES smooths low-RAM dips.
# Uses tools/machine-lock.sh with its own lock directory and fake vm_stat / memory_pressure. ~25 s.
set -u
ROOT=${0:A:h:h:h}
T=$(mktemp -d -t ne-withlock)
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $T' EXIT
export IJAH_LOCK_SCRIPT=$ROOT/tools/machine-lock.sh IJAH_LOCK_DIR=$T/lock LOCK_OWNER=wltest WL_POLL_S=0.2
WL=$ROOT/tools/with_lock.sh
rc=0
ok() { print "ok   $1"; }
bad() { print "FAIL $1"; rc=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi }
mklock() { mkdir $T/lock; printf "owner: %s\njob: %s\nsince: %s\nuntil: ?\npid: %s\n" "$1" "held" "$2" "$3" > $T/lock/info; }
now_minus() { date -v-${1}M '+%Y-%m-%d %H:%M'; }

# 1. a lock whose pid is dead is cleared and the job runs
sleep 0 & dead=$!; wait $dead
mklock someone "$(now_minus 5)" $dead
out=$($WL "stale test" 23:59 -- echo job-ran 2> $T/err); r=$?
check "dead holder: the job runs (rc $r, '$out')" '[ $r = 0 ] && [ "$out" = job-ran ]'
check "dead holder: logged as a stale lock" 'grep -q "clearing a stale lock" $T/err'
check "the lock is released at the end" '[ ! -d $T/lock ]'

# 2. a live holder is respected
sleep 60 & live=$!
mklock someone "$(now_minus 5)" $live
out=$($WL "held test" 23:59 -- echo job-ran 2> $T/err); r=$?
check "live holder of another owner: exit 75, job not run (rc $r)" '[ $r = 75 ] && [ -z "$out" ] && [ -d $T/lock ]'
rm -rf $T/lock

# 3. our own owner's lock whose pid is another (younger) process: a reused pid -> cleared; another owner's is not
mklock wltest "$(now_minus 30)" $live
out=$($WL "reuse test" 23:59 -- echo job-ran 2> $T/err); r=$?
check "own owner, pid started after the lock was taken: cleared, the job runs (rc $r)" '[ $r = 0 ] && [ "$out" = job-ran ] && grep -q "reused pid" $T/err'
mklock someone "$(now_minus 30)" $live
out=$($WL "reuse test 2" 23:59 -- echo job-ran 2> $T/err); r=$?
check "another owner's lock with a live pid is left alone (rc $r)" '[ $r = 75 ] && [ -d $T/lock ]'
rm -rf $T/lock
mklock wltest "$(now_minus 0)" $live
out=$($WL "fresh lock test" 23:59 -- echo job-ran 2> $T/err); r=$?
check "own owner, pid older than the lock: respected (rc $r)" '[ $r = 75 ]'
rm -rf $T/lock; kill $live

# fake system tools
cat > $T/vm_stat <<EOF
#!/bin/sh
echo "Swapouts:                      \$(cat $T/sw)."
EOF
cat > $T/mempressure <<EOF
#!/bin/sh
echo "System-wide memory free percentage: \$(cat $T/free)%"
EOF
chmod +x $T/vm_stat $T/mempressure
export WL_VM_STAT=$T/vm_stat WL_MEMPRESSURE=$T/mempressure MAX_SWAPOUT_MB=100
bump() { echo $(( $(cat $T/sw) + $1 )) > $T/sw; }   # pages of 16 KB: 320 = 5 MB
trickle='for i in {1..28}; do echo $(( $(cat '$T'/sw) + 320 )) > '$T'/sw; sleep 0.25; done'   # 5 MB / 0.25 s = 20 MB/s for 7 s = 140 MB

# 4. batch behaviour (no window): the total since the start counts -> killed
echo 1000 > $T/sw; echo 60 > $T/free
t0=$SECONDS; $WL "batch swap" 23:59 -- zsh -c "$trickle" 2> $T/err; r=$?
check "batch job (cumulative): 140 MB of swapouts > 100 MB kills it (rc $r after $((SECONDS - t0)) s)" '[ $r != 0 ] && grep -q "since the job started" $T/err'

# 5. server behaviour: the same trickle is 40 MB per 2 s window -> survives
export SWAP_WINDOW_S=2
echo 1000 > $T/sw
$WL "server swap" 23:59 -- zsh -c "$trickle" 2> $T/err; r=$?
check "server (SWAP_WINDOW_S=2): the same 20 MB/s trickle is not killed (rc $r)" '[ $r = 0 ] && ! grep -q swapouts $T/err'
echo 1000 > $T/sw; t0=$SECONDS
$WL "server burst" 23:59 -- zsh -c "sleep 1; echo 9000 > $T/sw; sleep 20" 2> $T/err; r=$?
check "server: a 130 MB burst inside the window is killed within ~1 s (rc $r after $((SECONDS - t0)) s)" '[ $r != 0 ] && grep -q "within ~2 s" $T/err && [ $((SECONDS - t0)) -lt 6 ]'
unset SWAP_WINDOW_S

# 6. low RAM: one sample kills a batch job; MIN_FREE_SAMPLES=4 rides out a short dip
echo 1000 > $T/sw; echo 60 > $T/free
t0=$SECONDS; $WL "batch ram" 23:59 -- zsh -c "sleep 0.5; echo 5 > $T/free; sleep 0.5; echo 60 > $T/free; sleep 3" 2> $T/err; r=$?
check "low free RAM for 0.5 s kills a batch job (rc $r)" '[ $r != 0 ] && grep -q "free RAM 5%" $T/err'
echo 60 > $T/free
MIN_FREE_SAMPLES=4 $WL "server ram" 23:59 -- zsh -c "sleep 0.5; echo 5 > $T/free; sleep 0.4; echo 60 > $T/free; sleep 1.5" 2> $T/err; r=$?
check "MIN_FREE_SAMPLES=4: a 0.4 s dip is ridden out (rc $r)" '[ $r = 0 ]'
check "no lock left behind" '[ ! -d $T/lock ]'
exit $rc
