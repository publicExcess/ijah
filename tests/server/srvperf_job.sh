#!/bin/zsh
# tests/server/srvperf_job.sh   lab-srvperf whole-model job (ONE with_lock job, ~8-12 min):
#   NEED_GB=85 LOCK_WAIT=60 LOCK_OWNER=lab-srvperf tools/with_lock.sh "srvperf (~81-100 GB)" HH:MM -- tests/server/srvperf_job.sh
# 1. one-model server (base): RAM; HTTP step / MTP tok/s (+ the server's host_us_per_token); greedy reference text
# 2. two-model server (--model-id base,Swift): RAM; tok/s on both; greedy texts; model-switch time to first token
# 3. tools/generate.py direct on the same prompt (no MTP, MTP d2; and Swift greedy for the text check)
# 4. memory (vm_stat: wired / anonymous / compressor) for one model, two models, two models without the shared
#    prefill buffers (NE_SRV_SHARE_PF=0): idle, then after an ~9K-token prefill on each model (greedy texts compared)
# 5. --share-state (both models over one KV + recurrent state) vs separate states: memory, greedy texts, a conversation
#    continued on A after B ran (texts must match between the two configs), model-switch time to first token
# Env: NE_LIB (default build/libengine.dylib), J_CTX (262144), J_N (1024 tokens), J_PARTS (123; 4, 5). Results:
# tests/server/results/.
set -u
ROOT=${0:A:h:h:h}
cd $ROOT
source ${0:A:h:h:h}/tools/env.sh; ijah_need_py
PY=(${=IJAH_PY} -I -B)
PORT=${J_PORT:-8142}
export NE_LIB=${NE_LIB:-$ROOT/build/libengine.dylib}
OUT=$ROOT/tests/server/results
mkdir -p $OUT
STAMP=$(date +%m%d-%H%M)
LOGP=$OUT/srvperf-$STAMP
CTX=${J_CTX:-262144}
N=${J_N:-1024}
PARTS=${J_PARTS:-123}
BASE=Qwen3.8-Flash-Next-native
SWIFT=Swift-1.5-Flash-Next-native
KEYF=$(mktemp -t srvperf-key)
( umask 077; $PY -c 'import secrets; print(secrets.token_urlsafe(32), end="")' > $KEYF ); chmod 600 $KEYF
SP=
stop_server() {
  [ -n "$SP" ] || return 0
  kill -TERM $SP 2>/dev/null
  for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill -0 $SP 2>/dev/null && kill -KILL $SP
  wait $SP 2>/dev/null
  SP=
}
trap 'stop_server; rm -f $KEYF' EXIT INT TERM
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT busy"; exit 1; }
free_gb() { local t f; t=$(sysctl -n hw.memsize); f=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); echo $(( t / 1073741824 * ${f:-0} / 100 )); }
# the server's own memory: physical footprint (dirty + wired) and resident size incl. mapped weight pages
mem_of() { echo "footprint $(footprint -p $1 2>/dev/null | sed -n "s/.*Footprint: \([0-9.]* [KMG]B\).*/\1/p" | head -1); rss $(( $(ps -o rss= -p $1) / 1048576 )) GB; free $(free_gb) GB"; }
serve() {   # args: tag, server args...
  local tag=$1; shift
  $PY server/ne_server.py --port $PORT --ctx $CTX --lib $NE_LIB --key-file $KEYF "$@" 2> $LOGP-$tag-server.log &
  SP=$!
  local t0=$(date +%s)
  for i in {1..200}; do
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && break
    kill -0 $SP 2>/dev/null || { echo "server died:"; tail -20 $LOGP-$tag-server.log; exit 1; }
    sleep 1
  done
  echo "[$tag] up in $(( $(date +%s) - t0 )) s: $(grep 'engine ready' $LOGP-$tag-server.log | cut -c1-200)"
  echo "[$tag] memory with the server up (ctx $CTX): $(mem_of $SP)"
}
vmsnap() {   # GB: wired, anonymous, compressor-occupied, free (16 KB pages)
  vm_stat | awk -v t="$1" '/page size of/ {ps=$8} /Pages wired down/ {w=$4} /Anonymous pages/ {a=$3} /occupied by compressor/ {c=$5} /Pages free/ {f=$3}
    END {g=ps/1073741824; printf "VM %s: wired %.1f anon %.1f compressor %.1f free %.1f GB\n", t, w*g, a*g, c*g, f*g}'
}
B() { $PY tests/server/bench_http.py --url http://127.0.0.1:$PORT --key-file $KEYF "$@"; }
echo "free before: $(free_gb) GB; swap $(sysctl -n vm.swapusage)"
if [[ $PARTS == *1* ]]; then
  serve one --model-id $BASE
  B speed --n $N --tag one
  B speed --n 48 --paths step --greedy --tag one-greedy
  echo "[one] memory after the benches: $(mem_of $SP)"
  stop_server
  grep -v "^\s*$" $LOGP-one-server.log | tail -6
fi
if [[ $PARTS == *2* ]]; then
  serve two --model-id $BASE,$SWIFT
  B speed --n 48 --paths step --greedy --model $BASE --tag two-base-greedy
  B speed --n 48 --paths step --greedy --model $SWIFT --tag two-swift-greedy
  B switch --models $BASE,$SWIFT --rounds 3 --tag two
  B speed --n $((N / 2)) --model $SWIFT --tag two-swift
  B speed --n $((N / 2)) --model $BASE --tag two-base
  echo "[two] memory after the benches: $(mem_of $SP)"
  stop_server
  grep -v "^\s*$" $LOGP-two-server.log | tail -14
fi
if [[ $PARTS == *3* ]]; then
  Q="Write a detailed, long essay about the history of the printing press, its technology and its effects on Europe."
  for m in "" "--mtp 2"; do
    echo "--- direct tools/generate.py ${m:-no MTP}"
    $PY tools/generate.py "$Q" --max $N --seed 7 ${=m} 2>&1 | grep -- '^--- [0-9]'
  done
  SP_PACK=$($PY server/ne_server.py --list-models | sed -n "s|^$SWIFT: \([^ ]*\) .*|\1|p")
  echo "--- direct tools/generate.py Swift greedy 48 ($SP_PACK)"
  $PY tools/generate.py "$Q" --max 48 --greedy --pack $SP_PACK 2>&1 | tail -4
fi
if [[ $PARTS == *4* ]]; then
  LONG="$(cat docs/server.md)

Summarise the document above in one sentence."
  for cfg in one two two-noshare; do
    vmsnap "$cfg before"
    case $cfg in
      one) serve m-$cfg --model-id $BASE ;;
      two) serve m-$cfg --model-id $BASE,$SWIFT ;;
      two-noshare) export NE_SRV_SHARE_PF=0; serve m-$cfg --model-id $BASE,$SWIFT; unset NE_SRV_SHARE_PF ;;
    esac
    vmsnap "$cfg up"
    for m in $BASE $SWIFT; do
      [ $cfg = one ] && [ $m = $SWIFT ] && continue
      B speed --n 32 --paths step --greedy --model $m --prompt "$LONG" --tag $cfg-long
      B speed --n 48 --paths step --greedy --model $m --tag $cfg-greedy
    done
    vmsnap "$cfg after prefills"
    echo "[$cfg] $(mem_of $SP)"
    grep "shared by" $LOGP-m-$cfg-server.log
    stop_server
    vmsnap "$cfg stopped"
  done
fi
if [[ $PARTS == *5* ]]; then   # --share-state vs separate states: memory, texts after handovers, switch cost
  for cfg in ${=J_CFGS:-two two-share}; do
    vmsnap "$cfg before"
    if [ $cfg = two ]; then serve s-$cfg --model-id $BASE,$SWIFT; else serve s-$cfg --model-id $BASE,$SWIFT --share-state; fi
    vmsnap "$cfg up"
    B speed --n 48 --paths step --greedy --model $BASE --tag $cfg-greedy
    B speed --n 48 --paths step --greedy --model $SWIFT --tag $cfg-greedy
    B turns --models $BASE,$SWIFT --tag $cfg
    B switch --models $BASE,$SWIFT --rounds 3 --tag $cfg
    vmsnap "$cfg after"
    echo "[$cfg] $(mem_of $SP)"
    stop_server
    grep -v "^\s*$" $LOGP-s-$cfg-server.log | tail -16 | cut -c1-260
  done
fi
echo "free after: $(free_gb) GB; swap $(sysctl -n vm.swapusage)"
