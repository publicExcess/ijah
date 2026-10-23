#!/bin/zsh
# tests/server/test_ijah_cli.sh   CPU-only (mock engine, no machine lock): tools/ijah over tools/serve.sh (release).
# A scratch NE_HOME, port 8146, a stand-in for with_lock.sh: ijah serve start MODEL -- args -> up; status; models
# names the served id and its drafter; restart reuses the last start's model + args; stop; download --dry-run pins
# every revision; doctor runs to the end.
set -u
ROOT=${0:A:h:h:h}
T=$(mktemp -d -t ne-ijah)
export NE_HOME=$T NE_SERVE_PORT=8146 NE_SERVE_WRAP=$T/wrap.sh NE_SKIP_BUILD=1 IJAH_DRAFTERS=$T/drafters
# (IJAH_PACKS stays the caller's: the mock server reads the packs' tokenizer; download / link use a scratch one)
IJ=$ROOT/tools/ijah
FAIL=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; FAIL=1; }
cleanup() { $IJ serve stop >/dev/null 2>&1; rm -rf $T; }
trap cleanup EXIT
printf '#!/bin/zsh\nshift 2; [ "${1:-}" = "--" ] && shift\nexec "$@"\n' > $T/wrap.sh; chmod +x $T/wrap.sh
( umask 077; print -n "test-key-$RANDOM$RANDOM$RANDOM$RANDOM" > $T/api-key )
mkdir -p $T/drafter && for f in config.json model.safetensors; do echo '{}' > $T/drafter/$f; done
M=Swift-1.5-Flash-Next-native
$IJ serve start $M -- --mock --ctx 8192 --drafter-path $T/drafter > $T/out 2>&1 && grep -q "up on 127.0.0.1:8146" $T/out \
  && ok "ijah serve start MODEL -- args: up" || bad "start: $(tail -2 $T/out)"
$IJ serve status | grep -q "listening on 8146" && ok "ijah serve status: listening" || bad "status: $($IJ serve status)"
$IJ models | grep -q "serving on 127.0.0.1:8146: $M (dflash2)" && ok "ijah models: the served id with its drafter" \
  || bad "models: $($IJ models | tail -1)"
$IJ serve restart > $T/out2 2>&1 && grep -q "stopped" $T/out2 && grep -q "up on" $T/out2 \
  && $IJ models | grep -q "serving on 127.0.0.1:8146: $M (dflash2)" && ok "ijah serve restart: same model and args" \
  || bad "restart: $(tail -3 $T/out2)"
$IJ serve log 5 | grep -q . && ok "ijah serve log N" || bad "log"
$IJ serve stop | grep -q stopped && ! lsof -nP -iTCP:8146 -sTCP:LISTEN >/dev/null 2>&1 && ok "ijah serve stop" || bad "stop"
( exec $IJ serve run $M -- --mock --ctx 8192 --drafter none > $T/run.out 2>&1 ) &
RP=$!
for i in {1..60}; do curl -s -m 1 http://127.0.0.1:8146/health 2>/dev/null | grep -q ok && break; sleep 0.5; done
$IJ models | grep -q "serving on 127.0.0.1:8146: $M (mtp)" && ok "ijah serve run (foreground, --drafter none): up, MTP head" \
  || bad "run: $(tail -2 $T/run.out)"
$IJ serve stop >/dev/null; wait $RP 2>/dev/null
! lsof -nP -iTCP:8146 -sTCP:LISTEN >/dev/null 2>&1 && ok "ijah serve stop ends a foreground run" || bad "run still listening"
D=$(IJAH_PACKS=$T/packs $IJ download all --dry-run)
[ $(print -r -- $D | grep -c -- '--revision [0-9a-f]\{40\}') = 5 ] && ok "download all --dry-run: 5 pinned revisions" \
  || bad "download: $D"
print -r -- $D | grep -q "ngram_table.bin --revision .* --local-dir $T/packs/Qwen3.8-Flash-Next-EXL3-Ijah\$" \
  && ok "the n-gram table goes into the base pack folder" || bad "ngram dest"
mkdir -p $T/packs/Qwen3.8-Flash-Next-EXL3-Ijah $T/packs/Swift-1.5-Flash-Next-EXL3-Ijah
echo table > $T/packs/Qwen3.8-Flash-Next-EXL3-Ijah/ngram_table.bin
IJAH_PACKS=$T/packs $IJ link > $T/out3 2>&1
[ $(stat -f %i $T/packs/Qwen3.8-Flash-Next-EXL3-Ijah/ngram_table.bin) = $(stat -f %i $T/packs/Swift-1.5-Flash-Next-EXL3-Ijah/ngram_table.bin 2>/dev/null || echo x) ] \
  && ok "ijah link: the n-gram table hard-linked into the Swift pack" || bad "link: $(cat $T/out3)"
$IJ doctor > $T/out4 2>&1; grep -qE "^(all good|[0-9]+ problem)" $T/out4 && ok "ijah doctor runs to the end" || bad "doctor: $(tail -3 $T/out4)"
$IJ bogus >/dev/null 2>&1; [ $? = 64 ] && ok "unknown command: exit 64" || bad "unknown command"
echo; [ $FAIL = 0 ] && echo "all passed" || echo "FAILED"
exit $FAIL
