#!/bin/zsh
# kld_gguf.sh GGUF NAME SETS : teacher KLD of a GGUF through llama.cpp on kld_teacher's sets and positions (lab splashcmp).
# Per set: tools/gguf_logits dumps every window's full-vocabulary f32 logits (llama.cpp, all layers on Metal), then
# tools/kld_teacher.py --logits-dir scores them exactly as it scores the engine (same teacher, positions, blocks), then
# the dump is deleted (set C: 4 x 32.5 GB). Writes tests/quality/results/kld-NAME-SET.json (+ -pos.npz).
# Needs: llama.cpp (brew install llama.cpp) and build-gguf/gguf_logits (build line at the top of tools/gguf_logits.cpp);
# the bf16 teacher run (NE_TEACHER_RUN, see tools/kld_teacher.py; not part of this repository).
# Run under tools/with_lock.sh with NEED_GB sized to the GGUF (~105 for a 100 GB file). Tokens: kld_teacher --dump-tokens.
# Env: KLD_WORK (default ~/.cache/ijah/kld-work: tokens + logits dumps), KLD_OUT (default tests/quality/results).
set -eu
G=$1; NAME=$2; SETS=$3
R=${0:A:h:h}; W=${KLD_WORK:-$HOME/.cache/ijah/kld-work}
source $R/tools/env.sh; ijah_need_py
NEPY=$IJAH_PY
[ -x $R/build-gguf/gguf_logits ] || { echo "no $R/build-gguf/gguf_logits: build it (see tools/gguf_logits.cpp)"; exit 2; }
[ -d $W/tok ] || $NEPY -I -B $R/tools/kld_teacher.py --dump-tokens $W/tok --sets A,B,C
for S in ${(s:,:)SETS}; do
  L=$W/logits-$NAME-$S; rm -rf $L; mkdir -p $L
  names=($(cd $W/tok && ls $S-*.tok | sed 's/\.tok$//' | sort -t- -k2 -n))
  echo "$(date +%H:%M:%S) $NAME $S: dumping ${#names} windows"
  $R/build-gguf/gguf_logits $G $W/tok $L $names 2>&1 | grep -vE "^(llama_|load|print_info|ggml_|graph_|sched_|init_|set_|create_|alloc_|llm_|.*Metal)" | tail -${#names}
  echo "$(date +%H:%M:%S) $NAME $S: scoring"
  ( cd $R && $NEPY -I -B tools/kld_teacher.py --logits-dir $L --pack $G --name $NAME-$S --sets $S --out ${KLD_OUT:-$R/tests/quality/results} ) 2>&1 | tail -3
  [ -f ${KLD_OUT:-$R/tests/quality/results}/kld-$NAME-$S.json ] && rm -rf $L
done
