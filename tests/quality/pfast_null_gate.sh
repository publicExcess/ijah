#!/bin/zsh
# pfast night job (NOT run yet): calibrate the paired teacher-KLD gate with null arms, then judge the parked trunk
# changes (NE_T6_FOLD_MIN=1024, NE_T6_SK_NARROW=128) against the null spread. docs/prefill.md 4i.
#
# Why: on 23 Oct a pure f32 reduction-order change (narrow split-K, 47-row chunks) "failed" set B (+0.00179
# [+0.00027, +0.00411]) and the fold threshold swung set C by -0.00228 / +0.00174 at prefill 100 / 500: per-window
# noise from the arithmetic floor (any rounding change moves the logits ~1e-2 KL at this depth) is as large as the
# effects judged. A null arm = a change that only reorders f32 sums / moves a rounding point with no expected quality
# effect, exercised at the same chunk sizes as the candidate:
#   48-row chunks: NE_T6_SK=64 (split-K for every Linear, reorder only), NE_T6_TG_MIN=17 (tensor-op vs fragment GEMM)
#   100 / 500-row chunks (fold path on master): NE_T6_DFOLD=0 (dense + fold_k unfused), NE_PF_SCAN=tb (GDN op order)
# Rule (fix before looking): a candidate is kept if, on every set and prefill size, |its paired mean diff| <= the
# largest |paired mean diff| of the null arms at that size, and it is not worse than master on the pooled mean.
#
# Usage (whole model, ~85 GB, one job after another; ~5 min per run, 26 runs ~2.2 h):
#   zsh tests/quality/pfast_null_gate.sh [OUTDIR]          then the summary at the end (also re-runnable: SUMMARY_ONLY=1)
set -u
ROOT=${0:A:h:h:h}; OUT=${1:-$ROOT/tests/quality/results/pfast-null}; mkdir -p $OUT
PY=${PY:-/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3}
export NE_TEACHER_RUN=${NE_TEACHER_RUN:-$HOME/Models/gguf/kld-work/teacher-view}
run() {  # name prefill maxpos sets env...
  local name=$1 pf=$2 mp=$3 sets=$4; shift 4
  [ -f $OUT/kld-$name.json ] && return
  (cd $ROOT && env "$@" NEED_GB=85 LOCK_OWNER=pfast-night LOCK_WAIT=30 zsh tools/with_lock.sh "pfast null gate $name (~85 GB)" 06:00 -- \
    $PY -I -B tools/kld_teacher.py --name $name --sets $sets --prefill $pf --max-pos $mp --out $OUT > $OUT/$name.log 2>&1)
  echo "$name exit $? $(date +%H:%M)"
}
if [ -z "${SUMMARY_ONLY:-}" ]; then
  for pf in 48 100 500; do
    mp=$((pf + 1024)); M=(NE_PF_NULL=master)
    run pf$pf-master $pf $mp A,B,C $M
    if [ $pf = 48 ]; then
      run pf$pf-null-sk64 $pf $mp A,B,C NE_T6_SK=64
      run pf$pf-null-tg17 $pf $mp A,B,C NE_T6_TG_MIN=17
      run pf$pf-cand-narrow $pf $mp A,B,C NE_T6_SK_NARROW=128
    else
      run pf$pf-null-dfold0 $pf $mp A,B,C NE_T6_DFOLD=0
      run pf$pf-null-scantb $pf $mp A,B,C NE_PF_SCAN=tb
      run pf$pf-cand-fold $pf $mp A,B,C NE_T6_FOLD_MIN=1024
      run pf$pf-cand-both $pf $mp A,B,C NE_T6_FOLD_MIN=1024 NE_T6_SK_NARROW=128
    fi
  done
  # set C has 4 windows: a second draw at another position (one folded 4096-row chunk in every arm, then the 499-row one)
  run pf4596-master 4596 5620 C NE_PF_NULL=master
  run pf4596-null-dfold0 4596 5620 C NE_T6_DFOLD=0
  run pf4596-null-scantb 4596 5620 C NE_PF_SCAN=tb
  run pf4596-cand-fold 4596 5620 C NE_T6_FOLD_MIN=1024
fi
for pf in 48 100 500 4596; do
  fs=($OUT/kld-pf$pf-master.json $OUT/kld-pf$pf-null-*.json(N) $OUT/kld-pf$pf-cand-*.json(N))
  [ -f $fs[1] ] || continue
  echo "=== prefill $pf"
  (cd $ROOT && $PY -I -B tools/kld_teacher.py --report $fs)
done
