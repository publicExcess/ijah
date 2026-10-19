# tools/env.sh -- sourced by the shell scripts (zsh): the interpreter and pack defaults (see tools/ijah_paths.py).
#   IJAH_PY           default <repo>/.venv/bin/python (make venv / tools/setup.sh)
#   IJAH_PACKS        default ~/.cache/ijah/packs
#   IJAH_PACK         default $IJAH_PACKS/Qwen3.8-Flash-Next-EXL3-Ijah
#   IJAH_GOLDEN_PACK  default $IJAH_PACK
IJAH_ROOT=${${(%):-%x}:A:h:h}
export IJAH_PY=${IJAH_PY:-$IJAH_ROOT/.venv/bin/python}
export IJAH_PACKS=${IJAH_PACKS:-$HOME/.cache/ijah/packs}
export IJAH_PACK=${IJAH_PACK:-$IJAH_PACKS/Qwen3.8-Flash-Next-EXL3-Ijah}
export IJAH_GOLDEN_PACK=${IJAH_GOLDEN_PACK:-$IJAH_PACK}
ijah_need_py() {   # exit 2 with a hint when the venv is missing
  [ -x ${IJAH_PY%% *} ] || { echo "no Python at $IJAH_PY: run 'make venv' (or set IJAH_PY)" >&2; exit 2; }
}
ijah_need_pack() {   # ijah_need_pack WHAT [DIR]: SKIP (exit 0) when the pack is missing
  local d=${2:-$IJAH_PACK}
  [ -f $d/config.json ] || { echo "SKIP $1: no pack at $d (set IJAH_PACK, or: hf download SiliconSpecies/${d:t} --local-dir $d)"; exit 0; }
}
