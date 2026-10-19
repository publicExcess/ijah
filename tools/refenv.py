"""Paths and import setup shared by the reference tools (lab-ref).

Every script runs with the project venv (make venv; requirements.txt, plus requirements-ref.txt for MLX):
    .venv/bin/python -I -B tools/<script>.py          (or $IJAH_PY)
The numpy-only tools (compare, golden, engine) need nothing else. The MLX reference (fn_exl3 + mlx-vlm 0.7.7
qwen4_exp), the EXL3 codec (exl3_codec, encode) and the KLD scorer (kld_core) are not part of this repository: point
IJAH_REF_DIR at a tree with convert/, kernels/, model/ and pack/ holding them; the scripts that need them say so.
Pack locations: tools/ijah_paths.py (IJAH_PACKS, IJAH_PACK, IJAH_GOLDEN_PACK).
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ijah_paths  # noqa: E402

PY = ijah_paths.python()
EXL3 = ijah_paths.ref_dir() or Path("/nonexistent-IJAH_REF_DIR")
PACK = Path(ijah_paths.golden_pack())
GOLDEN = Path(os.environ.get("NE_GOLDEN", ROOT / "tests/golden/data"))


def add_paths(mlx=False):
    """the EXL3 reference (IJAH_REF_DIR: codec always, fn_exl3 with mlx=True) on sys.path, when it is there"""
    want = [str(EXL3 / "convert"), str(EXL3 / "kernels")]
    if mlx:
        want = [str(EXL3 / "model"), str(EXL3 / "convert"), str(EXL3 / "kernels")]
    for p in reversed(want):
        if p not in sys.path and Path(p).is_dir():
            sys.path.insert(0, p)


def need_ref(what):
    """SKIP (exit 0) unless IJAH_REF_DIR holds the reference"""
    return ijah_paths.ref_or_skip(what)


def mlx_vlm_file(rel):
    """a source file of the installed mlx-vlm package (the long golden prompt is mlx-vlm 0.7.7's qwen4_exp source)"""
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("mlx-vlm is not installed (pip install -r requirements-ref.txt)")
    return Path(list(spec.submodule_search_locations)[0]) / rel
