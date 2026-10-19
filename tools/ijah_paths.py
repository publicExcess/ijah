"""Where Ijah finds its interpreter, packs and optional reference code (stdlib only; imported by the server, the tools
and the tests).

    IJAH_PY           Python for scripts that start other scripts (default: <repo>/.venv/bin/python)
    IJAH_PACKS        directory of downloaded packs (default ~/.cache/ijah/packs); the server's model ids resolve here:
                        Qwen3.8-Flash-Next-native   -> <IJAH_PACKS>/Qwen3.8-Flash-Next-EXL3-Ijah
                        Swift-1.5-Flash-Next-native -> <IJAH_PACKS>/Swift-1.5-Flash-Next-EXL3-Ijah
                      (hf download SiliconSpecies/<repo> --local-dir <IJAH_PACKS>/<repo>)
    IJAH_PACK         the pack tests and tools use by default (default: the Qwen3.8 pack above)
    IJAH_GOLDEN_PACK  the pack the layer goldens were generated from (default: IJAH_PACK)
    IJAH_REF_DIR      the MLX / numpy EXL3 reference (fn_exl3, exl3_codec, encode, kld_core: not part of this
                      repository); only the golden / reference / converter tools need it
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_REPO = "Qwen3.8-Flash-Next-EXL3-Ijah"
SWIFT_REPO = "Swift-1.5-Flash-Next-EXL3-Ijah"
HF_ORG = "SiliconSpecies"


def python():
    return os.environ.get("IJAH_PY") or str(ROOT / ".venv/bin/python")


def packs_dir():
    return Path(os.environ.get("IJAH_PACKS") or "~/.cache/ijah/packs").expanduser()


def pack():
    """the default pack directory (str; may not exist)"""
    p = os.environ.get("IJAH_PACK") or os.environ.get("NE_PACK")
    return str(Path(p).expanduser()) if p else str(packs_dir() / BASE_REPO)


def golden_pack():
    p = os.environ.get("IJAH_GOLDEN_PACK")
    return str(Path(p).expanduser()) if p else pack()


def ref_dir():
    p = os.environ.get("IJAH_REF_DIR")
    return Path(p).expanduser() if p else None


def download_hint(repo=BASE_REPO):
    return f"hf download {HF_ORG}/{repo} --local-dir {packs_dir() / repo}"


def skip(what, why):
    """print a SKIP line and exit 0 (a test that cannot run here is not a failure)"""
    print(f"SKIP {what}: {why}")
    sys.exit(0)


def pack_or_skip(what, path=None, need=("tokenizer.json",)):
    """-> the pack directory, or SKIP (exit 0) when the files `need` are missing. For CPU tests that only read the
    tokenizer and chat template, the small files of a pack are enough:
        hf download SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah --include '*.json' '*.jinja' --local-dir ..."""
    d = Path(path or pack())
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        skip(what, f"no pack at {d} ({', '.join(missing)} missing); set IJAH_PACK or run: {download_hint()}")
    return str(d)


def ref_or_skip(what):
    d = ref_dir()
    if d is None or not d.is_dir():
        skip(what, "needs the MLX / numpy EXL3 reference (IJAH_REF_DIR; not part of this repository)")
    return d
