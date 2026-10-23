"""The server's layout, stdlib only: where the registry, packs, drafters, API key and prefix cache live, and the
default port. server/ne_server.py and tools/ijah both import it, so the CLI reads exactly what the server uses.
(release, 23 Oct 2026)"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
import ijah_paths  # noqa: E402

# tools/serve.sh: NE_HOME (tests) puts the key and the run files in one scratch directory
RUN_DIR = Path(os.environ.get("NE_HOME") or "~/.cache/ijah").expanduser()        # log, pid, supervisor state
KEY_FILE = Path(os.environ.get("IJAH_KEY_FILE") or Path(os.environ.get("NE_HOME") or "~/.config/ijah") / "api-key"
                ).expanduser()                                                    # = tools/serve.sh keygen's
CACHE_DIR = Path.home() / ".cache/ijah/prefix-cache"
PORT = int(os.environ.get("NE_SERVE_PORT") or 8140)   # tools/serve.sh passes --port $NE_SERVE_PORT (default 8140)
# tools/ijah download: where the published packs and the drafter go (the public layout; server/models.json lists them)
PACKS_DIR = ijah_paths.packs_dir()   # IJAH_PACKS, default ~/.cache/ijah/packs
DRAFTERS_DIR = Path(os.environ.get("IJAH_DRAFTERS") or "~/.cache/ijah/drafters").expanduser()
NGRAM_FILE = "ngram_table.bin"   # the n-gram embedding table every pack folder needs (102 GB; one file, hard-linked)
# name -> Hugging Face repo, pinned revision, destination, approximate size (GB), what it is
DOWNLOADS = {
    "base": dict(repo="SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah", rev="e65cb66033b85cb4fca8e02ab4f2693a9df20c0e",
                 dest=PACKS_DIR / "Qwen3.8-Flash-Next-EXL3-Ijah", gb=83, model_id="Qwen3.8-Flash-Next-native", what="Qwen3.8-Flash-Next T6 pack (preferred)"),
    "swift": dict(repo="SiliconSpecies/Swift-1.5-Flash-Next-EXL3-Ijah", rev="ee339e0dddc2d927e29347a55ebc467792c07fef",
                  dest=PACKS_DIR / "Swift-1.5-Flash-Next-EXL3-Ijah", gb=83, model_id="Swift-1.5-Flash-Next-native",
                  what="Swift-1.5 T6 pack (non-commercial licence)"),
    "base-4bit": dict(repo="SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah-4bit",
                      rev="b7bf698360afc7c58cbd124ec62fdf51789b7736",
                      dest=PACKS_DIR / "Qwen3.8-Flash-Next-EXL3-Ijah-4bit", gb=67,
                      model_id="Qwen3.8-Flash-Next-native-4bit",
                      what="Qwen3.8 with 4-bit experts: faster, lower fidelity (T6 is preferred)"),
    "drafter": dict(repo="incoai/Qwen3.8-Flash-Next-DFlash2", rev="a22a1744e85c17364dcc54a6a420af7135d4c7f6",
                    dest=DRAFTERS_DIR / "Qwen3.8-Flash-Next-DFlash2", gb=1.2,
                    what="Inco's DFlash2 drafter (Apache-2.0; on by default when present)"),
    "ngram": dict(repo="SiliconSpecies/Qwen3.8-Flash-Next-ngram-table", rev="c605f0a86063e75674820eb251911d309679c484",
                  dest=None, gb=102, files=[NGRAM_FILE],
                  what="n-gram table for every pack (one copy, hard-linked into each pack folder)"),
}
PACK_DOWNLOADS = ("base", "swift", "base-4bit")
MODELS = HERE / "models.json"


def pack_complete(d):
    """A pack directory is servable: config + index + tokenizer present, and a T6 pack (trunk6 metadata) has all its
    layers converted plus the bf16 mixer (a partial T6 pack is a valid but mixed model: not served by id)."""
    d = Path(d)
    if not all((d / f).exists() for f in ("config.json", "model.safetensors.index.json", "tokenizer.json")):
        return False
    t6 = json.loads((d / "model.safetensors.index.json").read_text()).get("metadata", {}).get("trunk6")
    if t6 is None:
        return True
    n = json.loads((d / "config.json").read_text()).get("text_config", {}).get("num_hidden_layers", 48)
    return len(t6.get("layers", [])) == n and bool(t6.get("mixer_bf16"))


DRAFTER_FILES = ("config.json", "model.safetensors")
DRAFTER_HINT = ("hf download incoai/Qwen3.8-Flash-Next-DFlash2 --revision a22a1744e85c "
                "--local-dir ~/.cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2")


def canonical_id(model_id, registry=MODELS):
    """release (23 Oct): (id, wants_drafter). A -dflash id is an alias ("alias_of") of its plain id that requires the
    DFlash2 drafter; any other id (or None = the default) maps to itself, False."""
    if not model_id:
        return model_id, False
    e = json.loads(Path(registry).read_text())["models"].get(model_id, {})
    return (e["alias_of"], True) if e.get("alias_of") else (model_id, False)


def load_registry(path=MODELS):
    return json.loads(Path(path).read_text())


def drafter_candidates(model_id, override=None, registry=MODELS):
    """the checkpoint dirs a model id's drafter is looked for in (override = --drafter-path; '@IJAH_DRAFTERS/' = the
    drafters directory, IJAH_DRAFTERS, default ~/.cache/ijah/drafters), [] = no drafter entry"""
    model_id = canonical_id(model_id, registry)[0]
    d = json.loads(Path(registry).read_text())["models"].get(model_id, {}).get("drafter")
    if not d:
        return []
    if override:
        return [str(Path(override).expanduser())]
    return [str(DRAFTERS_DIR / c[len("@IJAH_DRAFTERS/"):]) if c.startswith("@IJAH_DRAFTERS/") else str(Path(c).expanduser())
            for c in d.get("paths", [])]


def find_drafter(cands):
    """the first candidate dir holding the DFlash2 checkpoint files, or None"""
    return next((c for c in cands if all((Path(c) / f).is_file() for f in DRAFTER_FILES)), None)


def resolve_model(model_id, pack=None, registry=MODELS):
    """(model id, pack dir, note). --pack wins (the id is then only the name reported); else the registry's first
    complete pack for the id ('@IJAH_PACKS/' = the packs directory, IJAH_PACKS, default ~/.cache/ijah/packs); unknown
    id without --pack -> SystemExit listing the known ids."""
    reg = json.loads(Path(registry).read_text())
    model_id = canonical_id(model_id or reg["default"], registry)[0]   # a -dflash alias -> its plain id
    if pack:
        return model_id, str(Path(pack).expanduser()), "--pack"
    if model_id not in reg["models"]:
        raise SystemExit(f"unknown --model-id {model_id!r} (known: {', '.join(reg['models'])}; or pass --pack DIR)")
    tried = []
    for c in reg["models"][model_id]["packs"]:
        d = str(PACKS_DIR / c[len("@IJAH_PACKS/"):]) if c.startswith("@IJAH_PACKS/") else str(Path(c).expanduser())
        if pack_complete(d):
            return model_id, d, f"registry ({'first choice' if not tried else 'fallback; incomplete or missing: ' + ', '.join(tried)})"
        tried.append(d)
    dl = next((n for n, x in DOWNLOADS.items() if x.get("model_id") == model_id), None)
    hint = f"; download it: tools/ijah download {dl}" if dl else ""
    raise SystemExit(f"--model-id {model_id}: no complete pack among {tried}{hint} (or pass --pack DIR)")
