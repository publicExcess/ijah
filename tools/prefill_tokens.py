"""lab-prefill: write a token-id file (int32) for build/pf-check from real text (the pack's tokenizer).
usage: $PY tools/prefill_tokens.py OUT N [files...]   (default text: this repo's docs + sources, repeated if short)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402
refenv.add_paths()
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PACK = Path(refenv.ijah_paths.pack())
out, n = sys.argv[1], int(sys.argv[2])
root = Path(__file__).resolve().parent.parent
files = [Path(f) for f in sys.argv[3:]] or sorted(root.glob("docs/*.md")) + sorted(root.glob("models/flash_next/*.mm")) \
    + sorted(root.glob("kernels/*.metal")) + sorted(root.glob("tools/*.py"))
text = "\n\n".join(f.read_text(errors="replace") for f in files)
tok = Tokenizer.from_file(str(PACK / "tokenizer.json"))
ids = tok.encode(text, add_special_tokens=False).ids
while len(ids) < n:
    ids = ids + ids
np.asarray(ids[:n], dtype=np.int32).tofile(out)
print(f"{out}: {n} tokens from {len(files)} files ({len(ids)} available)")
