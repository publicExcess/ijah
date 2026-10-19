"""Load golden cases written by tools/gen_goldens.py (numpy only).

    import golden
    golden.cases()                       # ['L00_decode1', ...]
    c = golden.load("L00_decode1")
    c.m                                  # manifest dict (step, rows, position_start, token_ids, cache_len_in, ...)
    c["gdn.qkv"]                         # np.float32 (or int) array, reference shape
    c.bf16("gdn.qkv")                    # the same as raw bf16 bits (np.uint16), for uploading to a bf16 kernel
    c.list(), c.shape(n), c.ref_dtype(n), c.weights(n), c.desc(n)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402


def root(path=None):
    return Path(path) if path else refenv.GOLDEN


def cases(path=None):
    idx = root(path) / "INDEX.json"
    if idx.exists():
        return json.loads(idx.read_text())["cases"]
    return sorted(p.name for p in root(path).iterdir() if (p / "manifest.json").exists())


def f32_to_bf16_bits(x):
    """Round-to-nearest-even f32 -> bf16, returned as uint16 bits (exact for values that already are bf16)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16
    return r.astype(np.uint16)


def bf16_bits_to_f32(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


class Case:
    def __init__(self, name, path=None):
        self.name = name
        self.dir = root(path) / name
        self.m = json.loads((self.dir / "manifest.json").read_text())
        self.t = self.m["tensors"]
        self._c = {}

    def __contains__(self, n):
        return n in self.t

    def __getitem__(self, n):
        if n not in self._c:
            self._c[n] = np.load(self.dir / self.t[n]["file"])
        return self._c[n]

    def get(self, n, default=None):
        return self[n] if n in self.t else default

    def bf16(self, n):
        assert self.t[n]["ref_dtype"] == "bfloat16", (n, self.t[n]["ref_dtype"])
        return f32_to_bf16_bits(self[n])

    def list(self):
        return list(self.t)

    def shape(self, n):
        return tuple(self.t[n]["shape"])

    def ref_dtype(self, n):
        return self.t[n]["ref_dtype"]

    def weights(self, n):
        return self.t[n]["pack_weights"]

    def desc(self, n):
        return self.t[n]["desc"]

    @property
    def layer(self):
        return self.m["layer"]

    @property
    def prefix(self):
        return f"language_model.model.layers.{self.m['layer']}."

    @property
    def positions(self):
        s = self.m["position_start"]
        return np.arange(s, s + self.m["rows"])


def load(name, path=None):
    return Case(name, path)


if __name__ == "__main__":
    for cn in (sys.argv[1:] or cases()):
        c = load(cn)
        print(f"{cn}: layer {c.layer} {c.m['layer_type']} step {c.m['step']} rows {c.m['rows']} pos {c.m['position_start']} "
              f"path {c.m['attn_path']} {c.m['bytes'] / 2**20:.1f} MiB")
        for n in c.list():
            print(f"   {n:28s} {str(c.shape(n)):22s} {c.ref_dtype(n):9s} {', '.join(c.weights(n))}")
