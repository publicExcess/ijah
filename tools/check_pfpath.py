"""pfast: old vs new prefill arithmetic on the same input, by the logits that follow the chunk.

    NEED_GB=85 LOCK_OWNER=pfast tools/with_lock.sh "check_pfpath (~85 GB)" HH:MM -- $PY tools/check_pfpath.py \
        [--ctx-tokens 16384] [--sizes 45,100,128,500,1000] [--steps 32] [--floor 1]

From a context of real text (model prefill, snapshot), per chunk size n: the same n tokens through a fresh Prefill
under each arm's env (engine knobs are read at record time), then `--steps` teacher-forced decode steps with the head
(the logits every later token sees through the chunk's KV rows and states). Arms: old = NE_T6_FOLD_MIN=64
NE_T6_SK_NARROW=0 (master), new = this branch's defaults, old2 = old again (determinism), narrow = only the narrow
split-K (fold as master), and floor = the n chunk rows as 1-row decode steps (the decode path's arithmetic).
Per pair: KL(a || b) of the f32 log-softmax, mean and max over the steps, max |logit diff|, argmax agreement.
"""
import argparse
import ctypes as C
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import backend as B  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ARMS = {"old": {"NE_T6_FOLD_MIN": "64", "NE_T6_SK_NARROW": "0"}, "new": {}, "old2": {"NE_T6_FOLD_MIN": "64",
        "NE_T6_SK_NARROW": "0"}, "narrow": {"NE_T6_FOLD_MIN": "64"}}
KNOBS = ("NE_T6_FOLD_MIN", "NE_T6_SK_NARROW")


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=(ROOT / "PACK_READY").read_text().strip())
    ap.add_argument("--ctx-tokens", type=int, default=16384)
    ap.add_argument("--sizes", default="45,100,128,500,1000")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--floor", type=int, default=1)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    files = sorted(ROOT.glob("docs/*.md")) + sorted(ROOT.glob("models/flash_next/*.mm")) + sorted(ROOT.glob("server/*.py"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files), add_special_tokens=False).ids
    be = B.RealBackend(ROOT, a.pack, a.ctx_tokens + 8192, 2, ROOT / "build/libengine.dylib", None, print)
    lib, vp = be.eng.lib, C.c_void_p
    mp = vp(be.m.ptr)
    lib.ne_fn_model_snapshot.restype = vp
    lib.ne_fn_model_snapshot.argtypes = [vp]
    lib.ne_fn_model_restore.argtypes = [vp, vp]
    lib.ne_fn_snap_free.argtypes = [vp]
    lib.ne_fn_prefill_create.restype = vp
    lib.ne_fn_prefill_create.argtypes = [vp, C.c_int, C.c_int]
    lib.ne_fn_prefill_free.argtypes = [vp]
    lib.ne_fn_prefill_tokens.argtypes = [vp, vp, C.c_int, vp]
    lib.ne_fn_model_step.argtypes = [vp, C.c_int, C.c_int, vp, vp, vp, vp, vp]
    lib.ne_fn_model_prefill.argtypes = [vp, vp, C.c_int, vp, vp, vp]
    ctx = np.array(ids[:a.ctx_tokens], np.int32)
    t = time.time()
    assert lib.ne_fn_model_prefill(mp, ctx.ctypes.data, len(ctx), None, None, None) == 0, be.eng.last_error()
    print(f"context {len(ctx)} tokens: {time.time() - t:.1f} s", flush=True)
    snap = lib.ne_fn_model_snapshot(mp)
    assert snap, be.eng.last_error()
    V = B.RealBackend.VOCAB
    lg = np.zeros((a.steps, V), np.float32)
    pos = a.ctx_tokens
    rows = []
    for n in [int(x) for x in a.sizes.split(",") if x]:
        chunk = np.array(ids[pos:pos + n], np.int32)
        follow = ids[pos + n:pos + n + a.steps]
        pos += n + a.steps
        res = {}
        arms = list(ARMS) + (["floor"] if a.floor else [])
        for arm in arms:
            saved = {k: os.environ.get(k) for k in KNOBS}
            for k in KNOBS:
                os.environ.pop(k, None)
            os.environ.update(ARMS.get(arm, {}))
            assert lib.ne_fn_model_restore(mp, snap) == 0, be.eng.last_error()
            if arm == "floor":
                for x in chunk:
                    assert lib.ne_fn_model_step(mp, int(x), 0, None, None, None, None, None) == 0, be.eng.last_error()
            else:
                h = lib.ne_fn_prefill_create(mp, 1024, 0)
                assert h, be.eng.last_error()
                assert lib.ne_fn_prefill_tokens(h, chunk.ctypes.data, n, None) == 0, be.eng.last_error()
                lib.ne_fn_prefill_free(h)
            for i, x in enumerate(follow):
                assert lib.ne_fn_model_step(mp, int(x), 1, lg[i].ctypes.data, None, None, None, None) == 0, \
                    be.eng.last_error()
            res[arm] = lg.copy()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for x, y in [("old", "old2"), ("old", "new"), ("old", "narrow"), ("old", "floor")]:
            if x not in res or y not in res:
                continue
            p, q = logsm(res[x]), logsm(res[y])
            kl = (np.exp(p) * (p - q)).sum(-1)
            md = float(np.abs(res[x] - res[y]).max())
            agree = int((res[x].argmax(-1) == res[y].argmax(-1)).sum())
            rows.append((n, f"{x} vs {y}", kl.mean(), kl.max(), md, agree))
            print(f"n {n:5d} {x:>4s} vs {y:<6s}: KL mean {kl.mean():.3e} max {kl.max():.3e}  max|dlogit| {md:.3f}  "
                  f"argmax {agree}/{a.steps}", flush=True)
    lib.ne_fn_snap_free(snap)
    print(f"\n| rows | pair | KL mean | KL max | max abs logit diff | argmax agree (of {a.steps}) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.2e} | {r[3]:.2e} | {r[4]:.3f} | {r[5]} |")


if __name__ == "__main__":
    main()
