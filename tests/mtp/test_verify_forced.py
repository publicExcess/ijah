"""lab-loop: the MTP loop's multi-row verify vs the 1-row decode, teacher-forced on the same tokens.

    NEED_GB=85 LOCK_OWNER=lab-loop tools/with_lock.sh "verify-forced (~83 GB)" HH:MM -- $PY tests/mtp/test_verify_forced.py
        [--n 128] [--rows 2,3,4,5] [--prompts think|fixed]

Per prompt: the greedy no-MTP continuation (n tokens) is the token stream; reference = 1-row decode logits after each
token (4-row prefill + ne_fn_model_step, the no-MTP path); test = ne_fn_mtp_verify_forced at R rows with a keep
pattern (all R rows kept, then a mixed pattern with rollbacks: KV truncation, lazy GDN / PLE commits). Reports top-1
agreement, KL(ref || verify) mean / max, and the reference's top-1 - top-2 logit gap at every disagreement (a flip
at a gap of ~0.1 or less is the multi-row arithmetic, not a state bug).
"""
import argparse
import ctypes as C
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
os.environ.setdefault("NE_FN_ROUND", "f32")
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--rows", default="2,3,4,5")
    ap.add_argument("--prompts", default="think")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    prompts = B.THINK if a.prompts == "think" else B.FIXED
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    rows = [int(x) for x in a.rows.split(",")]
    m = eng.fn_model(pack, 1024 + a.n + 64)
    mtp = B.Mtp(eng, m, max(rows) - 1, 4)
    f = eng.lib.ne_fn_mtp_verify_forced
    f.restype = C.c_int
    f.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p]
    V = E.FnModel.VOCAB
    bad = 0
    for name, q in prompts.items():
        ids = tok.encode(B.chat(a.pack, q), add_special_tokens=False).ids
        m.set_sampler(0.0, 1.0, 1, 0)
        m.reset()
        cont, _, _ = m.generate(ids, a.n, -1)
        m.reset()
        m.prefill(ids)
        ref = np.empty((len(cont), V), np.float32)
        for j, t in enumerate(cont):
            _, _, _, lg = m.step(t, True, logits=True)
            ref[j] = lg
        lr = logsm(ref)
        pr = np.exp(lr)
        ra = ref.argmax(-1)
        srt = np.sort(ref, -1)
        gap = srt[:, -1] - srt[:, -2]
        p = np.ascontiguousarray(ids, np.int32)
        c = np.ascontiguousarray(cont, np.int32)
        print(f"== {name}: {len(ids)} prompt + {len(cont)} tokens", flush=True)
        for R in rows:
            for pat in ([R], [1, R, 2, R - 1, R, 1, R]):
                kp = np.ascontiguousarray([min(R, max(1, x)) for x in pat], np.int32)
                out = np.zeros((len(cont), V), np.float32)
                k = f(C.c_void_p(mtp.ptr), p.ctypes.data, len(p), c.ctypes.data, len(c), R, kp.ctypes.data, len(kp),
                      out.ctypes.data)
                if k < 0:
                    raise E.EngineError(eng.last_error())
                lv = logsm(out[:k])
                kld = (pr[:k] * (lr[:k] - lv)).sum(-1)
                va = out[:k].argmax(-1)
                dis = np.nonzero(va != ra[:k])[0]
                worst = max((gap[i] for i in dis), default=0.0)
                ok = all(gap[i] < 0.25 for i in dis) and kld.mean() < 2e-2
                bad += not ok
                print(f"  R {R} keep {list(kp)}: {k} rows, top-1 {100 * (1 - len(dis) / k):.1f}% "
                      f"({len(dis)} flips, ref gaps {[round(float(gap[i]), 3) for i in dis][:6]}), KLD mean "
                      f"{kld.mean():.2e} max {kld.max():.2e}: {'OK' if ok else 'CHECK'}", flush=True)
    print("verify-forced:", "all OK" if not bad else f"{bad} cases to check")
    pack.close()


if __name__ == "__main__":
    main()
