"""lab-t6wire: engine-vs-engine logits KLD on the WHOLE prompts (CPU): KL(A || B) per position from the logits saved by
tools/compare_logits.py --save A (e.g. the K5 pack) and --save B (the T6 pack), same tokens (teacher-forced).

    $PY -I -B tests/trunk6/engine_kld.py DIR_A DIR_B
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    allk, allt = [], []
    out = {}
    for fa in sorted(A.glob("*.engine.npy")):
        fb = B / fa.name
        if not fb.exists():
            continue
        a, b = f32(np.load(fa)), f32(np.load(fb))
        la, lb = logsm(a), logsm(b)
        k = (np.exp(la) * (la - lb)).sum(-1)
        t = (a.argmax(-1) == b.argmax(-1)).astype(np.float64)
        name = fa.name.split(".")[0]
        out[name] = {"n": len(k), "kld_mean": float(k.mean()), "kld_median": float(np.median(k)),
                     "kld_max": float(k.max()), "top1": float(t.mean())}
        print(f"{name:6s} n={len(k)}: KLD mean {k.mean():.2e} median {np.median(k):.2e} max {k.max():.2e} "
              f"top1 {t.mean():.3f}")
        allk.append(k)
        allt.append(t)
    k, t = np.concatenate(allk), np.concatenate(allt)
    out["all"] = {"n": len(k), "kld_mean": float(k.mean()), "kld_median": float(np.median(k)), "top1": float(t.mean())}
    print(f"ALL {len(k)} positions: KL(A || B) mean {k.mean():.2e} median {np.median(k):.2e} top1 {t.mean():.4f}")
    print(json.dumps(out["all"]))


if __name__ == "__main__":
    main()
