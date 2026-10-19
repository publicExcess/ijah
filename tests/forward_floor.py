"""lab-integrate: the decode path's "floor" -- how far two prefill paths of the same tokens end up apart in the logits.

    NEED_GB=85 LOCK_OWNER=... LOCK_WAIT=30 tools/with_lock.sh "forward floor (~81 GB)" HH:MM -- \
        $PY tests/forward_floor.py [--round on,f32] [--streams real64,greedy,real2k]

Per stream (N prompt tokens, then K teacher-forced 1-row decode steps): reference A = N + K 1-row decode steps;
paths: A2 = A again (run-to-run determinism), B4 = lab-forward's 4-row prefill (NE_FN_PREFILL=steps) of the N tokens,
BC = lab-prefill's chunked prefill of N - 1 + the decode step, each followed by the same K 1-row steps.
Reports KL(A || path) mean / max over the K + 1 positions, top-1 agreement, the rms of the (centred) raw logit
difference (entropy-independent size of the perturbation), the reference's mean entropy (KLD between two nearby
distributions scales with it), B4 vs BC, and A with reference rounding ON vs plain f32.
Streams: real64 / real2k = repository text (tools/prefill_tokens.py's, = lab-prefill's pf-check model setup, N = 64 /
2048, K = 32); greedy = a chat prompt + the model's own greedy continuation (prompt + 32 continuation tokens
prefilled, the next 32 measured: lab-loop's kind of positions).
"""
import argparse
import ctypes as C
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402

V = E.FnModel.VOCAB


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def real_tokens(pack, n):
    root = refenv.ROOT
    files = sorted(root.glob("docs/*.md")) + sorted(root.glob("models/flash_next/*.mm"))
    tok = Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files), add_special_tokens=False).ids
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def run(m, eng, toks, N, K, path):
    """logits [K + 1][V] of positions N - 1 .. N + K - 1 after `path` prefilled toks[:N]"""
    m.reset()
    out = []
    if path in ("A", "A2"):
        for t in toks[:N - 1]:
            m.step(t, head=False)
        out.append(m.step(toks[N - 1], head=True, logits=True)[3])
    else:
        os.environ["NE_FN_PREFILL"] = "steps" if path == "B4" else "chunked"
        out.append(m.prefill(toks[:N], logits=True)[1])
        os.environ.pop("NE_FN_PREFILL", None)
    for t in toks[N:N + K]:
        out.append(m.step(t, head=True, logits=True)[3])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.golden_pack())  # IJAH_GOLDEN_PACK: the pack the goldens / MLX reference were made from
    ap.add_argument("--round", default="on,f32")
    ap.add_argument("--streams", default="real64,greedy,real2k")
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    K = a.k
    A_by = {}  # (stream) -> {round: raw logits of A}

    def report(rnd, sname, N, path, ref, raw_ref, raw_q):
        pr = np.exp(ref)
        ent = -(pr * ref).sum(-1)
        lq = logsm(raw_q)
        kl = (pr * (ref - lq)).sum(-1)
        top = (ref.argmax(-1) == lq.argmax(-1)).mean()
        low = ent < 0.5
        d = raw_ref.astype(np.float64) - raw_q
        d -= d.mean(-1, keepdims=True)
        sd = np.sqrt((d * d).mean(-1)).mean()  # raw logit perturbation, entropy-independent
        print(f"round {rnd:3s} {sname:7s} N {N:4d} {path:6s}: KLD mean {kl.mean():.2e} max {kl.max():.2e} "
              f"median {np.median(kl):.1e} top-1 {top * 100:.0f}% | logit diff rms {sd:.3f} | ref entropy mean "
              f"{ent.mean():.2f} (KLD at entropy < 0.5: {kl[low].mean() if low.any() else float('nan'):.1e} over "
              f"{low.sum()}, >= 0.5: {kl[~low].mean() if (~low).any() else float('nan'):.1e} over {(~low).sum()})",
              flush=True)

    for rnd in a.round.split(","):
        eng.set_ref_rounding(rnd == "on")
        m = eng.fn_model(pack, 2048 + 2 * K + 128)
        m.set_sampler(0.0, 1.0, 1, 0)
        for sname in a.streams.split(","):
            if sname.startswith("real"):
                N = 64 if sname == "real64" else 2048
                toks = real_tokens(a.pack, N + K)
            elif sname in A_by:  # the same tokens as the first rounding mode's run
                N, toks = A_by[sname]["toks"]
            else:
                ids = tok.encode(B.chat(a.pack, B.THINK["code_lru"]), add_special_tokens=False).ids
                m.reset()
                cont, _, _ = m.generate(ids, 2 * K + 1, -1)
                m.reset()
                toks = ids + cont
                N = len(ids) + K
            raw = {p: run(m, eng, toks, N, K, p) for p in ("A", "A2", "B4", "BC")}
            ref = logsm(raw["A"])
            for path in ("A2", "B4", "BC"):
                report(rnd, sname, N, path, ref, raw["A"], raw[path])
            report(rnd, sname, N, "B4|BC", logsm(raw["B4"]), raw["B4"], raw["BC"])
            A_by.setdefault(sname, {})[rnd] = (N, raw["A"])
            A_by[sname]["toks"] = (N, toks)
        m.free()
    for sname, d in A_by.items():  # the same 1-row path, reference rounding ON vs plain f32
        if "on" in d and "f32" in d:
            N = d["on"][0]
            report("on|f32", sname, N, "A", logsm(d["f32"][1]), d["f32"][1], d["on"][1])


if __name__ == "__main__":
    main()
