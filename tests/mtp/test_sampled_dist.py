"""lab-loop: end-to-end distribution check of sampled MTP (exact speculative sampling with sampled drafts).

    NEED_GB=85 LOCK_OWNER=lab-loop tools/with_lock.sh "sampled-dist (~83 GB)" HH:MM -- $PY tests/mtp/test_sampled_dist.py
        [--runs 400] [--depth 3] [--prompt text_crisis]

The model's sampler seed is fixed, so the prefill's sample x_P is the same in every run; the MTP seed varies. The
first round's first emitted token (out[1]: d_1 if accepted, else the correction) must then be distributed exactly as
the target's filtered distribution p(. | prompt, x_P) (temperature 1, top-k 20, top-p 0.95), computed here from the
1-row decode logits. Reports the empirical vs p total variation, the TV that sampling noise alone gives at this
many runs (simulated from p), a chi-square over p's support, the share of rounds whose d_1 was accepted, and the
same for out[2] vs the target's distribution given (x_P, out[1]) (computed per observed out[1]).
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
os.environ.setdefault("NE_FN_ROUND", "f32")
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402

TEMP, TOP_P, TOP_K = 1.0, 0.95, 20


def filt(lg):
    """temperature -> top-k -> top-p (smallest prefix with mass >= p), lowest id on ties: {id: prob}"""
    idx = np.lexsort((np.arange(len(lg)), -lg.astype(np.float64)))[:TOP_K]
    w = np.exp((lg[idx].astype(np.float64) - lg[idx[0]]) / TEMP)
    c = np.cumsum(w / w.sum())
    n = int(np.searchsorted(c, TOP_P) + 1) if TOP_P < 1 else len(idx)
    n = min(n, len(idx))
    w = w[:n] / w[:n].sum()
    return dict(zip(idx[:n].tolist(), w.tolist()))


def compare(name, samples, p, rng):
    n = len(samples)
    cnt = Counter(samples)
    tv = 0.5 * sum(abs(cnt.get(t, 0) / n - p.get(t, 0.0)) for t in set(cnt) | set(p))
    ids = list(p)
    pr = np.array([p[t] for t in ids])
    sim = []
    for _ in range(2000):
        c = rng.multinomial(n, pr)
        sim.append(0.5 * np.abs(c / n - pr).sum())
    sim = np.array(sim)
    outside = sum(v for t, v in cnt.items() if t not in p)
    chi = sum((cnt.get(t, 0) - n * p[t]) ** 2 / (n * p[t]) for t in ids)
    pval = float((sim >= tv).mean())
    print(f"  {name}: {n} samples, {len(cnt)} distinct, TV {tv:.4f} (noise: median {np.median(sim):.4f}, 95% "
          f"{np.quantile(sim, 0.95):.4f}; P(noise >= TV) {pval:.2f}), chi2 {chi:.1f} over {len(ids)} ids, "
          f"{outside} outside p's support")
    return pval >= 0.01 and outside == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--prompt", default="text_crisis")
    ap.add_argument("--dflash", default="", help="lab-dflash: DFlash2 checkpoint dir (its gated rounds draft)")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    ids = tok.encode(B.chat(a.pack, B.THINK[a.prompt]), add_special_tokens=False).ids
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    m = eng.fn_model(pack, 1024)
    mtp = B.Mtp(eng, m, max(4, a.depth), 4)
    if a.dflash:  # lab-dflash (docs/mtp.md 13)
        import ctypes as C
        en = eng.lib.ne_fn_dfl_enable
        en.restype, en.argtypes = C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int]
        if en(C.c_void_p(mtp.ptr), a.dflash.encode(), 8, max(4, a.depth), 8, 1):
            raise E.EngineError(eng.last_error())
    # find a spot with entropy: along the greedy continuation, position j whose token is certain under the filter
    # (so x_P = cont[j] whatever the seed) and whose next token has >= 5 ids and top <= 0.7
    m.set_sampler(0.0, 1.0, 1, 0)
    m.reset()
    cont, _, _ = m.generate(ids, 160, -1)
    m.reset()
    _, lg, _ = m.prefill(ids, logits=True)
    dists = [filt(lg)]
    for t in cont[:-1]:
        _, _, _, lg = m.step(t, True, logits=True)
        dists.append(filt(lg))
    j = next((j for j in range(1, len(cont) - 1) if len(dists[j]) == 1 and len(dists[j + 1]) >= 5
              and max(dists[j + 1].values()) <= 0.7), None)
    assert j is not None, "no position with a certain token followed by an uncertain one"
    ids = ids + cont[:j]
    print(f"prompt extended by {j} greedy tokens; next {tok.decode([cont[j]])!r} is certain, then "
          f"{len(dists[j + 1])} ids (top {max(dists[j + 1].values()):.2f})")
    m.set_sampler(TEMP, TOP_P, TOP_K, 1234)
    outs = []
    for s in range(a.runs):
        o, st = mtp.generate(ids, 3, a.depth, seed=1000 + s)
        outs.append(o)
    xp = {o[0] for o in outs}
    assert len(xp) == 1, f"prefill sample varies: {xp}"
    xp = xp.pop()
    # reference distributions from the 1-row decode
    m.reset()
    m.prefill(ids)
    _, _, _, lg1 = m.step(xp, True, logits=True)
    p1 = filt(lg1)
    rng = np.random.default_rng(0)
    print(f"== {a.prompt}: depth {a.depth}, x_P = {xp} {tok.decode([xp])!r}; p(. | x_P) has {len(p1)} ids, top "
          f"{max(p1.values()):.3f}")
    ok = compare("out[1] vs p(. | x_P)", [o[1] for o in outs], p1, rng)
    # out[2] given out[1]: pool the per-value comparisons through a mixture p2 = sum_v P(out1 = v) p(. | x_P, v)
    by = {}
    for o in outs:
        by.setdefault(o[1], []).append(o[2])
    for v, lst in sorted(by.items(), key=lambda kv: -len(kv[1]))[:2]:
        if len(lst) < 60:
            continue
        m.reset()
        m.prefill(ids)
        m.step(xp, True)
        _, _, _, lg2 = m.step(v, True, logits=True)
        ok &= compare(f"out[2] | out[1] = {tok.decode([v])!r}", lst, filt(lg2), rng)
    print("sampled-dist:", "OK" if ok else "CHECK")
    pack.close()


if __name__ == "__main__":
    main()
