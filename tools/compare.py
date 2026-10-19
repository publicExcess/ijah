"""Compare engine (or op-reference) outputs with goldens: rel L2, max|d| / max|ref|, cosine, pass/fail per tensor kind.

Library:
    import compare
    r = compare.check("gdn.qkv", got, ref)        # -> dict(name, kind, rel, maxrel, cos, tol, ok)
    compare.report([r, ...])                      # table; returns True if all pass
CLI (engine dumps as .npy files named like the golden tensors, e.g. out/gdn.qkv.npy):
    $PY tools/compare.py CASE OUT_DIR [--only gdn.] [--tol gdn.qkv=1e-2] [--json res.json]
    $PY tools/compare.py --self-test              # golden vs itself / vs bf16-perturbed copy
Exit status 0 = all compared tensors pass, 1 = a failure, 2 = nothing compared.
"""
import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden  # noqa: E402
import numpy as np  # noqa: E402

# (kind, patterns, rel-L2 tolerance, min cosine); first match wins. "exact" = equality (ids: as sets per row).
KINDS = [
    ("exact", ["*_ids", "*selected_blocks", "*token_history*"], 0.0, 1.0),
    ("state", ["gdn.ssm_state*"], 1e-4, 0.999999),
    ("expert", ["moe.routed_out"], 5e-3, 0.99998),
    ("pointwise", ["*.normed", "*.mix", "*.inject_w", "gdn.g", "gdn.q", "gdn.k", "gdn.v", "gdn.beta", "gdn.conv_out",
                   "ple.gate", "*_rope", "attn.q_normed", "attn.k_normed", "attn.idx_q", "*conv_state*",
                   "moe.router_probs", "moe.topk_weights", "moe.shared_act", "moe.shared_scale", "ple.keys",
                   "ple.queries", "ple.embeddings", "embed", "attn.index_block_keys*", "attn.idx_scores",
                   "attn.gated", "ple.gated_values", "*_cache_*", "attn.index_keys*", "attn.idx_raw_keys"],
     5e-3, 0.99999),
    ("projection", ["*.down", "*.up", "*_proj", "gdn.qkv", "gdn.z", "gdn.a", "gdn.b", "moe.router_logits",
                    "*.inject_logits", "attn.idx_qk", "attn.gate", "ple.values"], 5e-3, 0.99999),
    ("block", ["*"], 1e-2, 0.9999),
]


def kind_of(name):
    for k, pats, tol, cos in KINDS:
        if any(fnmatch.fnmatch(name, p) for p in pats):
            return k, tol, cos
    raise AssertionError(name)


def metrics(got, ref):
    g = np.asarray(got, np.float64).reshape(-1)
    r = np.asarray(ref, np.float64).reshape(-1)
    fin = np.isfinite(r)
    g, r = g[fin], r[fin]
    d = g - r
    nr = np.linalg.norm(r)
    rel = float(np.linalg.norm(d) / nr) if nr > 0 else float(np.linalg.norm(d))
    mr = float(np.abs(r).max()) if r.size else 0.0
    maxrel = float(np.abs(d).max() / mr) if mr > 0 else float(np.abs(d).max() if d.size else 0.0)
    ng = np.linalg.norm(g)
    cos = float(g @ r / (ng * nr)) if ng > 0 and nr > 0 else (1.0 if ng == nr else 0.0)
    return rel, maxrel, cos


def check(name, got, ref, tol=None, cos_scale=1.0):
    """cos_scale multiplies the cosine slack (1 - min cosine): a chain test that scales the rel-L2 tolerance by s
    should pass cos_scale = s**2 (1 - cos ~ rel**2 / 2)."""
    k, t, c = kind_of(name)
    c = 1.0 - (1.0 - c) * cos_scale
    if tol is not None:
        t = tol
    got = np.asarray(got)
    ref = np.asarray(ref)
    res = {"name": name, "kind": k, "tol": t, "shape": list(ref.shape)}
    if got.shape != ref.shape:
        if got.size == ref.size:
            got = got.reshape(ref.shape)
        else:
            res.update(ok=False, err=f"shape {list(got.shape)} != {list(ref.shape)}", rel=np.inf, maxrel=np.inf, cos=0.0)
            return res
    if k == "exact":
        if name.endswith("_ids") or name.endswith("selected_blocks"):
            gs = [set(map(int, x)) - {-1} for x in got.reshape(-1, got.shape[-1])]
            rs = [set(map(int, x)) - {-1} for x in ref.reshape(-1, ref.shape[-1])]
            bad = sum(a != b for a, b in zip(gs, rs))
            diff = sum(len(a ^ b) for a, b in zip(gs, rs))
            res.update(ok=bad == 0, rel=float(diff), maxrel=float(bad), cos=1.0 if bad == 0 else 0.0,
                       note=f"{bad} rows differ ({diff} ids)" if bad else "sets equal")
        else:
            eq = np.array_equal(got, ref)
            res.update(ok=bool(eq), rel=0.0 if eq else 1.0, maxrel=0.0 if eq else 1.0, cos=1.0 if eq else 0.0)
        return res
    rel, maxrel, cos = metrics(got, ref)
    res.update(rel=rel, maxrel=maxrel, cos=cos, ok=bool(rel <= t and cos >= c))
    return res


def check_topk(name, got_ids, ref_ids, ref_probs):
    """Top-k ids as sets per row, tie-aware: a row also passes if its set differs only among ids whose reference
    probability equals the k-th largest (bf16 probabilities tie often; argpartition picks either)."""
    g = np.asarray(got_ids).reshape(-1, np.shape(ref_ids)[-1])
    r = np.asarray(ref_ids).reshape(g.shape)
    p = np.asarray(ref_probs).reshape(g.shape[0], -1)
    bad = ties = 0
    for i in range(g.shape[0]):
        a, b = set(map(int, g[i])), set(map(int, r[i]))
        if a == b:
            continue
        kth = np.sort(p[i])[-g.shape[1]]
        if all(p[i][x] == kth for x in a ^ b):
            ties += 1
        else:
            bad += 1
    return {"name": name, "kind": "exact", "tol": 0.0, "shape": list(np.shape(ref_ids)), "ok": bad == 0,
            "rel": float(bad), "maxrel": float(ties), "cos": 1.0 if bad == 0 else 0.0,
            "note": f"{bad} rows differ, {ties} rows differ only in exact ties"}


def report(results, title=None, file=sys.stdout):
    if title:
        print(title, file=file)
    print(f"  {'tensor':30s} {'kind':10s} {'rel L2':>9s} {'max/max':>9s} {'cosine':>10s} {'tol':>7s}  ok", file=file)
    allok = True
    for r in results:
        allok &= r["ok"]
        extra = r.get("err") or r.get("note") or ""
        print(f"  {r['name']:30s} {r['kind']:10s} {r['rel']:9.2e} {r['maxrel']:9.2e} {r['cos']:10.7f} {r['tol']:7.0e}  "
              f"{'PASS' if r['ok'] else 'FAIL'} {extra}", file=file)
    return allok


def compare_dir(case, out_dir, only=None, tols=None):
    c = golden.load(case)
    out_dir = Path(out_dir)
    res = []
    for n in c.list():
        if only and not any(n.startswith(o) for o in only):
            continue
        f = out_dir / f"{n}.npy"
        if f.exists():
            res.append(check(n, np.load(f), c[n], (tols or {}).get(n)))
    return res


def self_test():
    """Golden vs itself (all pass, rel 0) and vs a copy with 1-ulp-of-bf16 relative noise (must pass, rel ~1e-3)
    and with 3% noise (must fail)."""
    ok = True
    rng = np.random.default_rng(0)
    for case in golden.cases()[:3]:
        c = golden.load(case)
        same = [check(n, c[n], c[n]) for n in c.list()]
        ok &= all(r["ok"] and (r["rel"] == 0) for r in same)
        for n in ("hc_out", "moe.routed_out"):
            r = c[n]
            noisy = r * (1 + 2.0 ** -9 * rng.standard_normal(r.shape))
            ok &= check(n, noisy, r)["ok"]
            ok &= not check(n, r * (1 + 0.03 * rng.standard_normal(r.shape)), r)["ok"]
        ids = c["moe.topk_ids"]
        ok &= check("moe.topk_ids", ids[..., ::-1], ids)["ok"]          # order does not matter
        ok &= not check("moe.topk_ids", np.roll(ids, 1, axis=1) if ids.shape[1] > 1 else ids + 1, ids)["ok"]
    print("compare self-test:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--only", action="append")
    ap.add_argument("--tol", action="append", default=[], help="name=rel")
    ap.add_argument("--json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    tols = {k: float(v) for k, v in (t.split("=") for t in a.tol)}
    res = compare_dir(a.case, a.out_dir, a.only, tols)
    if not res:
        print("nothing compared (no matching .npy in", a.out_dir, ")")
        sys.exit(2)
    ok = report(res, f"{a.case} vs {a.out_dir}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
