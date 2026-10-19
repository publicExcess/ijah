"""lab-snap: the server's own request path (Session + RealBackend, in-process): restored shared-prefix snapshot vs cold.

    NEED_GB=85 LOCK_OWNER=snap LOCK_WAIT=30 tools/with_lock.sh "check_shared_sess (~81 GB)" HH:MM -- \\
        $PY tools/check_shared_sess.py [--arms base,B:AHEAD=False,B:REST_FIRST=False]

Per arm (module switches of server/backend.py, as tools/bench_cold.py): sessions A (cold: the tools block snapshot) and
B (restores it, captures the learned one) on the MTP path, then session E on the step path with logprobs: restored
(the learned snapshot) vs cold (reuse off) vs cold again. Prints the first difference and the largest |d logprob|.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/server"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import backend as B  # noqa: E402
import chat  # noqa: E402
import shared_http as SH  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

EX = {"bias": None, "repetition": 1.0, "min_p": 0.0, "presence": 0.0, "frequency": 0.0, "logprob_k": 8, "rng0": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--arms", default="base")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok, "xhigh")
    b = B.RealBackend(ROOT, a.pack, a.ctx, 2, ROOT / "build/libengine.dylib", None, print)
    mk = [tok.encode("</IMPORTANT>", add_special_tokens=False).ids]

    def gen(S, p, reuse=True, mtp=False, capture=True, lp=True):
        out, tops = [], []

        def on(t, top=None):
            out.extend(t)
            if top is not None:
                tops.append(top)
        sl, cand = S.pick(p, reuse)
        g = S.run(sl, cand, p, a.steps, (0.0, 1.0, 20, 0), mtp, on, lambda: False, tpl.im_end,
                  extra=EX if lp and not mtp else None, capture=capture)
        while True:
            try:
                next(g)
            except StopIteration as e:
                return out, tops, e.value

    def cmp(nm, r, s):
        (ta, la, sa), (tb_, lb, sb) = r, s
        d = 0.0
        for i in range(min(len(ta), len(tb_))):
            if ta[i] != tb_[i]:
                top = sorted(lb[i][1])[-2:]
                print(f"  {nm}: first {i} tokens equal, then {ta[i]} vs {tb_[i]} (top-2 gap {top[1] - top[0]:.3f}); "
                      f"max |dlogprob| before {d:.2e} (cached {sa['cached_tokens']} / {sb['cached_tokens']})", flush=True)
                return
            ia, pa = la[i]
            ib, pb = lb[i]
            if ta[i] in list(ia) and tb_[i] in list(ib):
                d = max(d, abs(float(pa[list(ia).index(ta[i])]) - float(pb[list(ib).index(tb_[i])])))
        print(f"  {nm}: all {min(len(ta), len(tb_))} tokens equal, max |dlogprob| {d:.2e}"
              f"{' (bit-identical)' if d == 0 else ''} (cached {sa['cached_tokens']} / {sb['cached_tokens']})", flush=True)

    for ai, arm in enumerate(a.arms.split(",")):
        old = {}
        if arm.startswith("B:"):  # (arm "step...": sessions A and B on the step path)
            k, v = arm[2:].split("=")
            old[k] = getattr(B, k)
            setattr(B, k, eval(v))
        S = B.Session(b, a.ctx, 4, "auto", log=lambda *x: None, markers=mk)
        nonce = f"sess{ai}"
        P = {s: tpl.render_ids(SH.first_turn(nonce, s))[0] for s in "ABE"}
        P["F"] = tpl.render_ids(SH.first_turn(nonce, "F", SH.SYSTEM2))[0]
        mtp_ab = not arm.startswith("step")
        ra = gen(S, P["A"], mtp=mtp_ab, lp=False)
        rb = gen(S, P["B"], mtp=mtp_ab, lp=False)
        if arm == "trace":  # the piece schedules of F restored / cold (NE_SRV_TRACE lines on stderr)
            B.TRACE = True
            print("--- F restored", file=sys.stderr, flush=True)
            f_rest = gen(S, P["F"], capture=False)
            print("--- F cold", file=sys.stderr, flush=True)
            f_cold = gen(S, P["F"], reuse=False, capture=False)
            B.TRACE = False
            cmp("F restored vs cold", f_rest, f_cold)
            continue
        if arm == "diag":  # the tools block snapshot A made vs a plain export at the same position
            import numpy as np
            S1 = min(S.cache.shared(), key=lambda c: len(c.toks))
            x = len(S1.toks)
            sn = S1.snaps[x]
            seg = S1.segs[0]
            lh = b.snap_head(sn.meta) or 0
            b.reset()
            b.feed(P["A"][:x], sample=False)
            buf2, meta2 = b.state_save()
            kv2, cap2, hcap2 = b.kv_save(0, x, 0, lh)
            nm = int(b.eng.lib.ne_fn_kv_bytes(B.C.c_void_p(b.m.ptr), cap2))
            print(f"diag x {x}: meta {list(sn.meta)} vs {list(meta2)}; seg {seg[:2]} cap {seg[3]} head {seg[4:7]} vs cap "
                  f"{cap2} hcap {hcap2}; state bytes differing {int(np.count_nonzero(sn.buf != buf2))} of {buf2.size}; "
                  f"KV main differing {int(np.count_nonzero(seg[2][:nm] != kv2[:nm]))} of {nm}; head KV differing "
                  f"{int(np.count_nonzero(seg[2][nm:] != kv2[nm:]))} of {kv2.size - nm}", flush=True)
            continue
        print(f"arm {arm}: A cached {ra[2]['cached_tokens']} made {ra[2].get('shared_made')}, B cached "
              f"{rb[2]['cached_tokens']} made {rb[2].get('shared_made')}", flush=True)
        f_rest = gen(S, P["F"], capture=False)
        f_cold = gen(S, P["F"], reuse=False, capture=False)
        print(f"  F restored: cached {f_rest[2]['cached_tokens']} path {f_rest[2]['path']} shared {f_rest[2]['shared_src']}")
        cmp("F (tools block snapshot) restored vs cold", f_rest, f_cold)
        e_rest = gen(S, P["E"], capture=False)
        print(f"  E restored: cached {e_rest[2]['cached_tokens']} path {e_rest[2]['path']} shared {e_rest[2]['shared_src']}, "
              f"head_ok {b.head_ok()}")
        e_cold = gen(S, P["E"], reuse=False, capture=False)
        e_cold2 = gen(S, P["E"], reuse=False, capture=False)
        e_rest2 = gen(S, P["E"], capture=False)
        cmp("cold vs cold again", e_cold2, e_cold)
        cmp("restored vs cold", e_rest, e_cold)
        cmp("restored again vs restored", e_rest2, e_rest)
        S.cache.close()
        for k, v in old.items():
            setattr(B, k, v)


if __name__ == "__main__":
    main()
