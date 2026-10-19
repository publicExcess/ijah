"""lab-snap: is a restored shared-prefix snapshot exact? (docs/server.md 3f). In-process, one RealBackend (whole model).

    NEED_GB=85 LOCK_OWNER=snap LOCK_WAIT=30 tools/with_lock.sh "check_shared (~81 GB)" HH:MM -- $PY tools/check_shared.py

The synthetic agent-like first turn of tests/server/shared_http.py (session E). Same feed schedule in every arm:
[0, x) | [x, last) | [last, n) with the sample, then greedy steps (the MTP head kept in step, as the server's step path):
  split     reset, the schedule
  restore   reset, feed [0, x), export the state + KV rows [0, x) + the head's rows (what a shared snapshot holds),
            overwrite the cache with other text, reset, load them back, then [x, last) | [last, n): must equal split
            bit for bit (the park / restore exactness)
  cold      reset, [0, last) in one feed, [last, n): the piece-boundary effect alone (no restore) vs split
Per arm: the first STEPS greedy tokens and their logprobs (logprob_k 8). Prints the first difference and the largest
|d logprob| of the shared tokens.
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
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--x", default="tools,learned", help="snapshot positions: tools (the tools block end), learned "
                    "(the system end - 46, rounded to 64), or numbers")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok, "xhigh")
    b = B.RealBackend(ROOT, a.pack, a.ctx, 2, ROOT / "build/libengine.dylib", None, print)
    b.set_sampler(0.0, 1.0, 20, 0)
    P, _ = tpl.render_ids(SH.first_turn("check", "E"))
    S = B.Session(b, a.ctx, log=lambda *x: None, markers=[tok.encode("</IMPORTANT>", add_special_tokens=False).ids])
    n, last, sysb = len(P), S.last_start(P), S.system_boundary(P, tpl.im_end)
    tb = S.struct_boundary(P, tpl.im_end)
    xs = []
    for v in a.x.split(","):
        xs.append(tb if v == "tools" else (sysb - 46) // 64 * 64 if v == "learned" else int(v))
    print(f"prompt {n}, tools end {tb}, system end {sysb}, generation prompt {last}; x = {xs}", flush=True)
    junk = tok.encode(SH.SYSTEM2 * 9, add_special_tokens=False).ids[:9000]  # past every x: other rows everywhere

    def decode(first):
        toks, lps = [first], [b.last_top()]
        for _ in range(a.steps - 1):
            toks.append(b.step(toks[-1]))
            lps.append(b.last_top())
        return toks, lps

    def tail(x):
        b.feed(P[x:last], sample=False)
        return decode(b.feed(P[last:], sample=True))

    def arm_split(x):
        b.reset()
        b.set_extra(EX, P)
        b.feed(P[:x], sample=False)
        return tail(x)

    def arm_restore(x):
        b.reset()
        b.set_extra(EX, P)
        b.feed(P[:x], sample=False)
        buf, meta = b.state_save()
        buf, meta = buf.copy(), meta.copy()
        lh = b.snap_head(meta) or 0
        kv, cap, hcap = b.kv_save(0, x, 0, lh)
        b.reset()
        b.feed(junk, sample=False)  # other rows / state in the buffers
        b.reset()
        b.kv_load(0, x, kv, cap, 0, hcap, hcap)
        b.state_load(buf, meta)
        b.arm()
        print(f"  x {x}: head in step {bool(meta[3])}, head rows {lh}, head_ok after load {b.head_ok()}", flush=True)
        return tail(x)

    def arm_noreset(x, junk_steps):
        """as Session._prepare restores (no reset between the slot's previous use and the load)"""
        b.reset()
        b.set_extra(EX, P)
        b.feed(P[:x], sample=False)
        buf, meta = b.state_save()
        buf, meta = buf.copy(), meta.copy()
        lh = b.snap_head(meta) or 0
        kv, cap, hcap = b.kv_save(0, x, 0, lh)
        b.reset()
        nx = b.feed(junk, sample=True)  # the slot's previous conversation, ended by a reply
        for _ in range(junk_steps):
            nx = b.step(nx)
        b.kv_load(0, x, kv, cap, 0, hcap, hcap)
        b.state_load(buf, meta)
        b.arm()
        return tail(x)

    def arm_deferred(x, mtp):
        """as the server does it: the state at x exported during the prefill, the KV rows only after the reply"""
        b.reset()
        b.set_extra(EX, P)
        b.feed(P[:x], sample=False)
        buf, meta = b.state_save()
        buf, meta = buf.copy(), meta.copy()
        lh = b.snap_head(meta) or 0
        b.feed(P[x:last], sample=False)
        if mtp:
            b.clear_extra()
            b.feed(P[last:-1], sample=False)
            b.mtp_from(P[-1:], 64, 2, 0, lambda *z: None, lambda: False)
        else:
            decode(b.feed(P[last:], sample=True))
        kv, cap, hcap = b.kv_save(0, x, 0, lh)
        b.reset()
        b.feed(junk, sample=False)
        b.reset()
        b.set_extra(EX, P)
        b.kv_load(0, x, kv, cap, 0, hcap, hcap)
        b.state_load(buf, meta)
        b.arm()
        return tail(x)

    def arm_cold(_):
        b.reset()
        b.set_extra(EX, P)
        b.feed(P[:last], sample=False)
        return decode(b.feed(P[last:], sample=True))

    def cmp(nm, r, s):
        (ta, la), (tb_, lb) = r, s
        d, i = 0.0, 0
        for i in range(min(len(ta), len(tb_))):
            if ta[i] != tb_[i]:
                ids, lp = lb[i]
                top = sorted(lp)[-2:]
                print(f"  {nm}: first {i} tokens equal, then {ta[i]} vs {tb_[i]} (top-2 gap {top[1] - top[0]:.3f}); "
                      f"max |dlogprob| before {d:.2e}", flush=True)
                return
            ia, pa = la[i]
            ib, pb = lb[i]
            ja = list(ia).index(ta[i]) if ta[i] in list(ia) else None
            jb = list(ib).index(tb_[i]) if tb_[i] in list(ib) else None
            if ja is not None and jb is not None:
                d = max(d, abs(float(pa[ja]) - float(pb[jb])))
        print(f"  {nm}: all {i + 1} tokens equal, max |dlogprob| {d:.2e}"
              + (" (bit-identical)" if d == 0.0 else ""), flush=True)

    cold = arm_cold(0)
    cold2 = arm_cold(0)
    cmp("cold vs cold again (run-to-run)", cold2, cold)
    for x in xs:
        sp = arm_split(x)
        rs = arm_restore(x)
        cmp(f"x {x}: restore vs split (same pieces)", rs, sp)
        cmp(f"x {x}: split vs cold (piece boundary at x)", sp, cold)
        cmp(f"x {x}: restore vs cold", rs, cold)
        cmp(f"x {x}: restore without a reset (after a prefill) vs split", arm_noreset(x, 0), sp)
        cmp(f"x {x}: restore without a reset (after a reply) vs split", arm_noreset(x, 20), sp)
    print("text (cold):", repr(tok.decode(cold[0])[:160]))


if __name__ == "__main__":
    main()
