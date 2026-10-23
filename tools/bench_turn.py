"""lab-ttft: time to first token of small reused agent turns at long context (docs/server.md 3e).

    NEED_GB=85 LOCK_OWNER=lab-ttft LOCK_WAIT=30 tools/with_lock.sh "bench_turn (~90 GB)" HH:MM -- $PY tools/bench_turn.py \
        [--levels 32768,100000,160000] [--sizes 50,150,500] [--reps 2] [--profile 1] [--ctx 262144]

A synthetic agent conversation through the server's Session + RealBackend in-process (no HTTP; the server's request
path from Session.generate on): a system prompt + a long user message of repository text up to each context level,
then turns that add s new tokens each (the previous reply's end, a user message of filler, the generation prompt
'<|im_start|>assistant\\n<think>\\n') on the live state, greedy, 8 new tokens per turn. Per turn: TTFT (generate()
call -> first streamed token), the server's prefill_ms, the engine calls it made (each ne_fn_mtp_feed piece / plain
prefill piece / step / checkpoint export / ne_fn_mtp_generate_from with its prefill part) and their wall ms.
--profile: at each level, single chunks of n rows (no head) through a profile-mode Prefill (GPU ms per stage) and a
plain one (wall), the 1-row and 4-row steps, from an MTP snapshot (restored after each).
Out: one line per turn + markdown tables at the end.
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
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import backend as B  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

STAGES = ["embed", "ple", "hc_glue", "trunk_gemm", "gdn", "attn", "moe_glue", "moe_gemm"]


class Filler:
    """real text token ids (this repo's docs and sources), consumed in order"""
    def __init__(self, tok):
        files = sorted(ROOT.glob("docs/*.md")) + sorted(ROOT.glob("models/flash_next/*.mm")) + \
            sorted(ROOT.glob("kernels/**/*.metal")) + sorted(ROOT.glob("server/*.py")) + sorted(ROOT.glob("tools/*.py"))
        self.ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files), add_special_tokens=False).ids
        self.i = 0

    def take(self, n):
        out = []
        while len(out) < n:
            k = min(n - len(out), len(self.ids) - self.i)
            out += self.ids[self.i:self.i + k]
            self.i = (self.i + k) % len(self.ids)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--levels", default="32768,100000,160000")
    ap.add_argument("--sizes", default="50,150,500")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--pf-sizes", default="1,4,16,32,64,128,256,512")
    ap.add_argument("--prof-env", default="", help="'|'-separated env settings (K=V;K=V) the profile runs under in turn, "
                    "each with new Prefills (engine knobs read at record time); '-' = as is")
    ap.add_argument("--real", type=int, default=1, help="profile with fresh real-text tokens (else a fixed synthetic run)")
    ap.add_argument("--lib", default=str(ROOT / "build/libengine.dylib"))
    ap.add_argument("--dflash", default="", help="DFlash2 drafter dir (the served -dflash id: block 8, kmax 5, 8-bit, gate 1)")
    ap.add_argument("--arms", default="", help="comma list of arms run in turn in this process (each from a reset):\n"
                    "binary = NE_SRV_PIECES=binary (the old piece split), single = the new; env:K=V[;K=V] sets "
                    "os.environ for that arm (engine knobs read per call)")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    IMS, IME = B.Session.IM_START, B.Session.IM_END
    NL = enc("\n")
    GEN = [IMS] + enc("assistant\n<think>\n")
    UHEAD = [IMS] + enc("user\n")
    fill = Filler(tok)
    t0 = time.time()
    drafter = dict(kind="dflash2", path=a.dflash, block=8, kmax=5, bits=8, gate=1, max_ctx=8192) if a.dflash else None
    be = B.RealBackend(ROOT, a.pack, a.ctx, 2, Path(a.lib), None, print, drafter=drafter)
    print(f"load {time.time() - t0:.1f} s, mtp_cont {be.mtp_cont}", flush=True)
    S = None

    # ---- instrumentation: every engine call the request path makes, in order, with wall ms
    ev = []
    f_feed = be._f["ne_fn_mtp_feed"] if be.mtp_cont else None

    def w_mfeed(p, d, n):
        t = time.perf_counter()
        rc = f_feed(p, d, n)
        ev.append(("mfeed", n, (time.perf_counter() - t) * 1e3))
        return rc
    if f_feed is not None:
        be._f["ne_fn_mtp_feed"] = w_mfeed
    o_prefill = be.m.prefill

    def w_prefill(toks):
        t = time.perf_counter()
        r = o_prefill(toks)
        ev.append(("prefill", len(toks), (time.perf_counter() - t) * 1e3))
        return r
    be.m.prefill = w_prefill
    for name in ("step", "state_save"):
        o = getattr(be, name)

        def w(*x, _o=o, _n=name):
            t = time.perf_counter()
            r = _o(*x)
            ev.append((_n, len(x[0]) if _n == "feed" else 1, (time.perf_counter() - t) * 1e3))
            return r
        setattr(be, name, w)
    o_from = be.mtp_from

    def w_from(toks, *x, **k):
        t = time.perf_counter()
        r = o_from(toks, *x, **k)
        ev.append(("mtp_from", len(toks), (time.perf_counter() - t) * 1e3, r[1].get("prefill_ms", 0.0)))
        return r
    be.mtp_from = w_from

    greedy = (0.0, 1.0, 1, 0)
    rows = []
    split = []

    def turn(S, P, label):
        ev.clear()
        first = []
        out = []
        tq = time.perf_counter()

        def on_tokens(t, *_):
            if not first:
                first.append(time.perf_counter())
            out.extend(t)
        use = S.choose_mtp(len(P), S.plan(P)[0], a.max_new, be.mtp_ptr is not None) if a.policy == "auto" else \
            a.policy == "always"
        st = S.generate(P, a.max_new, greedy, use, on_tokens, lambda: False, IME)
        ttft = (first[0] - tq) * 1e3
        new = len(P) - st["cached_tokens"]
        calls = " ".join(f"{e[0]}:{e[1]}={e[2]:.0f}" + (f"(pf {e[3]:.0f})" if len(e) > 3 else "") for e in ev)
        print(f"{label}: prompt {len(P)} cached {st['cached_tokens']} new {new} path {st['path']}: TTFT {ttft:.0f} ms, "
              f"prefill_ms {st['prefill_ms']:.0f}, snapshot {st.get('snapshot_ms', 0):.0f}, restore "
              f"{st.get('restore_ms', 0):.0f} | {calls} | out {out}", flush=True)
        rows.append((label, len(P), new, st["path"], ttft, st["prefill_ms"], calls))
        # (pfast) the turn's prefill split: chunked feeds (>= 16 tokens), step feeds (< 16: the generation-prompt
        # tail), checkpoint exports, the MTP start's own prefill part, the rest (Python, restore, bookkeeping)
        ch = sum(e[2] for e in ev if e[0] in ("mfeed", "prefill") and e[1] >= 16)
        tl = sum(e[2] for e in ev if e[0] in ("mfeed", "prefill") and e[1] < 16) + sum(e[2] for e in ev if e[0] == "step")
        ck = sum(e[2] for e in ev if e[0] == "state_save")
        ms0 = sum(e[3] for e in ev if e[0] == "mtp_from")
        split.append((label, len(P), new, ttft, st["prefill_ms"], ch, tl, ck, ms0, st["prefill_ms"] - ch - tl - ck - ms0))
        return out

    def extend(P, out, n_new):
        """next prompt: P + reply (+ <|im_end|>) + '\n' + user message + generation prompt, ~n_new new tokens"""
        tail = ([] if out and out[-1] == IME else [IME]) + NL
        body = n_new - 1 - len(tail) - len(UHEAD) - 2 - len(GEN)
        return P + out + tail + UHEAD + fill.take(max(1, body)) + [IME] + NL + GEN

    sizes = [int(x) for x in a.sizes.split(",") if x]
    levels = [int(x) for x in a.levels.split(",") if x]
    arms = [x for x in a.arms.split(",") if x] or ["default"]
    prof = []
    outs = {}
    for ai, arm in enumerate(arms):
        if arm == "binary":
            B.BINARY_PIECES = True
        elif arm == "single":
            B.BINARY_PIECES = False
        elif arm.startswith("env:"):
            for kv in arm[4:].split(";"):
                k, v = kv.split("=", 1)
                os.environ[k] = v
        print(f"=== arm {arm} (binary pieces {B.BINARY_PIECES})", flush=True)
        be.reset()
        S = B.Session(be, a.ctx, 4, a.policy, log=print)
        fill.i = 0
        outs[arm] = []
        # warm-up at a short context: recordings / pipelines of every piece length the turns use
        sysm = [IMS] + enc("system\nYou are a coding agent. Reference material follows.\n") + fill.take(2000) + [IME] + NL
        P = sysm + UHEAD + fill.take(1500) + [IME] + NL + GEN
        out = turn(S, P, f"{arm} warm cold")
        for s in sizes:
            P2 = extend(P, out, s)
            out = turn(S, P2, f"{arm} warm s{s}")
            P = P2
        for L in levels:
            P2 = extend(P, out, L - len(P) - len(out))
            out = turn(S, P2, f"{arm} grow {L // 1000}K")
            outs[arm].append(out)
            P = P2
            for r in range(a.reps):
                for s in sizes:
                    P2 = extend(P, out, s)
                    out = turn(S, P2, f"{arm} T{L // 1000}K s{s} r{r}")
                    outs[arm].append(out)
                    P = P2
            if a.profile and ai == len(arms) - 1:
                for pe in (a.prof_env.split("|") if a.prof_env else ["-"]):
                    saved = dict(os.environ)
                    if pe != "-":
                        for kv in pe.split(";"):
                            k, v = kv.split("=", 1)
                            os.environ[k] = v
                    print(f"--- profile env {pe}", flush=True)
                    prof += [(pe,) + r for r in profile(be, L, [int(x) for x in a.pf_sizes.split(",") if x],
                                                         fill if a.real else None)]
                    os.environ.clear()
                    os.environ.update(saved)
    if len(arms) > 1:
        a0 = arms[0]
        for arm in arms[1:]:
            same = [x == y for x, y in zip(outs[a0], outs[arm])]
            print(f"greedy outputs {arm} vs {a0}: {sum(same)}/{len(same)} turns identical; first difference at turn "
                  f"{same.index(False) if False in same else '-'}")

    print("\n| turn | prompt | new | path | TTFT ms | prefill_ms | engine calls (wall ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.0f} | {r[5]:.0f} | {r[6]} |")
    print("\n| turn | prompt | new | TTFT | prefill_ms | chunk feeds | tail steps | checkpoint | MTP start | rest |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in split:
        print(f"| {r[0]} | {r[1]} | {r[2]} | " + " | ".join(f"{x:.0f}" for x in r[3:]) + " |")
    if prof:
        print("\n| env | T | n | path | wall ms | GPU ms | " + " | ".join(STAGES) + " |")
        print("|---|---|---|---|---|---|" + "---|" * len(STAGES))
        for p in prof:
            print(f"| {p[0]} | {p[1]} | {p[2]} | {p[3]} | {p[4]:.1f} | {p[5]:.1f} | " + " | ".join(f"{x:.1f}" for x in p[6]) + " |")


def profile(be, L, ns, fill=None):
    """single chunks of n rows at the current T (from an MTP snapshot, restored after each)"""
    lib = be.eng.lib
    vp = C.c_void_p
    lib.ne_fn_mtp_snapshot.restype = vp
    lib.ne_fn_mtp_snapshot.argtypes = [vp]
    lib.ne_fn_mtp_restore.argtypes = [vp, vp]
    lib.ne_fn_mtp_snap_free.argtypes = [vp]
    lib.ne_fn_prefill_create.restype = vp
    lib.ne_fn_prefill_create.argtypes = [vp, C.c_int, C.c_int]
    lib.ne_fn_prefill_free.argtypes = [vp]
    lib.ne_fn_prefill_tokens.argtypes = [vp, vp, C.c_int, vp]
    lib.ne_fn_model_step.argtypes = [vp, C.c_int, C.c_int, vp, vp, vp, vp, vp]
    snap = lib.ne_fn_mtp_snapshot(vp(be.mtp_ptr))
    assert snap, be.eng.last_error()
    T = be.T
    toks = np.arange(1000, 1000 + max(ns), dtype=np.int32)
    res = []
    hp = lib.ne_fn_prefill_create(vp(be.m.ptr), 4096, 1)
    hn = lib.ne_fn_prefill_create(vp(be.m.ptr), 4096, 0)
    assert hp and hn, be.eng.last_error()
    stt = np.zeros(12, np.float64)
    ms = C.c_double(0)
    for n in ns:
        if fill is not None:  # fresh real text each size (n-gram rows not in the page cache yet, as in a turn)
            toks = np.array(fill.take(n), np.int32)
        if n < 16:  # the step path the server takes below 16 tokens: 4-row steps + 1-row steps (head off)
            for rep in range(2):
                lib.ne_fn_mtp_restore(vp(be.mtp_ptr), vp(snap))
                t = time.perf_counter()
                for i in range(n):
                    lib.ne_fn_model_step(vp(be.m.ptr), int(toks[i]), 0, None, None, None, None, C.byref(ms))
                w = (time.perf_counter() - t) * 1e3
            res.append((T, n, "1-row steps", w, ms.value * n, [0.0] * 8))
            continue
        for h, name in ((hn, "chunk"), (hp, "chunk profile")):
            for rep in range(2):  # second = warm (recording made)
                lib.ne_fn_mtp_restore(vp(be.mtp_ptr), vp(snap))
                t = time.perf_counter()
                assert lib.ne_fn_prefill_tokens(vp(h), toks.ctypes.data, n, stt.ctypes.data) == 0, be.eng.last_error()
                w = (time.perf_counter() - t) * 1e3
                if rep == 0:
                    first = w
            res.append((T, n, f"{name} (first {first:.0f})", w, stt[1], stt[4:12].tolist()))
            print(f"profile T {T} n {n} {name}: wall {w:.1f} (first {first:.1f}) GPU {stt[1]:.1f} stages "
                  + " ".join(f"{k} {v:.1f}" for k, v in zip(STAGES, stt[4:12])), flush=True)
    lib.ne_fn_mtp_restore(vp(be.mtp_ptr), vp(snap))
    lib.ne_fn_prefill_free(vp(hp))
    lib.ne_fn_prefill_free(vp(hn))
    lib.ne_fn_mtp_snap_free(vp(snap))
    return res


if __name__ == "__main__":
    main()
