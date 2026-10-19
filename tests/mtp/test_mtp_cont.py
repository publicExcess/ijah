"""lab-mtpcont: MTP from the model's current state (docs/mtp.md 10.9) vs MTP from a fresh prefill.

    # partial model (layers 0-3 + embedding / head / MTP head, ~10 GB; meaningless text, the real state machinery;
    # 4 since the lab-vision merge: the model load needs an attention layer, layer 3):
    NE_FN_LAYERS=4 $PY tests/mtp/test_mtp_cont.py [--prefill steps|chunked|both]
    # whole model (~83 GB, under the lock):
    NEED_GB=85 LOCK_OWNER=... tools/with_lock.sh "mtp cont (~83 GB)" HH:MM -- $PY tests/mtp/test_mtp_cont.py --whole

Greedy, depth 2 (and adaptive). Checks:
  exact   fresh mtp_generate(P) == restore(snapshot after mtp_feed(P[:k])) + mtp_generate_from(P[k:]), with the KV rows
          past k overwritten by another run in between. NE_FN_PREFILL=steps and k % 4 == 0 make both target prefills the
          same arithmetic, so tokens AND the acceptance histogram must be identical (head state bit-identical). With the
          chunked prefill (other chunk boundaries) tokens may fork at an exact tie only: reported.
  live    after an MTP run the state = P + out[:-1] (T, head Lh + pending == T, snapshot-able); continuing it with new
          tokens vs a fresh run over the whole sequence (same up to ties: the verify's multi-row arithmetic built it).
  cut     runs cut by max_new inside a round commit only up to the last emitted token (T == n + k - 1), and continue.
  step    mtp_step keeps the head in step; a plain ne_fn_model_step makes it stale (generate_from refuses).
  lazy    (lab-headlazy) N mtp_steps with the lazy head (rows only queued) then MTP == the eager head (pending cap 0)
          == the full-layer head recordings (NE_MTP_HEAD_KV=full): tokens AND histogram (head KV bit-identical);
          also through a snapshot (which catches the head up first) and an explicit ne_fn_mtp_catch_up.
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

KEYS = ("prefill_ms", "head_prefill_ms", "decode_ms", "tokens", "rounds", "draft_gpu_ms", "verify_gpu_ms",
        "gather_us", "host_us", "max_depth")


class Mtp:
    def __init__(self, eng, model, maxd=4, bits=4):
        L = eng.lib
        vp, ip = C.c_void_p, C.c_int
        sig = {
            "ne_fn_mtp_create": (vp, [vp, ip, ip]),
            "ne_fn_mtp_generate": (ip, [vp, vp, ip, ip, ip, ip, C.c_uint64, vp, vp, vp]),
            "ne_fn_mtp_generate_from": (ip, [vp, vp, ip, ip, ip, ip, C.c_uint64, vp, vp, vp, vp, vp]),
            "ne_fn_mtp_feed": (ip, [vp, vp, ip]),
            "ne_fn_mtp_step": (ip, [vp, ip, C.POINTER(ip), vp]),
            "ne_fn_mtp_head_ok": (ip, [vp]),
            "ne_fn_mtp_head_state": (ip, [vp, C.POINTER(ip), C.POINTER(ip)]),
            "ne_fn_mtp_snapshot": (vp, [vp]),
            "ne_fn_mtp_restore": (ip, [vp, vp]),
            "ne_fn_mtp_snap_T": (ip, [vp]),
            "ne_fn_mtp_snap_free": (None, [vp]),
            "ne_fn_model_snapshot": (vp, [vp]),
            "ne_fn_mtp_catch_up": (ip, [vp, C.POINTER(C.c_double)]),
            "ne_fn_mtp_set_pend_cap": (ip, [vp, ip]),
            "ne_fn_kv_bytes": (C.c_int64, [vp, ip]),
            "ne_fn_kv_io": (ip, [vp, ip, ip, ip, vp, ip]),
            "ne_fn_mtp_kv_bytes": (C.c_int64, [vp, ip]),
            "ne_fn_mtp_kv_io": (ip, [vp, ip, ip, ip, vp, ip]),
            "ne_fn_mtp_state_bytes": (C.c_int64, [vp]),
            "ne_fn_mtp_state_io": (ip, [vp, ip, vp, C.c_int64, vp]),
            "ne_fn_snap_free": (None, [vp]),
        }
        for k, (r, a) in sig.items():
            f = getattr(L, k)
            f.restype, f.argtypes = r, a
        self.eng, self.L, self.m, self.maxd = eng, L, model, maxd
        self.p = L.ne_fn_mtp_create(model.ptr, maxd, bits)
        if not self.p:
            raise E.EngineError(eng.last_error())

    def _ok(self, rc, what):
        if rc is None or rc < 0:
            raise E.EngineError(f"{what}: {self.eng.last_error()}")
        return rc

    def _gen(self, fn, ids, n, depth, stop):
        p = np.ascontiguousarray(ids, np.int32)
        out = np.empty(n, np.int32)
        st = np.zeros(10, np.float64)
        M1 = self.maxd + 1
        hist = np.zeros(M1 * M1, np.int32)
        args = [self.p, p.ctypes.data, len(p), n, stop, depth, 0, out.ctypes.data, st.ctypes.data, hist.ctypes.data]
        if fn == "ne_fn_mtp_generate_from":
            args += [None, None]
        k = self._ok(getattr(self.L, fn)(*args), fn)
        s = dict(zip(KEYS, st.tolist()))
        s["hist"] = hist.tolist()
        return out[:k].tolist(), s

    def generate(self, ids, n, depth=2, stop=-1):
        return self._gen("ne_fn_mtp_generate", ids, n, depth, stop)

    def generate_from(self, ids, n, depth=2, stop=-1):
        return self._gen("ne_fn_mtp_generate_from", ids, n, depth, stop)

    def feed(self, ids):
        p = np.ascontiguousarray(ids, np.int32)
        self._ok(self.L.ne_fn_mtp_feed(self.p, p.ctypes.data, len(p)), "feed")

    def step(self, tok):
        nx = C.c_int(-1)
        self._ok(self.L.ne_fn_mtp_step(self.p, int(tok), C.byref(nx), None), "mtp_step")
        return nx.value

    def head_ok(self):
        return self._ok(self.L.ne_fn_mtp_head_ok(self.p), "head_ok") == 1

    def head_state(self):
        a, b = C.c_int(0), C.c_int(0)
        self._ok(self.L.ne_fn_mtp_head_state(self.p, C.byref(a), C.byref(b)), "head_state")
        return a.value, b.value

    def set_cap(self, cap):
        self._ok(self.L.ne_fn_mtp_set_pend_cap(self.p, cap), "set_pend_cap")

    def catch_up(self):
        ms = C.c_double(0)
        self._ok(self.L.ne_fn_mtp_catch_up(self.p, C.byref(ms)), "catch_up")
        return ms.value

    def snapshot(self):
        h = self.L.ne_fn_mtp_snapshot(self.p)
        if not h:
            raise E.EngineError(f"snapshot: {self.eng.last_error()}")
        return h

    def restore(self, h):
        self._ok(self.L.ne_fn_mtp_restore(self.p, h), "restore")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--whole", action="store_true", help="whole model (else NE_FN_LAYERS must be set)")
    ap.add_argument("--prefill", default="both", choices=["steps", "chunked", "both"])
    ap.add_argument("--n", type=int, default=48, help="new tokens per run")
    ap.add_argument("--depths", default="2,0")
    ap.add_argument("--dflash", default="", help="lab-dflash: DFlash2 checkpoint dir: every check with its rounds on")
    a = ap.parse_args()
    if not a.whole and not os.environ.get("NE_FN_LAYERS"):
        sys.exit("set NE_FN_LAYERS=4 (partial model) or pass --whole")
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    text = (refenv.ROOT / "docs" / "mtp.md").read_text()
    P = tok.encode(text[:3000], add_special_tokens=False).ids[:404]  # 404: P[:k] with k % 4 == 0 for k = 200, 256
    Q = tok.encode(text[5000:6500], add_special_tokens=False).ids[:150]
    X = tok.encode(text[8000:8600], add_special_tokens=False).ids[:40]
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    m = eng.fn_model(pack, 2048)
    m.set_sampler(0.0, 1.0, 1, 0)
    M = Mtp(eng, m)
    os.environ["NE_MTP_HEAD_KV"] = "full"  # the pre-headlazy head recordings (full layer per head step)
    MF = Mtp(eng, m)
    del os.environ["NE_MTP_HEAD_KV"]
    if a.dflash:  # lab-dflash (docs/mtp.md 13): the block drafter drafts the rounds (greedy: 3 drafts), both loops
        en = eng.lib.ne_fn_dfl_enable
        en.restype, en.argtypes = C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int]
        for MM in (M, MF):
            if en(MM.p, a.dflash.encode(), 8, 4, 8, 1):
                raise E.EngineError(eng.last_error())
        print(f"dflash on: {a.dflash}", flush=True)
    fails = []

    def check(name, ok, msg=""):
        print(f"{'PASS' if ok else 'FAIL'} {name} {msg}", flush=True)
        if not ok:
            fails.append(name)

    def gap(prefix, s1, s2, d):
        """the 1-row decode's logit gap between the two forked tokens (fresh prefill of everything before them)"""
        if d is None or d >= min(len(s1), len(s2)):
            return ""
        m.reset()
        _, lg, _ = m.prefill(prefix + s2[:d], logits=True)
        top = np.sort(lg)[::-1]
        return (f" (tokens {s1[d]} / {s2[d]}: 1-row logits {lg[s1[d]]:.3f} / {lg[s2[d]]:.3f}, top-2 gap "
                f"{top[0] - top[1]:.3f})")

    modes = ["steps", "chunked"] if a.prefill == "both" else [a.prefill]
    depths = [int(x) for x in a.depths.split(",")]
    for mode in modes:
        os.environ["NE_FN_PREFILL"] = mode
        for depth in depths:
            tag = f"[{mode} d{depth}]"
            A, sa = M.generate(P, a.n, depth)
            A2, sa2 = M.generate(P, a.n, depth)
            check(f"fresh run repeats {tag}", A == A2 and (depth == 0 or sa["hist"] == sa2["hist"]))
            T = m.info()["T"]
            Lh, npend = M.head_state()
            check(f"live-state {tag}", T == len(P) + len(A) - 1 and Lh + npend == T and M.head_ok(),
                  f"T {T} = {len(P)} + {len(A)} - 1, head {Lh} + {npend}")
            for k in (200, 256):
                m.reset()
                M.feed(P[:100])
                M.feed(P[100:k])  # two pieces: the head's 8-row steps must not depend on the split
                s = M.snapshot()
                M.generate_from(Q, 24, depth)  # overwrites the KV rows past k (target and head), as the server does
                M.restore(s)
                B, sb = M.generate_from(P[k:], a.n, depth)
                M.L.ne_fn_mtp_snap_free(s)
                d = first_diff(A, B)
                if mode == "steps":
                    # adaptive depth (d0) picks depths from measured round times: only the tokens must match there
                    check(f"exact k={k} {tag}", d is None and (depth == 0 or sa["hist"] == sb["hist"]),
                          f"first diff {d}; rounds {sa['rounds']:.0f} / {sb['rounds']:.0f}")
                else:
                    print(f"INFO chunked k={k} {tag}: tokens identical {d is None} (first diff {d} of {len(A)})"
                          f"{gap(P, A, B, d)}; "
                          f"hist equal {sa['hist'] == sb['hist']}", flush=True)
            # live continuation: state = P + B[:-1]; continue with B[-1] + X
            m.reset()
            M.feed(P[:256])
            B, _ = M.generate_from(P[256:], a.n, depth)
            ms = M.L.ne_fn_model_snapshot(m.ptr)
            check(f"snapshot after MTP {tag}", bool(ms), eng.last_error() if not ms else "")
            if ms:
                M.L.ne_fn_snap_free(ms)
            Cc, _ = M.generate_from([B[-1]] + X, a.n, depth)
            D, _ = M.generate(P + B + X, a.n, depth)
            d = first_diff(Cc, D)
            print(f"INFO live continuation {tag}: identical {d is None} (first diff {d} of {len(D)})"
                  f"{gap(P + B + X, Cc, D, d)}", flush=True)
            # cut runs: max_new inside a round
            for mx in (2, 3, 5, 6, 7):
                o, _ = M.generate(P, mx, depth)
                T = m.info()["T"]
                Lh, npend = M.head_state()
                ok = T == len(P) + len(o) - 1 and Lh + npend == T
                Cn, _ = M.generate_from([o[-1]], a.n, depth)
                both = min(len(A), len(o) + len(Cn))
                d = first_diff((o + Cn)[:both], A[:both])
                check(f"cut max_new={mx} {tag}", ok, f"T {T}, head {Lh} + {npend}; continued == uncut: {d is None} "
                      f"(first diff {d})")
        # step path keeps the head; a plain step makes it stale
        m.reset()
        M.feed(P)
        y = None
        for t in X:
            y = M.step(t)
        Lh, npend = M.head_state()
        check(f"mtp_step head [{mode}]", M.head_ok() and Lh + npend == m.info()["T"], f"head {Lh} + {npend}")
        S1, h1 = M.generate_from([y], a.n, 2)
        # the same target arithmetic without mtp_step: 1-token body-only feeds (= the step's body), then MTP
        m.reset()
        M.feed(P)
        for t in X:
            M.feed([t])
        S3, h3 = M.generate_from([y], a.n, 2)
        d = first_diff(S1, S3)
        check(f"mtp_step == 1-token feeds [{mode}]", d is None and h1["hist"] == h3["hist"], f"first diff {d}")
        S2, _ = M.generate(P + X + [y], a.n, 2)
        d = first_diff(S1, S2)
        # the plain (no MTP) greedy decode from the same two states: where 1-row vs prefill arithmetic alone forks
        m.reset()
        M.feed(P)
        for t in X:
            M.step(t)
        g1, nx = [], y
        for _ in range(a.n):
            g1.append(nx)
            nx = m.step(nx)[0]
        m.reset()
        g2, _, _ = m.generate(P + X + [y], a.n + 1)
        print(f"INFO mtp_step then MTP vs fresh MTP [{mode}]: first diff {d}{gap(P + X + [y], S1, S2, d)}; plain "
              f"decode from the same two states: first diff {first_diff(g1[1:], g2[:-1])}", flush=True)
        m.reset()
        M.feed(P)
        m.step(X[0])
        stale = not M.head_ok()
        try:
            M.generate_from(X[1:], 8, 2)
            refused = False
        except E.EngineError:
            refused = True
        check(f"stale head refused [{mode}]", stale and refused)
        # state export / import with a stale head (lazy head: many pending rows): exported as no head, imported as an
        # empty stale head; an import whose head bookkeeping does not add up to T is refused before touching the model
        def state_io(to_model, buf, meta):
            return M.L.ne_fn_mtp_state_io(M.p, to_model, buf.ctypes.data, buf.nbytes, meta.ctypes.data)
        M.set_cap(4096)
        m.reset()
        M.feed(P)
        y = None
        for t in X:
            y = M.step(t)
        npend0 = M.head_state()[1]
        m.step(y)  # plain step: the head goes stale with its pending rows
        buf = np.empty(M.L.ne_fn_mtp_state_bytes(M.p), np.uint8)
        meta = np.zeros(6, np.int64)
        rc = state_io(0, buf, meta)
        check(f"stale head: state export [{mode}]", rc == 0 and meta[3] == 0 and meta[5] == 0,
              f"rc {rc}, meta {meta.tolist()}, pending before {npend0}")
        rc = state_io(1, buf, meta)
        check(f"stale head: state import clears the head [{mode}]", rc == 0 and not M.head_ok()
              and M.head_state() == (0, 0), f"rc {rc}, head {M.head_state()}")
        m.reset()
        M.feed(P)
        for t in X[:3]:
            M.step(t)
        rc = state_io(0, buf, meta)
        T0 = m.info()["T"]
        bad = meta.copy()
        bad[4] += 1
        rc2 = state_io(1, buf, bad)
        check(f"head state import with Lh + pending != T refused [{mode}]", rc == 0 and rc2 != 0
              and m.info()["T"] == T0 and M.head_ok(), f"rc {rc}/{rc2}, T {T0} -> {m.info()['T']}")
        M.set_cap(1024)
        # a run reaching the context end leaves a clean state (one round of headroom for the flush; it used to stop at
        # T >= tmax - 1 with the last round's rows unflushed: dirty)
        tmax = 2048
        for dep in (2, 4):
            long_p = (P * 6)[:tmax - 40]
            S, _ = M.generate(long_p, 100, dep)
            T = m.info()["T"]
            Lh, npend = M.head_state()
            snap_ok = True
            try:
                M.L.ne_fn_mtp_snap_free(M.snapshot())
            except E.EngineError:
                snap_ok = False
            check(f"run to the context end d{dep} [{mode}]", T == len(long_p) + len(S) - 1 and T >= tmax - 4
                  and M.head_ok() and Lh + npend == T and snap_ok, f"T {T} of {tmax}, {len(S)} tokens, head {Lh} + "
                  f"{npend}, snapshot {snap_ok}")
        # lazy head: feed P, 40 forced + 60 own greedy mtp_steps, then MTP
        def lazy_run(MM, cap, how):
            MM.set_cap(cap)
            m.reset()
            MM.feed(P)
            y = None
            hist = list(P)  # the tokens in the state (the copy rule's history after a park, as the server passes it)
            for t in X:
                hist.append(t)
                y = MM.step(t)
            for _ in range(60):
                hist.append(y)
                y = MM.step(y)
            npend = MM.head_state()[1]
            if how == "snap":
                s = MM.snapshot()
                npend = (npend, MM.head_state()[1])
                MM.generate_from(Q, 24, 2)
                MM.restore(s)
                MM.L.ne_fn_mtp_snap_free(s)
            elif how == "park":  # lab-srvcache's parking: state export (catches up) + KV rows of both, then back
                buf = np.empty(MM.L.ne_fn_mtp_state_bytes(MM.p), np.uint8)
                meta = np.zeros(6, np.int64)
                MM._ok(MM.L.ne_fn_mtp_state_io(MM.p, 0, buf.ctypes.data, buf.nbytes, meta.ctypes.data), "state out")
                T, lh = int(meta[0]), int(meta[4])
                kb = np.empty(MM.L.ne_fn_kv_bytes(m.ptr, T), np.uint8)
                hb = np.empty(MM.L.ne_fn_mtp_kv_bytes(MM.p, lh), np.uint8)
                MM._ok(MM.L.ne_fn_kv_io(m.ptr, 0, 0, T, kb.ctypes.data, T), "kv out")
                MM._ok(MM.L.ne_fn_mtp_kv_io(MM.p, 0, 0, lh, hb.ctypes.data, lh), "head kv out")
                npend = (npend, int(meta[5]))
                m.reset()
                MM.generate(Q, 24, 2)  # overwrites both caches' rows
                MM._ok(MM.L.ne_fn_kv_io(m.ptr, 1, 0, T, kb.ctypes.data, T), "kv in")
                MM._ok(MM.L.ne_fn_mtp_kv_io(MM.p, 1, 0, lh, hb.ctypes.data, lh), "head kv in")
                MM._ok(MM.L.ne_fn_mtp_state_io(MM.p, 1, buf.ctypes.data, buf.nbytes, meta.ctypes.data), "state in")
                # the copy rule knows no tokens after an import; the server gives them back (ne_fn_mtp_set_context)
                sc = MM.L.ne_fn_mtp_set_context
                sc.restype, sc.argtypes = C.c_int, [C.c_void_p, C.c_void_p, C.c_int]
                hs = np.ascontiguousarray(hist, np.int32)
                MM._ok(sc(MM.p, hs.ctypes.data, len(hs)), "set_context")
            elif how == "catch":
                MM.catch_up()
                npend = (npend, MM.head_state()[1])
            S, h = MM.generate_from([y], a.n, 2)
            return S, h["hist"], npend
        ref = lazy_run(M, 0, "")
        for name, MM, cap, how in (("lazy", M, 4096, ""), ("lazy+snapshot", M, 4096, "snap"),
                                   ("lazy+catch_up", M, 4096, "catch"),
                                   ("lazy+park (state / KV export)", M, 4096, "park"), ("lazy cap 64", M, 64, ""),
                                   ("full-layer head eager", MF, 0, ""), ("full-layer head lazy", MF, 4096, "")):
            r = lazy_run(MM, cap, how)
            d = first_diff(ref[0], r[0])
            check(f"{name} == eager [{mode}]", d is None and r[1] == ref[1], f"first diff {d}; pending {r[2]} "
                  f"(eager {ref[2]})")
        M.set_cap(1024)
    print("SUMMARY", "all pass" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
