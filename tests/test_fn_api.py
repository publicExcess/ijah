"""lab-integrate: whole-model API checks (~83 GB, lock): snapshot / restore, the MTP token callback (streaming / abort),
prefill() routing (chunked from 16 tokens vs NE_FN_PREFILL=steps: same greedy first tokens).

    NEED_GB=85 LOCK_OWNER=... LOCK_WAIT=30 tools/with_lock.sh "fn api (~83 GB)" HH:MM -- $PY tests/test_fn_api.py
"""
import ctypes as C
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
os.environ.setdefault("NE_FN_ROUND", "f32")
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402


def main():
    pk = refenv.ijah_paths.pack()
    tok = Tokenizer.from_file(str(Path(pk) / "tokenizer.json"))
    eng = E.Engine()
    lib = eng.lib
    pack = eng.pack_open(pk)
    m = eng.fn_model(pack, 2048)
    m.set_sampler(0.0, 1.0, 1, 0)
    ids = tok.encode(B.chat(pk, B.THINK["code_lru"]), add_special_tokens=False).ids
    ok = True

    # snapshot / restore: the same greedy steps after a restore, bit-identical logits
    lib.ne_fn_model_snapshot.restype = C.c_void_p
    lib.ne_fn_model_snapshot.argtypes = [C.c_void_p]
    lib.ne_fn_model_restore.argtypes = [C.c_void_p, C.c_void_p]
    lib.ne_fn_snap_T.argtypes = [C.c_void_p]
    lib.ne_fn_snap_bytes.restype = C.c_int64
    lib.ne_fn_snap_bytes.argtypes = [C.c_void_p]
    lib.ne_fn_snap_free.argtypes = [C.c_void_p]
    def run16(t):
        out, lg = [], []
        for _ in range(16):
            t, _, _, l = m.step(t, head=True, logits=True)
            out.append(t)
            lg.append(l)
        return out, np.stack(lg)

    # lab-gdnlazy: reference without a snapshot (the decode steps keep deferred GDN rows; the snapshot materialises
    # them, so the run after it must match this one bit for bit)
    m.reset()
    nx, _, _ = m.prefill(ids)
    r0, lr0 = run16(nx)
    m.reset()
    nx, _, _ = m.prefill(ids)
    snap = lib.ne_fn_model_snapshot(C.c_void_p(m.ptr))
    assert snap, eng.last_error()

    a, la = run16(nx)
    lazy_same = a == r0 and np.array_equal(la, lr0)
    print(f"lazy GDN store: 16 greedy steps after a materialising snapshot vs none: "
          f"{'identical' if lazy_same else 'DIFFER'}")
    ok = ok and lazy_same
    # the KV rows past the snapshot's T are overwritten by the second run, as the restore contract allows
    rc = lib.ne_fn_model_restore(C.c_void_p(m.ptr), C.c_void_p(snap))
    b, lb = run16(nx)
    same = rc == 0 and a == b and np.array_equal(la, lb)
    print(f"snapshot T {lib.ne_fn_snap_T(C.c_void_p(snap))} ({lib.ne_fn_snap_bytes(C.c_void_p(snap)) / 1e6:.0f} MB): "
          f"restore rc {rc}, 16 greedy steps {'identical' if same else 'DIFFER'}")
    ok &= same
    lib.ne_fn_snap_free(C.c_void_p(snap))

    # prefill() routing: chunked (default from 16 tokens) vs the old 4-row / 1-row path
    m.reset()
    gc, _, sc = m.generate(ids, 24, -1)
    os.environ["NE_FN_PREFILL"] = "steps"
    m.reset()
    gs, _, ss = m.generate(ids, 24, -1)
    os.environ.pop("NE_FN_PREFILL")
    k = next((i for i in range(24) if gc[i] != gs[i]), 24)
    print(f"generate: chunked prefill {sc['prefill_ms']:.0f} ms vs steps {ss['prefill_ms']:.0f} ms ({len(ids)} tokens); "
          f"greedy identical for {k} of 24 tokens")
    ok &= k >= 1

    # MTP callback: tokens streamed in order, abort after >= 10
    mtp = B.Mtp(eng, m, 4, 4)
    f = lib.ne_fn_mtp_generate_cb
    CB = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_int), C.c_int)
    f.restype = C.c_int
    f.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_void_p,
                  C.c_void_p, CB, C.c_void_p]
    full, _ = mtp.generate(ids, 64, 2, seed=7)
    seen = []

    def cb(ctx, toks, n):
        seen.extend(toks[i] for i in range(n))
        return 1 if len(seen) >= 10 else 0

    cbf = CB(cb)
    p = np.ascontiguousarray(ids, np.int32)
    out = np.empty(64, np.int32)
    n = f(C.c_void_p(mtp.ptr), p.ctypes.data, len(p), 64, -1, 2, 7, out.ctypes.data, None, None, cbf, None)
    good = 10 <= n <= 13 and seen == out[:n].tolist() and seen == full[:n]
    print(f"mtp callback: stopped at {n} tokens (asked >= 10), streamed == out: {seen == out[:n].tolist()}, "
          f"== uncallbacked run's prefix: {seen == full[:n]}")
    ok &= good
    print("test_fn_api:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
