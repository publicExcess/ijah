"""DFlash2 drafter on the engine vs the MLX reference (tools/dflash_ref.py) on the same context (lab-dflash; docs/mtp.md 13).

    NEED_GB=85 LOCK_OWNER=dflash tools/with_lock.sh "test_dflash (~84 GB)" HH:MM -- $PY tests/mtp/test_dflash.py

Per prompt: the target's features of every prompt position from the engine's verify path (ne_fn_dfl_verify, rows of
<= 5), the same rows into both drafters' contexts, then one block draft from an anchor: final-normed hidden (rel L2 per
row), the 16 candidates (overlap with the reference's exact top 16; logit error on the shared ones), greedy drafts;
then the sampled proposals q of the engine vs the reference's selector on the engine's candidates (TV). Checks:
hn rel L2 < 3e-2 (8-bit drafter weights), candidate overlap >= 12 / 16 per row, q TV < 2e-2, first greedy draft
equal.
"""
import ctypes as C
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402

os.environ.setdefault("NE_FN_ROUND", "f32")
refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from bench_mtp import Mtp, chat  # noqa: E402
from bench_draft import H2H  # noqa: E402
from dflash_ref import DFlash2, Host, DRAFTER  # noqa: E402


def main():
    pack_dir = (ROOT / "PACK_READY").read_text().strip()
    tok = Tokenizer.from_file(str(Path(pack_dir) / "tokenizer.json"))
    eng = E.Engine()
    pack = eng.pack_open(pack_dir)
    m = eng.fn_model(pack, 1024)
    mtp = Mtp(eng, m, 5, 4)
    lib = eng.lib
    lib.ne_fn_dfl_enable.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int]
    lib.ne_fn_dfl_ctx_reset.argtypes = [C.c_void_p, C.c_int]
    lib.ne_fn_dfl_ctx_push.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    lib.ne_fn_dfl_probe.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_uint32, C.c_void_p, C.c_void_p,
                                    C.c_void_p, C.c_void_p, C.c_void_p]
    P = C.c_void_p(mtp.ptr)
    B = 8
    eng._ok(lib.ne_fn_dfl_enable(P, str(DRAFTER).encode(), B, 5, 8, 1), "dfl_enable")
    host = Host(eng, mtp)
    ref = DFlash2(pack=pack_dir)
    fails = []
    for name in ("code_lru", "text_crisis", "math_proof"):
        ids = tok.encode(chat(pack_dir, H2H[name]), add_special_tokens=False).ids
        host.reset()
        feats = []
        i = 0
        last = None
        while i < len(ids):
            r = min(5, len(ids) - i)
            if len(ids) - i - r == 1:
                r -= 1
            tk, lg, ft, _ = host.verify(ids[i:i + r])
            host.commit(ids[i:i + r], r)
            feats.append(ft)
            last = (tk[r - 1], lg[r - 1])
            i += r
        F = np.ascontiguousarray(np.concatenate(feats), np.float32)
        anchor = int(last[0][int(np.argmax(last[1]))])
        eng._ok(lib.ne_fn_dfl_ctx_reset(P, 0), "ctx_reset")
        eng._ok(lib.ne_fn_dfl_ctx_push(P, 0, F.ctypes.data, len(F)), "ctx_push")
        ref.reset()
        ref.add_context(F)
        hn = np.zeros((B - 1, 2560), np.float32)
        sl = np.zeros((B - 1, 16, 2), np.float32)
        qo = np.zeros((B - 1, 16, 2), np.float32)
        dr = np.zeros(B - 1, np.int32)
        ms = C.c_double()
        eng._ok(lib.ne_fn_dfl_probe(P, anchor, 0, 1.0, 1, hn.ctypes.data, sl.ctypes.data, dr.ctypes.data, None,
                                    C.byref(ms)), "probe")
        mx = ref.mx
        rh = ref.forward(anchor, B).astype(mx.float32)
        rl = ref.logits(rh).astype(mx.float32)
        mx.eval(rh, rl)
        rh, rl = np.array(rh), np.array(rl)
        rel = np.linalg.norm(hn - rh, axis=1) / np.linalg.norm(rh, axis=1)
        cid = sl[:, :, 0].copy().view(np.uint32).astype(np.int64)
        over, lerr = [], []
        for j in range(B - 1):
            top = set(np.argsort(-rl[j])[:16].tolist())
            over.append(len(top & set(cid[j].tolist())))
            sh = [c for c in cid[j] if c in top]
            lerr.append(max(abs(float(sl[j][list(cid[j]).index(c)][1]) - float(rl[j][c])) for c in sh) if sh else -1)
        rng = np.random.default_rng(0)
        rd, _ = ref.propose(anchor, B, 0.0, rng)
        # sampled: the engine's q vs the reference selector over the engine's candidates (same unary logits)
        eng._ok(lib.ne_fn_dfl_probe(P, anchor, 1, 1.0, 7, None, sl.ctypes.data, dr.ctypes.data, qo.ctypes.data,
                                    C.byref(ms)), "probe sampled")
        s_ = np.array(ref.lin(mx.array(hn), "candidate_selector.hidden_projection").astype(mx.float32))
        pc = ref.w["candidate_selector.predecessor_codebook"]
        sc = ref.w["candidate_selector.successor_codebook"]
        prev = anchor
        tv = []
        for j in range(B - 1):
            c_ = sl[j, :, 0].copy().view(np.uint32).astype(np.int64)
            pv = np.array(pc[prev].astype(mx.float32))
            cs = np.array(sc[mx.array(c_)].astype(mx.float32))
            sco = sl[j, :, 1] + cs @ (pv * s_[j])
            q = np.exp(sco - sco.max())
            q /= q.sum()
            tv.append(0.5 * np.abs(q - qo[j, :, 1]).sum())
            prev = int(dr[j])
        print(f"{name}: T {len(F)} anchor {anchor}: hn rel L2 {np.round(rel, 4).tolist()}, candidates shared "
              f"{over}, logit err {np.round(lerr, 3).tolist()}, greedy drafts engine {dr.tolist() if False else ''}"
              f"{[int(x) for x in dr]} (sampled) / reference greedy {rd}, q TV {np.round(tv, 4).tolist()}, "
              f"draft GPU {ms.value:.2f} ms", flush=True)
        if rel.max() > 3e-2:
            fails.append(f"{name}: hn rel L2 {rel.max():.3g}")
        if min(over) < 12:
            fails.append(f"{name}: candidate overlap {min(over)}")
        if max(tv) > 2e-2:  # (the reference selector uses the bf16 hidden_projection, the engine its 8-bit copy)
            fails.append(f"{name}: q TV {max(tv):.3g}")
    # greedy drafts: probe again greedily on the last context and compare with the reference
    eng._ok(lib.ne_fn_dfl_probe(P, anchor, 0, 1.0, 1, None, None, dr.ctypes.data, None, C.byref(ms)), "probe")
    print(f"greedy drafts engine {dr.tolist()} reference {rd}")
    if int(dr[0]) != rd[0]:
        fails.append("first greedy draft differs")
    print("FAIL: " + "; ".join(fails) if fails else "ALL PASS")
    pack.close()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
