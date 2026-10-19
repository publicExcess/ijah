"""lab-mtpimg: MTP after an image prompt vs the step path, in-process through the server's RealBackend (whole model).

    NEED_GB=85 LOCK_OWNER=... tools/with_lock.sh "mtp image (~83 GB)" HH:MM -- $PY tests/vision/mtp_image.py
        [--n 1024] [--depth 2] [--text-n 512] [--out FILE.json]
    NE_MTP_HEAD_POS=text|mrope   the MTP head's rope positions (models/flash_next/mtp_loop.mm mtp_create)

Per prompt (the chart image prompt of tests/vision/e2e_client.py; a text thinking prompt for the acceptance baseline):
greedy step path (feed + 1-row steps) vs greedy MTP (feed of all but the last token + ne_fn_mtp_generate_from, as
the server's Session runs it): decode tok/s, tokens / round, first mismatch and the step path's logit gap there
(teacher-forced 1-row steps), and for the chart whether the answer lists the nine values.
"""
import argparse
import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import backend as B  # noqa: E402
import chat  # noqa: E402
import vision as V  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as BM  # noqa: E402

CHART = Path(os.environ.get("IJAH_TEST_CHART", ROOT / "tests/vision/assets/flow.png"))  # any bar-chart PNG (IJAH_TEST_CHART)
FLOW = ROOT / "tests/vision/assets/flow.png"
VALUES = ["0.208", "0.136", "0.104", "0.059", "0.199", "0.082", "0.137", "0.063", "0.207"]


def build(tpl, msgs, effort):
    srcs = []
    text, _ = tpl.render({"messages": msgs, "reasoning_effort": effort}, srcs)
    prompt = tpl.encode(text)
    images = []
    for s in srcs:
        img, _ = V.load_image(s)
        images.append(V.VisionImage(img))
    if images:
        prompt, _, _ = V.expand_prompt(prompt, images)
    return prompt, {im.d40: im for im in images}


def step_run(be, keys, n):
    be.reset()
    t0 = time.perf_counter()
    nx = be.feed(keys)
    t1 = time.perf_counter()
    out = [nx]
    while len(out) < n and out[-1] not in be.STOPS:
        out.append(be.step(out[-1]))
    t2 = time.perf_counter()
    return out, (t1 - t0) * 1e3, (len(out) - 1) / (t2 - t1)


def mtp_run(be, keys, n, depth):
    be.reset()
    t0 = time.perf_counter()
    be.feed(keys[:-1], sample=False)
    feed_ms = (time.perf_counter() - t0) * 1e3
    # = RealBackend.mtp_from with the (depth, accepted) round histogram
    be._image_rows(keys[-1:])
    p = np.ascontiguousarray(V.keys_to_tokens(keys[-1:]), np.int32)
    o = np.full(n, -1, np.int32)
    sv = np.zeros(10, np.float64)
    hist = np.zeros(25, np.int32)
    k = be._f["ne_fn_mtp_generate_from"](C.c_void_p(be.mtp_ptr), p.ctypes.data, 1, n, be.STOPS[0], depth, 0,
                                         o.ctypes.data, sv.ctypes.data, hist.ctypes.data, be._CB(lambda *_: 0), None)
    assert k > 0, be.eng.last_error()
    out, st = o[:k].tolist(), dict(zip(be.MTP_KEYS, sv.tolist()))
    st["hist"] = hist.reshape(5, 5)[depth, :depth + 1].tolist()
    tps = st["tokens"] / max(st["decode_ms"], 1e-3) * 1e3
    tpr = st["tokens"] / max(st["rounds"], 1)
    return out, feed_ms + st["prefill_ms"], tps, tpr, st


def gap_at(be, keys, base, mm):
    """the step path's logits at base[mm] (teacher-forced 1-row steps): logit(base) - logit(mtp's), top-2 gap"""
    be.reset()
    lg = np.zeros(be.VOCAB, np.float32)
    nx = C.c_int(-1)
    be.feed(keys[:-1], sample=False)
    L = be.eng.lib
    toks = V.keys_to_tokens(keys[-1:]) + base[:mm]
    for i, t in enumerate(toks):
        last = i == len(toks) - 1
        rc = L.ne_fn_model_step(be.m.ptr, t, 1, lg.ctypes.data if last else None, C.byref(nx), None, None, None)
        assert rc == 0, be.eng.last_error()
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--text-n", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--effort", default="low", help="the image prompts' reasoning effort (e2e_client's); text: medium")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    hp = os.environ.get("NE_MTP_HEAD_POS", "text") or "text"
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok, a.effort)
    be = B.RealBackend(ROOT, a.pack, a.ctx, 4, ROOT / "build/libengine.dylib", None, print)
    print(f"head positions {hp}; mtp_cont {be.mtp_cont}; load {be.load_s:.1f} s", flush=True)
    be.set_sampler(0.0, 1.0, 1, 0)
    cases = [("chart", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": str(CHART)}},
                                                     {"type": "text", "text": "List every bar with its exact value."}]}],
              a.n, a.effort),
             ("mermaid", [{"role": "user", "content": [
                 {"type": "text", "text": "Convert this diagram to a Mermaid flowchart. Reply with the Mermaid code only."},
                 {"type": "image_url", "image_url": {"url": str(FLOW)}}]}], a.n, a.effort),
             ("text", [{"role": "user", "content": BM.THINK["code_lru"]}], a.text_n, "medium")]
    res = {"head_pos": hp, "depth": a.depth}
    for name, msgs, n, effort in cases:
        keys, images = build(tpl, msgs, effort)
        be.prepare(keys, images)  # positions of the whole prompt (+ the tower once): survives reset()
        nimg = sum(1 for k in keys if V.is_key(k))
        base, pf_s, tps_s = step_run(be, keys, n)
        out, pf_m, tps_m, tpr, st = mtp_run(be, keys, n, a.depth)
        mm = next((i for i in range(min(len(base), len(out))) if base[i] != out[i]), None)
        r = dict(prompt=len(keys), image_tokens=nimg, step_tokens=len(base), mtp_tokens=len(out), step_tps=tps_s,
                 mtp_tps=tps_m, tokens_per_round=tpr, rounds=st["rounds"], step_prefill_ms=pf_s, mtp_prefill_ms=pf_m,
                 head_prefill_ms=st["head_prefill_ms"], first_mismatch=mm,
                 draft_ms=st["draft_gpu_ms"], verify_ms=st["verify_gpu_ms"], hist=st["hist"])
        extra = "identical" if mm is None and len(base) == len(out) else f"first mismatch at {mm}"
        if mm is not None and mm < len(base) and mm < len(out):
            lg = gap_at(be, keys, base, mm)
            srt = np.sort(lg)
            r["gap"] = float(lg[base[mm]] - lg[out[mm]])
            r["top2_gap"] = float(srt[-1] - srt[-2])
            extra += (f" ({tok.decode([base[mm]])!r} vs {tok.decode([out[mm]])!r}; step-path logit gap {r['gap']:.3f},"
                      f" top-2 {r['top2_gap']:.3f})")
        for path, o in (("step", base), ("mtp", out)):
            txt = tok.decode(o, skip_special_tokens=True)
            ans = txt.split("</think>", 1)[-1]
            r[f"{path}_answer"] = ans.strip()
            if name == "chart":
                r[f"{path}_values"] = sum(v in ans for v in VALUES)
        print(f"{name}: prompt {len(keys)} ({nimg} image tokens); step {tps_s:.1f} tok/s ({len(base)} tokens, prefill "
              f"{pf_s:.0f} ms); MTP d{a.depth} {tps_m:.1f} tok/s, {tpr:.2f} tokens/round over {st['rounds']:.0f} "
              f"rounds (accepted 0..{a.depth}: {st['hist']}) ({len(out)} tokens, prefill {pf_m:.0f} ms, head {st['head_prefill_ms']:.0f} ms); {extra}"
              + (f"; values step {r['step_values']}/9, mtp {r['mtp_values']}/9" if name == "chart" else ""),
              flush=True)
        res[name] = r
    print("\nchart MTP answer:\n" + res["chart"]["mtp_answer"][:1500])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
