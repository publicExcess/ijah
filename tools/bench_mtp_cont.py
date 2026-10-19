"""lab-mtpcont: a multi-turn conversation through the server's Session + RealBackend in-process (no HTTP): MTP from the
reused state vs the 1-row step path, per turn: cached / prompt tokens, prefill ms, decode tok/s (docs/mtp.md 10.9).

    NEED_GB=85 LOCK_OWNER=... tools/with_lock.sh "mtp cont bench (~83 GB)" HH:MM -- $PY tools/bench_mtp_cont.py
        [--turns 3] [--n 512] [--sys-docs 1] [--temp 1.0]

Pass "mtp": every turn on the MTP path (policy always; continues from the restored checkpoint / live state).
Pass "step": every turn on the step path (the feed / step still keep the MTP head in step: the cost of that).
Then a micro check: 256 decode steps through ne_fn_mtp_step vs plain ne_fn_model_step from the same state.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import backend as B  # noqa: E402
import chat  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as BM  # noqa: E402

FOLLOW = ["Now make the cache size configurable at runtime and explain what changes.",
          "Summarise the whole discussion in five bullet points.",
          "What would you test first in production, and why?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--sys-docs", type=int, default=1, help="repository docs in the system prompt (0 = short)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ctx", type=int, default=32768)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok, "medium")
    be = B.RealBackend(ROOT, a.pack, a.ctx, 2, ROOT / "build/libengine.dylib", None, print)
    print(f"mtp_cont {be.mtp_cont}, load {be.load_s:.1f} s")
    sysmsg = "You are a careful senior engineer."
    if a.sys_docs:
        docs = [p for p in BM.DOCS if p.exists()][:a.sys_docs]
        sysmsg += "\n\nReference material:\n" + "\n\n".join(p.read_text()[:24000] for p in docs)
    sampler = (a.temp, 0.95, 20, 7) if a.temp >= 0.01 else (0.0, 1.0, 1, 0)
    res = {}
    for path in ("mtp", "step"):
        be.reset()
        s = B.Session(be, a.ctx, 6, "always" if path == "mtp" else "never", log=print)
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": BM.THINK["code_lru"]}]
        rows = []
        for t in range(a.turns):
            text, _ = tpl.render({"messages": msgs})
            p = tpl.encode(text)
            out = []
            t0 = time.perf_counter()
            st = s.generate(p, a.n, sampler, path == "mtp", out.extend, lambda: False, tpl.im_end)
            wall = (time.perf_counter() - t0) * 1e3
            ok = be.T == len(p) + len(out) - 1
            dec = (len(out) - 1) / max(st["decode_ms"], 1e-3) * 1e3
            tail = wall - st["prefill_ms"] - st["decode_ms"]
            rows.append((t + 1, st["path"], len(p), st["cached_tokens"], st["prefill_ms"], len(out), dec, tail, ok))
            print(f"{path} turn {t + 1}: path {st['path']}, prompt {len(p)}, cached {st['cached_tokens']}, prefill "
                  f"{st['prefill_ms']:.0f} ms, {len(out)} tokens at {dec:.1f} tok/s, rest {tail:.0f} ms, "
                  f"T == prompt + out - 1: {ok}", flush=True)
            reply = tok.decode(out, skip_special_tokens=True)
            r, c = reply.split("</think>", 1) if "</think>" in reply else ("", reply)
            msgs = msgs + [{"role": "assistant", "content": c.strip(), "reasoning_content": r.strip()},
                           {"role": "user", "content": FOLLOW[t % len(FOLLOW)]}]
        res[path] = rows
    print("\n| turn | path | prompt | cached | prefill ms | tokens | decode tok/s |")
    print("|---|---|---|---|---|---|---|")
    for path, rows in res.items():
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.0f} | {r[5]} | {r[6]:.1f} |")
    # micro: decode cost of keeping the head in step on the step path
    import ctypes as C
    p = tok.encode(sysmsg[:4000], add_special_tokens=False).ids[:800]
    be.set_sampler(0.0, 1.0, 1, 0)
    times = {}
    for kind in ("mtp_step", "plain"):
        be.reset()
        nx = be.feed(p)
        t0 = time.perf_counter()
        for _ in range(256):
            if kind == "mtp_step":
                nx = be.step(nx)
            else:
                rc = be.eng.lib.ne_fn_model_step(be.m.ptr, nx, 1, None, C.byref(be._nx), None, None, None)
                assert rc == 0
                nx = be._nx.value
        times[kind] = (time.perf_counter() - t0) * 1e3 / 256
    print(f"decode step: ne_fn_mtp_step {times['mtp_step']:.2f} ms vs ne_fn_model_step {times['plain']:.2f} ms "
          f"per token ({(times['mtp_step'] / times['plain'] - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
