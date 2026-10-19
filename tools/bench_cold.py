"""lab-pfpath: time to first token of COLD prompts (no prefix-cache hit) through the server's own request path,
in-process (chat.Template render + encode, Session.generate on a RealBackend: the code the HTTP server runs, minus
the socket), with the engine's per-chunk wall split (NE_FEED_LOG=1, NE_PF_RECLOG=1 in the environment).

    NEED_GB=85 LOCK_OWNER=pfpath LOCK_WAIT=30 tools/with_lock.sh "bench_cold (~81 GB)" HH:MM -- \\
        env NE_FEED_LOG=1 $PY tools/bench_cold.py [--sizes 10240,32768] [--reps 2] [--arms base,noahead] [--mtp 0]

Prompts: the head-to-head harness's new-text stream (IJAH_NEW_STREAM, a JSON list of token ids: 1,024-token prose /
web / code blocks; not shipped), disjoint slices behind a fresh nonce, the same request body as the harness's prefill
phase (one user
message, thinking off, sampled T=1 p=0.95 k=20 with a seed: the step path, as the harness's requests take).
--mtp 1: the MTP path instead (no seed). Arms run in turn in this process, interleaved per prompt size:
  base           the defaults
  noahead        NE_SRV_AHEAD off (no next-piece n-gram prefetch)
  B:NAME=V       a server/backend.py module switch for that arm (e.g. B:REST_FIRST=0)
  env:K=V[;K=V]  os.environ for that arm (knobs read per call)
TTFT = generate() call (incl. render + encode) -> first streamed token; prefill tok/s = prompt tokens / TTFT.
"""
import argparse
import json
import os
import statistics as stat
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

ASK = "\n\nIn one sentence, what is the first part of the text above about?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--sizes", default="10240,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--arms", default="base")
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--warm", default="4000,10240", help="unmeasured warm-up prompt sizes (recordings, pipelines)")
    ap.add_argument("--idle", type=float, default=0.0, help="seconds of idle before each measured request")
    ap.add_argument("--ping", type=float, default=0.0, help="during the idle: ne_fn_gpu_ping every PING s (0 = none); "
                    "< 0: one ping right before the request")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok, "xhigh")
    ids = json.load(open(os.environ.get("IJAH_NEW_STREAM") or sys.exit("set IJAH_NEW_STREAM: a JSON list of token ids of text the model has not seen (not shipped)")))
    pos = [200000]
    t0 = time.time()
    be = B.RealBackend(ROOT, a.pack, a.ctx, 2, ROOT / "build/libengine.dylib", None, print)
    print(f"load {time.time() - t0:.1f} s, mtp_cont {be.mtp_cont}, feed_ahead {'ne_fn_mtp_feed_ahead' in be._f}",
          flush=True)
    S = B.Session(be, a.ctx, 4, "auto", log=print)
    IME = B.Session.IM_END

    def prompt(size):
        text = f"[new {size}-{time.time_ns()}] " + tok.decode(ids[pos[0]:pos[0] + size - 40]) + ASK
        pos[0] += size
        return {"messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}}

    def run(req, label):
        first = []
        t = time.perf_counter()
        P, _ = tpl.render_ids(req)
        t_enc = time.perf_counter() - t

        def on_tokens(x, *_):
            if not first:
                first.append(time.perf_counter())
        sampler = (1.0, 0.95, 20, 0 if not a.mtp else 1234)
        st = S.generate(P, a.max_new, sampler, bool(a.mtp), on_tokens, lambda: False, IME)
        ttft = first[0] - t
        print(f"{label}: prompt {len(P)} cached {st['cached_tokens']} path {st['path']}: TTFT {ttft * 1e3:.0f} ms "
              f"({len(P) / ttft:.0f} tok/s), render+encode {t_enc * 1e3:.1f} ms, prefill_ms {st['prefill_ms']:.0f}, "
              f"restore {st.get('restore_ms', 0):.0f}, snapshot {st.get('snapshot_ms', 0):.0f}", flush=True)
        return len(P), ttft

    arms = [x for x in a.arms.split(",") if x]

    keys = {kv.split("=", 1)[0] for arm in arms if arm.startswith("env:") for kv in arm[4:].split(";")}
    base_env = {k: os.environ.get(k) for k in keys}
    base_mod = {kv.split("=", 1)[0]: getattr(B, kv.split("=", 1)[0]) for arm in arms if arm.startswith("B:")
                for kv in arm[2:].split(";")}

    def set_arm(arm):
        for k, v in base_env.items():  # every arm starts from the process's own environment
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        B.AHEAD = arm != "noahead"
        for k, v in base_mod.items():
            setattr(B, k, v)
        if arm.startswith("B:"):  # backend module switches, e.g. B:REST_FIRST=0
            for kv in arm[2:].split(";"):
                k, v = kv.split("=", 1)
                setattr(B, k, type(getattr(B, k))(int(v)) if isinstance(getattr(B, k), bool) else v)
        if arm.startswith("env:"):
            for kv in arm[4:].split(";"):
                k, v = kv.split("=", 1)
                os.environ[k] = v

    for w in [int(x) for x in a.warm.split(",") if x]:
        run(prompt(w), f"warm {w}")
    import ctypes
    pms = ctypes.c_double(0)
    if a.ping:
        be.eng.lib.ne_fn_gpu_ping.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    res = {}
    for size in [int(x) for x in a.sizes.split(",") if x]:
        for rep in range(a.reps):
            for arm in arms:
                set_arm(arm)
                if a.idle:
                    t_end = time.perf_counter() + a.idle
                    while a.ping > 0 and time.perf_counter() + a.ping < t_end:
                        time.sleep(a.ping)
                        be.eng.lib.ne_fn_gpu_ping(ctypes.c_void_p(be.m.ptr), ctypes.byref(pms))
                    time.sleep(max(0.0, t_end - time.perf_counter()))
                    if a.ping < 0:
                        be.eng.lib.ne_fn_gpu_ping(ctypes.c_void_p(be.m.ptr), ctypes.byref(pms))
                        print(f"ping before the request: {pms.value:.1f} ms", flush=True)
                n, t = run(prompt(size), f"{arm} {size} r{rep}")
                res.setdefault((arm, size), []).append(n / t)
    print("\n| arm | size | prefill tok/s (prompt / TTFT) | reps |")
    print("|---|---|---|---|")
    for (arm, size), v in res.items():
        sd = stat.stdev(v) if len(v) > 1 else 0.0
        print(f"| {arm} | {size} | {stat.mean(v):.0f} +- {sd:.0f} | {', '.join(f'{x:.0f}' for x in v)} |")


if __name__ == "__main__":
    main()
