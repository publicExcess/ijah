"""Whole-model test of the multi-conversation prefix cache and the two-slot scheduler (lab-srvcache). ONE with_lock job:

    NEED_GB=85 LOCK_WAIT=30 LOCK_OWNER=lab-srvcache tools/with_lock.sh "srvcache test (~83 GB)" HH:MM -- \\
        $PY tests/server/cache_job.py [--lib build/libengine.dylib] [--ctx 32768]

1. KV + state round trip: prefill, save, decode greedy; overwrite the slot with another prompt; load, decode again:
   identical tokens. Times of the copies vs re-prefilling the same rows (T = 8K, 16K, 28K).
2. Three conversations (docs/*.md as system prompts) round robin on ONE slot (each turn parks the previous
   conversation): cached tokens and prefill ms per turn; greedy output vs a fresh run (no reuse) of the same prompt.
3. Two streams at once on two slots (interleaved token by token, as the server's worker): both outputs vs each alone;
   aggregate tok/s vs one stream.
Results: tests/server/results/cache-<stamp>.json + stdout.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

import backend as B  # noqa: E402
import chat  # noqa: E402

GREEDY = (0.0, 1.0, 1, 0)
RES = {}


def free_gb():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return 128 * int(m.group(1)) / 100 if m else -1


def say(*a):
    print(*a, flush=True)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def decode(b, nx, k):
    out = [nx]
    for _ in range(k - 1):
        nx = b.step(nx)
        out.append(nx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(ROOT / "build/libengine.dylib"))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--only", default="123", help="sections to run")
    ap.add_argument("--mtp", type=int, default=2, help="MTP depth of both slots (0 = no MTP head)")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok, "xhigh")
    RES["free_gb_start"] = free_gb()
    t0 = time.time()
    b0 = B.RealBackend(ROOT, a.pack, a.ctx, a.mtp, Path(a.lib).resolve(), None, say)
    RES["free_gb_slot0"] = free_gb()
    b1 = B.RealBackend(ROOT, a.pack, a.ctx, a.mtp, Path(a.lib).resolve(), None, say, share=b0)
    RES["free_gb_slot1"] = free_gb()
    say(f"loaded 2 slots in {time.time() - t0:.1f} s; free {RES['free_gb_start']:.0f} -> {RES['free_gb_slot0']:.0f} "
        f"-> {RES['free_gb_slot1']:.0f} GB; kvio {b0.kvio}; state {b0.snap_bytes / 2**20:.1f} MB, KV "
        f"{b0.kv_row_bytes / 1024:.2f} KB per token; MTP cont {b0.mtp_cont}, head io {b0.head_io}")
    assert b0.kvio
    for b in (b0, b1):
        b.set_sampler(*GREEDY)

    docs = [ROOT / "docs" / n for n in ("kernels.md", "mtp.md", "trunk6.md", "reference.md", "attn.md",
                                        "research-prefill.md", "forward.md", "prefill.md", "core.md")]
    long = tok.encode("\n\n".join(d.read_text() for d in docs), add_special_tokens=False).ids

    # ---- 1. round trip + copy vs re-prefill
    rt = []
    for T in ((8192, 16384, 28672) if '1' in a.only else ()):
        p = long[:T]
        b0.reset()
        t1 = time.perf_counter()
        b0.feed(p[:-1], sample=False)
        pf_ms = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        sbuf, smeta = b0.state_save()
        t2 = time.perf_counter()
        lh = b0.snap_head(smeta) or 0
        kbuf, cap, hcap = b0.kv_save(0, T - 1, 0, lh)
        t3 = time.perf_counter()
        if b0.mtp_cont:  # greedy MTP from the state: tokens AND rounds (= the head's drafts) must repeat
            ref, sr = b0.mtp_from(p[-1:], 48, 2, 0, lambda t: None, lambda: False)
        else:
            ref = decode(b0, b0.feed(p[-1:]), 32)
        b0.reset()
        b0.feed(long[-3000:])  # overwrite rows [0, 3000), the head's rows and the state
        t4 = time.perf_counter()
        b0.kv_load(0, T - 1, kbuf, cap, 0, lh, hcap)
        t5 = time.perf_counter()
        b0.state_load(sbuf, smeta)
        t6 = time.perf_counter()
        if b0.mtp_cont:
            got, sg = b0.mtp_from(p[-1:], 48, 2, 0, lambda t: None, lambda: False)
        else:
            got = decode(b0, b0.feed(p[-1:]), 32)
            sr = sg = {}
        r = {"T": T, "prefill_ms": pf_ms, "state_save_ms": (t2 - t1) * 1e3, "kv_save_ms": (t3 - t2) * 1e3,
             "kv_load_ms": (t5 - t4) * 1e3, "state_load_ms": (t6 - t5) * 1e3, "kv_gb": kbuf.nbytes / 2**30,
             "head_rows": lh, "identical": got == ref, "first_diff": first_diff(got, ref),
             "rounds": [sr.get("rounds"), sg.get("rounds")]}
        rt.append(r)
        say(f"[1] T {T}: prefill {pf_ms:.0f} ms vs park (state {r['state_save_ms']:.1f} + KV {r['kv_save_ms']:.1f} ms, "
            f"{r['kv_gb']:.3f} GB incl. {lh} head rows) / restore (KV {r['kv_load_ms']:.1f} + state "
            f"{r['state_load_ms']:.1f} ms); after restore identical: {r['identical']}, MTP rounds {r['rounds']}")
        del kbuf, sbuf
    RES["roundtrip"] = rt

    # ---- 2. three conversations rotating on two slots, MTP turns (continuing from each restored conversation)
    use = b0.mtp_cont
    s = B.Session([b0, b1], a.ctx, max_ckpts=4, mtp_policy="always" if use else "never", log=say,
                  cache_ram=8 << 30)
    convs = []
    for c, d in enumerate(docs[:3] if '2' in a.only else []):
        text = d.read_text()[:14000]
        convs.append([{"role": "system", "content": f"You answer questions about this document.\n\n{text}"},
                      {"role": "user", "content": "In one sentence: what is this document about?"}])
    qs = ["Name one number this document reports.", "Which file or tool does it mention first?",
          "Give a one-line summary of its last section."]
    turns = []
    for t in range(3 if convs else 0):
        for c in range(3):
            req = {"messages": convs[c], "reasoning_effort": "none"}
            text, _ = tpl.render(req)
            p = tpl.encode(text)
            out = []
            st = s.generate(p, 96, GREEDY, use, out.extend, lambda: False, tpl.im_end)
            fresh = []
            stf = s.generate(p, 96, GREEDY, use, fresh.extend, lambda: False, tpl.im_end, reuse=False)
            r = {"conv": c, "turn": t, "prompt": len(p), "cached": st["cached_tokens"], "path": st["path"],
                 "slot": st["slot"], "tokens": len(out),
                 "decode_tok_s": (len(out) - 1) / max(st["decode_ms"], 1e-3) * 1e3,
                 "prefill_ms": st["prefill_ms"], "restore_ms": st.get("restore_ms", 0),
                 "fresh_prefill_ms": stf["prefill_ms"], "same_as_fresh": out == fresh,
                 "first_diff": first_diff(out, fresh), "ram_gb": s.cache.ram_bytes() / 2**30}
            turns.append(r)
            say(f"[2] conv {c} turn {t}: {r['path']} slot {r['slot']} prompt {len(p)}, cached {r['cached']}, "
                f"decode {r['decode_tok_s']:.1f} tok/s ({len(out)}), prefill {r['prefill_ms']:.0f} ms "
                f"(restore {r['restore_ms']:.0f}) vs fresh {r['fresh_prefill_ms']:.0f} ms; = fresh: {r['same_as_fresh']}"
                f" (first diff {r['first_diff']}); RAM {r['ram_gb']:.2f} GB; {s.cache.summary()}")
            convs[c] = convs[c] + [{"role": "assistant", "content": tok.decode(out[:-1]).strip()},
                                   {"role": "user", "content": qs[t]}]
    RES["three_convs"] = turns
    RES["cache_stats"] = dict(s.cache.stats)
    del s

    if "3" not in a.only:
        RES["free_gb_end"] = free_gb()
        out = ROOT / "tests/server/results" / f"cache-{time.strftime('%m%d-%H%M')}.json"
        out.write_text(json.dumps(RES, indent=1))
        say(f"results -> {out}")
        return
    # ---- 3. two streams on two slots
    s = B.Session([b0, b1], a.ctx, max_ckpts=4, mtp_policy="never", log=say, cache_ram=8 << 30)
    ps = []
    for d, q in ((docs[6], "Write a detailed essay about this document."), (docs[7], "Explain this document step by "
                                                                                       "step, at length.")):
        text, _ = tpl.render({"messages": [{"role": "system", "content": d.read_text()[:6000]},
                                           {"role": "user", "content": q}], "reasoning_effort": "none"})
        ps.append(tpl.encode(text))
    N = 384
    solo = []
    for i, p in enumerate(ps):
        out = []
        st = s.generate(p, N, GREEDY, False, out.extend, lambda: False, tpl.im_end, reuse=False)
        solo.append((out, st))
        say(f"[3] solo {i}: prompt {len(p)}, {len(out)} tokens, decode {(len(out) - 1) / st['decode_ms'] * 1e3:.1f} tok/s")
    outs = [[], []]
    gens = []
    for i, p in enumerate(ps):
        sl, cand = s.pick(p, reuse=False)
        gens.append([i, s.run(sl, None, p, N, GREEDY, False, outs[i].extend, lambda: False, tpl.im_end,
                              others=lambda: len(gens) > 1)])
    t1 = time.perf_counter()
    stats = {}
    first_both = None
    while gens:
        for g in list(gens):
            try:
                next(g[1])
            except StopIteration as e:
                stats[g[0]] = e.value
                gens.remove(g)
        if first_both is None and all(len(o) > 0 for o in outs):
            first_both = time.perf_counter()
    t2 = time.perf_counter()
    dec_tokens = sum(len(o) - 1 for o in outs)
    # aggregate over the window both decode: from the second stream's first token to the end
    agg = dec_tokens / (t2 - first_both)
    one = sum((len(o) - 1) / (st["decode_ms"] / 1e3) for o, st in solo) / 2
    RES["two_streams"] = {"tokens": [len(o) for o in outs], "same_as_solo": [outs[i] == solo[i][0] for i in range(2)],
                          "first_diff": [first_diff(outs[i], solo[i][0]) for i in range(2)],
                          "slots": [stats[i]["slot"] for i in range(2)], "aggregate_tok_s": agg,
                          "single_tok_s": one, "per_stream_tok_s": [(len(outs[i]) - 1) / (stats[i]["decode_ms"] / 1e3)
                                                                    for i in range(2)],
                          "wall_s": t2 - t1}
    say(f"[3] together: slots {RES['two_streams']['slots']}, aggregate {agg:.1f} tok/s vs one stream {one:.1f}; "
        f"per stream {[round(x, 1) for x in RES['two_streams']['per_stream_tok_s']]}; = solo: "
        f"{RES['two_streams']['same_as_solo']} (first diff {RES['two_streams']['first_diff']})")
    RES["free_gb_end"] = free_gb()
    out = ROOT / "tests/server/results" / f"cache-{time.strftime('%m%d-%H%M')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(RES, indent=1))
    say(f"results -> {out}")


if __name__ == "__main__":
    main()
