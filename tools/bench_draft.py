"""MTP drafting at real settings (lab-draft, 18 Oct): per-round profile, pruned draft vocabulary, copy rule. docs/mtp.md 11.

    NEED_GB=85 LOCK_OWNER=draft LOCK_WAIT=30 tools/with_lock.sh "bench_draft (~83 GB)" HH:MM -- \\
        $PY tools/bench_draft.py --configs full,pruned [--depths 2,3,0] [--prompts h2h|edit|all] [--n 1024]
        [--rounds 1] [--prof] [--json OUT] [--trace OUT.trace]

Settings = the same-client head-to-head harness's (not in this repository): temperature 1, top-p 0.95, top-k 20, thinking on
(reasoning_effort medium, the chat template's), seeds 500 + 10 i (+ rep). Prompts: h2h = its six THINK decode prompts
(1,024 tokens each); edit = two whole-file edits of files in this repo (the model shown a file and asked for the whole
updated file; 4,096 tokens). Configs (comma list, run alternately in one process, one model load):
    full    = the full draft vocabulary, no copy rule        pruned = the pruned draft vocabulary (Splish set)
    copy    = pruned + the copy rule (16-token windows)      fullcopy = full + the copy rule
    gate    = pruned + the gated verify (NE_MTP_QGATE)       gatecopy = gate + the copy rule
    a config may add "@key" options: vocab=<NE_MTP_DRAFT_VOCAB value>, copymin=<N>, copyd=<depth>, qgate=<0/1>,
    split=<0/1>, dedupe=<0/1/N> (lab-verify: routed experts read once per verify for all rows routed to them),
    group=<0/N> (lab-moegroup: routed experts grouped by expert from R = N rows on), vc=<0/1/-1> (DFlash2 gate cost
    table: per pair (default) / grouped / by the MoE path), e.g. copy@copyd=3;
    lab-dflash (--dflash DIR): dfl=<0/1> (the DFlash2 block drafter drafts the rounds), dgate=<0/1>, dk=<kmax>,
    e.g. copy@dfl=1 vs copy (run with --max-depth 5 for kmax 5)
tok/s = tokens after the prefill's first / decode wall (tools/bench_mtp.py's definition). Per round the loop writes a
trace line (NE_MTP_TRACE; columns in mtp_loop.mm generate_impl); the summary splits a round into draft GPU, verify A /
B GPU and the host gap, by depth. --prof: ne_fn_mtp_prof (each part of a round on its own) at T 512 and 2048.
"""
import argparse
import ctypes as C
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

os.environ.setdefault("NE_FN_ROUND", "f32")
refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from bench_mtp import Mtp, accept_table, chat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H2H = {  # = the head-to-head harness's THINK prompts, in its order (seed 500 + 10 i)
    "code_lru": "Implement a thread-safe LRU cache in Python with O(1) get and put, a per-entry TTL, and an eviction "
                "callback. Use type hints, explain the design briefly, and include pytest tests covering expiry, "
                "eviction order and concurrent access.",
    "text_trains": "Two trains leave stations 420 km apart at 9:00, heading toward each other at 80 km/h and "
                   "100 km/h. A bird starts at the first train at 9:30, flying at 150 km/h back and forth between "
                   "the trains until they meet. How far does the bird fly in total, and at what time and where do the "
                   "trains meet? Then: if the faster train had left 20 minutes late instead, how would each answer "
                   "change?",
    "text_crisis": "Explain the main causes of the 2008 financial crisis to a smart high-school student, then draw "
                   "three lessons for regulators today, with one concrete policy example for each.",
    "code_parser": "Write a recursive-descent parser in Python for arithmetic expressions with +, -, *, /, unary minus, "
                   "parentheses and integer and float literals; return an AST, then evaluate it. Include error messages "
                   "with the column of the problem and a few unit tests.",
    "text_climate": "Compare heat pumps, gas boilers and electric resistance heating for a 1930s UK semi-detached house: "
                    "running cost, upfront cost, carbon, comfort and the retrofit work each needs. End with a "
                    "recommendation for a household on a tight budget.",
    "math_proof": "Prove that there are infinitely many primes of the form 4k + 3. Then explain why the same argument "
                  "does not work for primes of the form 4k + 1, and sketch how one can prove that case.",
}


X16 = {  # lab-moegroup: ten more short prompts (with H2H: the 16-prompt greedy identity set, --prompts x16)
    "py_dedupe": "Write a Python function that removes duplicates from a list while keeping the first occurrence order, "
                 "and explain its complexity.",
    "sql_rank": "Write a SQL query that returns, for each department, the three highest-paid employees, with ties "
                "handled explicitly. Explain the window function you use.",
    "rust_borrow": "Explain Rust's borrow checker to a C++ programmer with two short code examples: one that compiles "
                   "and one that does not, and why.",
    "haiku_set": "Write five haiku about autumn in a northern city, each from a different person's point of view.",
    "physics_ball": "A ball is thrown upward at 20 m/s from a 15 m tall building. When does it hit the ground and "
                    "with what speed? Show the steps.",
    "history_print": "Summarise how the printing press changed European science and religion between 1450 and 1650, "
                     "in about 300 words.",
    "regex_email": "Write a regular expression that validates simple email addresses, list five strings it accepts "
                   "and five it rejects, and say what it gets wrong.",
    "js_debounce": "Implement debounce and throttle in JavaScript with TypeScript types, and show a small usage "
                   "example for each.",
    "stats_bayes": "A test for a disease is 99% sensitive and 95% specific; the disease affects 1 in 1000 people. "
                   "What is the chance a person who tests positive has it? Explain intuitively.",
    "plan_trip": "Plan a three-day walking trip around Kyoto for someone who likes gardens and food, with a rough "
                 "budget in yen.",
}


def edit_prompts():
    sf = os.environ.get("IJAH_EDIT_FILE")   # the harness's summary.py (a function `med`); not shipped: prompt skipped
    a = Path(sf).read_text() if sf else None
    b = (ROOT / "tools/pack_ab.py").read_text()
    out = {
        "edit_summary": "Here is a Python file:\n\n```python\n" + (a or "") + "```\n\nRename the function `med` to `median_or_none` "
                        "everywhere it is used and add a one-line docstring to every function that has none. Reply with "
                        "the complete updated file in one ```python code block and nothing else.",
        "edit_pack_ab": "Here is a Python file:\n\n```python\n" + b + "```\n\nRename the function `logsm` to "
                        "`log_softmax` everywhere and make every `print(` call pass `flush=True`. Reply with the "
                        "complete updated file in one ```python code block and nothing else.",
    }
    if a is None:
        del out["edit_summary"]
    return out


LONG_Q = {  # lab-dflong: the question after the long context (thinking on; each needs the whole context)
    "prose": "\n\n---\nThe text above is the opening of a novel. Answer in detail: (1) how the narrator's mood and "
             "situation change from the first page to the last page shown, (2) which characters matter most so far and "
             "what we learn about each, (3) quote three short passages that best show the book's tone and explain why.",
    "code": "\n\n---\nThe files above are part of a Metal inference engine for a mixture-of-experts language model. "
            "Explain how the code shown is organised, pick the most important function in it and walk through what it "
            "does step by step, then point out one likely bug or performance problem and write a corrected version.",
    "web": "\n\n---\nThe text above is a collection of web pages. List the main topics it covers, then pick the two "
           "most technical passages and explain each in depth to a newcomer, with a worked example for each.",
}


def long_sources():
    """lab-dflong: the long-context bodies: prose = novel_en (from "Call me Ishmael."), code = this repo's engine sources
    (models/flash_next *.h / *.mm / *.metal, then tools/*.py, each behind a file header), web = mixedweb-v1"""
    # novel_en.txt / mixedweb-v1: the bench corpora oMLX ships (IJAH_BENCH_CORPORA: any directory holding them)
    obc = Path(os.environ.get("IJAH_BENCH_CORPORA", "/Applications/oMLX.app/Contents/Resources/omlx/admin/bench_corpora"))
    novel = (obc / "novel_en.txt").read_text(encoding="utf-8")
    code = []
    for pat in ("models/flash_next/*.h", "models/flash_next/*.mm", "models/flash_next/*.metal", "tools/*.py"):
        for f in sorted(f for f in ROOT.glob(pat) if f.name != "draft_vocab.h"):   # (a 100K-token id table)
            code.append(f"==== {f.relative_to(ROOT)} ====\n" + f.read_text(encoding="utf-8"))
    web = (obc / "mixedweb-v1.txt").read_text(encoding="utf-8")
    return {"prose": novel[novel.find("Call me Ishmael."):], "code": "\n".join(code), "web": web}


def long_prompt(tok, pack, body, question, target):
    """chat-rendered ids of (the body cut at a line boundary) + question, ~target tokens in all"""
    ids = tok.encode(body, add_special_tokens=False).ids
    over = len(tok.encode(chat(pack, question), add_special_tokens=False).ids) + 8
    text = tok.decode(ids[:max(0, target - over)])
    text = text[:text.rfind("\n") + 1] or text
    return tok.encode(chat(pack, text + question), add_special_tokens=False).ids


CONFIGS = {"full": dict(vocab="full", copymin=0), "pruned": dict(vocab="1", copymin=0),
           "copy": dict(vocab="1", copymin=16), "fullcopy": dict(vocab="full", copymin=16),
           "gate": dict(vocab="1", copymin=0, qgate=1), "gatecopy": dict(vocab="1", copymin=16, qgate=1)}


def kept_ids():
    """the pruned draft vocabulary (models/flash_next/draft_vocab.h: ids < 98,304 + kDraftExtra)"""
    import re
    h = (ROOT / "models/flash_next/draft_vocab.h").read_text()
    extra = [int(x) for x in re.findall(r"\d+", h[h.index("kDraftExtra[") + 12:])][1:]
    return np.concatenate([np.arange(98304), np.array(extra)])


def parse_config(c):
    name, _, opts = c.partition("@")
    d = dict(CONFIGS[name])
    d.setdefault("copyd", 0)
    d.setdefault("qgate", 0)
    d.setdefault("split", -1)   # lab-verify: routed gate / up split by multiplicity (NE_FN_MOE_SPLIT)
    d.setdefault("dedupe", -1)  # lab-verify: routed-expert dedupe across verify rows (-1 = the engine's default)
    d.setdefault("vc", 0)       # lab-moegroup: DFlash2 gate cost table (0 per pair = default, 1 grouped, -1 by the MoE path)
    d.setdefault("group", -1)   # lab-moegroup: grouped routed experts from R = N on (0 off, -1 = the engine's default)
    d.setdefault("dfl", 0)      # lab-dflash: the DFlash2 drafter active
    d.setdefault("dgate", 1)
    d.setdefault("dk", 5)  # (explicit: ne_fn_dfl_set kmax 0 = keep would carry the last config's over; clamped to max depth)
    for o in filter(None, opts.split("@")):
        k, _, v = o.partition("=")
        d[k] = v if k == "vocab" else int(v)
    return d


def trace_rounds(lines):
    """round lines -> list of dicts (mtp_loop.mm generate_impl's trace columns)"""
    keys = ("D", "a", "c", "copy", "seed", "draft", "A", "B", "wall", "draft_wait", "B_commit", "host_pre_us",
            "gather_us", "drain_ms", "Dd")
    out = []
    for ln in lines:
        p = ln.split()
        if p and p[0] == "R":
            out.append(dict(zip(keys, [float(x) for x in p[1:]])))
    return out


def round_summary(rs):
    """per depth: rounds, mean draft / A / B GPU ms, wall, host gap = wall - draft - A - B, tokens / round, ms / token"""
    res = {}
    for D in sorted({int(r["D"]) for r in rs}):
        for cp in (0, 1):
            g = [r for r in rs if int(r["D"]) == D and int(r["copy"]) == cp]
            if not g:
                continue
            m = lambda k: statistics.fmean(r[k] for r in g)  # noqa: E731
            toks = statistics.fmean(r["c"] for r in g)
            res[f"d{D}{'c' if cp else ''}"] = dict(
                rounds=len(g), draft=m("draft"), A=m("A"), B=m("B"), wall=m("wall"),
                gap=m("wall") - m("draft") - m("A") - m("B"), host_pre_ms=m("host_pre_us") / 1e3,
                gather_ms=m("gather_us") / 1e3, drain=m("drain_ms"), tok_round=toks,
                ms_per_tok=(m("wall") + m("host_pre_us") / 1e3) / toks)
    return res


def long_summary(rows, names, ctx_of):
    """lab-dflong: per context length, per config: tok/s, tokens / round, verify / draft GPU ms per round; the paired
    ratio names[1] / names[0] over (prompt, seed) with a t 95% CI on the log ratio"""
    import math
    tq = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 11: 2.201}
    print("\n== long context (decode only; per round GPU ms)")
    for L in sorted(set(ctx_of.values())):
        g = [r for r in rows if ctx_of.get(r["prompt"]) == L]
        line = [f"ctx {L:6d} ({statistics.fmean(r['ctx'] for r in g):.0f} tok)"]
        for c in names:
            h = [r for r in g if r["config"] == c]
            if h:
                line.append(f"{c}: {statistics.fmean(r['tps'] for r in h):6.1f} tok/s, "
                            f"{statistics.fmean(r['tokens_per_round'] for r in h):.2f} tok/round, verify "
                            f"{statistics.fmean(r['verify_ms'] for r in h):.2f} draft {statistics.fmean(r['draft_ms'] for r in h):.2f}"
                            f" ms/round, prefill {statistics.fmean(r['ctx'] / r['prefill_ms'] * 1e3 for r in h):.0f} tok/s")
        if len(names) >= 2:
            a_ = {(r["prompt"], r["rep"]): r["tps"] for r in g if r["config"] == names[0]}
            b_ = {(r["prompt"], r["rep"]): r["tps"] for r in g if r["config"] == names[1]}
            lr = [math.log(b_[k] / a_[k]) for k in a_ if k in b_]
            if lr:
                mu = statistics.fmean(lr)
                hw = tq.get(len(lr) - 1, 2.0) * statistics.stdev(lr) / math.sqrt(len(lr)) if len(lr) > 1 else float("nan")
                line.append(f"{names[1]} / {names[0]} {math.exp(mu):.3f} [{math.exp(mu - hw):.3f}, {math.exp(mu + hw):.3f}],"
                            f" wins {sum(x > 0 for x in lr)} / {len(lr)}")
        print("  " + " | ".join(line), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--configs", default="full,pruned")
    ap.add_argument("--depths", default="2")
    ap.add_argument("--prompts", default="h2h", choices=["h2h", "edit", "all", "x16"])
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=0, help="new tokens (default 1024 h2h, 3072 edit)")
    ap.add_argument("--rounds", type=int, default=1, help="repetitions of the whole config cycle (seed + rep)")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--dflash", default="", help="lab-dflash: DFlash2 checkpoint dir (enables the drafter; configs @dfl=1)")
    ap.add_argument("--dflash-block", type=int, default=8)
    ap.add_argument("--dflash-bits", type=int, default=8)
    ap.add_argument("--dflash-max-ctx", type=int, default=1 << 30, help="lab-dflfix: the drafter's context cap (default none)")
    ap.add_argument("--long", default="", help="lab-dflong: decode after long contexts, e.g. 8192,32768,100000 (prompts "
                    "--long-prompts; --rounds seeds; configs ABBA per (prompt, seed): A B, then B A)")
    ap.add_argument("--long-prompts", default="prose,code,web")
    a = ap.parse_args()
    trace = a.trace or (a.json + ".trace" if a.json else "")
    if trace:
        os.environ["NE_MTP_TRACE"] = trace
        Path(trace).write_text("")
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    prompts = {}
    if a.prompts in ("h2h", "all"):
        prompts.update({k: (v, a.n or 1024, 500 + 10 * i) for i, (k, v) in enumerate(H2H.items())})
    if a.prompts == "x16":
        prompts.update({k: (v, a.n or 1024, 500 + 10 * i) for i, (k, v) in enumerate({**H2H, **X16}.items())})
    if a.prompts in ("edit", "all"):
        prompts.update({k: (v, a.n or 4096, 700 + 10 * i) for i, (k, v) in enumerate(edit_prompts().items())})
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
    enc = {k: tok.encode(chat(a.pack, q), add_special_tokens=False).ids for k, (q, _, _) in prompts.items()}
    ctx_of = {}
    if a.long:  # lab-dflong: replaces the short prompts
        prompts, enc = {}, {}
        src = long_sources()
        for L in [int(x) for x in a.long.split(",")]:
            for i, k in enumerate(a.long_prompts.split(",")):
                name = f"{k}@{L // 1024}k"
                enc[name] = long_prompt(tok, a.pack, src[k], LONG_Q[k], L)
                prompts[name] = ("", a.n or 1024, 500 + 10 * i)
                ctx_of[name] = L
    tmax = (max(len(enc[k]) + prompts[k][1] for k in prompts) + 256 + 63) // 64 * 64
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, tmax)
    mtp = Mtp(eng, m, a.max_depth, 4)
    lib = eng.lib
    lib.ne_fn_mtp_set_copy.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.ne_fn_mtp_set_draft_vocab.argtypes = [C.c_void_p, C.c_char_p]
    lib.ne_fn_mtp_set_qgate.argtypes = [C.c_void_p, C.c_int]
    lib.ne_fn_mtp_set_moe_dedupe.argtypes = [C.c_void_p, C.c_int]
    lib.ne_fn_mtp_set_moe_split.argtypes = [C.c_void_p, C.c_int]
    lib.ne_fn_mtp_set_moe_group.argtypes = [C.c_void_p, C.c_int]
    lib.ne_fn_dfl_set_vcost.argtypes = [C.c_void_p, C.c_int]
    lib.ne_fn_mtp_prof.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int]
    if a.dflash:
        lib.ne_fn_dfl_enable.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int]
        lib.ne_fn_dfl_set.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
        lib.ne_fn_dfl_last.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        t1 = time.time()
        eng._ok(lib.ne_fn_dfl_enable(C.c_void_p(mtp.ptr), a.dflash.encode(), a.dflash_block, a.max_depth,
                                     a.dflash_bits, 1), "dfl_enable")
        print(f"dflash enabled: block {a.dflash_block}, bits {a.dflash_bits}, {time.time() - t1:.1f} s", flush=True)
        if lib_has := hasattr(lib, "ne_fn_dfl_set_max_ctx"):  # lab-dflfix: bench every context length (no cap)
            lib.ne_fn_dfl_set_max_ctx.argtypes = [C.c_void_p, C.c_int]
            eng._ok(lib.ne_fn_dfl_set_max_ctx(C.c_void_p(mtp.ptr), a.dflash_max_ctx), "dfl_set_max_ctx")
    if a.long:
        lib.ne_fn_mtp_kv_bytes.restype = C.c_int64
        lib.ne_fn_mtp_kv_bytes.argtypes = [C.c_void_p, C.c_int]
        res_kv = {L: lib.ne_fn_mtp_kv_bytes(C.c_void_p(mtp.ptr), L) for L in set(ctx_of.values())}
        print("kv bytes (target + head, ne_fn_mtp_kv_bytes) by context: "
              + ", ".join(f"{L}: {v / 2**20:.1f} MiB" for L, v in sorted(res_kv.items())), flush=True)
    print(f"load {time.time() - t0:.1f} s, tmax {tmax}; prompts {[(k, len(enc[k])) for k in prompts]}", flush=True)
    res = {"rows": [], "prof": {}}
    cfgs = [(c, parse_config(c)) for c in a.configs.split(",")]

    def apply(cfg):
        eng._ok(lib.ne_fn_mtp_set_draft_vocab(C.c_void_p(mtp.ptr), cfg["vocab"].encode()), "set_draft_vocab")
        eng._ok(lib.ne_fn_mtp_set_copy(C.c_void_p(mtp.ptr), cfg["copymin"], cfg["copyd"]), "set_copy")
        eng._ok(lib.ne_fn_mtp_set_qgate(C.c_void_p(mtp.ptr), cfg["qgate"]), "set_qgate")
        if cfg["dedupe"] >= 0:
            eng._ok(lib.ne_fn_mtp_set_moe_dedupe(C.c_void_p(mtp.ptr), cfg["dedupe"]), "set_moe_dedupe")
        if cfg["split"] >= 0:
            eng._ok(lib.ne_fn_mtp_set_moe_split(C.c_void_p(mtp.ptr), cfg["split"]), "set_moe_split")
        if cfg["group"] >= 0:
            eng._ok(lib.ne_fn_mtp_set_moe_group(C.c_void_p(mtp.ptr), cfg["group"]), "set_moe_group")
        if a.dflash:
            eng._ok(lib.ne_fn_dfl_set(C.c_void_p(mtp.ptr), cfg["dgate"], cfg["dk"], cfg["dfl"]), "dfl_set")
            eng._ok(lib.ne_fn_dfl_set_vcost(C.c_void_p(mtp.ptr), cfg["vc"]), "dfl_set_vcost")
        elif cfg["dfl"]:
            raise SystemExit("config @dfl=1 needs --dflash DIR")

    if a.prof:
        names = ["head_step", "head_noreadout", "head_kv_only", "coarse_gemm", "topk_local", "shortlist", "draft_sample",
                 "full_head_argmax", "head_step_2row"] + [f"draft_n1_d{d}" for d in range(1, 5)] + \
                [f"draft_n2_d{d}" for d in range(1, 5)] + [f"verify_s_R{r}" for r in range(2, 6)] + \
                [f"verify_g_R{r}" for r in range(2, 6)]
        for c, cfg in cfgs:
            apply(cfg)
            for T in (512, 2048):
                if T + 16 > tmax:
                    continue
                out = np.zeros(32, np.float64)
                n = lib.ne_fn_mtp_prof(C.c_void_p(mtp.ptr), T, 20, out.ctypes.data, 32)
                if n < 0:
                    raise E.EngineError(eng.last_error())
                d = dict(zip(names, out[:n].round(4).tolist()))
                res["prof"][f"{c}@T{T}"] = d
                print(f"prof {c} T {T}: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()), flush=True)

    sp = (0.0, 1.0, 1, 0) if a.greedy else None
    base = {}
    if a.greedy:  # the no-MTP greedy text (1-row step path) for the identity check
        m.set_sampler(*sp)
        for name, (q, n, seed) in prompts.items():
            m.reset()
            base[name], _, _ = m.generate(enc[name], n, -1)

    def mismatch(name, out):
        b = base[name]
        mm = next((i for i in range(min(len(b), len(out))) if b[i] != out[i]), None)
        if mm is None:
            return None, None
        m.reset()
        m.prefill(enc[name])
        lg = None
        for t in b[:mm]:
            _, _, _, lg = m.step(t, True, logits=True)
        return mm, (float(lg[b[mm]] - lg[out[mm]]) if lg is not None else None)
    # configs alternate per prompt (lab-dflash: the GPU's state drifts over a long job; config-major order biased A/Bs)
    for rep in range(a.rounds):
        for depth in [int(x) for x in a.depths.split(",")]:
            for name, (q, n, seed) in prompts.items():
                order = cfgs if not (a.long and rep % 2) else cfgs[::-1]   # lab-dflong: ABBA over the seeds
                for c, cfg in order:
                    apply(cfg)
                    m.set_sampler(*(sp or (1.0, 0.95, 20, seed + rep)))
                    pos = Path(trace).stat().st_size if trace else 0
                    out, s = mtp.generate(enc[name], n, depth, seed=seed + rep)
                    lines = []
                    if trace:
                        with open(trace) as f:
                            f.seek(pos)
                            lines = f.read().splitlines()
                    rs = trace_rounds(lines)
                    tps = s["tokens"] / (s["decode_ms"] / 1e3)
                    cond, tpr = accept_table(s["hist"])
                    dr_, dt_ = C.c_int(0), C.c_int(0)
                    if a.dflash:
                        lib.ne_fn_dfl_last(C.c_void_p(mtp.ptr), C.byref(dr_), C.byref(dt_))
                    cr = sum(1 for r in rs if r["copy"])
                    ct = sum(r["c"] for r in rs if r["copy"])
                    row = dict(config=c, rep=rep, depth=depth, prompt=name, tps=tps, tokens_per_round=tpr, accept=cond,
                               rounds=s["rounds"], copy_rounds=cr, copy_tokens=ct, draft_ms=s["draft_gpu_ms"],
                               verify_ms=s["verify_gpu_ms"], host_us=s["host_us"], gather_us=s["gather_us"],
                               n=len(out), out=out, by_depth=round_summary(rs), dfl_rounds=dr_.value,
                               dfl_tokens=dt_.value, ctx=len(enc[name]), prefill_ms=s["prefill_ms"],
                               decode_ms=s["decode_ms"])
                    if a.greedy:
                        row["first_mismatch"], row["mismatch_gap"] = mismatch(name, out)
                    res["rows"].append(row)
                    if a.json and a.long:   # lab-dflong: long jobs keep every finished row
                        Path(a.json).write_text(json.dumps(res))
                    print(f"  {c:10s} d{depth} {name:13s}: {tps:6.1f} tok/s, {tpr:.2f} tok/round, accept "
                          f"{cond[:depth or a.max_depth]}, draft {s['draft_gpu_ms']:.2f} + verify "
                          f"{s['verify_gpu_ms']:.2f} ms/round, copy {cr} rounds / {ct:.0f} tokens"
                          + (f", dflash {dr_.value} rounds / {dt_.value} tokens" if dr_.value else "")
                          + (f"; vs no-MTP: first mismatch {row['first_mismatch']} (1-row logit gap "
                             f"{row['mismatch_gap']})" if a.greedy else ""), flush=True)
    if a.long:
        long_summary(res["rows"], [c for c, _ in cfgs], ctx_of)
        if a.json:
            res["kv_bytes"] = res_kv
    print("\n== medians over prompts (tok/s)")
    for c, _ in cfgs:
        for depth in [int(x) for x in a.depths.split(",")]:
            v = [r["tps"] for r in res["rows"] if r["config"] == c and r["depth"] == depth]
            if v:
                print(f"  {c:10s} d{depth}: median {statistics.median(v):6.1f}  mean {statistics.fmean(v):6.1f}  "
                      f"({', '.join(f'{x:.1f}' for x in v)})")
    # vocabulary coverage of the generated tokens (every config's outputs)
    kept = kept_ids()
    allt = [t for r in res["rows"] for t in r["out"][1:]]
    if allt:
        x = np.array(allt)
        res["coverage"] = dict(tokens=len(allt), lt_98304=float((x < 98304).mean()),
                               in_kept=float(np.isin(x, kept).mean()))
        print(f"coverage over {len(allt)} generated tokens: ids < 98,304 {res['coverage']['lt_98304']:.4%}, "
              f"in the pruned set {res['coverage']['in_kept']:.4%}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    pack.close()


if __name__ == "__main__":
    main()
