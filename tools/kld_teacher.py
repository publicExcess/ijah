"""Whole-model KLD of the native engine vs the bf16 teacher (lab-quality). Same sets, positions and scoring as
the lab tree's exl3/pack/whole_kld.py + kld_core.py (fn_exl3 in MLX), so the numbers sit next to
fn_exl3's on the same pack (K5: A 0.0357 / B 0.0105 / C 0.0163, RUN/kld/kld-L48-ours-K5.json).

  score : tools/with_lock.sh "kld_teacher (81 GB)" HH:MM -- $PY tools/kld_teacher.py --name NAME
              [--round f32|ref] [--kv bf16|kv8] [--sets A,B,C] [--max-windows N] [--max-pos P] [--pack DIR]
          -> tests/quality/results/kld-NAME.json (+ kld-NAME-pos.npz: per-position KLD)
  report: $PY tools/kld_teacher.py --report a.json b.json ...   (paired vs the first, bootstrap CI over windows;
          RUN/kld/*.json from whole_kld.py are accepted too)

Method (= kld_core.score_model): per window a fresh model (reset), every token fed by one decode step with the head
(teacher forcing; the engine's 4-row prefill has no per-position logits), logits = bf16 values in f32. Per position
KL(teacher || engine) over the full vocabulary, the teacher's log-probs rebuilt from its stored final hidden state hm
with the bf16 lm_head (MLX nn.Linear, bit-identical to the teacher run; checked against its stored top-64), f32
log-softmax on both sides in MLX, in 512-row blocks; the window's last position is left out (no target). top-1 =
argmax agreement; NLL of the real next token (engine and teacher). Set C also by position bin. Control: the first 64
positions of the first window run twice (rerun must be 0). Only the lm_head (1.3 GB) is in MLX: the MLX reference model
is never loaded next to the engine.
"""
import argparse
import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths()
RUN = Path(os.environ.get("NE_TEACHER_RUN", str(Path.home() / ".cache/ijah/teacher-run")))  # bf16 teacher refs + base trunks
KLDCORE = refenv.EXL3 / "pack"          # kld_core.py: windows, ref_path, load_head, bootstrap_ci (read-only import)
OUT = refenv.ROOT / "tests/quality/results"
SUB = 512
FN_EXL3 = {"K5": "kld-L48-ours-K5.json", "K5hc": "kld-L48-ours-K5hc.json", "K6hc": "kld-L48-ours-K6hc.json",
           "sushi": "kld-L48-sushi.json", "port": "kld-L48-port.json"}


def load_json(p):
    return json.loads(Path(p).read_text())


def report(files):
    sys.path.insert(0, str(KLDCORE))
    import numpy as np
    from kld_core import bootstrap_ci
    rs = [load_json(f) for f in files]
    for r, f in zip(rs, files):
        r["_file"] = Path(f)

    def per_window(r, S, P):
        """window name -> mean KLD, over the first P positions when r scored more (from its -pos.npz)"""
        x = r["sets"][S]
        if P and x.get("max_pos") != P:
            npz = r["_file"].with_name(r["_file"].stem + "-pos.npz")
            if not npz.exists():
                return {}
            z = np.load(npz)
            return {nm: float(z[f"{S}{i}"][:P].mean()) for nm, i in zip(x["windows"], x["window_idx"])}
        return dict(zip(x["windows"], x["per_window"]["kld"]))
    for S in ("A", "B", "C"):
        rows = [r for r in rs if S in r["sets"]]
        if not rows:
            continue
        b = rows[0]["sets"][S]
        print(f"set {S}: paired per window vs {rows[0]['name']} ({len(b['per_window']['kld'])} windows), 95% bootstrap CI")
        print(f"  {'name':22s} {'win':>3s} {'KLD':>8s} {'diff':>9s} {'95% CI':>20s} {'top-1':>7s} {'NLL':>7s} {'NLL t':>7s}")
        for r in rows:
            x = r["sets"][S]
            P = x.get("max_pos") or b.get("max_pos")
            mine, base = per_window(r, S, P), per_window(rows[0], S, P)   # paired by window name, same positions
            if not mine or not base:
                print(f"  {r['name']:22s} (positions differ from {rows[0]['name']} and no -pos.npz: skipped)")
                continue
            k = np.array(list(mine.values()))
            d = np.array([v - base[nm] for nm, v in mine.items() if nm in base])
            lo, hi = bootstrap_ci(d) if r is not rows[0] else (0.0, 0.0)
            note = f" (first {P} pos)" if P else ""
            print(f"  {r['name']:22s} {len(k):3d} {k.mean():8.5f} {d.mean():+9.5f} [{lo:+.5f}, {hi:+.5f}] "
                  f"{np.mean(x['per_window']['top1']):7.4f} {np.mean(x['per_window']['nll']):7.4f} "
                  f"{np.mean(x['per_window']['nll_teacher']):7.4f}{note}")
            if "kld_bins_mean" in x and not P:
                print(f"  {'':22s} bins {x['bins'][:-1]}+: " + " ".join(f"{v:.5f}" for v in x["kld_bins_mean"]))


class SelfTest:
    """stands in for FnModel: the teacher's own bf16 logits (lm_head(hm)) at the current position, on the CPU"""

    def __init__(self, head, run, W, blk):
        import mlx.core as mx
        import kld_core as K
        self.mx, self.K, self.head, self.run, self.blk = mx, K, head, run, blk
        self.refs, self.window, self.p = {}, None, 0

    def reset(self):
        self.p = 0

    def feed(self, tok, row):
        S, off = self.window
        if S not in self.refs:
            self.refs = {S: self.mx.load(str(self.K.ref_path(self.run, S, 48)))}
        hm = self.refs[S]["hm"][off + self.p:off + self.p + 1]
        self.blk[row] = __import__("numpy").array(self.head(hm).astype(self.mx.float32))[0]
        self.p += 1
        return 0.0

    def free(self):
        pass


class DumpedLogits:
    """lab splashcmp: stands in for FnModel with another engine's logits, dumped per window as raw f32 [T][V]
    (tools/gguf_logits.cpp: a GGUF through llama.cpp, teacher-forced over the same windows): DIR/{set}-{window}.f32"""

    def __init__(self, d, W, blk):
        import numpy as np
        self.np, self.d, self.W, self.blk = np, Path(d), W, blk
        self.window, self.p, self.key, self.mm = None, 0, None, None

    def reset(self):
        self.p = 0

    def feed(self, tok, row):
        S, off = self.window
        key = (S, self.W[S]["offsets"].index(off))
        if key != self.key:
            f = self.d / f"{key[0]}-{key[1]}.f32"
            T = len(self.W[S]["windows"][key[1]])
            self.mm = self.np.memmap(f, self.np.float32, "r")
            assert self.mm.size % T == 0, (f, self.mm.size, T)
            self.mm = self.mm.reshape(T, -1)
            self.key = key
        V = self.blk.shape[1]
        assert self.mm.shape[1] >= V, (self.mm.shape, V)
        self.blk[row] = self.mm[self.p, :V]
        self.p += 1
        return 0.0

    def free(self):
        self.mm = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--round", choices=["f32", "ref"], default="f32")
    ap.add_argument("--kv", choices=["bf16", "kv8", "f32"], default="bf16")
    ap.add_argument("--attn", choices=["qsa", "dense"], default="qsa")
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--windows", default="", help="per set window indices, e.g. C:0,2 (overrides --max-windows)")
    ap.add_argument("--max-pos", type=int, default=0, help="score only the first P positions of each window")
    ap.add_argument("--prefill", type=int, default=0, help="feed the first N tokens of each window through the chunked "
                    "prefill (ne_fn_model_prefill), then teacher-force and score from position N (prefill-path check)")
    ap.add_argument("--score-from", type=int, default=0, help="feed the first N tokens by decode steps unscored, score "
                    "from position N (the decode-only control for --prefill N)")
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", nargs="+")
    ap.add_argument("--selftest", action="store_true", help="CPU only, no engine: feed the teacher's own logits "
                    "through the scoring (KLD must be 0, NLL = the teacher's NLL in whole_kld.py's json)")
    ap.add_argument("--logits-dir", default="", help="lab splashcmp: score dumped logits (DIR/{set}-{window}.f32, "
                    "tools/gguf_logits.cpp) instead of the engine; --dump-tokens DIR first writes the windows' tokens")
    ap.add_argument("--dump-tokens", default="", help="write DIR/{set}-{window}.tok (int32) for --sets and exit")
    a = ap.parse_args()
    if a.dump_tokens:
        import numpy as np
        sys.path.insert(0, str(KLDCORE))
        import kld_core as K
        W = K.windows(Path(a.run))
        Path(a.dump_tokens).mkdir(parents=True, exist_ok=True)
        for S in a.sets.split(","):
            for wi, w in enumerate(W[S]["windows"]):
                np.asarray(w, np.int32).tofile(Path(a.dump_tokens) / f"{S}-{wi}.tok")
            print(S, len(W[S]["windows"]), "windows,", sum(map(len, W[S]["windows"])), "tokens")
        return
    if a.report:
        return report(a.report)
    # env before the library loads (layer graph + lab-attn's QSA read them at record time)
    os.environ["NE_FN_ROUND"] = "f32" if a.round == "f32" else "ref"
    os.environ["NE_KV"] = a.kv
    os.environ["NE_FN_ATTN"] = a.attn
    sys.path.insert(0, str(KLDCORE))
    import numpy as np
    import mlx.core as mx
    import kld_core as K
    import engine as E
    mx.set_cache_limit(2 << 30)
    if a.selftest:
        mx.set_default_device(mx.cpu)

    run = Path(a.run)
    W = K.windows(run)
    sets = a.sets.split(",")
    sel = {}
    for part in filter(None, a.windows.split(";")):
        s, ids = part.split(":")
        sel[s] = [int(i) for i in ids.split(",")]
    need = max(len(w) for S in sets for w in W[S]["windows"])
    if a.max_pos:
        need = min(need, a.max_pos)
    tmax = (need + 1 + 63) // 64 * 64
    t0 = time.time()
    head = K.load_head(run)
    V = E.FnModel.VOCAB
    blk = np.empty((SUB, V), np.float32)
    if a.selftest:
        m = SelfTest(head, run, W, blk)
    elif a.logits_dir:
        m = DumpedLogits(a.logits_dir, W, blk)
    else:
        eng = E.Engine()
        eng.set_ref_rounding(a.round == "ref")
        pack = eng.pack_open(a.pack)
        m = eng.fn_model(pack, tmax)
        m.set_sampler(0.0, 1.0, 1, 0)
    print(f"engine {E.LIB}, pack {a.pack}, tmax {tmax}, round {a.round}, kv {a.kv}, attn {a.attn}; "
          f"load {time.time() - t0:.1f} s", flush=True)
    nx, am, lp_, ms = C.c_int(-1), C.c_int(-1), C.c_float(0), C.c_double(0)

    def feed(tok, row):
        if a.selftest or a.logits_dir:
            return m.feed(tok, row)
        if eng.lib.ne_fn_model_step(m.ptr, int(tok), 1, blk[row].ctypes.data, C.byref(nx), C.byref(lp_), C.byref(am),
                                    C.byref(ms)) != 0:
            raise E.EngineError(eng.last_error())
        return ms.value

    name = a.name or f"engine-{a.round}-{a.kv}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"name": name, "pack": a.pack, "engine_lib": str(E.LIB), "round": a.round, "kv": a.kv, "attn": a.attn,
           "tmax": tmax, "method": "kld_core.score_model positions/scoring; engine teacher-forced decode steps",
           "loader": "ijah ne_fn_model", "sets": {}}
    pos = {}

    # control: the first 64 positions of the first window twice, logits bit for bit
    w0 = W[sets[0]]["windows"][0][:64]
    m.window = (sets[0], 0)
    ctl = []
    for _ in range(2):
        m.reset()
        rows = np.empty((len(w0), V), np.float32)
        for i, t in enumerate(w0):
            feed(t, 0)
            rows[i] = blk[0]
        ctl.append(rows)
    res["rerun_max_abs"] = float(np.abs(ctl[0] - ctl[1]).max())
    print(f"rerun control (64 pos): max |dlogit| {res['rerun_max_abs']:.1e}", flush=True)
    del ctl, rows

    for S in sets:
        Wd = W[S]
        offs, bins = Wd["offsets"], Wd.get("bins")
        idx = sel.get(S, list(range(len(Wd["windows"]))))
        if a.max_windows and S not in sel:
            idx = idx[:a.max_windows]
        ref = mx.load(str(K.ref_path(run, S, 48)))
        st = {"kld": [], "top1": [], "nll": [], "nll_teacher": [], "kld_max": [], "n": [], "gpu_ms": []}
        if bins:
            st["kld_bins"] = []
        rebuild = 0.0
        t1 = time.time()
        for wi in idx:
            w = Wd["windows"][wi]
            T = len(w)
            P = min(T, a.max_pos) if a.max_pos else T
            m.window = (S, offs[wi])
            m.reset()
            kl_all, t1_all, nl_all, nt_all, gms = [], [], [], [], []
            tw = time.time()
            start = a.prefill or a.score_from
            assert (start % SUB == 0 or a.prefill) and start < P, (start, SUB, P)  # pfkern: --prefill 256 for set A
            if a.prefill:
                m.prefill(np.asarray(w[:start], np.int32))
            else:
                for t in w[:start]:
                    feed(t, 0)
            p0s = [start] + list(range((start // SUB + 1) * SUB, P, SUB)) if start % SUB else list(range(start, P, SUB))
            for p0 in p0s:
                rows = min(SUB - p0 % SUB, P - p0)  # = SUB from a block boundary (unchanged); a partial first block
                for r in range(rows):
                    gms.append(feed(w[p0 + r], r))
                n = rows if p0 + rows < T else rows - 1      # the window's last position has no target
                if n <= 0:
                    continue
                lf = mx.array(blk[:n])
                lp = lf - mx.logsumexp(lf, -1, keepdims=True)
                o = offs[wi] + p0
                tp = K.logprobs(head, ref["hm"][o:o + rows])[:n]   # same row count as kld_core: bit-identical teacher
                nxt = mx.array(w[p0 + 1:p0 + 1 + n])[:, None]
                kl = (mx.exp(tp) * (tp - lp)).sum(-1)
                t1v = (mx.argmax(lp, -1) == mx.argmax(tp, -1)).astype(mx.float32)
                nl = -mx.take_along_axis(lp, nxt, axis=-1)[:, 0]
                nt = -mx.take_along_axis(tp, nxt, axis=-1)[:, 0]
                sv = mx.take_along_axis(tp, ref["top_index"][o:o + n], axis=-1)
                rb = mx.max(mx.abs(sv - ref["top_logprob"][o:o + n]))
                mx.eval(kl, t1v, nl, nt, rb)
                rebuild = max(rebuild, float(rb.item()))
                kl_all.append(np.array(kl)); t1_all.append(np.array(t1v))
                nl_all.append(np.array(nl)); nt_all.append(np.array(nt))
                del lf, lp, tp
                mx.clear_cache()
            kl = np.concatenate(kl_all)
            pos[f"{S}{wi}"] = kl.astype(np.float32)
            st["kld"].append(float(kl.mean())); st["kld_max"].append(float(kl.max()))
            st["top1"].append(float(np.concatenate(t1_all).mean()))
            st["nll"].append(float(np.concatenate(nl_all).mean()))
            st["nll_teacher"].append(float(np.concatenate(nt_all).mean()))
            st["n"].append(int(kl.size)); st["gpu_ms"].append(float(np.median(gms)))
            if bins:
                st["kld_bins"].append([float(kl[lo:hi].mean()) if lo < len(kl) else None
                                       for lo, hi in zip(bins[:-1], bins[1:])])
            print(f"{name} {S}[{wi}] {Wd['names'][wi]}: {kl.size} pos, KLD {kl.mean():.5f}, top-1 "
                  f"{st['top1'][-1]:.4f}, NLL {st['nll'][-1]:.4f} (teacher {st['nll_teacher'][-1]:.4f}), "
                  f"{time.time() - tw:.0f} s, GPU {np.median(gms):.2f} ms/step", flush=True)
            r = {"set": S, "layers": 48, "window_idx": idx[:len(st['kld'])], "max_pos": a.max_pos or None,
                 "windows": [Wd["names"][i] for i in idx[:len(st["kld"])]], "text": Wd.get("text"),
                 "teacher_rebuild_max_abs": rebuild, "per_window": st}
            for k in ("kld", "top1", "nll", "nll_teacher"):
                r[k + "_mean"] = float(np.mean(st[k]))
            if bins:
                b = np.array([[np.nan if x is None else x for x in rr] for rr in st["kld_bins"]], float)
                r["bins"] = bins
                r["kld_bins_mean"] = np.nanmean(b, axis=0).tolist()
            r["score_s"] = time.time() - t1
            res["sets"][S] = r
            (out / f"kld-{name}.json").write_text(json.dumps(res, indent=1))       # after every window: partial runs count
            np.savez_compressed(out / f"kld-{name}-pos.npz", **pos)
        print(f"{name} {S}: KLD {res['sets'][S]['kld_mean']:.5f}, top-1 {res['sets'][S]['top1_mean']:.4f}, teacher "
              f"rebuild {rebuild:.1e} ({time.time() - t1:.0f} s)", flush=True)
        del ref
    print(f"-> {out / f'kld-{name}.json'}", flush=True)
    m.free()


if __name__ == "__main__":
    main()
