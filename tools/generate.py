"""Generate with the native engine (whole Flash-Next model, 1 row per step). lab-forward.

    tools/with_lock.sh "generate" HH:MM -- $PY tools/generate.py "What is 17 x 23?" [--max 64] [--think]
        [--temp 1 --top-p 0.95 --top-k 20 --seed 0 | --greedy] [--raw] [--pack DIR] [--tmax 8192] [--mtp D]

--mtp D: the MTP token loop (docs/mtp.md section 10), draft depth D (1-4; 0 = adaptive), exact for sampled modes.

The prompt goes through the pack's chat template (jinja2; thinking off unless --think) unless --raw. Prints the
completion and the speed (prefill = 1-row steps, decode = full steps incl. the host n-gram gather and the sync).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

# Generation default: plain f32 (no reference bf16 rounding points): closest to the MLX reference at whole-model
# level (docs/forward.md section 2). --ref-round keeps the emulation. Set before the library loads (the layer graph
# and lab-attn's QSA kernels read NE_FN_ROUND).
os.environ.setdefault("NE_FN_ROUND", "ref" if "--ref-round" in sys.argv else "f32")  # (library default: f32)

refenv.add_paths()
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def chat_prompt(pack_dir, text, think=False):
    import jinja2

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
    t = env.from_string((Path(pack_dir) / "chat_template.jinja").read_text())
    return t.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=think)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--tmax", type=int, default=8192)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-round", action="store_true", help="keep the reference bf16 rounding points")
    ap.add_argument("--mtp", type=int, default=-1, help="MTP draft depth (0 = adaptive; default off)")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    text = a.prompt if a.raw else chat_prompt(a.pack, a.prompt, a.think)
    ids = tok.encode(text, add_special_tokens=False).ids
    stop = tok.token_to_id("<|im_end|>")
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, a.tmax)
    t_load = time.time() - t0
    if a.greedy:
        m.set_sampler(0.0, 1.0, 1, 0)
    else:
        m.set_sampler(a.temp, a.top_p, a.top_k, a.seed)
    if a.mtp >= 0:
        import bench_mtp
        mtp = bench_mtp.Mtp(eng, m, 4, 4)
        out, s = mtp.generate(ids, a.max, a.mtp, seed=a.seed, stop=stop if stop is not None else -1)
        print(text + tok.decode(out, skip_special_tokens=False))
        cond, tpr = bench_mtp.accept_table(s["hist"])
        print(f"\n--- {len(ids)} prompt tokens, {len(out)} generated; load+record {t_load:.1f} s; prefill "
              f"{s['prefill_ms'] / 1e3:.2f} s (head {s['head_prefill_ms'] / 1e3:.2f} s); decode "
              f"{s['tokens'] / (s['decode_ms'] / 1e3) if s['tokens'] else 0:.1f} tok/s with MTP (depth "
              f"{a.mtp or 'adaptive'}: {tpr:.2f} tokens / round, acceptance {cond}, draft {s['draft_gpu_ms']:.2f} + "
              f"verify {s['verify_gpu_ms']:.2f} ms GPU / round)", file=sys.stderr)
        pack.close()
        return
    out, lps, st = m.generate(ids, a.max, stop if stop is not None else -1)
    comp = tok.decode(out, skip_special_tokens=False)
    print(text + comp)
    n = st["decode_steps"]
    print(f"\n--- {len(ids)} prompt tokens, {len(out)} generated; load+record {t_load:.1f} s; prefill "
          f"{st['prefill_ms'] / 1e3:.2f} s ({len(ids) / (st['prefill_ms'] / 1e3):.1f} tok/s); decode "
          f"{(n / (st['decode_ms'] / 1e3)) if n else 0:.1f} tok/s (GPU {st['gpu_ms_mean']:.2f} ms/step, gather "
          f"{st['gather_us_mean']:.0f} us); {int(st['dispatches'])} dispatches/step", file=sys.stderr)
    m.free()
    pack.close()


if __name__ == "__main__":
    main()
