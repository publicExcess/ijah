"""Whole-model reference logits (rung 3): fn_exl3 (mlx-vlm 0.7.7 qwen4_exp + EXL3 experts, native rope) on the pack,
4 prompts x 64 positions, saved for tools/compare_logits.py. lab-forward.

    tools/with_lock.sh "ref logits (MLX whole model, ~80 GB, up to 30 min)" HH:MM -- \
        $PY tools/ref_logits.py [--pack DIR] [--out tests/golden/data/WHOLE] [--n 64] [--only chat,code,..]
        [--long 2600 --long-gen 32]   (long: chunked prefill 2048 + greedy decode; QSA sparse past 2048 tokens)

Per prompt: the prompt tokens are fed ONE TOKEN PER CALL (the decode path, as the engine runs them), then greedy
continuation up to n positions; logits of every position are saved (bf16 bits, exact). The same n tokens are then
run again as ONE prefill call (the reference's own prefill path) -> the reference's prefill-vs-decode spread, the
noise floor for judging the engine. Output: <out>/<name>.tokens.npy int32 [n + 1], <name>.decode.npy uint16 [n, V],
<name>.prefill.npy uint16 [n, V], manifest.json. RAM: the whole model (~65 GB in MLX); never next to the engine.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths(mlx=True)
import fn_exl3 as F  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PROMPTS = {
    "chat": "<|im_start|>user\nWhat is 17 multiplied by 23? Explain briefly.<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "prose": "Photosynthesis is the process by which green plants",
    "facts": "The capital of France is Paris. The capital of Germany is",
}


def bits(logits):
    """mx bf16 [.., V] -> numpy uint16 bf16 bits"""
    return np.array(logits.astype(mx.bfloat16).view(mx.uint16))


LONG_SRC = refenv.mlx_vlm_file("models/qwen4_exp/language.py")


def run_long(model, tok, out, man, n_long, n_gen, name="long"):
    """QSA past 2048 tokens: n_long tokens of mlx-vlm's qwen4_exp source prefilled in chunks of 2048 (the model's
    serving path, sparse indexer selection), then n_gen greedy decode steps. Saves the logits of the last prompt
    position + the n_gen decode positions."""
    ids = tok.encode(LONG_SRC.read_text(), add_special_tokens=False).ids
    while len(ids) < n_long:
        ids = ids + ids
    ids = ids[:n_long]
    cache = model.make_cache()
    ts = time.time()
    for s0 in range(0, n_long, 2048):
        lg = model(mx.array([ids[s0:s0 + 2048]]), cache=cache).logits[0, -1]
        mx.eval(lg)
    seq, dec = list(ids), [bits(lg)]
    for i in range(n_gen):
        seq.append(int(np.argmax(np.array(lg.astype(mx.float32)))))
        lg = model(mx.array([[seq[-1]]]), cache=cache).logits[0, -1]
        mx.eval(lg)
        dec.append(bits(lg))
    np.save(out / f"{name}.tokens.npy", np.array(seq, np.int32))
    np.save(out / f"{name}.decode.npy", np.stack(dec))
    text = tok.decode(seq[n_long:], skip_special_tokens=False)
    man["prompts"][name] = {"prompt": str(LONG_SRC), "prompt_len": n_long, "tokens": seq[-n_gen - 1:],
                              "first_pos": n_long - 1, "mode": "chunked prefill 2048 + decode", "completion": text,
                              "s": time.time() - ts}
    print(f"{name}: {n_long} prompt tokens (chunked prefill), {n_gen} decode; completion {text[:100]!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.golden_pack())  # IJAH_GOLDEN_PACK: the pack the goldens / MLX reference were made from
    ap.add_argument("--out", default=str(refenv.GOLDEN / "WHOLE"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--only", default=",".join(PROMPTS))
    ap.add_argument("--long", type=int, default=0, help="also a long prompt of this many tokens (QSA sparse > 2048)")
    ap.add_argument("--long-gen", type=int, default=32)
    ap.add_argument("--long-name", default="long")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pack = F.Pack(a.pack)
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    t0 = time.time()
    model, _ = F.build_model(pack)
    print(f"loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 2**30:.1f} GiB", flush=True)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    man.update({"pack": a.pack, "n": a.n, "rope": "native",
                "reference": "fn_exl3 (mlx-vlm 0.7.7 qwen4_exp, EXL3 experts)"})
    man.setdefault("prompts", {})
    if a.long:
        run_long(model, tok, out, man, a.long, a.long_gen, a.long_name)
        mf.write_text(json.dumps(man, indent=1))
    for name in [x for x in a.only.split(",") if x]:
        ids = tok.encode(PROMPTS[name], add_special_tokens=False).ids
        assert len(ids) < a.n
        cache = model.make_cache()
        seq = list(ids)
        dec = []
        ts = time.time()
        for i in range(a.n):
            lg = model(mx.array([[seq[i]]]), cache=cache).logits[0, -1]
            mx.eval(lg)
            dec.append(bits(lg))
            if i + 1 >= len(seq):
                seq.append(int(np.argmax(np.array(lg.astype(mx.float32)))))
        td = time.time() - ts
        cache = model.make_cache()
        lp = model(mx.array([seq[:a.n]]), cache=cache).logits[0]
        mx.eval(lp)
        pre = bits(lp)
        np.save(out / f"{name}.tokens.npy", np.array(seq[:a.n + 1], np.int32))
        np.save(out / f"{name}.decode.npy", np.stack(dec))
        np.save(out / f"{name}.prefill.npy", pre)
        text = tok.decode(seq[len(ids):], skip_special_tokens=False)
        man["prompts"][name] = {"prompt": PROMPTS[name], "prompt_len": len(ids), "tokens": seq[:a.n + 1],
                                "completion": text, "decode_s": td}
        print(f"{name}: {len(ids)} prompt tokens, {td:.1f} s decode; completion {text[:120]!r}", flush=True)
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
