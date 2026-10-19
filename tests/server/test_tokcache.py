"""lab-ttft: the incremental prompt encode (chat.Template.encode, an LRU of earlier texts reused up to an <|im_start|>)
must equal the whole-text encode on every request of an agent-like sequence: appended turns, the last assistant
message re-rendered without its reasoning, an edited earlier message, tool results, unicode (combining marks next to
the boundaries), a literal special-token string in a message, an unrelated conversation in between. CPU only (the
pack's tokenizer + chat template).

    $PY tests/server/test_tokcache.py
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import chat  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
    print(("ok   " if ok else "FAIL ") + what)


def main():
    pack = ijah_paths.pack_or_skip("tests/server/test_tokcache.py", need=("tokenizer.json", "chat_template.jinja"))
    tok = Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))
    tpl = chat.Template(pack, tok, "xhigh")
    rnd = random.Random(7)
    src = "\n\n".join(p.read_text(errors="replace") for p in sorted(ROOT.glob("docs/*.md"))[:6])
    odd = ["é", "́ combining first", "naïve café", "日本語のテキスト", "emoji 🧪🧬", "tab\tand\r\nCRLF",
           "a literal <|im_start|>user in the text", "trailing spaces   ", "\n\n\n", "<think> not really </think>"]

    def chunk(n):
        i = rnd.randrange(0, len(src) - n)
        return src[i:i + n] + rnd.choice(odd)

    def full(text):
        return tok.encode(text, add_special_tokens=False).ids

    n_ok = n = 0
    t_inc = t_full = 0.0
    for conv in range(3):
        msgs = [{"role": "system", "content": "You are an agent.\n" + chunk(20000)}]
        msgs.append({"role": "user", "content": chunk(30000)})
        for step in range(14):
            text, _ = tpl.render({"messages": msgs})
            t0 = time.perf_counter()
            a = tpl.encode(text)
            t1 = time.perf_counter()
            b = full(text)
            t2 = time.perf_counter()
            t_inc += t1 - t0
            t_full += t2 - t1
            n += 1
            n_ok += a == b
            if a != b:
                print(f"  conv {conv} step {step}: incremental differs at token "
                      f"{next(i for i, (x, y) in enumerate(zip(a + [-1], b + [-2])) if x != y)}")
            r = rnd.random()
            if r < 0.5:  # agent turn: the reply (with reasoning), a tool result / user message
                msgs.append({"role": "assistant", "content": chunk(300), "reasoning_content": chunk(200)})
                msgs.append({"role": rnd.choice(["user", "tool"]), "content": chunk(rnd.choice([50, 400, 3000]))})
            elif r < 0.65 and len(msgs) > 3:  # the last assistant message without its reasoning (re-render differs)
                for m in reversed(msgs):
                    if m["role"] == "assistant":
                        m.pop("reasoning_content", None)
                        break
                msgs.append({"role": "user", "content": chunk(200)})
            elif r < 0.8 and len(msgs) > 3:  # an earlier message edited
                k = rnd.randrange(1, len(msgs) - 1)
                msgs[k] = dict(msgs[k], content=msgs[k]["content"][:-40] + rnd.choice(odd))
            elif r < 0.9:  # an unrelated short conversation in between (other LRU entry)
                other = [{"role": "user", "content": chunk(12000)}]
                ot, _ = tpl.render({"messages": other})
                check(tpl.encode(ot) == full(ot), f"conv {conv} step {step}: unrelated conversation")
            else:  # regenerate: same messages again (the whole text is a cached prefix)
                pass
    check(n_ok == n, f"incremental encode == whole-text encode on {n_ok}/{n} requests "
                     f"(incremental {t_inc / n * 1e3:.1f} ms, whole {t_full / n * 1e3:.1f} ms per request)")
    import os
    os.environ["NE_TOK_CACHE"] = "0"
    t2 = chat.Template(pack, tok, "xhigh")
    text, _ = tpl.render({"messages": msgs})
    check(t2.encode(text) == full(text), "NE_TOK_CACHE=0: the whole-text encode")
    print("all passed" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
