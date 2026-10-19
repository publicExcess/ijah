"""Slot scheduler over HTTP (lab-sched), against the server on its mock engine with --slots 2 (tests/server/run_mock.sh):
an idle resident conversation on the other slot does not stop MTP; a request arriving during an MTP reply makes it
yield (timings.path "mtp+step"); every reply = the same request alone.

    $PY tests/server/test_sched_http.py --url http://127.0.0.1:8149 --key-file KEY
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client import FAILS, Client, check  # noqa: E402


def body(q):
    return {"model": "x", "temperature": 0, "messages": [{"role": "system", "content": "Scheduler HTTP test. " * 20},
                                                        {"role": "user", "content": q}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key-file", required=True)
    a = ap.parse_args()
    c = Client(a.url, Path(a.key_file).read_text().strip())

    def ask(q):
        st, r = c.chat(body(q))
        assert st == 200, (st, r)
        return r["choices"][0]["message"]["content"], r["timings"]

    _, tx = ask("short question X")
    ya, ty = ask("conversation Y: LONGTEST")
    check(ty["path"] == "mtp" and ty["slot"] != tx["slot"],
          f"http: one active + one idle resident conversation -> MTP on the other slot ({ty['path']}, slot {ty['slot']} "
          f"next to {tx['slot']})")
    ref = {"A": ya, "B": ask("request B: LONGTEST please")[0]}
    res = {}

    def run(name, q):
        res[name] = ask(q)
    ta = threading.Thread(target=run, args=("A", "conversation Y: LONGTEST"))
    tb = threading.Thread(target=run, args=("B", "request B: LONGTEST please"))
    ta.start()
    time.sleep(0.25)   # A is mid-reply on MTP when B arrives
    tb.start()
    ta.join()
    tb.join()
    pa, pb = res["A"][1]["path"], res["B"][1]["path"]
    check(pa == "mtp+step" and pb in ("step", "mtp+step") and res["A"][1]["slot"] != res["B"][1]["slot"],
          f"http: B arriving during A's MTP reply -> A yields to the step path ({pa}), B {pb}, two slots")
    check(res["A"][0] == ref["A"] and res["B"][0] == ref["B"], "http: both replies = the same request alone")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
