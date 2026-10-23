"""DFlash2 by default (release, 23 Oct), over HTTP against the server on its mock engine (tests/server/run_mock.sh).
The mock carries the drafter only as a label, so this checks the routing, not the drafting: with the drafter loaded
the plain id and its -dflash alias both answer (from the plain model) and /v1/models says "dflash2"; with
--drafter none (or no checkpoint) the plain id answers, /v1/models says "mtp" and the alias is a 400 naming the
download, so a request for the drafter is never answered by the MTP head.

    $PY tests/server/test_drafter_http.py --url URL --key-file KEY --model ID --drafter dflash2|mtp
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client import FAILS, Client, check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True, choices=["dflash2", "mtp"])
    a = ap.parse_args()
    c = Client(a.url, Path(a.key_file).read_text().strip())
    st, r = c.req("GET", "/v1/models")
    got = {m["id"]: m.get("drafter") for m in r.get("data", [])} if st == 200 else {}
    check(got == {a.model: a.drafter}, f"/v1/models: {a.model} with drafter {a.drafter} ({got})")

    def ask(model):
        return c.chat({"temperature": 0, "model": model, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "WHOAMI"}]})

    st, r = ask(a.model)
    check(st == 200 and r.get("model") == a.model, f"the plain id answers ({st})")
    st, r = ask(a.model + "-dflash")
    if a.drafter == "dflash2":
        check(st == 200 and r.get("model") == a.model, f"the -dflash alias answers from {a.model} ({st})")
    else:
        msg = r.get("error", {}).get("message", "")
        check(st == 400 and "needs its DFlash2 drafter" in msg and "hf download" in msg,
              f"no drafter loaded: the -dflash alias is a 400 that names the download ({st}: {msg[:80]})")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
