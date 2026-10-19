"""CPU unit tests of App.model_for: which model a request's `model` name selects (lab-dflash, 19 Oct).

The regression these exist for: with ONE model served, an unknown name used to be answered by the default model with
HTTP 200. pi had `Swift-1.5-Flash-Next-native-dflash` registered against a server serving
`Swift-1.5-Flash-Next-native`, so every "DFlash" request ran the plain MTP path and nothing - not the status code, not
the response, not pi - said so. A drafter benchmark measured on that server would have been the control arm, wearing
the drafter's label (docs/server.md 11, docs/mtp.md 13.7). No engine, no GPU, no port.

    $PY tests/server/test_model_id.py        ($PY = oMLX's python3 -I -B)
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import ne_server as N  # noqa: E402

PLAIN = "Swift-1.5-Flash-Next-native"
DFL = "Swift-1.5-Flash-Next-native-dflash"
FAILS = []
LOG = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAILS.append(what)


class Served:            # stands in for App.Served: model_for only reads .id
    def __init__(self, mid):
        self.id = mid


class Fake:              # stands in for App: model_for is called unbound on this
    def __init__(self, ids, allow_any_model=False):
        self.models = {i: Served(i) for i in ids}
        self.default = next(iter(self.models.values()))
        self.a = types.SimpleNamespace(allow_any_model=allow_any_model)
        self._lenient_seen = set()


def pick(app, name):
    """-> the selected id, or '400: <message>' when the server refuses"""
    try:
        return N.App.model_for(app, name).id
    except N.chat.RequestError as e:
        return f"400: {e}"


def main():
    N.log = lambda *a: LOG.append(" ".join(str(x) for x in a))
    one, dfl, two = Fake([PLAIN]), Fake([DFL]), Fake([PLAIN, DFL])

    check(pick(one, PLAIN) == PLAIN, "exact id -> that model (one model served)")
    check(pick(two, DFL) == DFL, "exact id -> that model (two served)")
    check(pick(two, PLAIN.upper()) == PLAIN, "case-insensitive match still selects it")
    check(pick(two, "") == PLAIN and pick(two, None) == PLAIN, "no name -> the default (first served id)")

    # the trap: asking for a drafter id the server does not serve
    check(pick(one, DFL).startswith("400:"), f"served {PLAIN}, asked {DFL} -> 400, not the default")
    check(pick(one, DFL).endswith(f"this server has {PLAIN}"), "the 400 lists the ids this server has, after the name")
    check(pick(two, "x").endswith(f"this server has {PLAIN}, {DFL}"), "the 400 lists every served id")
    check(pick(dfl, PLAIN).startswith("400:"), f"served {DFL}, asked {PLAIN} -> 400 (a drafter run must not be a control)")
    check(pick(one, "x").startswith("400:"), "single model served: an unknown name is a 400 (was: silent default)")
    check(pick(two, "x").startswith("400:"), "two models served: an unknown name is a 400 (as before)")

    # the opt-in escape hatch, for clients that send a placeholder name
    len_one = Fake([PLAIN], allow_any_model=True)
    check(pick(len_one, "x") == PLAIN, "--allow-any-model: a placeholder name gets the default")
    n = len(LOG)
    pick(len_one, "x")
    check(len(LOG) == n, "--allow-any-model warns once per name, not per request")
    pick(len_one, "y")
    check(len(LOG) == n + 1, "a new placeholder name warns once")
    check(pick(Fake([PLAIN, DFL], allow_any_model=True), "x").startswith("400:"),
          "--allow-any-model does not weaken a multi-model server (ambiguous which model was meant)")

    print(f"\n{len(FAILS)} failed" if FAILS else f"\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
