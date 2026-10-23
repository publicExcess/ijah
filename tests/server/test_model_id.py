"""CPU unit tests of App.model_for: which model a request's `model` name selects (lab-dflash, 19 Oct), and of the
DFlash2-by-default rules (release, 23 Oct: drafter_spec, canonical_id, models_all).

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
QWEN = "Qwen3.8-Flash-Next-native"
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
    def __init__(self, ids, allow_any_model=False, drafters=()):
        self.models = {i: Served(i) for i in ids}
        self.default = next(iter(self.models.values()))
        self.a = types.SimpleNamespace(allow_any_model=allow_any_model,
                                       drafters={i: ({"path": "x"} if i in drafters else None) for i in ids})
        self._lenient_seen = set()


def pick(app, name):
    """-> the selected id, or '400: <message>' when the server refuses"""
    try:
        return N.App.model_for(app, name).id
    except N.chat.RequestError as e:
        return f"400: {e}"


def main():
    N.log = lambda *a: LOG.append(" ".join(str(x) for x in a))
    # release (23 Oct): the plain ids use DFlash2 when the drafter is present; the -dflash ids are aliases of them
    one, two = Fake([PLAIN]), Fake([QWEN, PLAIN])                  # served with the MTP head only (no drafter)
    one_d, two_d = Fake([PLAIN], drafters=[PLAIN]), Fake([QWEN, PLAIN], drafters=[QWEN, PLAIN])

    check(pick(one, PLAIN) == PLAIN, "exact id -> that model (one model served)")
    check(pick(two, PLAIN) == PLAIN, "exact id -> that model (two served)")
    check(pick(two, PLAIN.upper()) == PLAIN, "case-insensitive match still selects it")
    check(pick(two, "") == QWEN and pick(two, None) == QWEN, "no name -> the default (first served id)")

    # the trap: asking for the drafter where the server runs the MTP head only
    check(pick(one, DFL).startswith("400:"), f"served {PLAIN} without its drafter, asked {DFL} -> 400, not the MTP head")
    check(f"serves {PLAIN} with the MTP head only" in pick(one, DFL) and "a22a1744e85c" in pick(one, DFL),
          "the 400 says the drafter is not loaded and how to get it")
    check(pick(one_d, DFL) == PLAIN, f"served {PLAIN} with its drafter: the alias {DFL} -> that model")
    check(pick(two_d, DFL.lower()) == PLAIN, "the alias matches case-insensitively")
    check(pick(one_d, PLAIN) == PLAIN, "the plain id on a drafter server -> the same model (DFlash2 is the default)")
    check(pick(Fake([QWEN], drafters=[QWEN]), DFL).startswith("400:") and "is not served" in
          pick(Fake([QWEN], drafters=[QWEN]), DFL), "an alias of a model this server does not serve -> 400")
    check(pick(Fake([PLAIN], allow_any_model=True), DFL).startswith("400:"),
          "--allow-any-model never answers a drafter alias with the MTP head")
    check(pick(one, "y").endswith(f"this server has {PLAIN}"), "the 400 lists the ids this server has, after the name")
    check(pick(two, "x").endswith(f"this server has {QWEN}, {PLAIN}"), "the 400 lists every served id")
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
    check(pick(Fake([QWEN, PLAIN], allow_any_model=True), "x").startswith("400:"),
          "--allow-any-model does not weaken a multi-model server (ambiguous which model was meant)")

    # the registry: plain ids carry the drafter, -dflash ids are aliases; --models all = the plain ids
    import json
    import tempfile
    reg = json.loads(N.MODELS.read_text())
    plain_all = N.models_all(reg)
    check(DFL not in plain_all and PLAIN in plain_all and plain_all[0] == reg["default"],
          f"--models all = the plain ids, default first ({', '.join(plain_all)})")
    check(N.models_all(reg, True) == plain_all, "--with-drafters (old command lines) changes nothing")
    check(all(reg["models"][m].get("in_all", True) for m in plain_all)
          and "Qwen3.8-Flash-Next-native-4bit" not in plain_all, "--models all leaves out in_all: false (the 4-bit pack)")
    check(N.canonical_id(DFL) == (PLAIN, True) and N.canonical_id(PLAIN) == (PLAIN, False),
          "canonical_id: alias -> (plain id, drafter required)")
    check(N.resolve_model(DFL, pack="/tmp/p")[0] == PLAIN, "resolve_model reports the plain id for an alias")
    check(all(reg["models"][m].get("drafter") for m in plain_all), "every plain id has a drafter entry")

    def ns(**k):
        d = dict(drafter_path=None, dflash_block=8, dflash_kmax=5, dflash_bits=8, dflash_gate=1, dflash_max_ctx=8192,
                 drafter="auto", mtp=2)
        d.update(k)
        return types.SimpleNamespace(**d)

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as empty:
        LOG.clear()
        check(N.drafter_spec(PLAIN, ns(drafter_path=empty)) is None, "auto, drafter absent -> None (the MTP head)")
        check(len(LOG) == 1 and "MTP head only" in LOG[0] and "ijah download drafter" in LOG[0],
              "auto, drafter absent -> ONE log line saying how to get it")
        for bad, what in ((ns(drafter_path=empty), "alias"), (ns(drafter_path=empty, drafter="dflash2"), "--drafter dflash2")):
            try:
                N.drafter_spec(DFL if what == "alias" else PLAIN, bad)
                check(False, f"{what}, drafter absent -> refused before any load")
            except SystemExit as e:
                check("needs its DFlash2 drafter checkpoint" in str(e), f"{what}, drafter absent -> SystemExit")
        for f in N.DRAFTER_FILES:
            (Path(td) / f).write_text("{}")
        d = N.drafter_spec(PLAIN, ns(drafter_path=td))
        check(d and d["path"] == str(Path(td)) and d["kmax"] == 5 and d["gate"] == 1 and d["max_ctx"] == 8192,
              "auto, drafter present -> DFlash2 (exact gate 1, 8192 context cap)")
        check(N.drafter_spec(DFL, ns(drafter_path=td))["path"] == str(Path(td)), "an alias -> its plain id's drafter")
        check(N.drafter_spec(PLAIN, ns(drafter_path=td, drafter="none")) is None, "--drafter none -> the MTP head")
        check(N.drafter_spec(PLAIN, ns(drafter_path=td, mtp=0)) is None, "--mtp 0 -> no drafter")
        try:
            N.drafter_spec(DFL, ns(drafter_path=td, drafter="none"))
            check(False, "an alias with --drafter none is refused")
        except SystemExit:
            check(True, "an alias with --drafter none is refused (asked for the drafter, told to turn it off)")

    print(f"\n{len(FAILS)} failed" if FAILS else f"\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
