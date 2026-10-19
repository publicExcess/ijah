"""Limits on what a request can make the server read (lab-fixsrv, 17 Oct). CPU.

Local image paths are read only under an allowlist (default $HOME, NE_VISION_ROOTS), symlinks resolved; an image over
NE_VISION_MAX_INPUT_PIXELS (default 40 megapixels) is refused from its header, before it is decoded; shutdown closes the
prefix cache of every served model, not only the default one.

    $PY tests/server/test_limits.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
from PIL import Image  # noqa: E402

import ne_server as S  # noqa: E402
import vision as V  # noqa: E402

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def refused(src, needle=None):
    try:
        V.load_image(src)
        return False
    except V.ImageError as e:
        return needle is None or needle in str(e)


def main():
    S.log = lambda *a: None
    os.environ.pop("NE_VISION_ROOTS", None)
    inside = Path(tempfile.mkdtemp(prefix=".ne-limits-", dir=ROOT / "tests/server"))   # under $HOME
    outside = Path(tempfile.mkdtemp(prefix="ne-limits-"))                                # /var/folders: not under $HOME
    try:
        Image.new("RGB", (64, 64), (10, 200, 30)).save(inside / "ok.png")
        Image.new("RGB", (64, 64), (10, 200, 30)).save(outside / "ok.png")
        check(not refused(str(inside / "ok.png")) and not refused("file://" + str(inside / "ok.png")),
              "a path under $HOME loads (plain and file://)")
        check(refused(str(outside / "ok.png"), "only read under"), "a path outside $HOME is refused with the reason")
        check(refused("/etc/hosts", "only read under"), "/etc/hosts is refused (not an image, not under $HOME)")
        check(refused(str(Path.home() / ".config/ijah/api-key")), "the API key file is refused (cannot be decoded)")
        (inside / "link.png").symlink_to(outside / "ok.png")
        check(refused(str(inside / "link.png"), "only read under"), "a symlink from the allowed tree to outside is refused")
        check(refused(str(inside / ".." / ".." / ".." / ".." / ".." / ".." / "etc/hosts"), "only read under")
              or refused(str(inside / ".." / ".." / ".." / ".." / ".." / ".." / "etc/hosts")), "../ out of the tree is refused")
        os.environ["NE_VISION_ROOTS"] = str(outside)
        check(not refused(str(outside / "ok.png")) and refused(str(inside / "ok.png"), "only read under"),
              "NE_VISION_ROOTS replaces the allowlist")
        os.environ["NE_VISION_ROOTS"] = ""
        check(refused(str(inside / "ok.png"), "no directory"), "NE_VISION_ROOTS empty: no local files at all")
        check(not refused("data:image/png;base64," + __import__("base64").b64encode((inside / "ok.png").read_bytes()).decode()),
              "data: URLs are not affected")
        os.environ.pop("NE_VISION_ROOTS")
        # pixel cap: refused from the header
        Image.new("L", (7000, 6000), 0).save(inside / "big.png")   # 42 megapixels, a few KB on disk
        check(refused(str(inside / "big.png"), "megapixels"), "a 42-megapixel image is refused before decoding (limit 40)")
        os.environ["NE_VISION_MAX_INPUT_PIXELS"] = "50000000"
        check(not refused(str(inside / "big.png")), "NE_VISION_MAX_INPUT_PIXELS raises the limit")
        os.environ["NE_VISION_MAX_INPUT_PIXELS"] = "1000"
        check(refused(str(inside / "ok.png"), "megapixels"), "... and lowers it")
        os.environ.pop("NE_VISION_MAX_INPUT_PIXELS")
    finally:
        shutil.rmtree(inside, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)

    # shutdown closes every model's cache
    closed = []

    def served(mid, ok=True, boom=False):
        def close(timeout):
            closed.append((mid, timeout))
            if boom:
                raise OSError("disk gone")
            return ok
        return SimpleNamespace(id=mid, session=SimpleNamespace(cache=SimpleNamespace(close=close)))
    app = SimpleNamespace(models={"a": served("a"), "b": served("b", ok=False), "c": served("c", boom=True), "d": served("d")})
    late = S.App.close(app, timeout=7)
    check([c[0] for c in closed] == ["a", "b", "c", "d"] and all(c[1] == 7 for c in closed) and late == ["b", "c"],
          f"close(): every model's cache closed ({[c[0] for c in closed]}), the late / failed ones reported ({late})")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
