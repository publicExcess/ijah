#!/usr/bin/env python3
"""Hard-link the files two packs have in common, so they take disk space once and the engine maps them once.

  tools/link_shared_packs.py PACK_A PACK_B [--dry-run]

A file in PACK_B is replaced by a hard link to PACK_A's file of the same name only when the two are byte-for-byte
identical (full comparison; files that merely share a name and size, such as fine-tuned trunk weights, are left
alone). Run it after downloading the Qwen3.8 and Swift-1.5 packs to serve both from one process (NE_MODEL=all): the
routed experts are shared and are then loaded once.
"""
import argparse
import filecmp
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack_a")
    ap.add_argument("pack_b")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    linked = saved = kept = 0
    for name in sorted(os.listdir(a.pack_a)):
        pa, pb = os.path.join(a.pack_a, name), os.path.join(a.pack_b, name)
        if not (os.path.isfile(pa) and os.path.isfile(pb)) or os.path.islink(pa) or os.path.islink(pb):
            continue
        sa, sb = os.stat(pa), os.stat(pb)
        if sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev:
            continue                                    # already the same file
        if sa.st_size != sb.st_size or not filecmp.cmp(pa, pb, shallow=False):
            kept += 1                                   # different contents: keep both
            continue
        if not a.dry_run:
            tmp = pb + ".link-tmp"
            os.link(pa, tmp)
            os.replace(tmp, pb)
        linked += 1
        saved += sb.st_size
    verb = "would link" if a.dry_run else "linked"
    print(f"{verb} {linked} identical files ({saved / 1e9:.1f} GB saved); {kept} same-named files differ and were kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
