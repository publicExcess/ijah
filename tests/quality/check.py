"""lab-quality: check an engine-vs-teacher result (tools/kld_teacher.py json) against the limits in docs/quality.md,
next to fn_exl3 (MLX, same pack) on the same windows.

    $PY tests/quality/check.py tests/quality/results/kld-NAME.json          # report + limits, exit 1 on a failure
    $PY tests/quality/check.py --report-only a.json b.json ...              # report only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import kld_teacher as KT  # noqa: E402

# Limits for the K5 pack (Qwen3.8-Flash-Next-EXL3-K5 / -a16), engine generation default (f32, bf16 KV, QSA).
# fn_exl3 on the same pack: A 0.0357 / B 0.0105 / C 0.0163, top-1 0.929 / 0.972 / 0.968 (whole_kld.py, 14 Oct).
# Engine measured 15 Oct (f32, bf16 KV): A 0.0361 / B 0.00995 / C 0.0163, top-1 0.928 / 0.974 / 0.968 (docs/quality.md
# section 2). Limits = measured + ~11-15% (the size of one bit-level nudge above the floor; rerun is exact, so any
# excess is a real change in the numerics, not noise).
LIMITS = {  # set: (max KLD mean, min top-1 mean)
    "A": (0.040, 0.920),
    "B": (0.0115, 0.968),
    "C": (0.0180, 0.962),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--vs", default="K5", help=f"fn_exl3 result to pair with: {', '.join(KT.FN_EXL3)}")
    a = ap.parse_args()
    base = KT.RUN / "kld" / KT.FN_EXL3[a.vs]
    KT.report([str(base)] + a.files)
    if a.report_only:
        return 0
    bad = []
    for f in a.files:
        r = json.loads(Path(f).read_text())
        for S, x in r["sets"].items():
            if x.get("max_pos") or len(x["per_window"]["kld"]) < len(KT_windows(S)):
                print(f"{r['name']} {S}: subset (windows {x['windows']}, max_pos {x.get('max_pos')}): limits not applied")
                continue
            kmax, t1min = LIMITS[S]
            ok = x["kld_mean"] <= kmax and x["top1_mean"] >= t1min
            print(f"{r['name']} {S}: KLD {x['kld_mean']:.5f} (<= {kmax}), top-1 {x['top1_mean']:.4f} (>= {t1min}): "
                  f"{'ok' if ok else 'FAIL'}")
            if not ok:
                bad.append(f"{r['name']} {S}")
        if r.get("rerun_max_abs", 0) != 0:
            bad.append(f"{r['name']} rerun {r['rerun_max_abs']}")
        for S, x in r["sets"].items():
            if x.get("teacher_rebuild_max_abs", 0) != 0:
                bad.append(f"{r['name']} {S} teacher rebuild {x['teacher_rebuild_max_abs']}")
    print("FAIL: " + ", ".join(bad) if bad else "test-quality: all limits ok")
    return 1 if bad else 0


def KT_windows(S):
    return json.loads((KT.RUN / "teacher" / "windows.json").read_text())["sets"][S]["windows"]


if __name__ == "__main__":
    sys.exit(main())
