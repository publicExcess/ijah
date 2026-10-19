"""GPU time of the full decode step (ne_fn_model_time) with the library NE_LIB, for the A/B of the sampler's token mask
(lab-api): the baseline library (no mask code) vs this build with the mask off (every unconstrained request) and with an
all-ones mask on (the cost of the bit read). One model load, N rounds x REPS replays at context T, sampler as the server
defaults (temperature 1, top-p 0.95, top-k 20). Prints one JSON line. Whole-model: run under tools/with_lock.sh.

    NE_LIB=/abs/libengine.dylib NEED_GB=85 tools/with_lock.sh "step A/B (~81 GB)" HH:MM -- $PY tests/server/step_ab.py
"""
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("NE_FN_ROUND", "f32")
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402

import engine as E  # noqa: E402


def main():
    T = int(os.environ.get("AB_T", 2048))
    rounds, reps = int(os.environ.get("AB_ROUNDS", 5)), int(os.environ.get("AB_REPS", 60))
    eng = E.Engine()
    pack = eng.pack_open(ijah_paths.pack())
    m = eng.fn_model(pack, 4096)
    m.set_sampler(1.0, 0.95, 20, 7)
    out = {"lib": os.environ.get("NE_LIB", "build/libengine.dylib"), "T": T, "reps": reps}
    modes = [("off", None)]
    if eng.has("ne_fn_model_set_mask"):
        modes.append(("mask_all", np.full((m.VOCAB + 31) // 32, -1, np.int32)))
    knobs = {}
    if eng.has("ne_fn_model_set_sampler_ex"):   # lab-sampler: the cost of each knob on the step path
        knobs = {"bias": dict(bias={5: -1.0}), "penalties": dict(presence=0.5, frequency=0.2, repetition=1.1),
                 "min_p": dict(min_p=0.05), "logprob20": dict(lp_k=20),
                 "all": dict(bias={5: -1.0}, presence=0.5, frequency=0.2, repetition=1.1, min_p=0.05, lp_k=20)}
        modes += [(k, None) for k in knobs] + [("off_again", None)]
    m.time(T, 10)   # warm-up
    for name, bits in modes:
        if eng.has("ne_fn_model_set_mask"):
            m.set_mask(bits)
        if knobs:
            kw = knobs.get(name, {})
            m.set_logit_bias(kw.get("bias"))
            m.set_sampler_ex(kw.get("min_p", 0.0), kw.get("presence", 0.0), kw.get("frequency", 0.0),
                             kw.get("repetition", 1.0), kw.get("lp_k", 0))
        r = [m.time(T, reps) for _ in range(rounds)]
        out[name] = {"gpu_ms_median_of_medians": round(statistics.median(x["gpu_ms_median"] for x in r), 4),
                     "gpu_ms_min": round(min(x["gpu_ms_min"] for x in r), 4),
                     "wall_ms_median_of_medians": round(statistics.median(x["wall_ms_median"] for x in r), 4),
                     "rounds": [round(x["gpu_ms_median"], 4) for x in r]}
    print("STEP_AB " + json.dumps(out))


if __name__ == "__main__":
    main()
