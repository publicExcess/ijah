// Recording helpers for lab-speed's faster layer kernels (kernels/fn_trunk.metal, fn_moe.metal). Owner: lab-speed.
// Drop-ins for single stages of models/flash_next/layer.mm record_stage (same Scratch / state buffers); each records
// its dispatch + barrier. Math and rounding are the old kernels' (outputs bit-identical); numbers: docs/kernels.md.
#pragma once
#include "models/flash_next/layer.h"

namespace ne::fn::krec {

// HC read v2 = two stages: fn_hc_pre2 (stage AttnHCPre / MlpHCPre: h -> per-stream partials s.hc_part, inj[4..7] =
// 1/rms) then fn_hc_mix2 (stage AttnHCMix / MlpHCMix: partials -> mix, inj[0..3], out). No atomics / counters:
// s.hc_ctr is not used. Works for the head's mixer too (inject = a zero matrix; inj[0..3] ignored).
void hc_pre(Recorder &r, const HcW &hw, int R, const Tensor &h, const Tensor &inject, const Tensor &inj,
            const Tensor &part);
// fn_hc_pre3 = fn_hc_pre2 without the threadgroup staging (any R; 1/rms may differ in the last bit), same outputs.
void hc_pre3(Recorder &r, const HcW &hw, int R, const Tensor &h, const Tensor &inject, const Tensor &inj,
             const Tensor &part);
void hc_mix(Recorder &r, const HcW &hw, int R, const Tensor &h, const Tensor &mix, const Tensor &inj,
            const Tensor &out, const Tensor &part);

}  // namespace ne::fn::krec
