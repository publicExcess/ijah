// lab-speed recording helpers (kernels/fn_rec.h).
#include "kernels/fn_rec.h"

namespace ne::fn::krec {

static std::string gname(const char *base, int g, int r) {
  return std::string(base) + "_g" + std::to_string(g) + "_r" + std::to_string(r);
}
static MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }

void hc_pre(Recorder &r, const HcW &hw, int R, const Tensor &h, const Tensor &inject, const Tensor &inj,
            const Tensor &part) {
  r.dispatch(pso(gname("fn_hc_pre2", hw.down.group, R)), {h, hw.norm, hw.down.w, hw.down.s, hw.down.b, inject, inj, part},
             sz((LR / 8 + 1) * HC), sz(256));
  r.barrier();
}
void hc_pre3(Recorder &r, const HcW &hw, int R, const Tensor &h, const Tensor &inject, const Tensor &inj,
             const Tensor &part) {
  r.dispatch(pso(gname("fn_hc_pre3", hw.down.group, R)), {h, hw.norm, hw.down.w, hw.down.s, hw.down.b, inject, inj, part},
             sz((LR / 8 + 1) * HC), sz(256));
  r.barrier();
}
void hc_mix(Recorder &r, const HcW &hw, int R, const Tensor &h, const Tensor &mix, const Tensor &inj,
            const Tensor &out, const Tensor &part) {
  r.dispatch(pso(gname("fn_hc_mix2", hw.up.group, R)), {h, hw.norm, hw.up.w, hw.up.s, hw.up.b, mix, inj, out, part},
             sz(HID / 8), sz(256));
  r.barrier();
}

}  // namespace ne::fn::krec
