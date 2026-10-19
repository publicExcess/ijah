# Third-party notices

Ijah is Apache-2.0 (LICENSE, NOTICE). It includes or adapts the material below, which stays under its own licence.
"Ported" = the code was translated or adapted from the upstream source; "follows" = the design or algorithm was taken
from it and the code written anew. The source comment at each site names the upstream file.

## MLX (Apple) -- MIT

https://github.com/ml-explore/mlx (0.32.3)

- `kernels/prefill/mlx/` (`attention_nax.h`, `attn_params.h`, `defines.h`, `gemm_nax.h`, `integral_constant.h`,
  `quantized_nax.h`, `type_traits.h`): vendored MLX headers (each keeps its Apple copyright line; `LICENSE.mlx`).
- `kernels/prefill/pf_gemm.metal`, `pf_attn.metal`, `kernels/attn/qsa_chunk.metal`, `kernels/vision/vision.metal`: GEMM
  / attention bodies built on those headers; `erf` / `expm1f` in `kernels/vision/vision.metal` are MLX's.
- `kernels/attn/qsa.metal`: the kv8 cache quantizer reproduces MLX's `affine_quantize` (8-bit, group 64).

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## expm1f (Norbert Juffa, through MLX) -- BSD 2-Clause

- `kernels/vision/vision.metal`: `expm1f` (MLX's `expm1f.h`, which carries this notice).

```
Copyright (c) 2015-2023 Norbert Juffa
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## sushi -- MIT

https://github.com/beamivalice/sushi

- Ported: `kernels/prefill/pf_moe.metal` (the EXL3 MoE prefill NAX GEMM body and trellis fragment decode, from
  `src/exl3/expert_exl3_kernels.zig`); `kernels/prefill/pf_gdn_pipe.metal` (from `src/kernels/gdn_pipelined.metal`,
  itself ported from oMLX, see below).
- Follows: the EXL3 decode rule, tensor-core slot mapping and simd-shuffle FWHT (`kernels/ne_common.h`,
  `kernels/trunk6/trunk6.metal`); the QSA radix select (`kernels/attn/qsa.metal`, after `sushi_qsa_select`); the NAX
  operand lane layout (`kernels/nax/nax_q8.metal`, after `nax_wfrag`); the f16 expert GEMM planes and HC write fold
  (`models/flash_next/prefill.mm`, `kernels/prefill/pf_trunk.metal`); the MTP acceptance modes and the prefix-cache
  design (docs/mtp.md, docs/server.md).

```
MIT License

Copyright (c) 2026 Theinruj Toranavikrai
Copyright (c) 2026 David Dalcu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## exllamav3 -- MIT

https://github.com/turboderp-org/exllamav3

- Follows: the EXL3 trellis format (MCG codebook, window 15, bit layout, tensor-core slot order) in
  `kernels/ne_common.h`, `kernels/trunk6/trunk6.metal`, `tools/opref.py`; `tools/convert_trunk_enc.py` re-implements
  the `quantize_exl3` encode recipe (Hadamard rotations, LDLQ, tail-biting Viterbi).

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mlx-vlm -- MIT

https://github.com/Blaizzy/mlx-vlm (0.7.7)

- Ported: `server/vision.py` (Qwen3-VL image preprocessing, `Qwen3VLImageProcessor`, and the MRoPE positions,
  `get_rope_index`).
- Follows: the reference model math the engine is tested against (`qwen4_exp` layers, the n-gram PLE ids, rope, the
  MTP drafter's accept / cache semantics) in `kernels/` and `models/flash_next/`; the reference tools import it.

```
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## oMLX -- Apache License 2.0

https://github.com/jundot/omlx -- Copyright 2025 oMLX contributors. Licence text: LICENSE (the same Apache License 2.0).

- Ported (through sushi): `kernels/prefill/pf_gdn_pipe.metal` derives from oMLX's
  `omlx/custom_kernels/qwen35_prefill/gdn.py` `gated_delta_pipelined`. Changes: our layouts, a constant-trip loop, f32
  output with the decode path's rounding point (see the file header).
- Follows (no code): the INT8-activation prefill (`--prefill-int8`, `kernels/trunk6/t6_a8.metal`, after oQ A8) and the
  norm-inject fold (`kernels/prefill/pf_trunk.metal`).

## Splish -- Apache License 2.0

https://github.com/publicExcess/splish -- an Apache-2.0 fork of Inco AI's Splash. Licence text: LICENSE.

- Copied: `models/flash_next/draft_vocab.h` (Splish's `runtime/model/flashnext/DraftVocab.hpp`, written by publicExcess).

## TensorFold

https://github.com/ashhart/TensorFold -- Copyright 2026 TensorFold contributors (Apache License 2.0 from 0.6.0).

- Follows (no code): the MTP copy-draft rule (`models/flash_next/mtp_loop.mm`) and the 98,304-id prefix of the pruned
  draft vocabulary.

## Python dependencies (installed by pip, not redistributed)

tokenizers (Apache-2.0), Jinja2 (BSD-3-Clause), NumPy (BSD-3-Clause), Pillow (MIT-CMU), xgrammar (Apache-2.0; pulls
in PyTorch, BSD-3-Clause, and transformers, Apache-2.0), jsonschema (MIT); optional: MLX (MIT), mlx-vlm (MIT).

## Model weights

Not part of this repository. The packs (SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah,
SiliconSpecies/Swift-1.5-Flash-Next-EXL3-Ijah on Hugging Face) carry their own licences; see their model cards.
