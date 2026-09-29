#!/usr/bin/env bash
# Clones the third-party baseline repositories at the commits used for the paper into pretrain/third_party/.
# They are not redistributed here; each keeps its own license.
# Lip-KL-GPA is not listed: its code is part of cvar_gpa/. t-EDM is not listed: it is implemented in pretrain/wrappers/tEDM/run_tedm.py.
set -euo pipefail
: "${CVAR_GPA_ROOT:?set CVAR_GPA_ROOT to the root of this repository}"
DST="${CVAR_GPA_ROOT}/pretrain/third_party"
mkdir -p "${DST}"; cd "${DST}"
get() { # get <dir> <url> <commit>
  [ -d "$1/.git" ] || git clone "$2" "$1"
  git -C "$1" checkout --quiet "$3"; echo "$1 @ $3"; }
get tailnflows                https://github.com/Tennessee-Wallaceh/tailnflows.git               61072f791fcbb351dca70ad6db3c1d8a57478cc0
get nflows                    https://github.com/Tennessee-Wallaceh/nflows.git                   f90af9faeaba417f683c0668aa4806d575c8af3c
get marginalTailAdaptiveFlow  https://github.com/Tennessee-Wallaceh/marginalTailAdaptiveFlow.git cc6268e00abdf4694bb0206e5e808759b7758a58
get mTAF_official             https://github.com/MikeLasz/marginalTailAdaptiveFlow.git           cbcfeb47469ed403e91b61acee0abc5bbb3f5190
get Tail-GAN                  https://github.com/chaozhang-ox/Tail-GAN.git                       bfbe9501a593d7797b2cfb80de1f20c78d99cda3
get heavy_tail_diffusion      https://github.com/huajianduzhuo-code/heavy_tail_diffusion.git     5f935870dddab8d3df9f7ab60dc08c196398bf43
get DLPM                      https://github.com/darioShar/DLPM.git                              faecd3a8bcea2f1eb59dfc8dc31b5d8a8d278dd5
