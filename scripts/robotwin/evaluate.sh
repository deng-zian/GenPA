#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export GENPA_ROBOTWIN_PACKAGE="$PWD"
: "${GENPA_ROBOTWIN_BENCHMARK:?Set the external RoboTwin benchmark directory}"
: "${GENPA_ROBOTWIN_CKPT:?Set the final RoboTwin checkpoint file}"
: "${GENPA_EVAL_OUTPUT:?Set an evaluation output directory}"
export GENPA_FLUX2_SRC="${GENPA_FLUX2_SRC:-$(cd third_party/flux2 && pwd)}"
: "${GENPA_FLUX2_WEIGHTS:?Set the FLUX.2 DiT weight file}"
: "${GENPA_FLUX2_AE:?Set the FLUX.2 VAE weight file}"
: "${GENPA_QWEN3:?Set the Qwen3-4B model directory}"
: "${GENPA_CUROBO_SRC:?Set the CuRobo source directory required by RoboTwin}"
export PYTHONPATH="$PWD:$PWD/src:$GENPA_FLUX2_SRC/src:$GENPA_FLUX2_SRC:$GENPA_CUROBO_SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GENPA_ROBOTWIN_STATS="${GENPA_ROBOTWIN_STATS:-$PWD/configs/robotwin/absolute_stats.json}"
python -m experiments.robotwin.run --config configs/robotwin/eval.yaml --output "$GENPA_EVAL_OUTPUT" "$@"
