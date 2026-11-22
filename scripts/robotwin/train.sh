#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${GENPA_OUTPUT_DIR:?Set an output directory}"
: "${GENPA_ROBOTWIN_DATA:?Set the processed RoboTwin data directory}"
: "${GENPA_ROBOTWIN_NUMERIC:?Set the processed numeric-array directory}"
: "${GENPA_ROBOTWIN_LANGUAGE:?Set the language index directory}"
: "${GENPA_ROBOTWIN_QWEN_0:?Set packed Qwen cache shard directory 0}"
: "${GENPA_ROBOTWIN_QWEN_1:?Set packed Qwen cache shard directory 1}"
: "${GENPA_ROBOTWIN_QWEN_2:?Set packed Qwen cache shard directory 2}"
export GENPA_FLUX2_SRC="${GENPA_FLUX2_SRC:-$(cd third_party/flux2 && pwd)}"
: "${GENPA_FLUX2_WEIGHTS:?Set the FLUX.2 DiT weight file}"
: "${GENPA_FLUX2_AE:?Set the FLUX.2 VAE weight file}"
: "${GENPA_QWEN3:?Set the Qwen3-4B model directory}"
export GENPA_ROBOTWIN_STATS="${GENPA_ROBOTWIN_STATS:-$PWD/configs/robotwin/absolute_stats.json}"
export PYTHONPATH="$PWD/src:$GENPA_FLUX2_SRC/src:$GENPA_FLUX2_SRC${PYTHONPATH:+:$PYTHONPATH}"
accelerate launch --config_file "${GENPA_ACCELERATE_CONFIG:-scripts/robotwin/accelerate_configs/accelerate_zero1_ds.yaml}" \
  --num_processes "${GENPA_GPUS:-8}" scripts/robotwin/train.py --config configs/robotwin/depth.yaml "$@"
