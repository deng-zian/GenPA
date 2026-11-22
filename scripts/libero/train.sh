#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
variant="${1:-depth}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in depth|action_only) ;; *) echo "Use depth or action_only" >&2; exit 2 ;; esac
: "${GENPA_OUTPUT_DIR:?Set an output directory}"
: "${GENPA_LIBERO_DATA:?Set the processed LIBERO LeRobot dataset directory}"
if [[ "$variant" == depth ]]; then
  : "${GENPA_LIBERO_DEPTH_LABELS:?Set the aligned current-depth label directory}"
fi
: "${GENPA_LIBERO_QWEN_CACHE:?Set the Qwen cache directory}"
export GENPA_FLUX2_SRC="${GENPA_FLUX2_SRC:-$(cd third_party/flux2 && pwd)}"
: "${GENPA_FLUX2_WEIGHTS:?Set the FLUX.2 DiT weight file}"
: "${GENPA_FLUX2_AE:?Set the FLUX.2 VAE weight file}"
: "${GENPA_QWEN3:?Set the Qwen3-4B model directory}"
export GENPA_LIBERO_STATS="${GENPA_LIBERO_STATS:-$PWD/configs/libero/dataset_stats.json}"
export GENPA_LIBERO_KEEP_RANGES="${GENPA_LIBERO_KEEP_RANGES:-$PWD/configs/libero/action_keep_ranges.json}"
export PYTHONPATH="$PWD/src:$GENPA_FLUX2_SRC/src:$GENPA_FLUX2_SRC${PYTHONPATH:+:$PYTHONPATH}"
accelerate launch --config_file "${GENPA_ACCELERATE_CONFIG:-scripts/libero/accelerate_configs/accelerate_zero1_ds.yaml}" \
  --num_processes "${GENPA_GPUS:-8}" scripts/libero/train.py --config-name "$variant" "$@"
