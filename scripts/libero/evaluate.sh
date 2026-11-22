#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash scripts/libero/evaluate.sh MODE [options] [Hydra overrides...]
Modes: standard | plus | plus_released
  --gpus 0,1             Physical GPU indices (default: 0)
  --workers-per-gpu 4    Independent policy + simulator workers on each GPU (default: 1)
  --chunk-size 20        Tasks sharing one model load (standard: 1, Plus: 20)
  --task-file PATH       Optional subset, one suite,task_id per line
  --trials N             Trials per task (standard: 50, Plus: 1)
  -h, --help             Show this help without loading models or benchmarks

Each worker runs policy inference and EGL simulation on the same GPU.
One GPU is enough; select 1, 2, 4, 8, or any other number using --gpus.
EOF
}
if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
mode="$1"
shift
case "$mode" in standard|plus|plus_released) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac
default_chunk=1
if [[ "$mode" != standard ]]; then default_chunk=20; fi
launch_args=()
overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus)
      [[ ${2:-} =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo '--gpus requires comma-separated nonnegative indices' >&2; exit 2; }
      launch_args+=("MULTIRUN.gpu_ids=[$2]"); shift 2 ;;
    --workers-per-gpu|--chunk-size|--trials)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo "$1 requires a positive integer" >&2; exit 2; }
      case "$1" in
        --workers-per-gpu) launch_args+=("MULTIRUN.max_tasks_per_gpu=$2") ;;
        --chunk-size) launch_args+=("MULTIRUN.chunk_size=$2") ;;
        --trials) launch_args+=("EVALUATION.num_trials=$2") ;;
      esac
      shift 2 ;;
    --task-file)
      [[ -n ${2:-} ]] || { echo '--task-file requires a path' >&2; exit 2; }
      launch_args+=("MULTIRUN.task_file=$2"); shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) overrides+=("$1"); shift ;;
  esac
done
cd "$(dirname "$0")/../.."
: "${GENPA_LIBERO_CKPT:?Set the final LIBERO checkpoint file}"
: "${GENPA_EVAL_OUTPUT:?Set an evaluation output directory}"
: "${GENPA_LIBERO_BENCHMARK:?Set the LIBERO or LIBERO-Plus checkout}"
: "${GENPA_LIBERO_QWEN_CACHE:?Set the matching Qwen cache directory}"
export GENPA_FLUX2_SRC="${GENPA_FLUX2_SRC:-$(cd third_party/flux2 && pwd)}"
: "${GENPA_FLUX2_WEIGHTS:?Set the FLUX.2 DiT weight file}"
: "${GENPA_FLUX2_AE:?Set the FLUX.2 VAE weight file}"
: "${GENPA_QWEN3:?Set the Qwen3-4B model directory}"
export GENPA_LIBERO_STATS="${GENPA_LIBERO_STATS:-$PWD/configs/libero/dataset_stats.json}"
export PYTHONPATH="$PWD/src:$GENPA_FLUX2_SRC/src:$GENPA_FLUX2_SRC:$GENPA_LIBERO_BENCHMARK${PYTHONPATH:+:$PYTHONPATH}"
export WORKER_PYTHONPATH="$PYTHONPATH"
export WORKER_ENV_SOURCE="${WORKER_ENV_SOURCE-}"
export LIBERO_PARALLEL_RUNNER=run_libero_parallel.sh
export LIBERO_CONFIG_PATH="${GENPA_LIBERO_CONFIG_PATH:-$HOME/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
benchmark_kind=libero
if [[ "$mode" != standard ]]; then benchmark_kind=libero_plus; fi
export GENPA_EVAL_BENCHMARK="$benchmark_kind"
args=(
  --config-name eval_libero
  "++EVALUATION.benchmark_kind=$benchmark_kind"
  task=libero_genpa
  "ckpt=$GENPA_LIBERO_CKPT"
  "EVALUATION.output_dir=$GENPA_EVAL_OUTPUT"
  "EVALUATION.dataset_stats_path=$GENPA_LIBERO_STATS"
  "model.flux2_src_path=$GENPA_FLUX2_SRC"
  "model.flux2_model_path=$GENPA_FLUX2_WEIGHTS"
  "model.ae_model_path=$GENPA_FLUX2_AE"
  "model.qwen3_model_spec=$GENPA_QWEN3"
  "data.train.qwen_text_cache_dir=$GENPA_LIBERO_QWEN_CACHE"
  data.train.qwen_context_len=128
  model.qwen_context_len=128
  model.proprio_dim=8
  model.checkpoint_blocks=false
  MULTIRUN.num_gpus=1
  'MULTIRUN.gpu_ids=[0]'
  MULTIRUN.max_tasks_per_gpu=1
  "MULTIRUN.chunk_size=$default_chunk"
  EVALUATION.action_horizon=16
  EVALUATION.replan_steps=12
  ++EVALUATION.use_qwen_text_cache=true
  ++EVALUATION.save_rollout_video=false
)
if [[ "$mode" == plus ]]; then
  : "${GENPA_CLEAN_INSTRUCTION_MANIFEST:?Generate the Clean-Instruction manifest first}"
  args+=(
    EVALUATION.num_trials=1
    ++EVALUATION.instruction_protocol=clean
    "++EVALUATION.clean_instruction_manifest_path=$GENPA_CLEAN_INSTRUCTION_MANIFEST"
  )
elif [[ "$mode" == plus_released ]]; then
  args+=(EVALUATION.num_trials=1 ++EVALUATION.instruction_protocol=official)
else
  args+=(EVALUATION.num_trials=50)
fi
if [[ ${#launch_args[@]} -gt 0 ]]; then args+=("${launch_args[@]}"); fi
if [[ ${#overrides[@]} -gt 0 ]]; then args+=("${overrides[@]}"); fi
python experiments/libero/run_libero_manager.py "${args[@]}"
