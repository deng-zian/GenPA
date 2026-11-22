# Evaluation

Start with the [upstream weights and benchmark environment](SETUP.md) and a [released checkpoint](CHECKPOINTS.md). Run each command from the code checkout.

## Standard LIBERO

Build the instruction cache once from the included prompt list. Only the benchmark and the model assets are needed to evaluate a trained policy.

```bash
export GENPA_LIBERO_QWEN_CACHE="$PWD/artifacts/cache/libero"
python scripts/libero/precompute_qwen_cache.py \
  --prompt-manifest configs/libero/prompts.json \
  --cache-dir "$GENPA_LIBERO_QWEN_CACHE" --model "$GENPA_QWEN3" \
  --flux2-src third_party/flux2
python scripts/libero/validate_qwen_cache.py \
  --prompt-manifest configs/libero/prompts.json --cache-dir "$GENPA_LIBERO_QWEN_CACHE"
```

Set the paths to your installed benchmark and the downloaded checkpoint. The default launch uses one GPU with its simulator and policy in the same worker.

```bash
export GENPA_LIBERO_CKPT="$PWD/artifacts/models/genpa_libero_depth.pt"
export GENPA_LIBERO_QWEN_CACHE="$PWD/artifacts/cache/libero"
export GENPA_LIBERO_BENCHMARK=/path/to/LIBERO
export GENPA_LIBERO_CONFIG_PATH=/path/to/libero-config
python scripts/libero/setup_benchmark_config.py \
  --benchmark-root "$GENPA_LIBERO_BENCHMARK" --config-dir "$GENPA_LIBERO_CONFIG_PATH"
export GENPA_EVAL_OUTPUT="$PWD/outputs/libero-eval"
bash scripts/libero/evaluate.sh standard
```

Standard evaluation uses 50 trials per task across the four suites. The policy predicts 16 actions and replans after 12 executed actions. For a short installation check, use one trial and a task file; see [the one-task example](#small-checks).

## LIBERO-Plus

LIBERO-Plus reuses the same weights without fine-tuning or checkpoint selection. Its main protocol removes appended perturbation identifiers from non-language categories, while preserving intended language perturbations:

```bash
export GENPA_LIBERO_CKPT="$PWD/artifacts/models/genpa_libero_depth.pt"
export GENPA_LIBERO_BENCHMARK=/path/to/LIBERO-plus
mkdir -p "$PWD/artifacts/cache/plus"
export GENPA_LIBERO_CONFIG_PATH=/path/to/libero-plus-config
python scripts/libero/setup_benchmark_config.py \
  --benchmark-root "$GENPA_LIBERO_BENCHMARK" --config-dir "$GENPA_LIBERO_CONFIG_PATH"
export PYTHONPATH="$PWD/src:$GENPA_LIBERO_BENCHMARK:third_party/flux2/src"
export LIBERO_CONFIG_PATH="$GENPA_LIBERO_CONFIG_PATH"
python scripts/libero/build_clean_instruction_manifest.py \
  --instruction-manifest "$PWD/artifacts/cache/plus/clean-instructions.json" \
  --prompt-manifest "$PWD/artifacts/cache/plus/clean-prompts.json" \
  --released-prompt-manifest "$PWD/artifacts/cache/plus/released-prompts.json" \
  --sample-ratio 1.0 --expected-task-count 10030
export GENPA_CLEAN_INSTRUCTION_MANIFEST="$PWD/artifacts/cache/plus/clean-instructions.json"
export GENPA_LIBERO_QWEN_CACHE="$PWD/artifacts/cache/plus-clean"
python scripts/libero/precompute_qwen_cache.py \
  --prompt-manifest "$PWD/artifacts/cache/plus/clean-prompts.json" \
  --cache-dir "$GENPA_LIBERO_QWEN_CACHE" --model "$GENPA_QWEN3" \
  --flux2-src third_party/flux2
python scripts/libero/validate_qwen_cache.py \
  --prompt-manifest "$PWD/artifacts/cache/plus/clean-prompts.json" --cache-dir "$GENPA_LIBERO_QWEN_CACHE"
export GENPA_EVAL_OUTPUT="$PWD/outputs/plus-clean"
bash scripts/libero/evaluate.sh plus --chunk-size 20
```

LIBERO-Plus uses one trial on each of 10,030 instances. Aggregate by total successes divided by total instances; do not average the seven category percentages. The launcher prints the resolved results directory. To check full coverage and produce the seven category scores, run:

```bash
python scripts/libero/summarize_plus.py --results /path/to/resolved-results-directory \
  --classification "$GENPA_LIBERO_BENCHMARK/libero/libero/benchmark/task_classification.json"
```

Use `--allow-partial` only for progress summaries; a complete report requires all 10,030 unique instances.

For released-instruction evaluation, build a separate cache from the unmodified prompts exported above and use a separate results directory:

```bash
export GENPA_LIBERO_QWEN_CACHE="$PWD/artifacts/cache/plus-official"
python scripts/libero/precompute_qwen_cache.py \
  --prompt-manifest "$PWD/artifacts/cache/plus/released-prompts.json" \
  --cache-dir "$GENPA_LIBERO_QWEN_CACHE" --model "$GENPA_QWEN3" \
  --flux2-src third_party/flux2
python scripts/libero/validate_qwen_cache.py \
  --prompt-manifest "$PWD/artifacts/cache/plus/released-prompts.json" --cache-dir "$GENPA_LIBERO_QWEN_CACHE"
export GENPA_EVAL_OUTPUT="$PWD/outputs/plus-official"
bash scripts/libero/evaluate.sh plus_released --chunk-size 20
```

## RoboTwin 2.0

From the repository root, in the RoboTwin simulator environment:

```bash
export GENPA_ROBOTWIN_BENCHMARK=/path/to/RoboTwin
export GENPA_CUROBO_SRC=/path/to/curobo/src
export GENPA_ROBOTWIN_CKPT=/path/to/genpa_robotwin_depth.pt
export GENPA_EVAL_OUTPUT=/path/to/new-robotwin-evaluation
bash scripts/robotwin/evaluate.sh --gpus 0,1,2,3,4,5,6,7 --trials 100
```

This evaluates the 50 tasks under clean and randomized conditions. Each call predicts 16 absolute joint-action targets, executed in full before replanning. Text is encoded when the instruction changes and reused within the episode. Scene generation and success criteria use the pinned upstream evaluator. A failed worker is recorded as an incomplete run rather than silently counted as a success.

## One worker contains the policy and simulator

Each worker owns one policy copy and one active `OffScreenRenderEnv`. CUDA inference and EGL rendering use the same physical GPU. There is no distributed model collective and no mandatory rendering-only GPU.

```bash
# Single GPU; useful as an installation check.
bash scripts/libero/evaluate.sh standard --gpus 0 --workers-per-gpu 1

# Two, four, or eight GPUs use the same entry point.
bash scripts/libero/evaluate.sh standard --gpus 0,1 --workers-per-gpu 4
bash scripts/libero/evaluate.sh standard --gpus 0,1,2,3 --workers-per-gpu 4
bash scripts/libero/evaluate.sh standard --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4
```

These are alternative launches, not commands to run together. The example worker count is a configurable concurrency setting, not a memory guarantee for every GPU. Start with one worker, then measure complete episodes at increasing concurrency. More workers help only while total throughput improves and memory retains sufficient headroom. The exact-prompt cache avoids keeping the text encoder in every worker.

The launcher assigns tasks across the selected GPU IDs, limits active workers on each GPU, and reuses free slots as processes finish. Physical GPU IDs are used for both `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`; inside each worker PyTorch sees its assigned GPU as `cuda:0`.

## Task grouping and reproducibility

`--chunk-size` is the number of consecutive tasks sharing one loaded model. Every task still creates and closes its own simulator. Standard evaluation defaults to one task per worker process and 50 trials per task. Both Plus modes default to chunks of 20 consecutive tasks and one trial per task, matching the grouping of the original full LIBERO-Plus Clean run:

```bash
bash scripts/libero/evaluate.sh plus --gpus 0,1 --workers-per-gpu 4 --chunk-size 20
```

Keep the task list, task order, chunk size, seed, initial states and instruction protocol fixed when comparing results. The evaluator seeds each worker once; changing task grouping can change the random state before a later environment is constructed. Increasing GPU count or workers changes where chunks run while retaining their task order and contents. Cross-GPU/driver closed-loop results are not guaranteed to be bitwise identical.

## Small checks

Create a UTF-8 task file with one `suite,task_id` per line. Task IDs are zero-based. For example:

```text
libero_spatial,0
libero_10,0
```

```bash
bash scripts/libero/evaluate.sh standard --gpus 0 --task-file /path/to/smoke.txt --trials 1
```

Duplicate, unknown, or out-of-range tasks are rejected before workers launch. A successfully completed subset is reported as complete for that subset, with `formal: false`; it is not a full benchmark result.

## Results and interruption

The manager prints the resolved output directory (including the checkpoint name). It contains the resolved configuration, canonical task list, per-task JSON, worker logs, `parallel_launch.json`, and a validated summary. Formal standard LIBERO requires 40 tasks × 50 trials; formal Plus requires all 10,030 instances × one trial.

A failed worker makes the command fail. Partial files remain available for diagnosis and are never presented as a complete benchmark. The launcher refuses to reuse a directory containing a launch manifest or task results. It does not silently resume, retry failed outcomes, or overwrite them. Interrupting the launcher terminates only the process groups it created.

Hydra overrides remain supported after these options. `bash scripts/libero/evaluate.sh --help` lists the common flags and works without model assets.

## Measured H100 configurations

The following runs used H100 80 GB GPUs and cached Qwen features:

| Protocol | GPUs × workers per GPU | Wall time including startup | Maximum observed memory per GPU |
| --- | ---: | ---: | ---: |
| Standard, 2,000 episodes | 8 × 5 | 10 min 22 s | 47.47 GiB |
| Plus Clean, 10,030 instances | 8 × 8 | 1 h 25 min 18 s | 75.48 GiB |

Standard has only 40 task processes with the original grouping, so five workers on each of eight GPUs launch the entire evaluation at once. Extra slots would not add concurrency. Eight Plus workers per H100 were validated for this checkpoint and cached-text setup; start with fewer on a smaller card. Memory figures include the model and renderer and depend on software and workload.

Some Plus Sensor Noise cases remain CPU-bound. The pinned benchmark's `glass_blur` implementation performs nested Python pixel swaps on every simulation step for noise IDs 41–50. Those instances can run much more slowly than other tasks even when GPU memory is occupied.

[Success rates from these runs](RESULTS.md#libero-re-evaluation)
