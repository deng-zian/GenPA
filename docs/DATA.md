# Training data

## LIBERO

The LIBERO training interface expects an aligned RGB-D replay dataset. The paper uses 1,808 successfully replayed demonstrations. Download the `paper/` directory from [GenPA-LIBERO](https://huggingface.co/datasets/Deduktive/GenPA-LIBERO) and use it as `GENPA_LIBERO_DATA`. The required fields are RGB, action/state arrays, and current metric depth; other perception targets are not needed.

```text
paper/
  data/chunk-*/episode_*.parquet
  videos/chunk-*/observation.images.image/episode_*.mp4
  videos/chunk-*/observation.images.wrist_image/episode_*.mp4
  depth/chunk-*/episode_*.hdf5
  meta/info.json
  meta/{episodes,tasks,episodes_stats}.jsonl
  manifests/{demonstrations,episode_map}.jsonl
  policy/{action_keep_ranges,dataset_stats}.json
  dataset.json
```

Episode IDs and frame indices are shared by the RGB, action and depth files. Each depth file has `agentview/depth_m` and `robot0_eye_in_hand/depth_m`, storing meters as losslessly preserved source values. `depth_targets.py` clips depth to [0.04, 3.5] m and applies the fixed log-inverse mapping to a three-channel target. Training applies the paired geometric transforms to RGB and depth, and photometric transforms only to RGB. Action-only training uses the same RGB/action subset without opening the depth files.

The included `configs/libero/action_keep_ranges.json` specifies retained action rows. `configs/libero/dataset_stats.json` provides the policy normalization. These statistics correspond to the paper's LIBERO action representation and must match the data used to train the checkpoint.

For an existing aligned replay dataset containing `visual_gt/`, run the following from the repository root to regenerate the portable subset:

```bash
python scripts/libero/export_training_data.py \
  --source /path/to/aligned-replay-dataset \
  --output /path/to/new-libero-rgbd --workers 4
```

The exporter copies RGB, actions and metadata from an aligned training view and extracts metric depth from its visual sidecars. It writes `manifests/demonstrations.jsonl`. The published dataset additionally includes the canonical episode mapping and the policy filters and statistics.

### Visual supervision

The full replay contains 2,000 demonstrations in 40 task archives. The 1,808 demonstrations in the paper training subset supply 295,442 frames and 293,976 action anchors after filtering. Use `paper/manifests/episode_map.jsonl` to match a training episode to its replay.

From the GenPA checkout, install the CPU data reader and download the first Spatial task with its labels. The archive expands under `canonical/libero_spatial/`.

```bash
python -m pip install -e '.[data]'
hf download Deduktive/GenPA-LIBERO --repo-type dataset --local-dir ./artifacts/GenPA-LIBERO \
  --include 'archives/libero_spatial/task_00.tar' 'labels/*' \
            'paper/manifests/episode_map.jsonl' 'paper/policy/action_keep_ranges.json'
tar -xf ./artifacts/GenPA-LIBERO/archives/libero_spatial/task_00.tar -C ./artifacts/GenPA-LIBERO
python scripts/libero/export_visual_targets.py --data ./artifacts/GenPA-LIBERO \
  --episode libero_spatial/task_00/demo_0 --frame 0 --output ./targets.npz
```

Use `--targets normals pointmap segmentation` to choose a subset. The default also emits current depth, grounding, future RGB and future depth. Metric depth is available directly as `paper/depth/chunk-*/episode_*.hdf5`. The main training launcher accepts the depth and action-only recipes. The visual target reader provides aligned labels for additional experiments.

### File formats and episode IDs

The full replay spans 338,575 frames. Each task archive contains 50 demonstrations, with two RGB videos, an action/state Parquet file, a visual HDF5 sidecar, a geometry table and scene XML per demonstration. Use `canonical/episodes.jsonl` to find the task archive for a source episode. The visual HDF5 includes metric depth, camera calibration, the stored-to-raw pixel transform, geometry IDs, simulator state and source frame indices. The scene XML uses `assets/libero/` and `assets/robosuite/` prefixes that resolve against the corresponding benchmark installations.

The paper view numbers its 1,808 successful replays independently. Join it to a full replay with `source_episode_id` in `paper/manifests/episode_map.jsonl`. Numeric episode IDs can designate different demonstrations across the two views. The half-open intervals in `paper/policy/action_keep_ranges.json` select 293,976 action anchors from 295,442 stored frames.

| Target | What the reader returns for each camera |
| --- | --- |
| Depth | The current metric depth, encoded with the fixed 0.04–3.5 m log-inverse mapping |
| Normals | Camera-frame surface normals derived from calibrated depth |
| Pointmap | Robot-base XYZ with the paper's fixed per-axis companding |
| Segmentation | Geometry IDs mapped to a common 37-class palette |
| Grounding | Operation object, goal or anchor, and active part in three binary channels |
| Future RGB and future depth | The frame 16 retained action steps ahead, clamped to the last retained step |

The ready-to-download `samples/*.npz` visual examples also carry the corresponding RGB observations in `current_rgb`. The reader exports the seven labels and their metadata. The source RGB for any frame is in the two camera videos.

The NPZ stores image-form arrays in `[0, 1]`, per-camera validity, pixel masks and the target frame indices. Choose a retained paper anchor for the future targets. The metric depth arrays under `paper/depth/` retain float32 meters. To download those arrays without the RGB videos:

```bash
hf download Deduktive/GenPA-LIBERO --repo-type dataset --local-dir artifacts/GenPA-LIBERO \
  --include 'paper/depth/*' 'paper/manifests/*' 'paper/policy/*'
```

### Re-render an original LIBERO demonstration

Install the [LIBERO simulator](SETUP.md) and download the selected task's demonstration HDF5 from the [original dataset revision](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a). The renderer uses the episode index to locate the matching HDF5 and BDDL file.

```bash
hf download Deduktive/GenPA-LIBERO canonical/episodes.jsonl --repo-type dataset --local-dir ./artifacts/GenPA-LIBERO
python scripts/libero/render_replay.py --index ./artifacts/GenPA-LIBERO/canonical/episodes.jsonl \
  --raw-root /path/to/original-libero-demonstrations --benchmark "$GENPA_LIBERO_BENCHMARK" \
  --episode libero_spatial/task_00/demo_0 --gpu 0 --output ./spatial-demo-replay
```

The result follows the same `canonical/<suite>/` layout as an extracted task archive. Repeat `--episode` for multiple episodes or use `--suite libero_spatial` to process a full suite into a new directory. The [original demonstrations](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) and benchmark assets are obtained separately.

## RoboTwin 2.0

Download [flex-pi/robotwin_3d](https://huggingface.co/datasets/flex-pi/robotwin_3d) at revision `bee164afe94041d8c3d7dd1203725c41163fc3f4`. Preserve its `meta/`, `data/`, `videos/` and `dataset_stats.json` layout. The input contains 27,500 RGB-D demonstrations. The preparation code uses a deterministic 99%/1% demonstration split with seed 42, filters idle rows, and calculates action statistics on the training split.

From the repository root, using the training environment:

```bash
export PYTHONPATH="$PWD/src"
export GENPA_ROBOTWIN_DATA=/path/to/robotwin_3d
export GENPA_ROBOTWIN_NUMERIC=/path/to/robotwin-prepared/numeric
export GENPA_ROBOTWIN_LANGUAGE=/path/to/robotwin-prepared/language
python scripts/robotwin/prepare_numeric.py \
  --root "$GENPA_ROBOTWIN_DATA" --output "$GENPA_ROBOTWIN_NUMERIC" \
  --official-filter scripts/robotwin/compute_nonidle_ranges.py
python scripts/robotwin/prepare_language.py \
  --root "$GENPA_ROBOTWIN_DATA" --numeric "$GENPA_ROBOTWIN_NUMERIC" \
  --output "$GENPA_ROBOTWIN_LANGUAGE"
python scripts/robotwin/prepare_absolute_stats.py \
  --numeric-dir "$GENPA_ROBOTWIN_NUMERIC" --output /path/to/absolute_stats.json
```

The evaluation configuration loads the included `configs/robotwin/absolute_stats.json`. Use `GENPA_ROBOTWIN_STATS` to select the statistics generated for a new training dataset. Its values must correspond to the checkpoint.

The numeric directory contains action/state arrays, retained-row indices, trajectory offsets, and the training/validation references. The language directory stores a per-frame instruction index and a SQLite instruction table. Instructions come from the selected current frame; they are not replaced with one arbitrary instruction per episode.

Build one complete exact-prompt cache:

```bash
python scripts/robotwin/export_qwen_prompts.py \
  --language-dir "$GENPA_ROBOTWIN_LANGUAGE" --output /path/to/prompts.jsonl
torchrun --nproc_per_node=1 scripts/robotwin/precompute_packed_qwen_cache.py \
  --tasks-jsonl /path/to/prompts.jsonl \
  --expected-prompts "$(wc -l < /path/to/prompts.jsonl)" \
  --model "$GENPA_QWEN3" --flux2-src third_party/flux2 \
  --output-dir /path/to/robotwin-text-cache
python scripts/robotwin/validate_packed_qwen_cache.py \
  --tasks-jsonl /path/to/prompts.jsonl --cache-dir /path/to/robotwin-text-cache
```

Assign the resulting directory to all three `GENPA_ROBOTWIN_QWEN_*` variables. Multiple cache directories may also be used; lookup must return the exact instruction. Caches can be large because this dataset has many distinct instructions. A cache is a preprocessing acceleration, not an additional learned model.
