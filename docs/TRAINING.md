# Training

Start in the GenPA checkout with the [training environment and upstream weights](SETUP.md#training). Pick the benchmark below.

## LIBERO

The prepared RGB-D dataset contains 1,808 successful replay demonstrations. From the code root, set its root and the exact-prompt cache location:

```bash
hf download Deduktive/GenPA-LIBERO --repo-type dataset --include 'paper/*' --local-dir ./artifacts/GenPA-LIBERO
export GENPA_LIBERO_DATA="$PWD/artifacts/GenPA-LIBERO/paper"
export GENPA_LIBERO_DEPTH_LABELS="$GENPA_LIBERO_DATA/depth"
export GENPA_LIBERO_QWEN_CACHE="$PWD/artifacts/cache/libero"
python scripts/libero/precompute_qwen_cache.py \
  --prompt-manifest configs/libero/prompts.json \
  --cache-dir "$GENPA_LIBERO_QWEN_CACHE" --model "$GENPA_QWEN3" \
  --flux2-src third_party/flux2
python scripts/libero/validate_qwen_cache.py \
  --prompt-manifest configs/libero/prompts.json --cache-dir "$GENPA_LIBERO_QWEN_CACHE"
```

Choose the training recipe. Each run writes to its own directory.

```bash
export GENPA_OUTPUT_DIR="$PWD/outputs/libero-depth"
bash scripts/libero/train.sh depth
```

For the action-only comparison, run this instead:

```bash
export GENPA_OUTPUT_DIR="$PWD/outputs/libero-action-only"
bash scripts/libero/train.sh action_only
```

The supplied recipe uses eight GPUs, batch size 10 per GPU, one accumulation step, 36,750 updates, BF16, and AdamW. The backbone/adapter learning rates are 1e-5/1e-4. Both variants use the same demonstrations, augmentation and action normalization; the perception loss is disabled in `action_only`. See [the data guide](DATA.md#libero) for the dataset schema and preparation tools. Changing GPU count changes the global batch unless accumulation or per-GPU batch size is adjusted.

## RoboTwin 2.0

Use the RGB-D release at [flex-pi/robotwin_3d](https://huggingface.co/datasets/flex-pi/robotwin_3d). The frozen dataset revision and preparation commands are given in [data preparation](DATA.md#robotwin-20). The recipe mixes the 2,500 clean and 25,000 randomized demonstrations, with the fixed held-out split encoded by the data preparation script. Run the following from the code root after preparing the data and text cache.

```bash
export GENPA_ROBOTWIN_DATA=/path/to/robotwin_3d
export GENPA_ROBOTWIN_NUMERIC=/path/to/robotwin-prepared/numeric
export GENPA_ROBOTWIN_LANGUAGE=/path/to/robotwin-prepared/language
export GENPA_ROBOTWIN_QWEN_0=/path/to/robotwin-text-cache
export GENPA_ROBOTWIN_QWEN_1="$GENPA_ROBOTWIN_QWEN_0"
export GENPA_ROBOTWIN_QWEN_2="$GENPA_ROBOTWIN_QWEN_0"
export GENPA_OUTPUT_DIR=/path/to/new-robotwin-run
bash scripts/robotwin/train.sh
```

The recipe uses eight GPUs, batch size 16 per GPU and three accumulation steps (global batch 384), for 70,847 updates. Backbone and adapter learning rates are both 1e-4, with AdamW and BF16. Depth is a training target; deployed policies receive RGB and proprioception.
