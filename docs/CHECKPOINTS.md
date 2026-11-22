# Checkpoints

The [GenPA model repository](https://huggingface.co/Deduktive/GenPA) contains three inference checkpoints trained with seed 42. Each file uses the training configuration and action statistics listed below.

| File | Selected step | Adapter width | Use |
|---|---:|---:|---|
| `genpa_libero_depth.pt` | 32,000 | 512 | LIBERO and LIBERO-Plus |
| `genpa_libero_action_only.pt` | 28,000 | 512 | LIBERO action-only comparison |
| `genpa_robotwin_depth.pt` | 64,000 | 1,024 | RoboTwin Clean and Randomized |

The [LIBERO depth](../configs/libero/depth.yaml) and [action-only](../configs/libero/action_only.yaml) configurations share [LIBERO statistics](../configs/libero/dataset_stats.json). The [RoboTwin depth configuration](../configs/robotwin/depth.yaml) uses [RoboTwin absolute-action statistics](../configs/robotwin/absolute_stats.json).

The files carry the fine-tuned transformer, action prefix and regression adapter in the `genpa` state dictionary. Training resumption also requires optimizer and scheduler state. The Qwen encoder and VAE come from their upstream releases.

```bash
hf download Deduktive/GenPA genpa_libero_depth.pt --local-dir ./artifacts/models
python tools/smoke_policy.py --benchmark libero --variant depth \
  --checkpoint ./artifacts/models/genpa_libero_depth.pt
```

Use `--variant action_only` with the corresponding LIBERO file, or `--benchmark robotwin` with the RoboTwin file. See [the setup](SETUP.md#run-a-checkpoint) for external model paths. An example prediction validates the interface. [Closed-loop evaluation](EVALUATION.md) measures task success.
