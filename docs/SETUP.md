# Environment and model assets

## Run a checkpoint

Start on Linux with an NVIDIA CUDA GPU and Python 3.11. Use the PyTorch wheel that matches your driver. The configuration below uses the paper's PyTorch 2.7.1 and CUDA 11.8 environment.

```bash
git clone https://github.com/deng-zian/GenPA.git
cd GenPA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e '.[inference]'
```

Download the original FLUX.2 base transformer and the compatible image autoencoder. The autoencoder comes from a gated repository. Accept its conditions with your own Hugging Face account and sign in with `hf auth login` before downloading it. [Base transformer](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) · [Autoencoder access](https://huggingface.co/black-forest-labs/FLUX.2-dev)

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors --local-dir artifacts/flux
hf download black-forest-labs/FLUX.2-dev ae.safetensors --local-dir artifacts/flux
export GENPA_FLUX2_WEIGHTS="$PWD/artifacts/flux/flux-2-klein-base-4b.safetensors"
export GENPA_FLUX2_AE="$PWD/artifacts/flux/ae.safetensors"
```

If these files are already on disk, point the two variables at your copies. Keep this shell open for the [checkpoint example](../README.md#run-a-checkpoint). Its cached instruction lets you run it before downloading a text encoder.

## Training

Use the environment and two FLUX paths above. Install the training stack and download Qwen3-4B to encode instructions. These commands run from the GenPA root. The same upstream files are used for benchmark evaluation. [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

```bash
python -m pip install -r requirements.txt
hf download Qwen/Qwen3-4B --include '*.json' '*.safetensors' '*.txt' '*.jinja' --local-dir artifacts/Qwen3-4B
export GENPA_QWEN3="$PWD/artifacts/Qwen3-4B"
```

The training recipes use Accelerate 1.12.0, DeepSpeed 0.18.5 and ZeRO-1. The defaults use eight H100 80 GB GPUs. When adapting batch size for smaller hardware, preserve the global batch size and update count for comparisons with the paper. Set `GENPA_ACCELERATE_CONFIG` to supply your own Accelerate configuration.

For RoboTwin training, install `torchcodec==0.5` in this environment and provide compatible system FFmpeg libraries. Follow the [training recipes](TRAINING.md) for data preparation, text caching and optimization.

## Standard LIBERO / LIBERO-Plus

Install the benchmark's dependencies, assets and BDDL/task files following the instructions for [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) or [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus). [BENCHMARKS.md](BENCHMARKS.md) records the benchmark versions and the Plus compatibility patch. The verified LIBERO runtime uses MuJoCo 3.3.2 and robosuite 1.4.0. Set `GENPA_LIBERO_BENCHMARK` to the checkout root and `GENPA_LIBERO_CONFIG_PATH` to its LIBERO configuration directory. The configuration must refer to that checkout's task and init-state files; use separate configuration directories for standard LIBERO and LIBERO-Plus.

Headless evaluation defaults to EGL. Set `MUJOCO_GL` and `PYOPENGL_PLATFORM` to another supported backend if required by the host. Evaluation uses independent single-GPU workers, not distributed data-parallel inference.

## RoboTwin

The verified simulator environment uses Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, SAPIEN 3.0.0b1, mplib 0.2.1 and SciPy 1.15.3. Use RoboTwin's simulator environment and install its assets, planner dependencies and CuRobo following the [RoboTwin instructions](https://github.com/RoboTwin-Platform/RoboTwin). `GENPA_ROBOTWIN_BENCHMARK` points to its checkout; `GENPA_CUROBO_SRC` points to the directory containing the importable `curobo` package. The verified CuRobo source revision is `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`. CUDA extension compilation must match the PyTorch/CUDA versions of this environment.

Install the GenPA inference dependencies in this simulator environment before evaluation:

```bash
pip install transformers==4.56.1 safetensors==0.5.3 einops==0.8.1 \
  hydra-core==1.3.2 omegaconf==2.3.0 huggingface-hub==0.36.2 \
  sentencepiece==0.2.1 rich PyYAML
pip install -e .
```

The supplied evaluator copies the task configuration into the evaluation output directory and invokes the pinned upstream evaluator in `third_party/robotwin/`. It does not edit the benchmark checkout. Set an unused output directory for each new evaluation, or use `--resume` only for the identical configuration.

## Model interface

`GENPA_FLUX2_WEIGHTS` is the 4B **base** DiT weight file, `GENPA_FLUX2_AE` is the compatible FLUX.2 autoencoder, and `GENPA_QWEN3` is Qwen3-4B with its tokenizer. Qwen features concatenate three hidden layers into width 7,680 with 128 positions. The learned 16-token Action Prefix is inserted by the policy; it must not be precomputed into the text cache.

Weights, pretrained assets and generated text caches remain outside the code directory. The frozen Qwen encoder and VAE are shared across the provided policy configurations.
