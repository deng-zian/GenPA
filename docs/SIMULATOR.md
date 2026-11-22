# Simulator execution

Each LIBERO worker runs one policy and one simulator on the same GPU. Use this default on a single GPU or distribute workers across several GPUs. [GPU and worker settings](EVALUATION.md#one-worker-contains-the-policy-and-simulator)

## Optional separate rendering GPUs

On hosts where EGL readback stalls when the simulator and policy share a GPU, you can assign rendering to another GPU. Set the renderer options as in the following example.

```bash
export GENPA_LIBERO_SIM_PROCESS=1
export GENPA_LIBERO_RENDER_GPU_MAP=0:4,1:5,2:6,3:7
export WORKER_OMP_NUM_THREADS=1 WORKER_MKL_NUM_THREADS=1
bash scripts/libero/evaluate.sh standard --gpus 0,1,2,3 --workers-per-gpu 1
```

The policy process receives the observations from the simulator process. Both use the same task, initial state and camera settings as the default launcher.
