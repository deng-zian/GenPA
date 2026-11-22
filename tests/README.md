# Result integrity tests

Run from the repository root:

```bash
pip install -e .
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

The result-integrity tests use Python's standard library only. They create temporary synthetic
result files and require no GPU, model weights, datasets, or simulator. The process-adapter
tests additionally require NumPy; they report an explicit skip when it is unavailable.

Coverage includes complete standard LIBERO and LIBERO-Plus evaluations,
instance-pooled aggregation, explicit partial reports, scoped smoke checks,
invalid or duplicate task/episode identities, malformed counts and JSON,
and mixed benchmark, instruction-protocol, seed, or checkpoint metadata.
The full-standard command is also checked against a narrowed task manifest.

These are artifact-integrity checks, not policy-performance or simulator tests.

Launcher tests start real subprocesses with a small CPU-only evaluator fixture. They check 1/2/4/8 GPU assignments, sparse physical GPU indices, CUDA/EGL device agreement, per-GPU concurrency limits, quoted arguments, failure reporting, cancellation, and refusal to overwrite results. They validate process management; they do not establish throughput or correctness on physical GPUs.

Process-adapter tests use real spawned CPU children with a small fake simulator backend. They check RPC values and lifecycle, Python/NumPy RNG handoff across two environments, child exception propagation, and cleanup after normal and abnormal exits. They do not establish CUDA/EGL or physical-simulator equivalence.

Repository-layout tests require the lightweight package dependencies installed by
`pip install -e .` (Hydra and PyYAML). They verify editable imports from outside the
checkout, composed configs, nested launcher paths and the relocated RoboTwin dry-run.
The shell tests intercept Python/Accelerate commands before any model is loaded.
The suite has 40 tests when NumPy and the package dependencies are available.
