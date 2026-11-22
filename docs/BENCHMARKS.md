# LIBERO benchmark setup

For standard LIBERO, install [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and its assets. The reported setup used MuJoCo 3.3.2 and robosuite 1.4.0. The original checkout's Git revision is unknown.

For LIBERO-Plus, install [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182). From the GenPA repository root, set `GENPA_LIBERO_BENCHMARK` to the location of the Plus checkout, then run the following once.

```bash
git -C "$GENPA_LIBERO_BENCHMARK" checkout 4976dc30028e805ff8094b55501d532c48fec182
git -C "$GENPA_LIBERO_BENCHMARK" apply "$PWD/patches/libero-plus-pytorch-path-compat.patch"
touch "$GENPA_LIBERO_BENCHMARK/libero/__init__.py"
```

The patch loads the benchmark's released initial states with PyTorch 2.7 and accepts path objects for BDDL files. The package marker enables Python imports. Next, [configure the benchmark paths](SETUP.md#standard-libero--libero-plus) and [prepare the Clean or Official instructions](EVALUATION.md#libero-plus).
