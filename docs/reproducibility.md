# Reproducibility notes

## Environment used for the reported run
- **OS:** Windows 11. **GPU:** NVIDIA RTX 5070 Laptop (8 GB, sm_120), driver supporting CUDA 13.1.
- **Python 3.12** via `uv`. **torch:** 2.14.1+cu130. **Lucent:** pinned to commit
  `2ccc434d80097e58f53cb2a70c60f72902fe93af`.
- Exact versions of every dependency are in `uv.lock`.
- **Wall-clock (confirmatory run, `logs/20261023_171349.log`):**
  - start-up (model load) plus E1: 13 s;
  - E2: 40.6 min;
  - E3: 15.7 min;
  - total: 56.5 min.
- **Post-hoc phase** (`logs/20261023_184009.log`): 22.3 min.
- **Audit phase** (`logs/20261023_210800.log`): 4.7 min.
- **Audit2 phase** (`logs/20261023_230809.log`): 4.0 min.
- **Provenance of `results/e1_replication.json`.** It was regenerated at 19:19 by a deterministic E1 rerun that adds
  the Lucent-LRN check (`logs/20261023_191858.log`). Its other numbers are identical to the confirmatory log.

## Determinism and seeds
- **Stimuli are deterministic given the jitter seed.**

  | Seed | Purpose |
  |---|---|
  | 0, 100 | pilot, discarded |
  | 1 | selection |
  | 2 | confirmatory |
  | 3, 4 | stability |
  | 101–104 | `--quick` smoke tests, which never touch confirmatory stimuli |

- **Minimal pairs** use `seed + 10000`.
- **Null neuron sets and bootstraps** draw from one `numpy.random.default_rng(confirm_seed)` stream in a fixed
  phase order. Changing the order of analyses therefore changes which null sets are drawn, but not their
  distribution.
  - A single phase (`--phase e2` or `--phase e3`) starts the stream fresh, so its null sets differ from those of the
    full run.
  - Only `uv run python main.py` reproduces the reported E2/E3 numbers exactly.
  - E1 runs first, so it is reproduced either way.
- **TF32 is disabled** (`torch.backends.cudnn.allow_tf32 = False`). cuDNN may still pick different fp32 convolution
  algorithms for different input sizes.
  - In fp32, full-map and cropped-window evaluation differ by up to about 3e-3 absolute, about 1e-5 of the value
    range.
  - In fp64 they are identical. A re-check measured a difference of 0, and
    `tests/test_model.py::test_cropped_readout_equals_full_map` asserts ≤ 1e-9.
  - Clean and patched readouts always use the same cropped code path.

## Data provenance
- **`data/circuits/*.json`:** copied from
  [distillpub/post--circuits-curve-detectors](https://github.com/distillpub/post--circuits-curve-detectors)
  (CC-BY 4.0). See `data/circuits/README.md`.
- **Imagenette-320:** downloaded from the fast.ai S3 mirror on the first E3 run and extracted to `data/imagenette/`,
  which is gitignored. Only relative file names of the confirmatory images are stored in
  `results/e3_natural.json`. The images are never redistributed, because Imagenette is an ImageNet subset.
- **Model weights:** `inception5h.pth` comes from
  [ProGamerGov/pytorch-old-tensorflow-models](https://github.com/ProGamerGov/pytorch-old-tensorflow-models) and is
  cached by `torch.hub`.

## Known platform issues and their fixes
- **Lucent's `setup.py` fails on Windows.** It reads its README with the cp1252 codec. `pyproject.toml` sets
  `[tool.uv] extra-build-variables = { torch-lucent = { PYTHONUTF8 = "1" } }`.
- **The PyPI `torch-lucent 0.1.8` pins `kornia<=0.4.1`.** We install Lucent from git (dev branch) instead.
- **On an 8 GB WDDM GPU, the PyTorch caching allocator can push total use past dedicated memory.** Windows then pages
  to shared memory and kernels slow down by orders of magnitude. `interventions.collect` calls
  `torch.cuda.empty_cache()`, and the full-network batch size defaults to 64 (`BATCH_SIZE` in `.env`).
