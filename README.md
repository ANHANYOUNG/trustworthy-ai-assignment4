trustworthy-ai-assignment4
==========================

Assignment 4 implementation for neural network verification with
alpha-beta-CROWN.

This repository verifies an external ONNX model, the MNIST wide MLP from
Assignment 3, under local L-infinity robustness specifications. The same model
and sample were previously checked with Marabou, so the results include a
direct alpha-beta-CROWN vs. Marabou comparison.

Environment
-----------

Tested environment:

- Conda environment: `trustworthy-ai-a4`
- Python: 3.11
- PyTorch: 2.8.0+cu128
- CUDA runtime in PyTorch: 12.8
- GPU used for the recorded run: NVIDIA A10
- alpha-beta-CROWN: local editable install from `./alpha-beta-CROWN`
- auto_LiRPA: local editable install from `./alpha-beta-CROWN/auto_LiRPA`

A CUDA-capable GPU is expected because the provided YAML files set
`general.device: cuda`.

Setup
-----

Clone or initialize alpha-beta-CROWN in the repository root:

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

If `alpha-beta-CROWN/` already exists, make sure its submodules are initialized:

```bash
git -C alpha-beta-CROWN submodule update --init --recursive
```

Create the conda environment from the official alpha-beta-CROWN environment
file:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -y -n trustworthy-ai-a4 \
  -f alpha-beta-CROWN/complete_verifier/environment_pyt280.yaml
```

Activate the environment and install the local verifier packages:

```bash
conda activate trustworthy-ai-a4
pip install --no-deps -e alpha-beta-CROWN/auto_LiRPA
pip install --no-deps -e alpha-beta-CROWN
```

Install from `requirements.txt` only if you need to recreate the package set
manually:

```bash
pip install -r requirements.txt
```

Environment Checks
------------------

Check CUDA and key packages:

```bash
conda activate trustworthy-ai-a4
python -c "import torch, auto_LiRPA, onnx, onnxruntime; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); \
print(auto_LiRPA.__version__, onnx.__version__, onnxruntime.__version__)"
```

Expected package versions from the recorded run:

- `torch 2.8.0+cu128`
- `cuda_available True`
- `auto_LiRPA 0.7.1`
- `onnx 1.19.1`
- `onnxruntime 1.23.2`

Check the alpha-beta-CROWN CLI:

```bash
conda activate trustworthy-ai-a4
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --help
cd ../..
```

Optional built-in example check:

```bash
conda activate trustworthy-ai-a4
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config exp_configs/tutorial_examples/custom_specs.yaml
cd ../..
```

The recorded built-in example result was `safe-incomplete` with final verified
accuracy `100.0%`.

Dependency check:

```bash
conda activate trustworthy-ai-a4
pip check
```

The recorded result was `No broken requirements found.`

Project Layout
--------------

- `prep/prepare_a3_assets.py`: copies the Assignment 3 ONNX model, metadata,
  sample, and Marabou baseline results into this repository.
- `prep/make_mnist_wide_vnnlib.py`: generates alpha-beta-CROWN VNNLIB specs
  for epsilons `1`, `5`, and `10`.
- `configs/`: alpha-beta-CROWN YAML files for the three epsilon values.
- `models/`: copied Assignment 3 MNIST-wide ONNX model and metadata.
- `data/`: copied Assignment 3 MNIST sample.
- `specs/mnist_wide/`: generated VNNLIB robustness specifications.
- `results/marabou_baseline/`: copied Assignment 3 Marabou results.
- `results/abcrown/`: alpha-beta-CROWN raw logs.
- `results/abcrown_summary.csv`: combined alpha-beta-CROWN vs. Marabou summary.
- `test.py`: reproducible runner for all three alpha-beta-CROWN experiments.

The submitted repository includes the prepared model, sample, VNNLIB specs, and
baseline result files. The `prep/` scripts are kept so the assets can be
regenerated from a local Assignment 3 checkout if needed.

Asset Preparation
-----------------

If the prepared files already exist, this step is optional. To regenerate them,
clone the Assignment 3 repository inside this project directory:

```bash
git clone https://github.com/ANHANYOUNG/trustworthy-ai-assignment3.git
```

The default preparation script expects the Assignment 3 repository at
`./trustworthy-ai-assignment3`. If the repository is somewhere else, pass its
path with `--assignment3-root`:

```bash
conda activate trustworthy-ai-a4
python prep/prepare_a3_assets.py
python prep/make_mnist_wide_vnnlib.py
```

Alternative path example:

```bash
python prep/prepare_a3_assets.py --assignment3-root /path/to/trustworthy-ai-assignment3
python prep/make_mnist_wide_vnnlib.py
```

Run Verification Experiments
----------------------------

Run all alpha-beta-CROWN experiments:

```bash
conda activate trustworthy-ai-a4
python test.py
```

`test.py` runs:

- `configs/mnist_wide_eps_1.yaml`
- `configs/mnist_wide_eps_5.yaml`
- `configs/mnist_wide_eps_10.yaml`

It writes raw logs to `results/abcrown/<experiment>/output.txt` and writes the
combined summary to `results/abcrown_summary.csv`.

Recorded Results
----------------

| Epsilon | alpha-beta-CROWN status | alpha-beta-CROWN wall time | Marabou baseline |
| --- | --- | ---: | --- |
| `1.0` | `verified` (`Result: unsat`) | `7.43s` | `UNSAT`, `1.66s` |
| `5.0` | `verified` (`Result: unsat`) | `7.56s` | `UNSAT`, `9.43s` |
| `10.0` | `falsified` (`Result: sat`) | `6.81s` | `SAT+TIMEOUT`, `145.86s` |

Interpretation:

For this MNIST-wide model and sample, alpha-beta-CROWN proves robustness for
pixel-space perturbations of epsilon `1` and `5`. At epsilon `10`, PGD finds a
counterexample, so the local robustness property is falsified. This is
consistent with the Assignment 3 Marabou baseline at epsilon `10`, where at
least one adversarial target was satisfiable while other target checks timed
out.
