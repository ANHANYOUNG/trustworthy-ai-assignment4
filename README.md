trustworthy-ai-assignment4
==========================

Environment setup for Assignment #4: neural network verification with
alpha-beta-CROWN.

Current Status
--------------

The conda environment setup has been completed on the Slurm GPU allocation
`674695` using `srun --overlap`.

Environment:

- Conda environment: `trustworthy-ai-a4`
- Python: 3.11
- PyTorch: 2.8.0+cu128
- CUDA runtime in PyTorch: 12.8
- Verified GPU node/device: `n015`, NVIDIA A10
- alpha-beta-CROWN: editable install from `./alpha-beta-CROWN`
- auto_LiRPA: editable install from `./alpha-beta-CROWN/auto_LiRPA`

Setup Commands
--------------

Initialize the alpha-beta-CROWN submodule:

```bash
git -C alpha-beta-CROWN submodule update --init --recursive
```

Create the conda environment from the official alpha-beta-CROWN environment
file:

```bash
srun --overlap --jobid=674695 bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -y -n trustworthy-ai-a4 \
  -f alpha-beta-CROWN/complete_verifier/environment_pyt280.yaml
'
```

Install the local verifier packages in editable mode:

```bash
srun --overlap --jobid=674695 bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trustworthy-ai-a4
pip install --no-deps -e alpha-beta-CROWN/auto_LiRPA
pip install --no-deps -e alpha-beta-CROWN
'
```

Activate the environment:

```bash
conda activate trustworthy-ai-a4
```

Environment Checks
------------------

Check CUDA and key packages:

```bash
srun --overlap --jobid=674695 bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trustworthy-ai-a4
python -c "import torch, auto_LiRPA, onnx, onnxruntime; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0)); \
print(auto_LiRPA.__version__, onnx.__version__, onnxruntime.__version__)"
'
```

Observed result:

- `torch 2.8.0+cu128`
- `cuda_available True`
- `device NVIDIA A10`
- `auto_LiRPA 0.7.1`
- `onnx 1.19.1`
- `onnxruntime 1.23.2`

alpha-beta-CROWN CLI check:

```bash
srun --overlap --jobid=674695 bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trustworthy-ai-a4
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --help
'
```

Provided Example Check
----------------------

A small built-in tutorial configuration was run successfully:

```bash
srun --overlap --jobid=674695 bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trustworthy-ai-a4
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config exp_configs/tutorial_examples/custom_specs.yaml
'
```

Observed result:

- Result: `safe-incomplete`
- Final verified accuracy: `100.0%`
- Total examples: `1`
- Total verified: `1`
- Total falsified: `0`
- Timeout: `0`
- Runtime: about `0.8` to `0.9` seconds

Dependency check:

```bash
srun --overlap --jobid=674695 bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trustworthy-ai-a4
pip check
'
```

Observed result:

- `No broken requirements found.`

Assignment Implementation Notes
-------------------------------

The environment setup is complete. The assignment-specific external model,
dataset sample, YAML configuration, `test.py`, and final report will be added
in the implementation stage.

Expected final deliverables:

- `requirements.txt` or environment file
- `test.py`
- YAML configuration file(s)
- `report.pdf`
- `README.md`
