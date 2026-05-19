# Server Environment

Target server environment:

- Python: 3.11
- CUDA runtime target for PyTorch wheels: 12.6
- Conda environment name: `tinyllm`

Do not store SSH passwords, tokens, or other credentials in this repository.

## Install

From the repository root on the server:

```powershell
conda env create -f requirements/server-conda.yml
conda activate tinyllm
python -m pip install -r requirements/torch-cu126.txt
python -m pip install -r requirements/dev.txt
```

The PyTorch packages are installed from the official CUDA 12.6 wheel index:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

## Validate

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest
python -m ruff check .
```
