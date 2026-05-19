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
python -m pip install -r requirements/dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

The PyTorch packages are installed from Aliyun's CUDA 12.6 wheel mirror:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://mirrors.aliyun.com/pytorch-wheels/cu126_full/
```

## Validate

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest
python -m ruff check .
```
