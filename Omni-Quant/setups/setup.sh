#!/bin/bash

srun --gres=gpu:1 --mem=32G bash << 'EOF'

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
unset LD_LIBRARY_PATH

source ~/yunjun_omni/bin/activate

python3 -m pip install --upgrade pip wheel packaging ninja
python3 -m pip install "setuptools==70.2.0"

# PyTorch 설치: CUDA 13.0
python3 -m pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# PyTorch import 확인
python3 - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# requirements 설치
python3 -m pip install -r ~/Quantization/Omni-Quant/requirements_linux.txt

# AutoGPTQ 설치
cd ~/Quantization/Omni-Quant/AutoGPTQ-bugfix
python3 -m pip install -v . --no-build-isolation

# OmniQuant 설치
cd ~/Quantization/Omni-Quant/OmniQuant
python3 -m pip install -e .

# Hugging Face 관련 설치
python3 -m pip install "huggingface_hub==0.36.0"

# calibration 데이터 다운로드
cd ~/Quantization/Omni-Quant/OmniQuant

git lfs install

if [ ! -d "act_shifts" ]; then
    git clone https://huggingface.co/ChenMnZ/act_shifts
fi

if [ ! -d "act_scales" ]; then
    git clone https://huggingface.co/ChenMnZ/act_scales
fi

# 모델 다운로드
python3 - << 'PY'
from huggingface_hub import snapshot_download

models = {
    "opt-125m": "facebook/opt-125m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",

    "llama1-7b": "huggyllama/llama-7b",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
}

for local_name, repo_id in models.items():
    print(f"\n[DOWNLOAD] {repo_id} -> ./models/{local_name}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=f"./models/{local_name}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )

print("\nAll model downloads completed.")
PY

echo "Setup done!"

EOF