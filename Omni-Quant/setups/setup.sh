#!/bin/bash

srun --gres=gpu:1 --mem=32G bash << 'EOF'

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/yunjun_omni/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging ninja

# PyTorch 설치
python -m pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# PyTorch import 확인
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY

# AutoGPTQ 설치
cd ~/Quantization/Omni-Quant/AutoGPTQ-bugfix

python -m pip install -v . --no-build-isolation

# OmniQuant 설치
cd ~/Quantization/Omni-Quant/OmniQuant
python -m pip install -e .

# requirements 설치
python -m pip install -r ~/Quantization/Omni-Quant/requirements_linux.txt

# Hugging Face 관련 설치
python -m pip install -U huggingface_hub

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
python - << 'PY'
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