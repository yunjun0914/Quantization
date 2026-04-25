#!/bin/bash

# GPU 할당
srun --gres=gpu:1 --mem=32G --pty bash << 'EOF'

# CUDA 환경변수 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# venv 활성화
source ~/yunjun_env/bin/activate

# PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# AutoGPTQ 설치
cd ~/Quantization/Omni-Quant/AutoGPTQ-bugfix && pip install -v . --no-build-isolation

# OmniQuant 설치
cd ~/Quantization/Omni-Quant/OmniQuant && pip install -e .

# requirements 설치
pip install -r ~/Quantization/Omni-Quant/requirements_linux.txt

# huggingface_hub 설치
pip install huggingface_hub

# calibration 데이터 다운로드
cd ~/Quantization/Omni-Quant/OmniQuant
git lfs install
git clone https://huggingface.co/ChenMnZ/act_shifts
git clone https://huggingface.co/ChenMnZ/act_scales

# OPT-125m 다운로드
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/opt-125m', local_dir='./models/opt-125m')
"

# LLaMA-1 7B 다운로드 (공개 변환본)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='huggyllama/llama-7b', local_dir='./models/llama-7b')
"

# LLaMA-2 7B 다운로드 (Meta 라이선스 동의 + HF 토큰 필요)
# 사전 준비: huggingface-cli login 또는 HF_TOKEN 환경변수 설정
# https://huggingface.co/meta-llama/Llama-2-7b-hf 에서 라이선스 동의 필요
python -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN')
snapshot_download(repo_id='meta-llama/Llama-2-7b-hf', local_dir='./models/llama2-7b', token=token)
"

echo "Setup done!"
EOF