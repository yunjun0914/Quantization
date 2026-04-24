#!/bin/bash

# venv 활성화
source ~/yunjun_env/bin/activate

# PyTorch 먼저 설치 (CUDA 12.1 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# AutoGPTQ 설치
cd ~/Omni-Quant/AutoGPTQ-bugfix && pip install -v . && cd ~/Omni-Quant

# OmniQuant 설치
cd ~/Omni-Quant/OmniQuant && pip install -e . && cd ~/Omni-Quant

# requirements 설치
pip install -r ~/Omni-Quant/requirements_linux.txt

# huggingface_hub 설치
pip install huggingface_hub

# calibration 데이터 다운로드
cd ~/Omni-Quant/OmniQuant
git lfs install
git clone https://huggingface.co/ChenMnZ/act_shifts
git clone https://huggingface.co/ChenMnZ/act_scales

# OPT-125m 다운로드
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/opt-125m', local_dir='./models/opt-125m')
"

echo "Setup done!"