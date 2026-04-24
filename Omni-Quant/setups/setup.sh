#!/bin/bash

# venv 활성화
source ~/yunjun_env/bin/activate

# PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# AutoGPTQ 설치
cd ~/Quantization/Omni-Quant/AutoGPTQ-bugfix && pip install -v .

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
snapshot_download(repo_id='facebook/opt-125m', local_dir='~/Quantization/Omni-Quant/OmniQuant/models/opt-125m')
"

echo "Setup done!"