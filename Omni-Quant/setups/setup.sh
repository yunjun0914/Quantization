#!/bin/bash

source /path/to/yunjun_env/bin/activate

cd Quantization

# AutoGPTQ 먼저 설치
cd AutoGPTQ-bugfix && pip install -v . && cd ..

# OmniQuant 설치
cd OmniQuant && pip install -e . && cd ..

# requirements 설치
pip install -r requirements_linux.txt

# calibration 데이터 다운로드
cd OmniQuant
git lfs install
git clone https://huggingface.co/ChenMnZ/act_shifts
git clone https://huggingface.co/ChenMnZ/act_scales

# OPT-125m 다운로드
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/opt-125m', local_dir='./models/opt-125m')
"

echo "Setup done!"