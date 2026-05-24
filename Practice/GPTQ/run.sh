#!/bin/bash
#SBATCH --partition=rabbit
#SBATCH --job-name=gptq_llama
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --output=/home/yunjun0914/Quantization/Practice/logs/%x_%j.out
#SBATCH --error=/home/yunjun0914/Quantization/Practice/logs/%x_%j.err

source ~/yunjun_env/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

mkdir -p ~/Quantization/Practice/logs

cd ~/Quantization/Practice/GPTQ

python -u main_llama.py \
    --model ~/Quantization/Omni-Quant/OmniQuant/models/llama2-7b \
    --bits 3 \
    --rot hadamard \
    --svd_rank 0 \
    --v2 \
    --compare \
    --percdamp 0.01 \
    --dev cuda:0 
