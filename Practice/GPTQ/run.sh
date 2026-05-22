#!/bin/bash
#SBATCH --partition=rabbit
#SBATCH --job-name=gptq_llama
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
    --model meta-llama/Llama-2-7b-hf \
    --bits 4 \
    --rot hadamard \
    --groupsize 128 \
    --compare \
    --dev cuda:0
