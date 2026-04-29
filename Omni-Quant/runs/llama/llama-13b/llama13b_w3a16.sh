#!/bin/bash
#SBATCH --account=judy
#SBATCH --job-name=llama13b_w3a16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yunjun0914/Quantization/Omni-Quant/logs/%x_%j.out
#SBATCH --error=/home/yunjun0914/Quantization/Omni-Quant/logs/%x_%j.err

source ~/yunjun_env/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ~/Quantization/Omni-Quant/OmniQuant

python main.py \
    --model ./models/llama-13b \
    --net llama-13b \
    --epochs 20 \
    --output_dir ./log/llama-13b-w3a16 \
    --eval_ppl \
    --wbits 3 \
    --abits 16 \
    --lwc \
    --act-scales ./act_scales/llama-13b.pt \
    --act-shifts ./act_shifts/llama-13b.pt \
    --save_dir ./output/llama13b_w3a16
