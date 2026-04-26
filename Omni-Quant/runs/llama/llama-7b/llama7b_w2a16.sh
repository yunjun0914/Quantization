#!/bin/bash
#SBATCH --account=rabbit
#SBATCH --job-name=llama7b_w2a16
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
    --model ./models/llama-7b \
    --net llama-7b \
    --epochs 20 \
    --output_dir ./log/llama-7b-w2a16 \
    --eval_ppl \
    --wbits 2 \
    --abits 16 \
    --lwc \
    --act-scales ./act_scales/llama-7b.pt \
    --act-shifts ./act_shifts/llama-7b.pt \
    --save_dir ./output/llama7b_w2a16
