#!/bin/bash
#SBATCH --partition=rabbit
#SBATCH --job-name=llama2_70b_w3a16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/home/yunjun0914/Quantization/Omni-Quant/logs/%x_%j.out
#SBATCH --error=/home/yunjun0914/Quantization/Omni-Quant/logs/%x_%j.err

source ~/yunjun_env/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ~/Quantization/Omni-Quant/OmniQuant

python main.py \
    --model ~/Quantization/Omni-Quant/OmniQuant/models/llama2-70b \
    --net Llama-2-70b \
    --epochs 20 \
    --output_dir ./log/llama2-70b-w3a16 \
    --eval_ppl \
    --wbits 3 \
    --abits 16 \
    --lwc \
    --act-scales ./act_scales/Llama-2-70b.pt \
    --act-shifts ./act_shifts/Llama-2-70b.pt \
    --save_dir ./output/llama2_70b_w3a16
