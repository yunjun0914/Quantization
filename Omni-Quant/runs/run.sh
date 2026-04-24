#!/bin/bash
#SBATCH --job-name=omniquant_w4a16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /path/to/yunjun_env/bin/activate

cd Quantization/OmniQuant

python main.py \
    --model ./models/opt-125m \
    --wbits 4 \
    --abits 16 \
    --act-scales ./act_scales/opt-125m.pt \
    --act-shifts ./act_shifts/opt-125m.pt \
    --save-dir ./output/omniquant_w4a16