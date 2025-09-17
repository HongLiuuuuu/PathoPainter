#!/bin/bash
# Set job requirements
#SBATCH -p gpu_a100
#SBATCH -n 16
#SBATCH -t 20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -o /home/hliu/extend_space/PathoPainter/output/%j_train.txt

# Source bashrc to ensure environment variables and functions are loaded
source ~/.bashrc

# Initialize conda environment correctly
conda activate ldm

# Change to the working directory
cd /home/hliu/extend_space/PathoPainter

# Execute the Python program
python3 main.py -t --gpus 0,1 --base configs/latent-diffusion/pathopainter.yaml