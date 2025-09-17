#!/bin/bash
# Set job requirements
#SBATCH -p gpu_a100
#SBATCH -n 16
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=2
#SBATCH -o /home/hliu/extend_space/HIPT/output/%j_infer_hipt_CATCH.txt

# Source bashrc to ensure environment variables and functions are loaded
source ~/.bashrc

# Initialize conda environment correctly
conda activate diff

cd /home/hliu/extend_space/HIPT
python extract_regional_feature.py \
  --mask_list /path/to/mask/list/ \
  --model_path /path/to/trained/hipt/model/ \
  --output_folder ssl_features/ \
  --gpu 0
