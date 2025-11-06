#!/bin/bash -l
#SBATCH --mem=64G
#SBATCH --job-name=def_diffusion
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=8
#SBATCH --requeue

vpkg_require anaconda
source activate torch
python train.py config='./config.yaml' run=1
