#!/bin/bash
#SBATCH --job-name=LMyan
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/wav2vec.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:30:00
#SBATCH --exclude=helios,eris
#eris,ether


python train.py train.yaml --output_folder results/wav2vec_OB3
