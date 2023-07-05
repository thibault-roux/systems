#!/bin/bash
#SBATCH --job-name=scratch
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/scratch.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=13:30:00
#SBATCH --exclude=helios,eris
#eris,ether


python scratch.py scratch.yaml --output_folder results/scratch
