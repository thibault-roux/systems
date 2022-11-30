#!/bin/bash
#SBATCH --job-name=finetuned
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/finetuned.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=13:30:00
#SBATCH --exclude=helios,eris
#eris,ether


python finetuned.py finetuned.yaml --output_folder results/finetuned
