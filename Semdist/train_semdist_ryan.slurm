#!/bin/bash
#SBATCH --job-name=train_semdist_ryan
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/train_semdist_ryan.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=7-01:30:00
#SBATCH --exclude=helios,eris
#eris,ether



python train_semdist_ryan.py hparams/semdist_ryan.yaml
