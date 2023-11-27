#!/bin/bash
#SBATCH --job-name=e2e2_semdist
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/e2e2_semdist.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=7-01:30:00
#SBATCH --exclude=helios,eris
#eris,ether



python train2.py hparams/train_fr_with_wav2vec7k_unigram250_semdist_lia_v2.yaml
