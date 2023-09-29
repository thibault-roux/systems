#!/bin/bash
#SBATCH --job-name=prepchar80
#SBATCH --partition=prepost
#SBATCH --output=output/prepchar80.txt
#SBATCH --mail-user=thibault.baneras.roux@gmail.com
#SBATCH --mail-type=END
#SBATCH --time=01:00:00
#SBATCH -A rbg@v100


python train.py hparams/bpe600.yaml
