#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/infer.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=7-01:30:00
#SBATCH --exclude=helios,eris
#eris,ether

# the two next lines can't be use automatically (need to be done manually)
# for folder in CKPT+2024* ; do echo $folder >> names.txt ; done
# cp results_lia_asr/$MYSYSTEM/1234/save/names.txt ckpt/$MYSYSTEM.txt

MYSYSTEM="ctc" # wer semdist semdist_ryan random random_xt

i=0
while read path; do
    i=$((i+1))
    echo $path
    python infer.py hparams/$MYSYSTEM.yaml --loaded_checkpoint $i --ckpt_path $path
done < ckpt/$MYSYSTEM.txt # path written as in Mickael/results/ckpt.txt
