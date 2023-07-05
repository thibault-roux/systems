#!/bin/bash
#SBATCH --job-name=yan
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=output/wav2vec.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --time=7-01:30:00
#SBATCH --exclude=helios,eris
#eris,ether


#python train.py train.yaml --data_parallel_backend --batch_size $SLURM_ARRAY_TASK_ID --output_folder results/CRDNN_OB$SLURM_ARRAY_TASK_ID/ #--device cuda:0
#python train.py train.yaml --data_parallel_backend --batch_size 7 --output_folder results/CRDNN_lr${learning_rate[$SLURM_ARRAY_TASK_ID]} --lr ${learning_rate[$SLURM_ARRAY_TASK_ID]}
# python train.py train.yaml --data_parallel_backend --batch_size 4 --output_folder results/wav2vec_OB3
python train.py bpe.yaml --output_folder results/wav2vec_bpe
