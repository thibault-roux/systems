#!/bin/bash
#SBATCH --job-name=unigram5000
#SBATCH --output=output/unigram5000.txt
#SBATCH --mail-user=thibault.baneras.roux@gmail.com
#SBATCH --mail-type=END
#SBATCH --time=60:30:00
#SBATCH -C v100
#SBATCH -A rbg@v100
#SBATCH --gpus-per-node=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# avec qps_gpu-t4, on peut mettre à 99h la limite de temps

# python -m torch.distributed.launch --nproc_per_node=2 train.py hparams/char80.yaml --distributed_launch --distributed_backend='nccl'

python train.py hparams/unigram5000.yaml --data_parallel_backend

