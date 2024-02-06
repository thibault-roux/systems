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


# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec1k_char_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec3k_char_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_char_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec53_char_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec53fr_char_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_bpe750_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_bpe1000_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_bpe250_lia.yaml
# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_bpe500_lia.yaml

# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec2base_char_lia.yaml

# python train_with_wav2vec_zenidoc_v2.py hparams/train_fr_with_wav2vec7k_bpe750_v2_lia.yaml
# python train_with_wav2vec_zenidoc_v2.py hparams/train_fr_with_wav2vec7k_bpe250_v3_lia.yaml # faut-il que j'utilise aussi le train v2 pour bpe750_v2 ??


# # liste=("1500" "900" "650" "150" "100" "50")
# liste=("250" "500" "750" "1000")
# for i in "${liste[@]}"
# do
#     python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_bpe${i}_lia.yaml
# done

# liste=("7" "3" "1")
# for i in "${liste[@]}"
# do
#     python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec${i}k_char_lia.yaml
# done


# liste=("1" "3" "5" "7")
# for i in "${liste[@]}"
# do
#     # # mkdir results_lia_asr/wav2vec2_ctc_fr_7k_epoch_${i}
#     # # mkdir results_lia_asr/wav2vec2_ctc_fr_7k_epoch_${i}/1234
#     # # mkdir results_lia_asr/wav2vec2_ctc_fr_7k_epoch_${i}/1234/save
#     # # cp data/*.csv results_lia_asr/wav2vec2_ctc_fr_7k_epoch_${i}/1234/save
#     # python train_epochs.py hparams/train_fr_with_wav2vec7k_char_lia_original.yaml --output_folder results_lia_asr/wav2vec2_ctc_fr_7k_epoch_${i}/1234 --number_of_epochs ${i}
#     python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_char_lia_original.yaml --output_folder results_lia_asr/wav2vec2_ctc_fr_7k_epoch_${i}/1234 --number_of_epochs ${i}
# done


# python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe1000_lia_without_space.yaml
# python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe750_lia_without_space.yaml
# python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe500_lia_without_space.yaml
# python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe250_lia_without_space.yaml
# python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe150_lia_without_space.yaml

# python train_with_wav2vec_zenidoc.py hparams/train_fr_with_wav2vec7k_LucasMaison.yaml # mettre en majuscule



# i=0
# while read path; do
#     i=$((i+1))
#     echo $path
#     python train_with_wav2vec_zenidoc_save_checkpoint.py hparams/train_fr_with_wav2vec7k_char_lia_50epochs.yaml --loaded_checkpoint $i --ckpt_path $path
# done < ckpt.txt

cp /users/rouvier/yanis_data/results_lia_asr/wav2vec2_ctc_fr_bpe1050_v2_7k/ results_lia_asr -r
# cp /users/rouvier/yanis_data/results_lia_asr/wav2vec2_ctc_fr_bpe500_v2_7k/ results_lia_asr -r
# python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe500_v2_lia.yaml
python infer_SP.py hparams/train_fr_with_wav2vec7k_bpe1050_v2_lia.yaml
