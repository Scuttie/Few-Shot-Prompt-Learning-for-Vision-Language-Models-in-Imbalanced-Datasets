#!/bin/bash

#SBATCH --job-name=lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=64
#SBATCH --partition=P3
#SBATCH --output=./slurm_log/lora/balance/S-%x.%j.out


source ${HOME}/.bashrc
source ${HOME}/anaconda/bin/activate
conda activate promptsrc

# custom config
# DATA=/home/shared/
DATA="/shared/s2/lab01/dataset"
TRAINER=LoRA

# DATASETS="eurosat sun397"
# DATASETS="oxford_pets dtd oxford_flowers"
# DATASETS="caltech101 dtd stanford_cars food101"
# DATASETS="fgvc_aircraft ucf101"
DATASETS="fgvc_aircraft"


# python main.py --root_path /shared/s2/lab01/dataset --dataset fgvc --seed 1
CFG=vit_b16_ep50
SEED=1
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
SAMPLER=RandomSampler # WeightedClassSampler RandomSampler
ENCODER=both # both vision text

TEXT_LOSS_WEIGHT=25
IMAGE_LOSS_WEIGHT=10
LOGITS_LOSS_WEIGHT=1.0

for DATASET in ${DATASETS}; do
    for SEED in 1; do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/${SAMPLER}_REG/seed${SEED}
        echo "Run this job and save the output to ${DIR}"
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATALOADER.TRAIN_X.SAMPLER ${SAMPLER} \
            TRAINER.LORA.ENCODER ${ENCODER} \
            TRAINER.LORA.TEXT_LOSS_WEIGHT ${TEXT_LOSS_WEIGHT} \
            TRAINER.LORA.IMAGE_LOSS_WEIGHT ${IMAGE_LOSS_WEIGHT} \
            TRAINER.LORA.LOGITS_LOSS_WEIGHT ${LOGITS_LOSS_WEIGHT}
    done
done

