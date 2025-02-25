#!/bin/bash

#SBATCH --job-name=lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=64
#SBATCH --partition=P2
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
# DATASETS="caltech101 dtd stanford_cars"
# DATASETS="fgvc_aircraft"
DATASETS="imagenet"


# python main.py --root_path /shared/s2/lab01/dataset --dataset fgvc --seed 1

CFG=vit_b16_ep50
# NCTX=16  # number of context tokens
# REG_TYPE=grad # svd spectral_norm grad
SEED=1
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
# REG_COEFF=10.0
SAMPLER=RandomSampler # WeightedClassSampler RandomSampler
# K=1

for DATASET in ${DATASETS}; do
    for SEED in 1; do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/${SAMPLER}/seed${SEED}
        echo "Run this job and save the output to ${DIR}"
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATALOADER.TRAIN_X.SAMPLER ${SAMPLER}
    done
done

