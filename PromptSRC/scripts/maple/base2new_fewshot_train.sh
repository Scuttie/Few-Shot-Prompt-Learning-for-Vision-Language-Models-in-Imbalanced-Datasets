#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000
# cd ../..

DATA="/home/jewonyeom/prompt_learning/OxfordPets"
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx

# (A) 단일 샷
# SHOTS=16
# PER_CLASS_SHOTS="[]"

# (B) 클래스별 샷
SHOTS=-1
PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
else
    echo "Run this job and save output to ${DIR}"
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
fi
