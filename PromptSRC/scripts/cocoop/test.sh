#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

DATA="/shared"
TRAINER=CoCoOp

DATASET=$1
SEED=$2
CFG=vit_b16_c4_ep10_batch1_ctxv1
LOADEP=10
SUB=all

# focal loss 사용 여부
USE_FOCAL=0
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

SHOTS=-21
PER_CLASS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_all/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "[TEST] Directory found => Resuming..."
    python3 train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.COCOOP.USE_FOCAL_LOSS ${FOCAL_ARG}
else
    echo "[TEST] Directory not found => First time"
    python3 train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.COCOOP.USE_FOCAL_LOSS ${FOCAL_ARG}
fi
