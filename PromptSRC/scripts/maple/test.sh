#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=laal_3090

DATA="/shared"
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx
LOADEP=5
SUB=all

USE_FOCAL=1  # 테스트 시에도 동일 옵션 적용 가정
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_LOSS_ARG=True
else
    FOCAL_LOSS_ARG=False
fi

SHOTS=-23
PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_all/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are in ${DIR}. Resuming..."

    python train.py \
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
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.MAPLE.USE_FOCAL_LOSS ${FOCAL_LOSS_ARG}
else
    echo "Evaluating model"
    echo "First run => saving output to ${DIR}"

    python train.py \
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
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.MAPLE.USE_FOCAL_LOSS ${FOCAL_LOSS_ARG}
fi
