#!/bin/bash
#SBATCH --job-name=base2new_linear_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=laal_3090

DATA="/shared"
TRAINER=LinearProbeCLIP

DATASET=$1
BACKBONE=$2
LOSS_TYPE=$3
SEED=$4
LOADEP=$5
SUB=$6

SHOTS=-22
PER_CLASS_SHOTS="[16,16,16,...,0]"  # 예시

CFG="vit_b16_ep50"
CONFIG_PATH="configs/trainers/${TRAINER}/${CFG}.yaml"

MODEL_DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${BACKBONE}_${LOSS_TYPE}/seed${SEED}
DIR=output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${BACKBONE}_${LOSS_TYPE}/seed${SEED}

if [ -z "$LOADEP" ]; then
    echo "Load best checkpoint"
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file ${CONFIG_PATH} \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only \
        MODEL.BACKBONE.NAME ${BACKBONE} \
        TRAINER.LINEAR_PROBE.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
else
    echo "Load epoch = $LOADEP"
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file ${CONFIG_PATH} \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        MODEL.BACKBONE.NAME ${BACKBONE} \
        TRAINER.LINEAR_PROBE.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
