#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000

# cd ../..

# custom config
DATA="/home/jewonyeom/prompt_learning/OxfordPets"
TRAINER=CoCoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
LOADEP=10
SUB=new

# --------------------------------------------------
# [A] 단일 정수 shots 모드
# --------------------------------------------------
# NUM_SHOTS=16
# PER_CLASS_SHOTS="[]"
# OUT_DIR_NAME="shots_${NUM_SHOTS}"

# --------------------------------------------------
# [B] 클래스별 shots 모드
# --------------------------------------------------
NUM_SHOTS=0
PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
OUT_DIR_NAME="shots_-2"

COMMON_DIR=${DATASET}/${OUT_DIR_NAME}/${TRAINER}/${CFG}/seed${SEED}

MODEL_DIR=output/base2new/train_all/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

echo "==== TEST SCRIPT SETTINGS ===="
echo "DATASET = ${DATASET}"
echo "SEED    = ${SEED}"
echo "NUM_SHOTS = ${NUM_SHOTS}"
echo "PER_CLASS_SHOTS = ${PER_CLASS_SHOTS}"
echo "OUTPUT DIR = ${DIR}"
echo "=============================="

if [ -d "$DIR" ]; then
    echo "[TEST] Evaluating model: directory already exists => Resuming..."
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
        DATASET.NUM_SHOTS ${NUM_SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
else
    echo "[TEST] Evaluating model: directory not found => First run"
    echo "Will save the output to ${DIR}"

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
        DATASET.NUM_SHOTS ${NUM_SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
