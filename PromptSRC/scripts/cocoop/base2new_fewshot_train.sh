#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000
#cd ../..

# custom config
DATA="/home/jewonyeom/prompt_learning/OxfordPets"
TRAINER=CoCoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1

# 1) 예: 단일 정수 shots
# SHOTS=16

# 2) 예: 클래스별 shots (예시로 37개를 모두 1~5 사이 랜덤으로 준 경우)
# 꼭 37개를 맞추어야 함(클래스 개수만큼)
PER_CLASS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
SHOTS=-2

DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

echo "Single-shot mode: NUM_SHOTS=${SHOTS}, PER_CLASS_SHOTS=${PER_CLASS}"

python3 train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.PER_CLASS_SHOTS ${PER_CLASS} \
DATASET.SUBSAMPLE_CLASSES all