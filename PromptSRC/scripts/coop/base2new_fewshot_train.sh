#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000

###############################################################################
# Usage:
#   bash base2new_train.sh <DATASET> <CFG> <CTP> <NCTX> <SHOTS> <CSC> <SEED>
#
# Example:
#   bash base2new_train.sh \
#       OxfordPets \
#       vit_b16_c2_ep20_batch4_4+4ctx \
#       end \
#       16 \
#       16 \
#       False \
#       1
###############################################################################

DATA="/home/jewonyeom/prompt_learning/OxfordPets"
TRAINER=CoOp

# 인자
DATASET=$1
CFG=$2
CTP=$3
NCTX=$4
SHOTS=$5
CSC=$6
SEED=$7

# 만약 "클래스별 샷"을 쓰고 싶다면,
# NUM_SHOTS=0 (or -1), PER_CLASS_SHOTS="[...]" 을 인자로 추가해야 함
# 여기서는 스크립트 변수를 하나 더 둘 수 있음:
# PER_CLASS_SHOTS="[]"

# [예] 만약 "클래스별 샷"을 해보고 싶다면:
PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
SHOTS=-1

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results exist in ${DIR}. Resuming..."
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
fi
