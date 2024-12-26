#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000

###############################################################################
# Usage:
#   bash base2new_train.sh <DATASET> <CFG> <CTP> <NCTX> <SHOTS> <CSC> <SEED>
#
# 예시:
#   bash base2new_train.sh \
#       OxfordPets \
#       vit_b16_c2_ep20_batch4_4+4ctx \
#       end \
#       16 \
#       16 \
#       False \
#       1
###############################################################################

# 경로와 트레이너 설정
DATA="/home/jewonyeom/prompt_learning/OxfordPets"      # 실제 데이터셋 경로로 수정
TRAINER=CoOp

# 인자 받기
DATASET=$1   # 예: OxfordPets
CFG=$2       # 예: vit_b16_c2_ep20_batch4_4+4ctx
CTP=$3       # 예: end
NCTX=$4      # 예: 16
SHOTS=$5     # 예: 16
CSC=$6       # 예: False
SEED=$7      # 예: 1

# output 디렉토리 설정 (base 학습)
DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
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
        DATASET.SUBSAMPLE_CLASSES base
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
        DATASET.SUBSAMPLE_CLASSES base
fi
