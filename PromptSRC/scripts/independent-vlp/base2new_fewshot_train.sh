#!/bin/bash
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000
# cd ../..

####################
# 사용자 환경 세팅 #
####################
DATA="/home/jewonyeom/prompt_learning/OxfordPets"   # 예시 경로
TRAINER=IVLP

DATASET=$1
SEED=$2

# 실험 CFG
CFG=vit_b16_c2_ep20_batch4_4+4ctx

# (A) 단일 샷
SHOTS=16
PER_CLASS_SHOTS="[]"

# (B) 클래스별 샷
#SHOTS=-1
#PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

####################
# 디렉토리 세팅    #
####################
DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

####################
# 스크립트 실행    #
####################
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
        DATASET.SUBSAMPLE_CLASSES base

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
        DATASET.SUBSAMPLE_CLASSES base
fi
