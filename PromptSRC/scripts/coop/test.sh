#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=laal_3090

###############################################################################
# Usage:
#   bash base2new_train.sh <DATASET> <CFG> <CTP> <NCTX> <SHOTS> <CSC> <SEED>
#
# Example:
#   bash base2new_train.sh \
#       oxford_pets \
#       vit_b16_ep50 \
#       end \
#       16 \
#       16 \
#       False \
#       1
###############################################################################

# custom config
# "/home/jewonyeom/prompt_learning/OxfordPets"
# "/home/shared"

DATA="/home/shared"
TRAINER=CoOp

DATASET=$1
CFG=$2
CTP=$3
NCTX=$4
SHOTS=$5
CSC=$6
SEED=$7

LOADEP=100
SUB=all

# focal loss 사용 여부
USE_FOCAL=0
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

SHOTS=4 
PER_CLASS_SHOTS="[]"
#PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

#PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Evaluating model on '${SUB}' classes"
    echo "Results are in ${DIR}, so resuming..."

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
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.COOP.USE_FOCAL_LOSS ${FOCAL_ARG}
else
    echo "Evaluating model on '${SUB}' classes"
    echo "Run first time => output to ${DIR}"

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
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.COOP.USE_FOCAL_LOSS ${FOCAL_ARG}
fi
