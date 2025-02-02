#!/bin/bash
#SBATCH --partition=laal_a6000
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

# custom config
# "/home/jewonyeom/prompt_learning/OxfordPets"
# "/shared"

DATA="/home/jewonyeom/prompt_learning/OxfordPets"
TRAINER=CoCoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1

# focal loss 사용 여부 (0 => False, 1 => True)
USE_FOCAL=1
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

# 예시: 클래스별 샷
SHOTS=-24
#PER_CLASS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
PER_CLASS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"


DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

echo "NUM_SHOTS=${SHOTS}, PER_CLASS_SHOTS=${PER_CLASS}, USE_FOCAL_LOSS=${FOCAL_ARG}"

python3 train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.PER_CLASS_SHOTS ${PER_CLASS} \
DATASET.SUBSAMPLE_CLASSES all \
DATALOADER.TRAIN_X.SAMPLER WeightedClassSampler \
TRAINER.COCOOP.USE_FOCAL_LOSS ${FOCAL_ARG}
