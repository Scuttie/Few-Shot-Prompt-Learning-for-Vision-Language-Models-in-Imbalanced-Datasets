#!/bin/bash
#SBATCH --job-name=independent_vlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=laal_a6000


# Pets: "/home/jewonyeom/prompt_learning/OxfordPets"
# Else: "/shared"
DATA="/shared"
TRAINER=IVLP

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep20_batch4_4+4ctx_kd

# focal loss 사용 (0 => False, 1 => True)
USE_FOCAL=0
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

SHOTS=-16
# fgvc_aircraft
PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
# (B) 클래스별 샷 (pets)
#PER_CLASS_SHOTS="[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"


DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

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
        DATASET.SUBSAMPLE_CLASSES all \
        DATALOADER.TRAIN_X.SAMPLER WeightedClassSampler \
        TRAINER.IVLP.USE_FOCAL_LOSS ${FOCAL_ARG} \
        TRAINER.PROMPTSRC.SIMCLR_ALPHA 0.0
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
        DATASET.SUBSAMPLE_CLASSES all \
        DATALOADER.TRAIN_X.SAMPLER WeightedClassSampler \
        TRAINER.IVLP.USE_FOCAL_LOSS ${FOCAL_ARG} \
        TRAINER.PROMPTSRC.SIMCLR_ALPHA 0.0
fi
#        DATALOADER.TRAIN_X.SAMPLER WeightedClassSampler
#        DATALOADER.TRAIN_X.SAMPLER RandomSampler
