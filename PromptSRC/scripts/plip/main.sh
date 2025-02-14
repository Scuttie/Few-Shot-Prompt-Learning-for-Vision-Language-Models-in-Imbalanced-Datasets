#!/bin/bash

# custom config
DATA=/home/shared/
TRAINER=PLIP

DATASET=fgvc_aircraft # oxford_pets imagenet
CFG=vit_b16_ep100
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
REG_TYPE=spectral_norm # svd spectral_norm grad

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_${REG_TYPE}_ep100/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.PLIP.N_CTX_TEXT ${NCTX} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.PLIP.REG_TYPE ${REG_TYPE}
    fi
done