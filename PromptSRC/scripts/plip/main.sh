#!/bin/bash

#SBATCH --job-name=caltech101
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=64
#SBATCH --partition=P2
#SBATCH --output=./slurm_log/plip/imbalance/S-%x.%j.out


source ${HOME}/.bashrc
source ${HOME}/anaconda/bin/activate
conda activate promptsrc

# custom config
# DATA=/home/shared/
DATA="/shared/s2/lab01/dataset"
TRAINER=PLIP

DATASET=oxford_pets # oxford_pets imagenet dtd oxford_flowers 
# caltech101 dtd eurosat stanford_cars
# ucf101 food101 sun397 fgvc_aircraft imagenet_r imagenet_a imagenetv2 imagenet imagenet_sketch

CFG=vit_b16_ep100
NCTX=16  # number of context tokens
REG_TYPE=grad # svd spectral_norm grad
SEED=1
# SHOTS=16  # number of shots (1, 2, 4, 8, 16)
SHOTS=-1
REG_COEFF=0.05
SAMPLER=RandomSampler # WeightedClassSampler RandomSampler
K=2

##############################################
NUM_CLASSES=37 # 37 102
PER_CLASS_SHOTS=()
HALF=$((NUM_CLASSES / 2))

for ((i=0; i<HALF; i++)); do
    PER_CLASS_SHOTS+=("16")
done

for ((i=HALF; i<NUM_CLASSES; i++)); do
    PER_CLASS_SHOTS+=("1")
done
echo "${PER_CLASS_SHOTS[@]}"
##############################################

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_${REG_TYPE}_REG_COEFF${REG_COEFF}_${SAMPLER}_K${K}_ep100/seed${SEED}
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --per_class_shots "${PER_CLASS_SHOTS[@]}" \
    --output-dir ${DIR} \
    TRAINER.PLIP.N_CTX_TEXT ${NCTX} \
    TRAINER.PLIP.REG_TYPE ${REG_TYPE} \
    TRAINER.PLIP.K ${K} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATALOADER.TRAIN_X.SAMPLER ${SAMPLER}
done

