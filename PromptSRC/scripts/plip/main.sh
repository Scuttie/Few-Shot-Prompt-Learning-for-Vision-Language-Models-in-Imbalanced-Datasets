#!/bin/bash

#SBATCH --job-name="dtd oxford_flowers"
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


DATASETS="oxford_pets dtd oxford_flowers"
# DATASETS="caltech101 dtd eurosat stanford_cars"
# DATASETS="ucf101 food101 sun397 fgvc_aircraft"
# DATASETS="imagenet"

CFG=vit_b16_ep100
NCTX=16  # number of context tokens
REG_TYPE=grad # svd spectral_norm grad
SEED=1
# SHOTS=16  # number of shots (1, 2, 4, 8, 16)
SHOTS=-1
REG_COEFF=0.01
SAMPLER=RandomSampler # WeightedClassSampler RandomSampler
K=1

##############################################
declare -A DATASET_CLASSES=(
    ["oxford_pets"]=37
    ["imagenet"]=1000
    ["dtd"]=47
    ["oxford_flowers"]=102
    ["caltech101"]=101
    ["eurosat"]=10
    ["stanford_cars"]=196
    ["ucf101"]=101
    ["food101"]=101
    ["sun397"]=397
    ["fgvc_aircraft"]=100
)

for DATASET in ${DATASETS}; do
    for SEED in 1; do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_${REG_TYPE}_REG_COEFF${REG_COEFF}_${SAMPLER}_K${K}/seed${SEED}
        echo "Run this job and save the output to ${DIR}"

        ##############################################
        if [[ -v DATASET_CLASSES[$DATASET] ]]; then
            NUM_CLASSES=${DATASET_CLASSES[$DATASET]}
            echo "NUM_CLASSES for $DATASET is set to $NUM_CLASSES"
        else
            echo "Error: Unknown dataset '$DATASET'"
            exit 1
        fi

        PER_CLASS_SHOTS=()
        HALF=$(( (NUM_CLASSES + 1) / 2 ))

        for ((i=0; i<HALF; i++)); do
            PER_CLASS_SHOTS+=("16")
        done

        for ((i=HALF; i<NUM_CLASSES; i++)); do
            PER_CLASS_SHOTS+=("1")
        done
        echo "${PER_CLASS_SHOTS[@]}"
        ##############################################


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
done
