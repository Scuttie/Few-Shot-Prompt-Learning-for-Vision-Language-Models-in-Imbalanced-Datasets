#!/bin/bash
#SBATCH --job-name=base2new_linear
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=laal_3090

DATA="/shared"
TRAINER=LinearProbeCLIP

DATASET=$1        # 예: fgvc_aircraft
BACKBONE=$2       # 예: ViT-B/16
LOSS_TYPE=$3      # "ce" 또는 "focal"
SEED=$4           # random seed
SAMPLER=$5        # 예: WeightedClassSampler
SHOTS=-32

# (Bash 함수) 전달한 값 n개를 배열 형태로 "n번" 반복 => 문자열로 만들어주는 함수
function repeat_value {
  local val=$1
  local count=$2
  local out=""
  for ((i=0; i<${count}; i++)); do
    out="$out,$val"
  done
  # 맨 앞의 콤마(,) 제거
  echo "${out#,}"
}

# 1) 첫 번째 절반(19개)에 16이 반복
PART1=$(repeat_value 16 50)   # "16,16,16,...(총19번)"
# 2) 두 번째 절반(18개)에 VAL2가 반복
PART2=$(repeat_value 1 50)

# 최종 PER_CLASS_SHOTS => [PART1, PART2]
PER_CLASS_SHOTS="[$PART1,$PART2]"

CFG="vit_b16_ep50"
CONFIG_PATH="configs/trainers/${TRAINER}/${CFG}.yaml"

DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${BACKBONE}_${LOSS_TYPE}/seed${SEED}

echo "======== base2new linear-probe training ========"
echo "DATASET = $DATASET"
echo "BACKBONE = $BACKBONE"
echo "LOSS_TYPE = $LOSS_TYPE"
echo "SEED = $SEED"
echo "SHOTS = $SHOTS"
echo "SAMPLER = $SAMPLER"
echo "DIR = $DIR"
echo "================================================"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file ${CONFIG_PATH} \
    --output-dir ${DIR} \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    TRAINER.LINEAR_PROBE.LOSS_TYPE ${LOSS_TYPE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.PER_CLASS_SHOTS ${PER_CLASS_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    DATALOADER.TRAIN_X.SAMPLER ${SAMPLER}
