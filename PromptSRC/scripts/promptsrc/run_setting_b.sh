#!/bin/bash
#SBATCH --job-name=promptsrc_sun_setting_b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --partition=laal_a6000

###############################################################################
# (0) Dataset + Config
###############################################################################
DATASET="sun397"
TRAINER="PromptSRC"

CFG="vit_b16_c2_ep20_batch4_4+4ctx_2" # config가 dataset마다 다른 것에 주의
CONFIG_PATH="configs/trainers/${TRAINER}/${CFG}.yaml"

DATA="/home/shared"
SEED=1
SAMPLER="RandomSampler"

###############################################################################
# (1) repeat_value 함수
###############################################################################
function repeat_value {
  local val=$1
  local count=$2
  local out=""
  for ((i=0; i<${count}; i++)); do
    out="$out,$val"
  done
  echo "${out#,}"
}

###############################################################################
# (2) Dataset별 (앞 절반, 뒷 절반) 사이즈 매핑
###############################################################################
function get_split_sizes {
  case $1 in
    "oxford_flowers")
       FIRST_HALF_SIZE=51
       SECOND_HALF_SIZE=51
       ;;
    "oxford_pets")
       FIRST_HALF_SIZE=19
       SECOND_HALF_SIZE=18
       ;;
    "caltech101")
       FIRST_HALF_SIZE=51
       SECOND_HALF_SIZE=50
       ;;
    "food101")
       FIRST_HALF_SIZE=51
       SECOND_HALF_SIZE=50
       ;;
    "ucf101")
       FIRST_HALF_SIZE=51
       SECOND_HALF_SIZE=50
       ;;
    "stanford_cars")
       FIRST_HALF_SIZE=98
       SECOND_HALF_SIZE=98
       ;;
    "sun397")
       FIRST_HALF_SIZE=199
       SECOND_HALF_SIZE=198
       ;;
    "dtd")
       FIRST_HALF_SIZE=24
       SECOND_HALF_SIZE=23
       ;;
    "eurosat")
       FIRST_HALF_SIZE=5
       SECOND_HALF_SIZE=5
       ;;
    "fgvc_aircraft")
       FIRST_HALF_SIZE=50
       SECOND_HALF_SIZE=50
       ;;
    "imagenet")
       FIRST_HALF_SIZE=500
       SECOND_HALF_SIZE=500
       ;;
    *)
       echo "Dataset not recognized!"
       exit 1
       ;;
  esac
}

###############################################################################
# (3) shot 쌍 목록
###############################################################################
PAIRS=(
  "16 0"
  "15 1"
  "14 2"
  "13 3"
  "12 4"
  "11 5"
  "10 6"
  "9 7"
  "8 8"
  "8 0"
  "7 1"
  "6 2"
  "5 3"
  "4 4"
  "4 0"
  "3 1"
  "2 2"
)

###############################################################################
# (4) 실행
###############################################################################
# 먼저 (앞 절반, 뒷 절반) 사이즈 할당
get_split_sizes "$DATASET"

for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  frontVal=$(echo "$pair" | awk '{print $1}')
  backVal=$(echo "$pair" | awk '{print $2}')

  # shots = 음수
  SHOTS=$((590 + i))
  SHOTS="-$SHOTS"

  PART1=$(repeat_value "${frontVal}" "${FIRST_HALF_SIZE}")
  PART2=$(repeat_value "${backVal}" "${SECOND_HALF_SIZE}")
  PER_CLASS_SHOTS="[${PART1},${PART2}]"

  DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

  echo "============================================================"
  echo ">>> Experiment #$((i+1))"
  echo "    DATASET=${DATASET}"
  echo "    pair=(${frontVal}, ${backVal})"
  echo "    SHOTS=${SHOTS}"
  echo "    FIRST_HALF_SIZE=${FIRST_HALF_SIZE}, SECOND_HALF_SIZE=${SECOND_HALF_SIZE}"
  echo "    PER_CLASS_SHOTS=${PER_CLASS_SHOTS}"
  echo "    Output dir: ${DIR}"
  echo "============================================================"

  if [ -d "$DIR" ]; then
      echo "Results exist in ${DIR}. Resuming..."
      python train.py \
          --root "${DATA}" \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file ${CONFIG_PATH} \
          --output-dir ${DIR} \
          DATASET.NUM_SHOTS ${SHOTS} \
          DATASET.PER_CLASS_SHOTS "${PER_CLASS_SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES all \
          DATALOADER.TRAIN_X.SAMPLER ${SAMPLER} \
          TRAINER.PROMPTSRC.LABEL_SCOPE default \
          TRAINER.PROMPTSRC.LOSS_TYPE ce \
          TRAINER.PROMPTSRC.SIMCLR_ALPHA 0.0
  else
      echo "Run this job and save the output to ${DIR}"
      mkdir -p ${DIR}

      python train.py \
          --root "${DATA}" \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file ${CONFIG_PATH} \
          --output-dir ${DIR} \
          DATASET.NUM_SHOTS ${SHOTS} \
          DATASET.PER_CLASS_SHOTS "${PER_CLASS_SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES all \
          DATALOADER.TRAIN_X.SAMPLER ${SAMPLER} \
          TRAINER.PROMPTSRC.LABEL_SCOPE default \
          TRAINER.PROMPTSRC.LOSS_TYPE ce \
          TRAINER.PROMPTSRC.SIMCLR_ALPHA 0.0
  fi

  echo
  echo
done
