#!/bin/bash
#SBATCH --job-name=coop_eurosat_setting_a
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --partition=laal_a6000

###############################################################################
# 고정 config
###############################################################################
# "oxford_flowers" (50 50)
# "oxford_pets"
# "caltech101"
# "food101"
# "ucf101"
# "stanford_cars" (98 98)
# "sun397" (199 198)
# "dtd" (24 23)
# "eurosat" (5 5)

DATASET="dtd"
CFG="vit_b16_ep100"
CTP="end"
NCTX=16
CSC=False
SEED=1
USE_FOCAL=0
LOSS_TYPE="ce"

# DATA 경로
# "/home/shared"
DATA="/home/shared"

TRAINER=CoOp

# focal 사용 여부
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

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

###############################################################################
# (A) 첫 번째 절반(FIRST_HALF)은 'FIRST_HALF_VALUE'로 고정 (예: 19번 반복)
###############################################################################
FIRST_HALF_SIZE=5
FIRST_HALF_VALUE=16

###############################################################################
# (B) 두 번째 절반(SECOND_HALF)은 1부터 16까지 각각에 대해 반복
#     즉, SECOND_HALF_LIST=(1 2 3 ... 16)
###############################################################################
SECOND_HALF_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# 총 16번 반복 (i=0..15)
for i in "${!SECOND_HALF_LIST[@]}"; do
  VAL2=${SECOND_HALF_LIST[$i]}

  # Shots 설정 (원하시는 로직대로 변경)
  # 여기서는 예시로 i=0이면 -346, i=1이면 -347, ... i=15이면 -361
  SHOTS=$((456 + i))
  SHOTS="-$SHOTS"

  # (1) 첫 번째 절반(19개)에 FIRST_HALF_VALUE(=16) 반복
  PART1=$(repeat_value "${FIRST_HALF_VALUE}" "${FIRST_HALF_SIZE}")

  # (2) 두 번째 절반(18개)에 VAL2 반복
  PART2_SIZE=5
  PART2=$(repeat_value "${VAL2}" "${PART2_SIZE}")

  # 최종 PER_CLASS_SHOTS => [PART1, PART2]
  PER_CLASS_SHOTS="[${PART1},${PART2}]"

  ###############################################################################
  # (1) 데이터셋 경로 매핑
  ###############################################################################
  case $DATASET in
      "fgvc_aircraft")
          DATASET_PATH="${DATA}"
          ;;
      "oxford_pets")
          DATASET_PATH="${DATA}"
          ;;
      "caltech101")
          DATASET_PATH="${DATA}"
          ;;
      *)
          DATASET_PATH="${DATA}"
          ;;
  esac

  ###############################################################################
  # (2) 출력 경로
  ###############################################################################
  DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

  echo "============================================================"
  echo ">>> Experiment #$((i+1))"
  echo "    VAL2=${VAL2}"
  echo "    SHOTS=${SHOTS}"
  echo "    PER_CLASS_SHOTS=${PER_CLASS_SHOTS}"
  echo "    Output dir: ${DIR}"
  echo "============================================================"

  # (3) 이미 해당 결과 폴더가 있으면 "Resuming...", 없으면 새로 실행
  if [ -d "$DIR" ]; then
      echo "Results exist in ${DIR}. Resuming..."
      python train.py \
          --root "${DATASET_PATH}" \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR} \
          TRAINER.COOP.N_CTX ${NCTX} \
          TRAINER.COOP.CSC ${CSC} \
          TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
          DATASET.NUM_SHOTS ${SHOTS} \
          DATASET.PER_CLASS_SHOTS "${PER_CLASS_SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES all \
          DATALOADER.TRAIN_X.SAMPLER RandomSampler \
          TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE}
  else
      echo "Run this job and save the output to ${DIR}"
      mkdir -p ${DIR}

      python train.py \
          --root "${DATASET_PATH}" \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
          --output-dir ${DIR} \
          TRAINER.COOP.N_CTX ${NCTX} \
          TRAINER.COOP.CSC ${CSC} \
          TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
          DATASET.NUM_SHOTS ${SHOTS} \
          DATASET.PER_CLASS_SHOTS "${PER_CLASS_SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES all \
          DATALOADER.TRAIN_X.SAMPLER RandomSampler \
          TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE}
  fi

  echo
  echo
done
