#!/bin/bash
#SBATCH --job-name=coop_caltech_setting_b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --partition=laal_a6000

###############################################################################
# (0) Dataset + Config
###############################################################################
# 사용할 데이터셋 이름을 여기서 지정 (아래 11개 중 하나)
# "oxford_flowers" (51 51)
# "oxford_pets" (19 18)
# "caltech101" (51 50)
# "food101" (51 50)
# "ucf101" (51 50)
# "stanford_cars" (98 98)
# "sun397" (199 198)
# "dtd" (24 23)
# "eurosat" (5 5)
# "fgvc_aircraft" (50 50)
# "imagenet" (500 500)

DATASET="caltech101"

CFG="vit_b16_ep100"
CTP="end"
NCTX=16
CSC=False
SEED=1
USE_FOCAL=0
LOSS_TYPE="ce"

TRAINER="CoOp"
DATA="/home/shared"

###############################################################################
# (1) focal 사용 여부 세팅
###############################################################################
if [ ${USE_FOCAL} -eq 1 ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

###############################################################################
# (2) (Bash 함수) 전달한 값 n개를 배열 형태로 "n번" 반복 => 문자열로 만들어주는 함수
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
# (3) Dataset별 (앞 절반, 뒷 절반) 사이즈를 자동 매핑
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
       echo "Dataset not recognized! Please check the dataset name."
       exit 1
       ;;
  esac
}

###############################################################################
# (4) shot 쌍
###############################################################################
PAIRS=(
  "16 0"
  "15 1"
  "14 2"
  "15 3"
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
# (5) 실제 실행 로직
###############################################################################
# 5-1) 먼저, FIRST_HALF_SIZE / SECOND_HALF_SIZE 결정
get_split_sizes "$DATASET"

# 5-2) for 루프
for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  frontVal=$(echo "$pair" | awk '{print $1}')
  backVal=$(echo "$pair" | awk '{print $2}')

  # shots = 음수(-556, -557, ...) 예시
  SHOTS=$((650 + i))
  SHOTS="-$SHOTS"

  # 앞 절반, 뒷 절반
  PART1=$(repeat_value "${frontVal}" "${FIRST_HALF_SIZE}")
  PART2=$(repeat_value "${backVal}" "${SECOND_HALF_SIZE}")
  PER_CLASS_SHOTS="[${PART1},${PART2}]"

  # output dir
  DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

  echo "============================================================"
  echo ">>> Experiment #$((i+1))"
  echo "    DATASET=${DATASET}"
  echo "    pair=(${frontVal}, ${backVal})"
  echo "    SHOTS=${SHOTS}"
  echo "    FIRST_HALF_SIZE=${FIRST_HALF_SIZE}, SECOND_HALF_SIZE=${SECOND_HALF_SIZE}"
  echo "    PER_CLASS_SHOTS=${PER_CLASS_SHOTS}"
  echo "    Output dir: ${DIR}"
  echo "============================================================"

  # 폴더 존재하면 Resuming, 없으면 새로 실행
  if [ -d "$DIR" ]; then
      echo "Results exist in ${DIR}. Resuming..."
      python train.py \
          --root "${DATA}" \
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
          --root "${DATA}" \
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
