#!/bin/bash
#SBATCH --job-name=coop
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --partition=laal_3090

###############################################################################
# 고정 config
###############################################################################
DATASET="imagenet"
CFG="vit_b16_ep50"
CTP="end"
NCTX=16
CSC=False
SEED=1
USE_FOCAL=0
LOSS_TYPE="ce"

# DATA 경로
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

# 2번째 절반(18개)에 들어갈 값들
SECOND_HALF_LIST=(1 2 3 4)

# 10번 반복 (i=0..9)
for i in "${!SECOND_HALF_LIST[@]}"; do
  # i=0일 때 SHOTS=-100, i=1이면 -101, ... i=9이면 -109
  # 즉 -((100 + i))
  SHOTS=$(( 365 + i ))
  SHOTS="-$SHOTS"

  # 두 번째 절반 값
  VAL2=${SECOND_HALF_LIST[$i]}

  # 1) 첫 번째 절반(19개)에 16이 반복
  PART1=$(repeat_value 4 500)   # "16,16,16,...(총19번)"
  # 2) 두 번째 절반(18개)에 VAL2가 반복
  PART2=$(repeat_value $VAL2 500)

  # 최종 PER_CLASS_SHOTS => [PART1, PART2]
  PER_CLASS_SHOTS="[$PART1,$PART2]"

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
