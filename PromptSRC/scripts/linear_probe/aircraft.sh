#!/bin/bash
#SBATCH --job-name=base2new_linear
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=laal_3090

###############################################################################
# (1) 고정값 세팅
###############################################################################
DATA="/shared"
TRAINER=LinearProbeCLIP

DATASET="fgvc_aircraft"        # 예: "fgvc_aircraft", "oxford_pets" 등
BACKBONE="ViT-B/16"       # 예: "ViT-B/16"
LOSS_TYPE="ce"      # "ce" 또는 "focal"
SEED=1           # random seed
SAMPLER="WeightedClassSampler"        # 예: "RandomSampler" 또는 "WeightedClassSampler"

# focal 사용 여부 (참조할 코드에서의 예시처럼 사용해보고 싶다면)
if [ "${LOSS_TYPE}" == "focal" ]; then
    FOCAL_ARG=True
else
    FOCAL_ARG=False
fi

###############################################################################
# (2) repeat_value 함수 정의
#     전달한 값 $val 을 $count 번 반복해서 "val,val,val,..." 형태의 문자열을 만들어 줌
###############################################################################
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
# (3) "2번째 절반"에 사용할 shot 값을 사전에 정의
#     예: 10번 반복을 위해 10개 정도 정의 (원하는 값 넣어서 실험)
###############################################################################
SECOND_HALF_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

###############################################################################
# (4) 루프 돌면서 여러 실험
#     i = 0..9 => SHOTS = -50, -51, -52, ... -59
###############################################################################
for i in "${!SECOND_HALF_LIST[@]}"; do
  # i=0 -> SHOTS=-50, i=1 -> SHOTS=-51, ...
  SHOTS=$(( 100 + i ))
  SHOTS="-$SHOTS"

  # 두 번째 절반 값을 SECOND_HALF_LIST에서 뽑아 사용
  VAL2=${SECOND_HALF_LIST[$i]}

  # 아래는 "참조할 코드"에서처럼, 
  # 앞 절반에는 16이 반복되고, 뒤 절반에는 VAL2가 반복되는 예시
  # 갯수(19, 18)는 원하는 대로 맞춰서 바꾸면 됨
  PART1=$(repeat_value 16 50)   # "16"을 19번
  PART2=$(repeat_value $VAL2 50)

  PER_CLASS_SHOTS="[$PART1,$PART2]"

  ###############################################################################
  # (5) config 설정
  #     - cfg 예시는 vit_b16_ep50
  ###############################################################################
  CFG="vit_b16_ep50"
  CONFIG_PATH="configs/trainers/${TRAINER}/${CFG}.yaml"

  # 출력 폴더 (원하는 디렉토리 구조에 맞춰 수정)
  DIR=output/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${BACKBONE}_${LOSS_TYPE}/seed${SEED}

  ###############################################################################
  # (6) 정보 출력
  ###############################################################################
  echo "============================================================"
  echo ">>> Experiment #$((i+1))"
  echo "    DATASET          = ${DATASET}"
  echo "    BACKBONE         = ${BACKBONE}"
  echo "    LOSS_TYPE        = ${LOSS_TYPE}"
  echo "    SEED             = ${SEED}"
  echo "    SAMPLER          = ${SAMPLER}"
  echo "    SHOTS            = ${SHOTS}"
  echo "    PER_CLASS_SHOTS  = ${PER_CLASS_SHOTS}"
  echo "    Output dir       = ${DIR}"
  echo "============================================================"

  ###############################################################################
  # (7) 이미 결과 폴더가 있으면 "Resuming...", 없으면 새로 실행
  ###############################################################################
  if [ -d "$DIR" ]; then
      echo "Results exist in ${DIR}. Resuming..."
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
          DATASET.PER_CLASS_SHOTS "${PER_CLASS_SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES all \
          DATALOADER.TRAIN_X.SAMPLER ${SAMPLER}
  else
      echo "Run this job and save the output to ${DIR}"
      mkdir -p ${DIR}

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
          DATASET.PER_CLASS_SHOTS "${PER_CLASS_SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES all \
          DATALOADER.TRAIN_X.SAMPLER ${SAMPLER}
  fi

  echo
  echo
done
