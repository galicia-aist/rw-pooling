#!/bin/bash

eval "$(/mnt/miniconda3/bin/conda shell.bash hook)"
conda activate rw-pooling

# ==============================
# Config
# ==============================
EXP_NAME="pooling_ratio_grid"
RUNS=2

DATASETS=("NCI1")

# Only methods that use pool_ratio (no K, no alpha)
METHODS=("topk")

POOL_RATIOS=(0.1 0.3 0.5 0.7 0.9)

# ==============================
# Run experiments
# ==============================

for DATASET in "${DATASETS[@]}"
do
  for METHOD in "${METHODS[@]}"
  do
    echo "======================================"
    echo "Dataset=${DATASET}, Method=${METHOD}"
    echo "======================================"

    for RATIO in "${POOL_RATIOS[@]}"
    do
      echo "Running ratio=${RATIO}"

      python main.py \
        --dataset "$DATASET" \
        --pmethod "$METHOD" \
        --pool_ratio "$RATIO" \
        --runs $RUNS \
        --exp_name "${EXP_NAME}_${METHOD}_r${RATIO}"

    done

  done
done

echo "All experiments completed."