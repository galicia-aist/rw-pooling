#!/bin/bash

eval "$(/mnt/miniconda3/bin/conda shell.bash hook)"
conda activate graph-pooling-v1


# ==============================
# Config
# ==============================
EXP_NAME="baseline_models"
RUNS=2

DATASETS=("ENZYMES" "IMDB-BINARY" "MUTAG" "PROTEINS" "REDDIT-BINARY")
METHODS=("mean" "uniform" "topk" "sag" "diffpool" "countsketch")

# ==============================
# Run experiments
# ==============================

for DATASET in "${DATASETS[@]}"
do
  for METHOD in "${METHODS[@]}"
  do
    echo "Running dataset=${DATASET}, method=${METHOD}..."

    python main.py \
      --dataset "$DATASET" \
      --pmethod "$METHOD" \
      --runs $RUNS \
      --exp_name "$EXP_NAME"

  done
done

echo "All experiments completed."