#!/usr/bin/env bash
#
# Run DMRS training jobs for all patterns (TDD and FDD) using train_acc_dmrs.py.
# Defaults: 8 GPUs via torchrun; override with NPROC.
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh           # run all
#   NPROC=4 ./run_training.sh   # override number of processes
#   PATTERNS="pattern1_type1_sparse pattern3_highmob_dense" ./run_training.sh
#
# Notes:
# - Creates output directories if missing (non-destructive).
# - Fails fast on any error.

set -euo pipefail

NPROC="${NPROC:-8}"

DEFAULT_PATTERNS="pattern1_type1_sparse pattern2_type2_densefreq pattern3_highmob_dense"
PATTERNS="${PATTERNS:-$DEFAULT_PATTERNS}"

train() {
  local pattern="$1" mode="$2" u2d="$3"
  local data_root="./dmrs_datasets/${pattern}/${mode}"
  local save_dir="./dmrs_model_weights/${pattern}/${mode}"
  local save_name="U2U_LLM4CP.pth"
  if [[ "$u2d" == "1" ]]; then
    save_name="U2D_LLM4CP.pth"
  fi

  local train_his="${data_root}/H_U_his_train.mat"
  local train_tgt="${data_root}/H_U_pre_train.mat"
  if [[ "$u2d" == "1" ]]; then
    train_tgt="${data_root}/H_D_pre_train.mat"
  fi

  mkdir -p "$save_dir"

  echo "=== ${pattern} | ${mode^^} | torchrun nproc=${NPROC} ==="
  echo "data:  $train_his"
  echo "tgt :  $train_tgt"
  echo "save:  ${save_dir}/${save_name}"

  torchrun --nproc_per_node="${NPROC}" train_acc_dmrs.py \
    --u2d "$u2d" \
    --train-his "$train_his" \
    --train-tgt "$train_tgt" \
    --save-path "${save_dir}/${save_name}"
}

for pattern in $PATTERNS; do
  train "$pattern" "TDD" 0
  train "$pattern" "FDD" 1
done

echo "All jobs dispatched."
