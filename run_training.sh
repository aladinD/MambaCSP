#!/usr/bin/env bash
#
# Run DMRS training jobs for all patterns (TDD and FDD) with a shared training script.
# Defaults: 8 GPUs via torchrun; override with NPROC.
#
# Configure runtime backbone through BACKBONE:
#   BACKBONE=gpt   (default, uses GPT-2)
#   BACKBONE=mamba (uses Mamba)
#
# Optional Mamba options:
#   USE_HF_MAMBA=1   to use HF Mamba instead of compact mamba-ssm
#   HF_NAME=state-spaces/mamba-130m-hf
#   MODEL_SUBDIR=mamba   stores weights in dmrs_model_weights/<pattern>/<mode>/mamba
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh                         # GPT-2
#   BACKBONE=mamba ./run_training.sh          # Mamba
#   BACKBONE=mamba USE_HF_MAMBA=1 MODEL_SUBDIR=mamba ./run_training.sh
#   NPROC=4 ./run_training.sh                # override number of processes
#   PATTERNS="pattern1_type1_sparse pattern3_highmob_dense" ./run_training.sh
#
# Notes:
# - Creates output directories if missing (non-destructive).
# - Fails fast on any error.

set -euo pipefail

NPROC="${NPROC:-8}"
BACKBONE="${BACKBONE:-gpt}"
BACKBONE="${BACKBONE,,}"
USE_HF_MAMBA="${USE_HF_MAMBA:-0}"
HF_NAME="${HF_NAME:-state-spaces/mamba-370m-hf}"
MODEL_SUBDIR="${MODEL_SUBDIR:-}"

if [[ "$BACKBONE" != "gpt" && "$BACKBONE" != "mamba" ]]; then
  echo "[ERROR] BACKBONE must be 'gpt' or 'mamba' (got '$BACKBONE')" >&2
  exit 1
fi

DEFAULT_PATTERNS="pattern1_type1_sparse pattern2_type2_densefreq pattern3_highmob_dense"
PATTERNS="${PATTERNS:-$DEFAULT_PATTERNS}"

train() {
  local pattern="$1" mode="$2" u2d="$3"
  local data_root="./dmrs_datasets/${pattern}/${mode}"
  local save_dir="./dmrs_model_weights/${pattern}/${mode}"
  if [[ -n "$MODEL_SUBDIR" ]]; then
    save_dir="${save_dir}/${MODEL_SUBDIR}"
  fi
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

  local mamba_args=()
  if [[ "$BACKBONE" == "mamba" ]]; then
    mamba_args+=(--backbone mamba)
    if [[ "$USE_HF_MAMBA" == "1" ]]; then
      mamba_args+=(--use-hf-mamba --hf-name "${HF_NAME}")
    fi
  else
    mamba_args+=(--backbone gpt)
  fi

  torchrun --nproc_per_node="${NPROC}" train_dmrs.py \
    --u2d "$u2d" \
    "${mamba_args[@]}" \
    --train-his "$train_his" \
    --train-tgt "$train_tgt" \
    --save-path "${save_dir}/${save_name}"
}

for pattern in $PATTERNS; do
  train "$pattern" "TDD" 0
  train "$pattern" "FDD" 1
done

echo "All jobs dispatched."
