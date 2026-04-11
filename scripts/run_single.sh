#!/bin/bash
# ============================================================
# Run contamination detection for a single dataset.
# Edit the variables below, then: bash scripts/run_single.sh
# ============================================================

DATASET_FILE="/home/dameli/DCQ-OpenAI-Version/8_datasets/mmlu_all.csv"
DATASET_NAME="MMLU"
MODEL_PATH="/data/models/Qwen3.5-2B"
MODEL_NAME="qwen3.5-2b"

# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate TTinLLMs
cd /home/dameli/TimeTravelinLLMs

bash scripts/run_qa_dataset.sh \
    "${DATASET_FILE}" \
    "${DATASET_NAME}" \
    "${MODEL_PATH}" \
    "${MODEL_NAME}"
