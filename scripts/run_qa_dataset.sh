#!/bin/bash
# ============================================================
# Reusable script for checking QA/MCQ datasets for contamination
# using the TimeTravelInLLMs method (ROUGE-L + BLEURT).
#
# Usage:
#   bash run_qa_dataset.sh <DATASET_FILE> <DATASET_NAME> <MODEL_PATH> <MODEL_NAME> [SPLIT] [BASE_URL]
#
# Examples:
#   bash run_qa_dataset.sh /home/dameli/8datasets/mmlu_first100.csv MMLU /data/models/Qwen3.5-2B qwen3.5-2b
#   bash run_qa_dataset.sh /home/dameli/8datasets/mmlu_pro_all.csv MMLU-Pro /data/models/Llama-3-8B llama3-8b test
# ============================================================

DATASET_FILE="${1}"
DATASET_NAME="${2}"
MODEL="${3}"
MODEL_NAME="${4}"
SPLIT="${5:-test}"
BASE_URL="${6:-http://localhost:23333/v1}"

if [[ -z "$DATASET_FILE" || -z "$DATASET_NAME" || -z "$MODEL" || -z "$MODEL_NAME" ]]; then
    echo "Usage: bash run_qa_dataset.sh <DATASET_FILE> <DATASET_NAME> <MODEL_PATH> <MODEL_NAME> [SPLIT] [BASE_URL]"
    echo "Example: bash run_qa_dataset.sh /home/dameli/8datasets/mmlu_first100.csv MMLU /data/models/Qwen3.5-2B qwen3.5-2b"
    exit 1
fi

DATASET_BASENAME=$(basename "${DATASET_FILE}" .csv)
EXPERIMENT="$(dirname "$0")/../results/${MODEL_NAME}/${DATASET_BASENAME}"

# Force TensorFlow (used by BLEURT) to run on CPU so it doesn't compete
# with vLLM for GPU memory.
export CUDA_VISIBLE_DEVICES=""

python "$(dirname "$0")/../src/run.py" \
    --experiment "${EXPERIMENT}" \
    --filepath "${DATASET_FILE}" \
    --task qa \
    --dataset "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --model "${MODEL}" \
    --base_url "${BASE_URL}" \
    --sleep_time 0 \
    --text_column question \
    --should_split_text \
    --min_p 40 \
    --max_p 70 \
    --sample_size 100 \
    --process_guided_replication \
    --process_general_replication \
    --rouge_eval \
    --bleurt_eval
