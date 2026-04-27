#!/bin/bash
# ============================================================
# One-sample smoke test for QA/MCQ contamination detection in
# thinking mode. This mirrors run_qa_dataset_thinking.sh but uses
# exactly one sampled row and saves under <MODEL_NAME>-thinking-one.
#
# Usage:
#   bash run_qa_dataset_thinking_one.sh <DATASET_FILE> <DATASET_NAME> <MODEL_PATH> <MODEL_NAME> [SPLIT] [BASE_URL]
#
# Example:
#   bash scripts/run_qa_dataset_thinking_one.sh /home/dameli/DCQ-OpenAI-Version/8_datasets/mmlu_all.csv MMLU /data/models/Qwen3.5-2B qwen3.5-2b
# ============================================================

DATASET_FILE="${1}"
DATASET_NAME="${2}"
MODEL="${3}"
MODEL_NAME="${4}"
SPLIT="${5:-test}"
BASE_URL="${6:-http://localhost:23333/v1}"

if [[ -z "$DATASET_FILE" || -z "$DATASET_NAME" || -z "$MODEL" || -z "$MODEL_NAME" ]]; then
    echo "Usage: bash run_qa_dataset_thinking_one.sh <DATASET_FILE> <DATASET_NAME> <MODEL_PATH> <MODEL_NAME> [SPLIT] [BASE_URL]"
    echo "Example: bash scripts/run_qa_dataset_thinking_one.sh /home/dameli/DCQ-OpenAI-Version/8_datasets/mmlu_all.csv MMLU /data/models/Qwen3.5-2B qwen3.5-2b"
    exit 1
fi

DATASET_BASENAME=$(basename "${DATASET_FILE}" .csv)
EXPERIMENT="$(dirname "$0")/../results/${MODEL_NAME}-thinking-one/${DATASET_BASENAME}"

# Force TensorFlow (used by BLEURT) to run on CPU so it doesn't compete
# with vLLM for GPU memory.
export CUDA_VISIBLE_DEVICES=""

/home/dameli/miniconda3/envs/TTinLLMs/bin/python "$(dirname "$0")/../src/run.py" \
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
    --sample_size 1 \
    --thinking_mode \
    --thinking_budget 12000 \
    --max_tokens 12000 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --sampling_min_p 0.0 \
    --presence_penalty 1.5 \
    --repetition_penalty 1.0 \
    --process_guided_replication \
    --process_general_replication \
    --rouge_eval \
    --bleurt_eval \
    --icl_eval \
    --icl_model gpt-5 \
    --icl_max_tokens 1000
