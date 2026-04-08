#!/bin/bash

# Run TimeTravelinLLMs contamination check on Qwen3.5-2B using AG News test split.
# Prerequisites:
#   1. vLLM server is running:  bash /home/dameli/model_serve/model_serve.sh
#   2. TTinLLMs conda env is active: conda activate TTinLLMs
#   3. Dependencies installed: pip install -r requirements.txt

python ../../../src/run.py \
    --experiment ../../../results/qwen3.5-2b/ag_news/test \
    --filepath ../../../data/ag_news/ag_news_test.csv \
    --task cls \
    --dataset "AG News" \
    --split test \
    --model /data/models/Qwen3.5-2B \
    --text_column text \
    --label_column label \
    --base_url http://localhost:23333/v1 \
    --sleep_time 0 \
    --process_guided_replication \
    --process_general_replication \
    --rouge_eval \
    --bleurt_eval
