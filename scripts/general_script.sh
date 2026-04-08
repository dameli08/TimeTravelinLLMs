#!/bin/bash

# -- Set the variables below with appropriate values --
EXPERIMENT="Path to Your Experiment Results"
FILEPATH="Path to Your Data File"
TASK="Task Corresponds to Your Dataset"       # cls | nli | sum | xsum
DATASET="Your Dataset Name"
SPLIT="Split Corresponds to Your Data"        # train | test | validation
MODEL="/data/models/YourModelName"            # path to local model on disk
BASE_URL="http://localhost:23333/v1"          # vLLM local endpoint
TEXT_COLUMN="Your Text Column Name"
# LABEL_COLUMN="column"  # uncomment for cls/nli tasks
# -- End of variables --

python ../src/run.py \
        --experiment "${EXPERIMENT}" \
        --filepath "${FILEPATH}" \
        --task "${TASK}" \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --model "${MODEL}" \
        --base_url "${BASE_URL}" \
        --sleep_time 0 \
        --text_column "${TEXT_COLUMN}" \
        --should_split_text \
        --process_guided_replication \
        --process_general_replication \
        --rouge_eval \
        --bleurt_eval \
        --max_p 70 \
        --min_p 40
        #--label_column "${LABEL_COLUMN}" \
