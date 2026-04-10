#!/bin/bash
# ============================================================
# Start vLLM server in THINKING mode (vLLM v1 engine).
#
# IMPORTANT: Do NOT set VLLM_USE_V1=0 — v1 engine is required
# for chat_template_kwargs enable_thinking to work.
#
# Usage:
#   bash scripts/serve_vllm_think.sh [MODEL_PATH] [PORT]
#
# Defaults:
#   MODEL_PATH = /data/models/Qwen3.5-2B
#   PORT       = 23333
# ============================================================

MODEL_PATH="${1:-/data/models/Qwen3.5-2B}"
PORT="${2:-23333}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate TTinLLMs

CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
