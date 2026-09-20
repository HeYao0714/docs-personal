#!/bin/bash
# 轻量级 bench_serving 单独测试
set -euo pipefail
export PYTHONPATH=/home/zhm/0729_sglang/alter/sglang/python:$PYTHONPATH
# export PYTHONPATH=/home/zhm/0729_sglang/baseline/sglang/python:$PYTHONPATH
# export PYTHONPATH=/home/zhm/0729_sglang/matsueyu/sglang/python:$PYTHONPATH

# MODEL_PATH="/home/zhm/myweights/Qwen3.5-35B-A3B"
MODEL_PATH=/home/weights/Qwen3.5-122B
# MODEL_PATH="/home/weights/Qwen3.5-35B-A3B"
HOST=127.0.0.1
PORT=8964

python3 -m sglang.benchmark.serving \
  --host ${HOST} \
  --port ${PORT} \
  --model ${MODEL_PATH} \
  --tokenizer ${MODEL_PATH} \
  --backend sglang \
  --dataset-name random-ids \
  --random-input-len 2048 \
  --random-output-len 32768 \
  --random-range-ratio 1 \
  --max-concurrency 208 \
  --num-prompts 208 \
  --request-rate inf \
  --seed 1234 \
  --tokenize-prompt

echo "single test done."
  # --random-output-len 32768 \
