export MODEL="glm"
export OPENAI_API_BASE_URL="http://10.246.63.17:18994/v1"
export OPENAI_API_KEY="EMPTY"

# generation-config 和智谱官方适配文档对齐
evalscope eval \
  --model "$MODEL" \
  --api-url "$OPENAI_API_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --eval-type openai_api \
  --datasets gpqa_diamond \
  --eval-batch-size 8 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":163840,"timeout":7200,"retries":2}' \
  --dataset-args '{"gpqa_diamond":{"filters":{"remove_until":"</think>"}}}'
