#MODEL_PATH=/home/weights/Qwen3.5-397B-A17B-w4a8
MODEL_PATH=/home/weights/Qwen3-4B

evalscope eval \
--model $MODEL_PATH \
--api-url http://127.0.0.1:31025/v1 \
--api-key EMPTY \
--eval-type openai_api \
--generation-config '{
      "max_tokens": 1024,
      "timeout": 600,
      "stream": true
}' \
--datasets gsm8k \
--dataset-args '{"gsm8k": {"dataset_id": "/home/q30063557/workspace/acc-test/dataset/gsm8k"}}' \
--eval-batch-size 64 \
--ignore-errors \
--limit 256
