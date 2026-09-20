# 122B
```sh
evalscope eval   --model /home/weights/Qwen3.5-122B   --api-url http://127.0.0.1:8964/v1   --api-key EMPTY   --eval-type openai_api   --generation-config '{
    "max_tokens": 200000,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 60000,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}
  }'   --datasets aime25   --dataset-args '{"aime25": {"dataset_id": "/home/zhm/0729_sglang/dataset/aime25"}}'   --eval-batch-size 128   --timeout 60000
```

```bash
evalscope eval \
  --model /home/weights/Qwen3.5-122B \
  --api-url http://127.0.0.1:8964/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gpqa_diamond \
  --dataset-args '{"gpqa_diamond": {"local_path": "/home/zhm/0729_sglang/dataset/gpqa_diamond"}}' \
  --eval-batch-size 104 \
  --timeout 60000 \
  --generation-config '{"temperature": 1.0, "max_tokens": 200000,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 60000}'
```

```bash
evalscope eval \
  --model /home/weights/Qwen3.5-122B \
  --api-url http://127.0.0.1:8964/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets mmmu \
  --dataset-args '{"mmmu": {"local_path": "/home/zhm/0729_sglang/dataset/MMMU"}}' \
  --eval-batch-size 104 \
  --timeout 60000 \
  --generation-config '{"temperature": 1.0, "max_tokens": 200000,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 60000}'
```


# 35B

```sh
evalscope eval   --model /home/weights/Qwen3.5-35B-A3B   --api-url http://127.0.0.1:8964/v1   --api-key EMPTY   --eval-type openai_api   --generation-config '{
    "max_tokens": 200000,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 60000,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}
  }'   --datasets aime25   --dataset-args '{"aime25": {"dataset_id": "/home/zhm/0729_sglang/dataset/aime25"}}'   --eval-batch-size 128   --timeout 60000
```

```bash
evalscope eval \
  --model /home/weights/Qwen3.5-35B-A3B \
  --api-url http://127.0.0.1:8964/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gpqa_diamond \
  --dataset-args '{"gpqa_diamond": {"local_path": "/home/zhm/0729_sglang/dataset/gpqa_diamond"}}' \
  --eval-batch-size 256 \
  --timeout 60000 \
  --generation-config '{"temperature": 1.0, "max_tokens": 200000,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 60000}'
```

```shell
evalscope eval \
  --model /home/weights/Qwen3.5-35B-A3B \
  --api-url http://127.0.0.1:8964/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets mmmu \
  --dataset-args '{"mmmu": {"local_path": "/home/zhm/0729_sglang/dataset/MMMU"}}' \
  --eval-batch-size 256 \
  --timeout 60000 \
  --generation-config '{"temperature": 1.0, "max_tokens": 200000,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 60000}'
```
