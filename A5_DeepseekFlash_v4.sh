#!/bin/bash
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
source /home/z00894098/CANN/cann/set_env.sh
source /home/z00894098/CANN/nnal/atb/set_env.sh
source /home/z00894098/CANN/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
source /home/z00894098/CANN/cann-9.1.0/opp/vendors/custom_transformer/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export TASK_QUEUE_ENABLE=1
export SGLANG_SET_CPU_AFFINITY=1
# export ASCEND_LAUNCH_BLOCKING=1
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
#export INF_NAN_MODE_FORCE_DISABLE=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# deepep
# HCCL buffer 保持不变
# export HCCL_BUFFSIZE=5120
# export HCCL_INTRA_DMA_ENABLE=1
# decode阶段token数
# export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

# 禁用长序列 combine（A5不支持）
# unset DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ

# dsv4
export IS_DEEPSEEK_V4=1 
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1

# skip gpu branch
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False
# export ROUTED_EXPERTS_FP8_ACTIVATION=True

# path
# export PYTHONPATH=/home/l00567497/sglang/python:$PYTHONPATH
export PYTHONPATH=/home/sjs/sglang/python:$PYTHONPATH
export MODEL_PATH=/home/weights/DeepSeek-V4-Flash

# PLOG
export COLLECT_LOGS_PATH=/home/sjs/plog  # 设置用于收集日志的环境路径变量
rm -rf "$COLLECT_LOGS_PATH"
mkdir "$COLLECT_LOGS_PATH"
export ASCEND_SLOG_PRINT_TO_STDOUT=0  # 1/0 Plog是否打屏（推荐为0）
export ASCEND_GLOBAL_LOG_LEVEL=3  # 日志等级 0: debug 1: info 2: warning 3: error
export ASCEND_PROCESS_LOG_PATH="$COLLECT_LOGS_PATH"  # 设置Plog存储路径

# export SGLANG_DSV4_NPU_DEBUG_REF_ATTN=1
# export SGLANG_DSV4_NPU_DEBUG_REF_ATTN_LAYER=0
# export SGLANG_DSV4_NPU_FORCE_TORCH_HC=1
# export SGLANG_DSV4_NPU_FORCE_TORCH_HC_HEAD=1
# export SGLANG_DSV4_NPU_DEBUG_MODEL=1
# export SGLANG_DSV4_NPU_DEBUG_MODEL_LAYER=0,1
# export SGLANG_DSV4_NPU_DEBUG_MODEL_SUMMARY=1
# export SGLANG_DSV4_DUMP_MOE=1

# export SGLANG_DSV4_NPU_DEBUG_LOGITS=1
# export SGLANG_DSV4_NPU_DEBUG_LM_HEAD_REF=1
# export SGLANG_DSV4_NPU_DEBUG_SYNC=1
# export SGLANG_DSV4_NPU_DEBUG_MAX_PRINTS=3

# export SGLANG_DEFAULT_THINKING=1
# export SGLANG_DSV4_REASONING_EFFORT=high

# 当前 custom_ops wheel 与 vllm_ascend 自带 custom OPP 不兼容
export VLLM_PLUGINS=""
# export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
# export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=150

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend dsv4 \
    --device npu \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.75 \
    --max-running-requests 40 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --quantization fp8 \
    --enable-dp-lm-head \
    --enable-dp-attention \
    --dp-size 4 \
    --cuda-graph-bs 1 2 4 8 10 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3
    # --enable-dp-lm-head \
    # --enable-prefill-cp \
    # --cp-strategy interleave \
    # --cuda-graph-bs 1 2 4 8 10 16 \
    # --skip-server-warmup
    # --speculative-algorithm EAGLE \
    # --speculative-num-steps 2 \
    # --speculative-eagle-topk 1 \
    # --speculative-num-draft-tokens 3
    # --moe-a2a-backend deepep \
    # --deepep-mode auto

#性能
python3 -m sglang.benchmark.serving \
  --dataset-name random \
  --dataset-path /home/sharegpt.json \
  --backend sglang \
  --host 127.0.0.1 \
  --port 6688 \
  --max-concurrency 40 \
  --input-len 8192 \
  --output-len 1024 \
  --num-prompts 120 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --seed 42 \
  --extra-request-body \
  '{"sampling_params":{"temperature":0,"max_new_tokens":1024,"ignore_eos":true,"skip_special_tokens":false}}'

#精度
    --model dsv \
    --api-url http://127.0.0.1:6688/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{
        "max_tokens": 163840,
        "timeout": 3600,
        "seed": 3407,
        "top_p": 1.0,
        "temperature": 0.0,
        "n": 1,
        "stream": false,
        "extra_body": {
            "reasoning_effort": "low"
        }
    }' \
    --datasets gsm8k \
    --dataset-hub local \
    --dataset-args '{
        "gsm8k": {
            "local_path": "/home/gsm8k",
            "eval_split": "test",
            "subset_list": ["default"],
            "default_subset": "default",
            "few_shot_num": 0
        }
    }' \
    --eval-batch-size 64

