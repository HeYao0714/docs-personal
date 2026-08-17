unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

#export SGLANG_DEBUG_TENSOR=1
#export SGLANG_DEBUG_TENSOR_FILE=/tmp/sglang_debug.log
rm -f /tmp/sglang_debug.log

pkill -9 python | pkill -9 sglang | pkill -9 VLLM
pkill -9 python | pkill -9 sglang | pkill -9 VLLM

# source /home/x00452483/cann-0506/cann/set_env.sh
# source /home/l00951279/cann-0612/cann/set_env.sh
# source /home/b050cann/cann/set_env.sh
# source /usr/local/Ascend/cann/set_env.sh

#export PYTHONPATH=/home/l00951279/sglang-0707/sglang/python:$PYTHONPATH
#export PYTHONPATH=/home/lws/sglang_main/python:$PYTHONPATH
export PYTHONPATH=/home/lws/sglang/python:$PYTHONPATH

# export HCCL_TOPO_FILE_PATH=/etc/superpod_1d_noroce.json
# export HCCL_TOPO_FILE_PATH=/etc/server_8p_noroce.json
export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68

export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# MODEL_PATH=/mnt/share/weight/GLM-5-FP8-mxw4a8_0518
MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4
#MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8

# [DEBUG]
# export ENABLE_PROFILING=1
# export PROFILING_BS=8
# export PROFILING_STAGE="decode"
# export PROFILING_step=50
# export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_GLOBAL_LOG_LEVEL=4
# export ASCEND_MODULE_LOG_LEVEL=RUNTIME=4

# [FIA]  
export ASCEND_USE_FIA=1
# export SGLANG_USE_FIA_NZ=1

# [MLAPO]  
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_NPU_GLM_NEXTN_BF16_KV_CACHE=1

# [DEEPEP]
# export MOE_ENABLE_TOPK_NEG_ONE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=50
# export SGLANG_NPU_DEEPEP_QUANT="MXFP4"
# export SGLANG_DEEPEP_BF16_DISPATCH=1

# [Prefill Delay]
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

export HCCL_IF_BASE_PORT=60001

# zbal
#export HCCL_BUFFSIZE=0
#unset PYTORCH_NPU_ALLOC_CONF
#export SGLANG_ZBAL_LOCAL_MEM_SIZE=80000
#export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
#export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:24669"
# zbal if use mix alloc
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
## zbal if support graph
#export ZBAL_ENABLE_GRAPH=1

# =========== @50ms ==============
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 8 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6677 \
--max-running-requests 4 \
--mem-fraction-static 0.85 \
--quantization modelslim \
--chunked-prefill-size 2048 \
--kv-cache-dtype "fp8_e4m3" \
--dp 2 \
--moe-a2a-backend deepep \
--deepep-mode auto \
--enable-dp-attention \
--enable-dp-lm-head \
--load-balance-method round_robin \
--enable-metrics \

# python3 aisbench_test.py --input_len 131072 --output_len 1024 --data_num 64 --concurrency 16 --dataset_type prefix_cache --repeat_rate 0.9 --dp 2 --prefix_test

#--enable-nsa-prefill-context-parallel \
#--nsa-prefill-cp-mode in-seq-split \
#--enable-nsa-prefill-cp-layer-split \
#--attn-cp-size 4 \

#性能
python3 -m sglang.benchmark.serving \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 31126 \
  --max-concurrency 16 \
  --num-prompts 16 \
  --random-range-ratio 1 \
  --intput-len 6204 \
  --output-len 1024 \
  --temperature 0 \
  --warmup-requests 1 \


#精度
evalscope eval \
  --model glm \
  --api-url http://127.0.0.1:6677/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gpqa_diamond \
  --eval-batch-size 8 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":163840,"timeout":7200,"retries":2}' \
  --dataset-args '{"gpqa_diamond":{"filters":{"remove_until":"</think>"}}}'

