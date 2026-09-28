# 卡TTFT 3s，2890.03
# P节点
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# hccl相关环境变量
export HCCL_BUFFSIZE=300
export HCCL_OP_EXPANSION_MODE=AIV

# 内存碎片管理
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

# 使用ascendc的flash attention
export ASCEND_USE_FIA=1

# PD分离配置MF_STORE
export ASCEND_MF_STORE_URL="tcp://141.61.45.113:24669"
export ASCEND_MF_TRANSFER_PROTOCOL="device_urma"

# deepep相关环境变量
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_HCCL_BUFFSIZE=2500
# 在调度阶段启用蚂蚁搬家
export DEEPEP_NORMAL_LONG_SEQ_ROUND=10
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=4096
# combine开启蚂蚁搬家，对性能有影响，如果能运行则无需开启
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_HOST_SOCKET_PORT_RANGE=auto

# USE_SCATTER_PA_KV_CACHE
export SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1

#export PYTHONPATH=/mnt/share/w00937173/sglang/python:$PYTHONPATH
MODEL_PATH=/mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash
DFLASH_PATH=/mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash/dflash

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 8 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size 16384 \
        --trust-remote-code --port 10000 \
        --host 141.61.45.113 --max-running-requests 64 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 800000 \
        --swa-full-tokens-ratio 0.3 \
        --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
        --disaggregation-bootstrap-port 8996 \
        --moe-a2a-backend deepep --deepep-mode normal \
        --disable-radix-cache

        # --dp-size 2 --enable-dp-attention --enable-dp-lm-head \
​




# D 节点
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# hccl相关环境变量
export HCCL_BUFFSIZE=300
export HCCL_OP_EXPANSION_MODE=AIV

# 内存碎片管理
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

# 使用eagle_worker_v2，并开启plan stream，掩盖草稿模型和主模型准备阶段
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# 使用ascendc的flash attention
export ASCEND_USE_FIA=1

# PD分离配置MF_STORE
export ASCEND_MF_STORE_URL="tcp://141.61.45.113:24669"
export ASCEND_MF_TRANSFER_PROTOCOL="device_urma"

# deepep相关环境变量
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_HCCL_BUFFSIZE=1200

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_HOST_SOCKET_PORT_RANGE=auto

# DFlash Target model's context_length (1048576) is greater than the derived context_length (262144)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# USE_SCATTER_PA_KV_CACHE
export SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1

# export PYTHONPATH=/mnt/share/w00937173/sglang/python:$PYTHONPATH
MODEL_PATH=/mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash
DFLASH_PATH=/mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash/dflash

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --attention-backend ascend \
    --device npu \
    --tp-size 8 --nnodes 1 --node-rank 0 \
    --trust-remote-code --port 10001 \
    --host 141.61.45.118 --max-running-requests 16 \
    --mem-fraction-static 0.9 \
    --swa-full-tokens-ratio 0.1 \
    --cuda-graph-bs-decode 1 2 4 8 12 16 \
    --disaggregation-mode decode --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8996 \
    --moe-a2a-backend deepep --deepep-mode low_latency \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path $DFLASH_PATH \
    --speculative-num-draft-tokens 8 \
    --enable-metrics \
    --disable-radix-cache
​


# Router
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://141.61.45.113:10000 \
    --decode  http://141.61.45.118:10001 \
    --host 141.61.45.113  \
    --port 8010 \
    --health-check-interval-secs 3600 \
    --mini-lb
​



# Bench serving P
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 141.61.45.113 \
  --port 8010 \
  --model /mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash \
  --dataset-path /mnt/share/w00937173/run_file/mimo-v2.5-pro/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --tokenize-prompt \
  --random-input-len 16384 \
  --random-output-len 1 \
  --request-rate 1.1 \
  --random-range-ratio 1 \
  --num-prompts 200 \
