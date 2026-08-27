#!/bin/bash

# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export DEEPEP_HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export TRANSFORMERS_VERBOSITY=error


export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_BS=16

export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1


python3 -m sglang.launch_server \
    --model-path /home/weights/GLM-5.2-w4a8 \
    --attention-backend ascend \
    --device npu \
    --tp-size 16 \
    --nnodes 1 \
    --dp-size 4 \
    --enable-dp-attention \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 32768 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --served-model-name glm5 \
    --cuda-graph-bs 4 \
    --max-running-requests 8 \
    --quantization modelslim \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --load-balance-method round_robin \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --host 0.0.0.0 \
    --port 8810
    
