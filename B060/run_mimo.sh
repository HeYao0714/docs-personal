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

export HCCL_BUFFSIZE=300
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export ASCEND_USE_FIA=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_HCCL_BUFFSIZE=2500
export DEEPEP_NORMAL_LONG_SEQ_ROUND=20
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=4096
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_HOST_SOCKET_PORT_RANGE=auto

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export PYTHONPATH=/mnt/share/z00937177/sglang/python:$PYTHONPATH
MODEL_PATH=/mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash
DFLASH_PATH=/mnt/share/weights/MiMo-V2.5-Pro-FP4-DFlash/dflash

sglang serve \
        --model-path $MODEL_PATH \
        --served-model-name $MODEL_PATH \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --mem-fraction-static 0.92 \
        --tp-size 8 --nnodes 1 --node-rank 0 \
        --host 127.0.0.1 \
        --port 9903 \
        --chunked-prefill-size 4096 \
        --max-total-tokens 80000 \
        --max-running-requests 32 \
        --moe-a2a-backend deepep --deepep-mode auto \
        --cuda-graph-bs-decode 1 2 4 8 16 \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path $DFLASH_PATH \
        --speculative-num-draft-tokens 8 \
        --dp-size 2 --enable-dp-attention --enable-dp-lm-head \
