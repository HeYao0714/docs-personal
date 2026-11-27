#!/bin/bash

export ASCEND_MF_TRANSFER_PROTOCOL="sdma"

# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# 绑核
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

unset PYTHONPATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 与 K3 启动脚本使用相同的 SGLang 源码。
#export PYTHONPATH=/home/z00842572/dev/sglang/python:$PYTHONPATH

export TRANSFORMERS_VERBOSITY=error

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# pd传输，IP设置为P节点
export ASCEND_MF_STORE_URL="tcp://192.168.25.216:24667"

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200


# 模型路径
MODEL_PATH=/home/weights/GLM-5.2-w4a8


export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export TASK_QUEUE_ENABLE=2
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0
export DEEPEP_HCCL_BUFFSIZE=2000
unset PYTORCH_NPU_ALLOC_CONF
# zbal if use mix alloc （开启混合分配减少内存碎片）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
### 这个不能去掉，开启pp时需要首层有topk_idnex
export SGLANG_PP_LAYER_PARTITION=18,20,20,20
### 性能优化all2all
export DEEP_USE_ALLTOALL_MODE=1
export SGLANG_TORCH_PROFILER_DIR="./sglang_profile"
export SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1
export PYTHONPATH=/home/z00842572/sgl/code/sglang/python:$PYTHONPATH
# P节点
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --port 8024 \
    --trust-remote-code \
    --tp-size 4 \
    --enable-dynamic-chunking \
    --mem-fraction-static 0.8 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --max-running-requests 16 \
    --served-model-name glm-5.2-w4a8 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-metrics \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --disable-cuda-graph \
    --dtype bfloat16 \
    --moe-dense-tp-size 1 \
    --pp-size 4
