# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
#cd <code_path>
#export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONPATH=/home/luochen/glm/sglang_glm52/sglang/python:$PYTHONPATH

# 内存碎片
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0

# model path
MODEL_PATH=/home/weights/GLM-5.2-0610-Provider-w4a8

P_IP=('61.47.19.68' '61.47.19.67')
P_MASTER="${P_IP[0]}:4567"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

# mtp环境变量
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

#################################### profiling
# export ENABLE_PROFILING=1

# export SGLANG_NPU_USE_MLAPO=1

#### 定位问题使用，测试性能关闭
# export ASCEND_LAUNCH_BLOCKING=1

# export HCCL_OP_EXPANSION_MODE=AIV

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

# export SGLANG_NPU_USE_MULTI_STREAM=1

# export SGLANG_USE_AG_AFTER_QLORA=1

### 调度优化
# export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

export TRANSFORMERS_VERBOSITY=error
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
################################### profiling
        echo "${P_IP[$i]}"
        export HCCL_BUFFSIZE=2500
        python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --dist-init-addr 61.47.19.68:5000 \
        --tp-size 32 --nnodes 2 --node-rank $i \
        --dp-size 8 --enable-dp-attention \
        --chunked-prefill-size 65536 --max-prefill-tokens 280000 \
        --trust-remote-code \
        --host ${P_IP[$i]} \
        --mem-fraction-static 0.60 \
        --port 6688 \
        --served-model-name glm-5 \
        --cuda-graph-max-bs 8 \
        --max-running-requests 64 \
        --quantization modelslim \
        --speculative-draft-model-quantization unquant \
        --moe-a2a-backend deepep --deepep-mode auto \
        --load-balance-method round_robin \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
        NODE_RANK=$i
        break
    fi
done
