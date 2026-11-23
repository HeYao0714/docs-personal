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

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0

#export PYTHONPATH=/home/h00848570/code/sglang-main/python:$PYTHONPATH

# model path
MODEL_PATH=/home/weights/GLM-5.3-w8a8c8

P_IP=('61.47.19.70' '61.47.19.71')

# mtp环境变量
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEPEP_HCCL_BUFFSIZE=900

LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

echo ""${LOCAL_HOST1}""
echo ""${LOCAL_HOST2}""

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1


export DEEPEP_HYBRID_DEPLOYMENT=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
for i in ""${!P_IP[@]}"";
do
    if [[ ""$LOCAL_HOST1"" == ""${P_IP[$i]}"" || ""$LOCAL_HOST2"" == ""${P_IP[$i]}"" ]];
    then
        echo ""${P_IP[$i]}""
        python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --dist-init-addr 61.47.19.70:5000 \
        --tp-size 32 --nnodes 2 --node-rank $i \
        --attn-dp-size 16 \
        --chunked-prefill-size 131072 \
        --trust-remote-code \
        --host ${P_IP[$i]} \
        --mem-fraction-static 0.84 \
        --port 5599 \
        --context-length 200000 \
        --served-model-name glm-5 \
        --cuda-graph-max-bs-decode 2 \
        --max-running-requests 32 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto \
        --load-balance-method round_robin \
        --served-model-name glm-5.3-w8a8 \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --enable-metrics \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
        break
    fi
done



#        --speculative-draft-model-quantization unquant \
