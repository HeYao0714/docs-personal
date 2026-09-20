# cpu高性能, performance 所有CPU核心的频率都锁定在最高频率 tee 把performance输出到屏幕和写入CPU核心的频率策略
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# 宁可杀进程，也不准用硬盘当内存
sysctl -w vm.swappiness=20
# 关闭NUMA自动平衡. numa_balancing是Linux的一个自动迁移机制，它会尝试把进程迁移到离内存更近的CPU上. 但大模型推理的内存占用巨大（几十GB），自动迁移会触发大量内存页迁移，产生不可预测的延迟抖动
sysctl -w kernel.numa_balancing=0
# 只有当进程在另一个CPU上运行能节省超过50000纳秒（50微秒）时，才允许迁移
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
# 把SGLang的工作线程绑定到特定的CPU核心上. SGLANG_SET_CPU_AFFINITY=1就是让SGLang内部自动做这件事
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
# cd /home/zhm/0811_sglang/

cd /home/zhm/0729_sglang/
export PYTHONPATH=/home/zhm/0729_sglang/alter/sglang/python:$PYTHONPATH
# export PYTHONPATH=/home/zhm/0729_sglang/baseline/sglang/python:$PYTHONPATH

# 这四行是在删除所有HTTP/HTTPS代理环境变量
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/9.0.0/bisheng_toolkit/set_env.sh

export ASCEND_CUSTOM_OPP_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/vendors/customize
export LD_LIBRARY_PATH=/usr/local/python3.11.13/lib/python3.11/site-packages/vendors/customize/op_api/lib:$LD_LIBRARY_PATH

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=16
# export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# export ASCEND_LAUNCH_BLOCKING=1

export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
# export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

# 可见卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GDN_ATTN_BACKEND_TRITON=1
export TEST_FIV_BACKEND_TRITON=1

export SGLANG_NPU_FUSED_MOE_MODE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

# DeepEP-Ascend normal 模式的 get_dispatch_layout 里 token 数超过了 round * per_round_tokens 上限
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
export HCCL_BUFFSIZE=2400

# 试图用来解决最后的stall问题

# profiling
export ENABLE_PROFILING=0

unset SGLANG_NPU_USE_MULTI_STREAM

# export SGLANG_MAMBA_SSM_DTYPE=bfloat16

MODEL_PATH=/home/weights/Qwen3.5-122B
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 8 \
	--dtype bfloat16 \
        --chunked-prefill-size 32768 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.87 \
        --max-running-requests 112 \
        --port 8964 \
        --cuda-graph-bs-decode 1 4 8 14 16\
        --dp 8 \
        --enable-dp-attention \
        --enable-dp-lm-head \
        --moe-a2a-backend deepep \
        --ep-size 8 \
        --stream-interval 64 \
        --schedule-conservativeness 0.4 \
        --enable-multimodal \
        --mm-attention-backend ascend_attn \
