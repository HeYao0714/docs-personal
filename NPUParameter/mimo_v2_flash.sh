# high performance cpu
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

export ASCEND_USE_FIA=1
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCCL_BUFFSIZE=800
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_STAGE="prefill"
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24669"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DEEPEP_BF16_DISPATCH=0
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export MODEL_PATH=/home/weights/MiMo-V2-Flash-w8a8-all-0512
export PYTHONPATH=/home/w00937173/sglang/python:$PYTHONPATH

sglang serve \
    --model-path ${MODEL_PATH} \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --mem-fraction-static 0.85 \
    --swa-full-tokens-ratio 0.95 \
    --reasoning-parser mimo \
    --attention-backend ascend \
    --disable-piecewise-cuda-graph \
    --host 127.0.0.1 \
    --port 8010 \
    --base-gpu-id 0 \
    --max-running-requests 64 \
    --cuda-graph-bs 1 2 4 8 16 \
    --dp-size 4 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --quantization modelslim \
    --skip-server-warmup \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-multi-layer-eagle \
    --speculative-draft-model-quantization unquant \
    --moe-a2a-backend deepep \
    --deepep-mode auto

from types import SimpleNamespace
import requests
from sglang.test.few_shot_gsm8k import run_eval
import os


OUTPUT_DIR = "./profiler_dir"

def _start_profile(**kwargs):
    """Start profiling with optional parameters."""
    requests.post(
        f"http://127.0.0.1:30000/start_profile", # 不需要改
        json=kwargs if kwargs else None,
    )
    # self.assertEqual(response.status_code, 200)

def gsm8k():
    # print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # _start_profile(start_step="200",num_steps=1)
    # print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    args = SimpleNamespace(
        num_shots=5,
        data_path="/home/w00937173/run_file/mimo-v2-flash/test.jsonl", # 改成test.jsonl的路径
        num_questions=200,   #1319
        max_new_tokens=2048,
        parallel=16,
        host="http://61.47.19.68", # 改ip
        port=8010, # 改端口
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    print(f"{metrics['accuracy']=}")
    # self.assertGreater(metrics["accuracy"], 0.7)


if __name__ == "__main__":
    # os.environ["SGLANG_TORCH_PROFILER_DIR"] = OUTPUT_DIR
    gsm8k()
