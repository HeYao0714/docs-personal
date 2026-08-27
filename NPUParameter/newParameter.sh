Function Type：Memory and scheduling
1.
参数名称：--retraction-policy
描述：The decode retraction policy to use when the KV cache is full. 'length' preserves the existing behavior and retracts short-output, long-input requests first. 'priority' retracts lower-priority requests first, using the same priority direction as priority scheduling.
Defaults :length 
Options :length, priority

备注：MODEL_PATH=/home/weights/Qwen3.5-9B
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 4 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.08 \
        --port 8555 \
        --enable-metrics \
        --log-level debug \
        --enable-priority-scheduling \
        --retraction-policy priority \
        --schedule-conservativeness 0.0

AI：(
  start=$(date +%s.%N)
  curl -s -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"low2: $INPUT_TEXT 以二为题目写一篇文章不少于50000字\"}],
      \"max_tokens\": 51000,
      \"priority\": 0
    }" > /dev/null
  end=$(date +%s.%N)
  echo "low2 start=$start end=$end" > result_low2.txt
) &

echo "=== 3. 发送高优先级请求 (priority=20) ==="
(
  start=$(date +%s.%N)
  curl -s -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"high: $INPUT_TEXT 以三为题目写一篇文章不少于50000字\"}],
      \"max_tokens\": 52000,
      \"priority\": 20
    }" > /dev/null
  end=$(date +%s.%N)
  echo "high start=$start end=$end" > result_high.txt
) &

# 等待所有后台进程完成
wait

echo "=== 完成时间对比 ==="
cat result_*.txt | sort

# 提取结束时间并比较
high_end=$(grep high result_high.txt | awk -F'end=' '{print $2}')
low1_end=$(grep low1 result_low1.txt | awk -F'end=' '{print $2}')
low2_end=$(grep low2 result_low2.txt | awk -F'end=' '{print $2}')

echo ""
echo "高优先级结束时间: $high_end"
echo "低优先级1结束时间: $low1_end"
echo "低优先级2结束时间: $low2_end"

if (( $(echo "$high_end < $low1_end" | bc -l) )); then
    echo "?  高优先级请求比低优先级1先完成！抢占生效！"
else
    echo "?  高优先级并未早于低优先级1完成。"
fi

if (( $(echo "$high_end < $low2_end" | bc -l) )); then
    echo "?  高优先级请求比低优先级2先完成！抢占生效！"
else
    echo "?  高优先级并未早于低优先级2完成。"


适配进展：通过发送3条max_token高的占满kvcache。高优先级最后发送。结果高优先级最先完成
参数说明：当KVcache满了之后会触发retraction，配置为priority，将会撤回优先级低的

2.
参数名称：--swa-full-tokens-ratio
描述：The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens.
Defaults :0.8
Options :Type: float
备注：只针对部分模型生效，配置该参数走不到相应分支
def is_hybrid_model(
    model_architectures: List[str],
    hybrid_kvcache_ratio: Optional[float],
    context_length: Optional[int],
    attention_chunk_size: Optional[int],
):
    if model_architectures[0] in [
        "MiMoV2FlashForCausalLM",
        "MiMoV2MTP",
    ]:
        return 1
    if hybrid_kvcache_ratio is None:
        return None
    elif (
        hybrid_kvcache_ratio > 0
        and model_architectures[0] == "Llama4ForConditionalGeneration"
        and context_length > attention_chunk_size
    ):
        return hybrid_kvcache_ratio
    else:
        return None

适配进展：https://github.com/sgl-project/sglang/pull/18032 https://github.com/sgl-project/sgl-kernel-npu/pull/357

3.
参数名称：--disable-hybrid-swa-memory
描述：Disable the hybrid SWA memory.
Defaults :FALSE
Options :bool flag (set to enable)
备注：服务正常启动，curl请求返回精度正常，调用处输出相关日志
适配进展：分析完成

4.
参数名称：--enable-dynamic-chunking
描述：Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.
Defaults : 
Options : 
备注：依赖PP  SR20260106538909
适配进展：PD分离适配 重新测试没有精度问题，pr已关闭
