--mm-processor-worker-num（HF processor 调用线程数）
功能 ：控制 process_and_combine_mm_data （resize/normalize、构建 pixel_values、token 展开）的并发线程数。>1 时每线程持有一份 HF processor 的 copy.deepcopy ，由 MultimodalProcessorExecutor 隔离调度；克隆失败或模型声明不支持并发时静默回退 1。
可观测行为 ：
● 启动日志 Multimodal processor concurrency enabled with N isolated worker threads (explicit/auto). —— N>1 的唯一证据
● 日志 不出现 该行 = 单线程（auto 被锁，或回退）
● 功能透明性：不同线程数下 greedy 输出必须一致
NPU 特有逻辑 ：fast image processor 跑在与 scheduler 同一颗 NPU 上，auto 恒锁 1（模型声明的值也被 cap）；显式传参或 --disable-fast-image-processor 可绕开。预处理同时会按 processor 类名打 Ascend 补丁（qwen-vl/MiniMaxVL/Glm46V）。
--mm-io-worker-num（数据加载/解码线程数）
功能 ：控制 URL 下载、base64 解码、PIL 图像解码、RGB 转换、视频音频读取的并发。纯 CPU，不碰加速器，NPU 上无 nvJPEG、全部 PIL 解码，调大无设备争抢风险。
可观测行为 ：
● 启动日志 Multimodal data loading enabled with N worker threads (explicit/environment/auto). —— 仅 N>4 时打印
● 三级优先级： 显式参数 > SGLANG_IO_WORKERS （仅参数为 0 时生效）> 模型默认 （qwen-vl 系 16，base 4）
NPU 手动测试步骤
用例 1 — 显式参数生效 + 功能正确
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --mm-processor-worker-num 4 --mm-io-worker-num 8 \
  --enable-multimodal --mm-attention-backend ascend_attn --port 30000
# base64 方式（io 线程的解码路径 + processor 线程的预处理路径）
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAIAAgBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APv6/9k="}},
        {"type": "text", "text": "What color is this image? Answer in one word."}
      ]
    }],
    "max_tokens": 32,
    "temperature": 0
  }'
验收：两条日志出现（processor 4 explicit / io 8 explicit）；发 base64 + URL 图片请求返回 200 且描述正确（多线程克隆路径无功能损坏，顺带覆盖 qwen Ascend patch 生效——无 reshape 报错）。
用例 2 — auto 锁 1（NPU 核心行为）
不传两个参数重启。验收：日志 无 concurrency 行（processor 锁 1）；io 行为 16 (auto)（qwen-vl 默认，>4 会打印）。
用例 3 — --disable-fast-image-processor 翻转
加该参数重启。验收：日志出现 concurrency enabled with 2 ... (auto) （预处理回 CPU，锁 1 解除）。
用例 4 — 环境变量优先级
SGLANG_IO_WORKERS=8 python -m sglang.launch_server ...            # 参数缺省
# 验收：io 日志 8 (environment)
SGLANG_IO_WORKERS=8 python -m sglang.launch_server ... --mm-io-worker-num 16
# 验收：io 日志 16 (explicit)，env 被压制
用例 5 — 功能透明性 + 性能
--mm-processor-worker-num 1 与 4 各起一次，同样图片 greedy 对比输出一致； python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16 对比准确率（应相同）与耗时；压测期间 watch -n 1 npu-smi info 观察 NPU 利用率是否被预处理抢占（线程调高可能涨也可能跌，这是 NPU 与 CPU 平台的行为差异点）。

核心验收点 ：① 显式参数能开多线程且功能正确；② auto 时 NPU 锁 1；③ disable-fast 后翻转为 2；④ env 仅在参数为 0 时生效；⑤ worker=1/4 输出一致；⑥ 整页图片无 Ascend reshape 报错。
