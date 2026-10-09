# --prefill-decode-interval：在调度下一个预填充批次之前，预填充批次之后需要运行的解码轮数。在数据并行注意力模式下，该间隔在所有DP（数据并行）等级之间同步。设置为0以禁用此功能。


export  SGLANG_LOG_FORWARD_ITERS=1

python3 -m sglang.launch_server \
  --model-path /home/weights/Qwen3-0.6B \
  --host 127.0.0.1 --port 6688 \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --prefill-decode-interval 400 \
  --max-running-requests 16

> --max-running-requests 客户端并发发来两条请求，这个参数参数只要大于等于2即可，保证这两个请求是允许并发的
> --prefill-decode-interval 观察Decode轮数大于等于400

for i in {1..2}
do
    curl --location 'http://127.0.0.1:6688/generate' --header 'Content-Type: application/json' --data '{
        "text": "Where is China",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 640,
            "ignore_eos": true
        }
    }' &
done

这里的 `"ignore_eos": true` 是为了保证稳定输出 640 个Token，这个640是为了大于`--prefill-decode-interval 400`并流出余量。

```log
[2026-08-27 08:30:07] Prefill batch [10], #new-seq: 1, #new-token: 128, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 1, #pending-token: 0, npu graph: False, input throughput (token/s): 2.66
[2026-08-27 08:30:07] Decode batch [42], #running-req: 1, #token: 128, token usage: 0.00, npu graph: True, gen throughput (token/s): 0.79, #queue-req: 1
[2026-08-27 08:30:07] Decode batch [82], #running-req: 1, #token: 128, token usage: 0.00, npu graph: True, gen throughput (token/s): 87.33, #queue-req: 1
[2026-08-27 08:30:08] Decode batch [122], #running-req: 1, #token: 128, token usage: 0.00, npu graph: True, gen throughput (token/s): 87.41, #queue-req: 1
[2026-08-27 08:30:08] Decode batch [162], #running-req: 1, #token: 256, token usage: 0.00, npu graph: True, gen throughput (token/s): 84.84, #queue-req: 1
[2026-08-27 08:30:09] Decode batch [202], #running-req: 1, #token: 256, token usage: 0.00, npu graph: True, gen throughput (token/s): 87.40, #queue-req: 1
[2026-08-27 08:30:09] Decode batch [242], #running-req: 1, #token: 256, token usage: 0.00, npu graph: True, gen throughput (token/s): 87.82, #queue-req: 1
[2026-08-27 08:30:10] Decode batch [282], #running-req: 1, #token: 384, token usage: 0.00, npu graph: True, gen throughput (token/s): 91.04, #queue-req: 1
[2026-08-27 08:30:10] Decode batch [322], #running-req: 1, #token: 384, token usage: 0.00, npu graph: True, gen throughput (token/s): 90.12, #queue-req: 1
[2026-08-27 08:30:11] Decode batch [362], #running-req: 1, #token: 384, token usage: 0.00, npu graph: True, gen throughput (token/s): 90.62, #queue-req: 1
[2026-08-27 08:30:11] Decode batch [402], #running-req: 1, #token: 512, token usage: 0.00, npu graph: True, gen throughput (token/s): 97.35, #queue-req: 1
[2026-08-27 08:30:11] Prefill batch [411], #new-seq: 1, #new-token: 128, #cached-token: 0, token usage: 0.00, #running-req: 1, #queue-req: 0, #pending-token: 0, npu graph: False, input throughput (token/s): 28.59
[2026-08-27 08:30:11] Decode batch [443], #running-req: 2, #token: 640, token usage: 0.00, npu graph: True, gen throughput (token/s): 160.73, #queue-req: 0
```

411 - 10 = 401 >= 400

```log
[2026-08-27 08:30:30] Prefill batch [1052], #new-seq: 1, #new-token: 128, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 1, #pending-token: 0, npu graph: False, input throughput (token/s): 6.88
[2026-08-27 08:30:30] Decode batch [1084], #running-req: 1, #token: 128, token usage: 0.00, npu graph: True, gen throughput (token/s): 3.05, #queue-req: 1
[2026-08-27 08:30:30] Decode batch [1124], #running-req: 1, #token: 128, token usage: 0.00, npu graph: True, gen throughput (token/s): 119.96, #queue-req: 1
[2026-08-27 08:30:31] Decode batch [1164], #running-req: 1, #token: 128, token usage: 0.00, npu graph: True, gen throughput (token/s): 119.67, #queue-req: 1
[2026-08-27 08:30:31] Decode batch [1204], #running-req: 1, #token: 256, token usage: 0.00, npu graph: True, gen throughput (token/s): 117.02, #queue-req: 1
[2026-08-27 08:30:31] Decode batch [1244], #running-req: 1, #token: 256, token usage: 0.00, npu graph: True, gen throughput (token/s): 113.28, #queue-req: 1
[2026-08-27 08:30:32] Decode batch [1284], #running-req: 1, #token: 256, token usage: 0.00, npu graph: True, gen throughput (token/s): 112.53, #queue-req: 1
[2026-08-27 08:30:32] Decode batch [1324], #running-req: 1, #token: 384, token usage: 0.00, npu graph: True, gen throughput (token/s): 120.64, #queue-req: 1
[2026-08-27 08:30:32] Decode batch [1364], #running-req: 1, #token: 384, token usage: 0.00, npu graph: True, gen throughput (token/s): 121.10, #queue-req: 1
[2026-08-27 08:30:33] Decode batch [1404], #running-req: 1, #token: 384, token usage: 0.00, npu graph: True, gen throughput (token/s): 120.43, #queue-req: 1
[2026-08-27 08:30:33] Decode batch [1444], #running-req: 1, #token: 512, token usage: 0.00, npu graph: True, gen throughput (token/s): 119.41, #queue-req: 1
[2026-08-27 08:30:33] Prefill batch [1453], #new-seq: 1, #new-token: 128, #cached-token: 0, token usage: 0.00, #running-req: 1, #queue-req: 0, #pending-token: 0, npu graph: False, input throughput (token/s): 37.52
[2026-08-27 08:30:33] Decode batch [1485], #running-req: 2, #token: 640, token usage: 0.00, npu graph: True, gen throughput (token/s): 198.71, #queue-req: 0
```

1453 - 1052 = 401 >= 400

Prefill batch 与 Decode batch 后的第一个数字表示全局forward的次数。
--prefill-decode-interval 也是按照forward的轮次算，但是因为 prefill 被阻塞，这里的forward只包含decode。

`[2026-08-27 08:30:07] Prefill batch [10]` 中一开始是10是因为warmup的时候 1次prefill + 8次的max_tokens的decode = 9，所以进来的第一条请求是第10次forward。
如果添加 --skil-server-warmup，就可以看到Prefill batch的初始值是从1开始的。

由于中间要求隔400轮的forward，因此理想情况下相邻的两个prefill请求就是间隔401。
