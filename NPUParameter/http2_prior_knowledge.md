# --http2-max-concurrent-streams ：每个 HTTP/2 连接上通告的最大并发流数（1 到 2^32 - 1）。仅在启用 --enable-http2 时适用。

python3 -m sglang.launch_server \
  --model-path /home/weights/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 6688 \
  --device npu \
  --trust-remote-code \
  --enable-http2 \
  --http2-max-concurrent-streams 2

拉起服务之后，发送请求`curl -v --http2-prior-knowledge http://127.0.0.1:6688/get_model_info`，
返回信息中包含`MAX_CONCURRENT_STREAMS == 2`，说明参数生效了，这个输出的默认值是200。

# MAX_CONCURRENT_STREAMS这个值表示的最大的并发的流的数量，是比较精准的观测点
