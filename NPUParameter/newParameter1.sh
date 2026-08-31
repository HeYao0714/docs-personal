Function Type：Optimization/debug options
1.
参数名称：--image-processor-backend
描述：Image processor backend. 'auto' lets Transformers select the best available backend.
Defaults :`auto`
Options :`auto`, `torchvision`, `pil`

备注：替换--disable-fast-image-processor
这个参数做自动化测试的价值有限，但建议做：

a. 参数解析测试（必做，10行代码）:


python
 
def test_image_processor_backend_parsing(self):
    for backend in ["auto", "torchvision", "pil"]:
        sa = self._parse(["--image-processor-backend", backend])
        self.assertEqual(sa.image_processor_backend, backend)
b. 废弃参数兼容测试（必做）:


python
 
def test_disable_fast_image_processor_migration(self):
    sa = self._parse(["--disable-fast-image-processor"])
    self.assertEqual(sa.image_processor_backend, "pil")
c. 端到端启动测试（可选）:


python
 
def test_server_starts_with_image_processor_backend(self):
    # 启动服务，验证能正常初始化 multimodal processor
    # 不需要真实图片输入
为什么不用验证 backend 实际差异：

这个参数只是透传，实际选择逻辑在 transformers 库
pil 和 torchvision 输出应该一致
测了也测不出 sglang 代码的问题
总结: 写 a 和 b 就够了，保证参数能解析、废弃参数能迁移。端到端测试根据团队要求决定。

2.
参数名称：--enable-deterministic-inference
描述：Enable deterministic inference mode with batch invariant ops.
Defaults :FALSE
Options :bool flag

3.
参数名称：--rl-on-policy-target
描述：The training system that SGLang needs to match for true on-policy
Defaults : None
Options : fsdp

4.
参数名称：--enable-attn-tp-input-scattered
描述：Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.
Defaults : FALSE
Options : bool flag(set to enable)

备注：SGLANG_USE_AG_AFTER_QLORA环境变量
