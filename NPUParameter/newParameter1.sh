Function Type：Optimization/debug options
1.
参数名称：--image-processor-backend
描述：Image processor backend. 'auto' lets Transformers select the best available backend.
Defaults :`auto`
Options :`auto`, `torchvision`, `pil`

备注：替换--disable-fast-image-processor

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
