"""
Test LoRA on models with tied lm_head (tie_word_embeddings=True).

When tie_word_embeddings=True, lm_head shares the same weight tensor as
embed_tokens. PyTorch's named_modules() deduplicates by object identity, so
lm_head won't appear as a separate module in the LoRAManager scan, and an
lm_head-only adapter would be silently skipped (#18634). This test validates
that SGLang unties lm_head before LoRA wrapping, applies the adapter, and
agrees numerically with the HuggingFace+PEFT reference.

Two assertions, because the two questions need different statistics:

1. "is the adapter actually applied?"  Compare SGLang against *itself* with and
   without the adapter, scoring one fixed token sequence.  Same framework, same
   kernels, same shapes -> the difference is exactly zero when LoRA is skipped
   (measured 0) and O(1e-1) when applied.  No cross-framework noise enters, so
   this is the assertion that actually guards #18634.

2. "does it agree with the reference implementation?"  Compare SGLang against
   HF+PEFT with both scoring the SAME token ids, teacher-forced.

Why the token ids must be shared (not just the prompt):

Comparing top-k logprobs positionally after letting each side greedily generate
its own continuation is unsound under bf16.  A ~1e-1 logit difference can flip
greedy's argmax mid-generation, after which every later logprob scores a
different context and the diff jumps to O(1..10).  The metric then measures
greedy-path agreement rather than numerical agreement, and no continuous
threshold separates the regimes: measured non-flip samples sit at 0.237-0.241
against the old 0.25 threshold while flip samples read 6.3-7.5.  With one
sequence forced through both sides the metric is continuous again -- residual
is pure bf16 noise (measured 0.049-0.144 over 3 repeats x 2 prompts).
"""

import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

import torch

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "peft", "--no-deps"])
    from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

try:
    from sglang.test.ascend.test_ascend_utils import QWEN3_5_4B_WEIGHTS_PATH
except ImportError:
    # Some trees do not ship the Ascend test utils; fall back to the local
    # weights path used by the NPU runners.
    QWEN3_5_4B_WEIGHTS_PATH = "/home/weights/Qwen3.5-4B"

os.environ["SGLANG_ENABLE_FAST_INPUT_LOGPROBS"] = "0"

register_npu_ci(est_time=180, suite="full-1-npu-a3", nightly=True)

# Use a small model with tie_word_embeddings=True
BASE_MODEL = QWEN3_5_4B_WEIGHTS_PATH
TEST_PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]

MAX_NEW_TOKENS = 16

# A skipped lm_head LoRA yields exactly 0 for the same-framework delta, so the
# bar only has to clear numerical noise.  The seeded adapter moves logprobs by
# ~3e-1 (measured), giving an order of magnitude of headroom.
LORA_APPLIED_MIN_DELTA = 1e-2

# Cross-framework bf16 noise. Measured: parity 0.049-0.144, while a real
# mismatch (adapter applied on one side only) reads 0.307.
LOGPROB_THRESHOLD = 2e-1


def create_lora_adapter_with_lm_head(base_model_name: str, output_dir: str):
    """
    Programmatically create a LoRA adapter that targets lm_head,
    using a model with tie_word_embeddings=True.

    The adapter uses randomly initialized LoRA weights (no training). This is
    sufficient to test that SGLang can load the adapter, actually applies it to
    the tied lm_head, and agrees with the HF reference.

    The RNG is seeded so the adapter -- and therefore the measured delta -- is
    reproducible across CI runs.
    """
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Verify the model actually has tied embeddings
    assert (
        model.config.tie_word_embeddings
    ), f"Expected tie_word_embeddings=True for {base_model_name}"

    # Only target lm_head to isolate the test to the tied-embedding scenario.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["lm_head"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    # PEFT initializes lora_B to zeros by default, which makes the adapter
    # produce identical output to the base model. Initialize lora_B with
    # non-zero random weights so the adapter has a visible effect.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    peft_model.save_pretrained(output_dir)

    # Verify the saved adapter contains lm_head keys
    from safetensors import safe_open

    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
    f = safe_open(safetensors_path, framework="pt")
    lm_head_keys = [k for k in f.keys() if "lm_head" in k]
    assert (
        len(lm_head_keys) > 0
    ), f"Expected lm_head LoRA weights in adapter, got keys: {sorted(f.keys())}"

    print(f"Created LoRA adapter at {output_dir}")
    print(f"  lm_head keys: {lm_head_keys}")

    # Clean up the model to free memory
    del peft_model, model
    torch.npu.empty_cache()


def greedy_generate(engine, token_ids, lora_path):
    """Greedy-generate from explicit token ids; return the generated token ids."""
    out = engine.generate(
        input_ids=[list(token_ids)],
        sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0},
        lora_path=lora_path,
    )
    # Batched input_ids returns a list of per-request results.
    if isinstance(out, list):
        out = out[0]
    return [int(t) for t in out["output_ids"]]


def score_sequence(engine, token_ids, lora_path):
    """Teacher-forced logprob of every position of ``token_ids``, under SGLang.

    ``input_token_logprobs`` entry 0 has no meaning in SGLang, so it is dropped;
    that also aligns it with HF's score of tokens 1..n-1.
    """
    out = engine.generate(
        input_ids=[list(token_ids)],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
        lora_path=lora_path,
    )
    if isinstance(out, list):
        out = out[0]
    return [t[0] for t in out["meta_info"]["input_token_logprobs"][1:]]


def hf_score_sequence(model, token_ids):
    """Teacher-forced logprob of ``token_ids[1:]`` in a single HF forward."""
    ids = torch.tensor([list(token_ids)])
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    return [logprobs[k - 1, int(token_ids[k])].item() for k in range(1, len(token_ids))]


class TestLoRATiedLMHead(CustomTestCase):
    """
    Test that LoRA works correctly on models with tied lm_head.
    """

    _adapter_dir = None

    @classmethod
    def setUpClass(cls):
        """Create a temporary LoRA adapter with lm_head targeting."""
        super().setUpClass()
        cls._adapter_dir = tempfile.mkdtemp(prefix="sglang_test_lora_tied_lm_head_")
        create_lora_adapter_with_lm_head(BASE_MODEL, cls._adapter_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary adapter directory."""
        if cls._adapter_dir and os.path.exists(cls._adapter_dir):
            shutil.rmtree(cls._adapter_dir)
        super().tearDownClass()

    def test_tied_lm_head_lora_hf_sgl_logprob_match(self):
        """
        Verify the tied lm_head adapter is applied and matches HF+PEFT.

        Both frameworks score one token sequence, so the parity diff is
        numerical noise rather than greedy-path divergence.
        """
        prompts = TEST_PROMPTS[:2]
        tokenizer = get_tokenizer(BASE_MODEL)

        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            model_type="generation",
            lora_paths=[self._adapter_dir],
            max_loras_per_batch=1,
            lora_backend="triton",
            lora_target_modules=["lm_head"],
            disable_cuda_graph=True,
            disable_radix_cache=True,
            mem_fraction_static=0.80,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
            attention_backend="ascend",
        ) as srt_runner:
            engine = srt_runner.engine
            results = []
            for prompt in prompts:
                prompt_ids = tokenizer.encode(prompt)
                gen_ids = greedy_generate(engine, prompt_ids, self._adapter_dir)
                seq = prompt_ids + gen_ids
                results.append(
                    (
                        seq,
                        torch.tensor(score_sequence(engine, seq, self._adapter_dir)),
                        torch.tensor(score_sequence(engine, seq, None)),
                    )
                )

        torch.npu.empty_cache()

        # Load HF directly rather than via HFRunner: HFRunner runs the model in a
        # child process, so the parent cannot teacher-force a chosen sequence
        # through it.
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cpu")
        from peft import PeftModel

        hf_model = PeftModel.from_pretrained(
            base_model,
            self._adapter_dir,
            torch_dtype=torch.bfloat16,
            is_trainable=False,
        ).to("cpu")

        for i, (seq, srt_lora_lp, srt_base_lp) in enumerate(results):
            # (1) The adapter must actually change SGLang's output. A skipped
            # lm_head LoRA (the #18634 failure) makes this exactly zero.
            applied_delta = (srt_lora_lp - srt_base_lp).abs().max().item()
            print(
                f"Prompt {i} lm_head LoRA applied delta (SGLang base vs LoRA): "
                f"{applied_delta:.6e}"
            )
            self.assertGreater(
                applied_delta,
                LORA_APPLIED_MIN_DELTA,
                f"Prompt {i}: lm_head LoRA appears not to be applied "
                f"(delta {applied_delta:.6e} <= {LORA_APPLIED_MIN_DELTA:.0e}). "
                f"The tied lm_head was probably not wrapped by LoRAManager.",
            )

            # (2) Numerical parity with the reference implementation, with both
            # sides scoring the same token ids.
            hf_lp = torch.tensor(hf_score_sequence(hf_model, seq))
            parity_diff = (srt_lora_lp - hf_lp).abs().max().item()
            print(
                f"Prompt {i} logprob max_diff (SGLang vs HF+PEFT): {parity_diff:.6e}"
            )
            self.assertLess(
                parity_diff,
                LOGPROB_THRESHOLD,
                f"Prompt {i}: logprob diff {parity_diff:.6e} exceeds threshold "
                f"{LOGPROB_THRESHOLD:.0e}",
            )

        del hf_model, base_model


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
