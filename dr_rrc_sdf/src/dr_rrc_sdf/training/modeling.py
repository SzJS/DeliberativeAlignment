"""STUB — model/tokenizer loading and LoRA configuration, shared by both training stages.

Contract
    load_model_and_tokenizer(cfg, stage) -> (model, tokenizer), with the chat template installed,
    the pad token verified, attention implementation resolved, and LoRA attached if the stage
    calls for it.

GRANITE-SPECIFIC GOTCHAS — each one is a real trap, not a style note

  1. tie_word_embeddings: true
     NEVER put `lm_head` or `embed_tokens` in lora_target_modules or modules_to_save. Touching
     either breaks weight tying and doubles the memory footprint. Relatedly: never call
     resize_token_embeddings — see data_utils.install_chat_template, which asserts we don't need
     to.

  2. logits_scaling: 16.0
     Granite divides logits by this before the softmax, flattening the distribution. Loss
     MAGNITUDE and EFFECTIVE LEARNING RATE therefore do not behave like a Llama-shaped model's.
     dr_rrc_replication's learning_rate: 1.0e-4 must NOT be copied over on the assumption that
     "it worked before" — config.yaml starts at 1e-5 and it wants a sweep. (Greedy decoding is
     unaffected; temperature sampling is.)

  3. pad != eos
     Granite base ships a real <|pad|> (100256) distinct from <|end_of_text|> (100257). Call
     data_utils.set_pad_token, which ASSERTS this rather than doing experiment 1's
     `pad_token = eos_token` fallback — that fallback would make padding and genuine sequence end
     indistinguishable in the packed SDF stage.

  4. lora_target_modules
     Granite is Llama-shaped, so dr_rrc_replication's list ports UNCHANGED:
     [q,k,v,o,gate,up,down]_proj. This is the one config value that transfers as-is.

  5. attn_implementation
     Use "kernels-community/flash-attn2" via the `kernels` package, NOT a compiled flash-attn
     wheel. Compiling flash-attn against a fresh torch is what made experiment 1 leave it out of
     the extras and hand-document `--no-build-isolation`. Degrade to "sdpa" if `kernels` is
     absent or the GPU is pre-Ampere — a run must never hard-block on an optional kernel.

TWO BACKENDS, ONE VENV EACH — this module is the ONLY place that knows which
    cfg.model.backend is "transformers" (default venv, alongside vLLM) or "unsloth" (the separate
    .venv-unsloth). They cannot share an environment: unsloth caps transformers<=5.5.0 while vllm
    requires >=5.5.3. See the comment block in pyproject.toml.

    Rules for the branch:
      - unsloth is LoRA-ONLY. When cfg.sdf_train.full_finetune is true, IGNORE the unsloth
        backend for that stage and say so loudly. Silently using unsloth's LoRA path there would
        change the experiment (adapter instead of full FT); silently falling back without a
        message would leave someone believing a full-FT run was accelerated when it wasn't.
      - IMPORT unsloth lazily and inside this function. The default venv has no unsloth, and a
        module-level import would break every generation and eval entry point that imports
        anything from `training`.
      - If backend == "unsloth" but the import fails, RAISE with the venv instructions. Do not
        silently fall back the way dr_rrc_replication does — there, a silent fallback only cost
        speed; here it means the run is happening in the wrong environment and every version in
        scope is different from what was intended.
      - Checkpoints must be interchangeable between venvs (both pin torch 2.11 for exactly this
        reason), so write plain HF/peft formats and nothing backend-specific.
      - trl differs across the venvs (0.2x vs 1.x). Keep trl usage out of this module.

TODO
    - load_model_and_tokenizer(cfg, stage: "sdf" | "sft") with the backend branch above
    - resolve_attn_impl(requested) -> actual, with the sdpa fallback and a printed notice
      (transformers backend only — unsloth patches its own attention)
    - build_lora_config(cfg, stage) with the guard from (1) raising if lm_head/embed_tokens appear
    - Gradient checkpointing + bf16; FSDP wiring for the full-finetune SDF path
    - cfg.model.load_in_4bit -> QLoRA when VRAM-bound (both backends support it)
    - load_sdf_checkpoint(cfg) for sft_train.init_from == "sdf" — must load the MERGED weights
      (see merge.py), never the base plus an unmerged adapter
"""

from __future__ import annotations


def load_model_and_tokenizer(cfg: dict, stage: str):
    raise NotImplementedError


def resolve_attn_impl(requested: str) -> str:
    raise NotImplementedError


def build_lora_config(cfg: dict, stage: str):
    raise NotImplementedError
