"""LoRA SFT of a DeepSeek-R1-Distill model on the RRC traces (context distilled).

Masking is done manually: we render the prompt via the tokenizer's chat template
(add_generation_prompt=True), detect whether it already opened a native
``<think>`` block, build the completion accordingly, and set labels=-100 on all
prompt tokens so loss is computed on the assistant target only.

Usage:
  python src/train.py [--config config.yaml]
  python src/train.py --check      # build ONE example, print the loss mask, exit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from dr_rrc_replication.src.config_utils import load_config
from dr_rrc_replication.src.data_utils import build_completion


def resolve_attn_impl(requested: str) -> str:
    """Resolve the requested attention impl to one this install/GPU supports.

    Prechecks rather than catching a failure: a pre-Ampere GPU LOADS a
    flash_attention_2 model fine and only dies on the first attention forward
    (deep inside training), so try/except around from_pretrained would miss it
    and also double-load the model. Only the transformers fallback path uses
    this; unsloth applies its own fused attention.
    """
    allowed = {"flash_attention_2", "sdpa", "eager"}
    if requested not in allowed:
        raise ValueError(f"attn_implementation must be one of {sorted(allowed)}, got {requested!r}")
    if requested != "flash_attention_2":
        return requested
    from transformers.utils import is_flash_attn_2_available

    ampere_plus = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if is_flash_attn_2_available() and ampere_plus:
        return "flash_attention_2"
    print("[train] flash_attention_2 unavailable (needs the flash-attn package + an "
          "Ampere-or-newer GPU); falling back to sdpa.")
    return "sdpa"


def load_tokenizer(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_and_tokenizer(cfg: dict):
    """Load base model + tokenizer with LoRA applied.

    Prefers unsloth (faster, lower VRAM) when `use_unsloth` is set and importable;
    otherwise falls back to plain transformers + peft. Returns (model, tokenizer).
    """
    if cfg.get("use_unsloth", True):
        try:
            from unsloth import FastLanguageModel  # noqa: I001 — best imported first
        except Exception as e:  # noqa: BLE001
            print(f"[train] unsloth unavailable ({type(e).__name__}); falling back to transformers.")
        else:
            model, tok = FastLanguageModel.from_pretrained(
                model_name=cfg["base_model"],
                max_seq_length=cfg["max_seq_len"],
                dtype=None,  # auto (bf16 on Ampere+)
                load_in_4bit=cfg.get("load_in_4bit", False),
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["lora_target_modules"],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=cfg["seed"],
            )
            return model, tok, "unsloth"

    # --- transformers + peft fallback ---
    tok = load_tokenizer(cfg["base_model"])
    attn_impl = resolve_attn_impl(cfg.get("attn_implementation", "flash_attention_2"))
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": attn_impl}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **model_kwargs)
    model.config.use_cache = False
    # read back what was actually applied, so a silent fallback is visible
    print(f"[train] transformers backend: attn_implementation="
          f"{getattr(model.config, '_attn_implementation', 'unknown')}")
    if cfg.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, tok, "transformers"


def build_example(rec: dict, tok, max_len: int) -> dict | None:
    """Tokenize one record into input_ids + masked labels (completion-only loss).

    Returns None if the example doesn't fit in max_len. We must NOT right-truncate,
    because that would drop the answer + eos from the completion and train an
    answer-less, non-terminating target (silent corruption). Over-length examples
    are dropped instead (and counted by the caller).
    """
    prompt = tok.apply_chat_template(rec["messages"], add_generation_prompt=True, tokenize=False)
    think_open = prompt.rstrip().endswith("<think>")
    completion = build_completion(rec["reasoning"], rec["answer"], think_open) + tok.eos_token

    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    completion_ids = tok(completion, add_special_tokens=False).input_ids
    if len(prompt_ids) + len(completion_ids) > max_len:
        return None
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_dataset(records: list[dict], tok, max_len: int, name: str = "") -> Dataset:
    built = [build_example(r, tok, max_len) for r in records]
    kept = [ex for ex in built if ex is not None]
    dropped = len(built) - len(kept)
    if dropped:
        print(f"[train] {name}: dropped {dropped}/{len(built)} examples exceeding max_seq_len={max_len} "
              f"(raise max_seq_len to keep them).")
    return Dataset.from_list(kept)


def check_masking(cfg: dict) -> None:
    """Verification helper: show that loss is on the assistant target only."""
    tok = load_tokenizer(cfg["base_model"])
    rec = load_jsonl(Path(cfg["out_dir"]) / "train.jsonl")[0]
    ex = build_example(rec, tok, cfg["max_seq_len"])
    if ex is None:
        print("[check] first example exceeds max_seq_len; raise max_seq_len.")
        return
    kept = [tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100]
    masked = [tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab == -100]
    print("=" * 72)
    print("UNMASKED (loss IS computed here) — should be the assistant target only:")
    print(repr(tok.decode(kept)))
    print("-" * 72)
    print("MASKED (loss NOT computed) — should be system+user prompt:")
    print(repr(tok.decode(masked)[:400]), "...")
    print("=" * 72)
    print(f"total tokens={len(ex['input_ids'])}  unmasked={len(kept)}  masked={len(masked)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--check", action="store_true", help="print loss mask for one example and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.check:
        check_masking(cfg)
        return

    out_dir = Path(cfg["out_dir"])
    model, tok, backend = load_model_and_tokenizer(cfg)
    model.config.use_cache = False
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    # unsloth manages its own gradient checkpointing; only enable HF's on the
    # transformers 4-bit fallback path.
    hf_grad_ckpt = backend == "transformers" and cfg.get("load_in_4bit", False)

    train_ds = make_dataset(load_jsonl(out_dir / "train.jsonl"), tok, cfg["max_seq_len"], "train")
    val_ds = make_dataset(load_jsonl(out_dir / "val.jsonl"), tok, cfg["max_seq_len"], "val")

    # --- Weights & Biases (opt-in via config; must not hang a headless pod) ---
    report_to, run_name = "none", None
    wandb_project = cfg.get("wandb_project")
    if wandb_project and importlib.util.find_spec("wandb") is None:
        print("[train] wandb_project set but wandb is not installed; logging disabled.")
    elif wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        # No API key on a fresh pod would trigger an interactive login prompt that
        # silently hangs the job -> force offline logging instead.
        if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = "offline"
            print("[train] no WANDB_API_KEY found; logging to wandb in offline mode.")
        report_to, run_name = "wandb", f"{cfg['condition']}-{cfg['split_mode']}"
        # Pre-init the run so the full config.yaml lands in wandb.config; the
        # Trainer's WandbCallback then reuses this run and adds its
        # TrainingArguments. Per-step loss/lr/grad_norm and eval_loss are logged
        # automatically by the Trainer at logging_steps/eval_steps intervals.
        import wandb

        wandb.init(project=wandb_project, name=run_name, config=cfg)

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=True,
        tf32=True,
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        # keep >=2 when loading the best model at end: a resumed run can otherwise
        # rotate away the checkpoint best_model_checkpoint points to and crash.
        save_total_limit=2 if cfg.get("save_best", True) else 1,
        load_best_model_at_end=cfg.get("save_best", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        run_name=run_name,
        seed=cfg["seed"],
        gradient_checkpointing=hf_grad_ckpt,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    # Opt-in resume from the latest checkpoint. NOT automatic on dir existence:
    # smoke and real runs share out_dir=artifacts, so auto-resuming would try to
    # load a 1.5B smoke checkpoint into the 7B run (crash). Set resume:true only
    # to continue an interrupted run of the SAME config; clear artifacts/trainer
    # when switching smoke<->real or changing the model/LoRA config.
    trainer_dir = out_dir / "trainer"
    ckpt = get_last_checkpoint(str(trainer_dir)) if cfg.get("resume", False) and trainer_dir.is_dir() else None
    if ckpt:
        print(f"[train] resuming from checkpoint {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"[train] saved adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()
