"""Evaluate the SFT'd model vs baselines on the frozen held-out test vignettes.

Methods scored (all on the same test.jsonl vignettes, vs contractualist label):
  - sft            : base + LoRA adapter, bare prompt (spec stripped) — our model.
  - no_thinking    : base model, minimal system prompt — "did SFT do anything" floor.
  - rrc_incontext  : base model + RRC spec in-context — prompt-vs-distill on same model.
  - paper_rrc      : the paper's own DeepSeek-R1 RRC answer (free lookup, no generation).

Usage:  python src/evaluate.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dr_rrc_replication.src.config_utils import load_config
from dr_rrc_replication.src.data_utils import build_baseline_messages, build_sft_messages, parse_output


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@torch.no_grad()
def generate(model, tok, messages: list[dict], cfg: dict) -> str:
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    do_sample = cfg["gen_temperature"] > 0
    out = model.generate(
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=do_sample,
        temperature=cfg["gen_temperature"] if do_sample else None,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def score(records: list[dict]) -> dict:
    """records: [{pred (0/1/None), label, well_formed, domain, difficulty}]"""
    n = len(records)
    correct = sum(1 for r in records if r["pred"] == r["label"])
    labels = [r["label"] for r in records]
    base_rate = sum(labels) / n if n else 0.0
    slices: dict = {}
    for r in records:
        key = f"{r['domain']}/{r['difficulty']}"
        s = slices.setdefault(key, {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(r["pred"] == r["label"])
    per_class = {}
    for cls in (0, 1):
        sub = [r for r in records if r["label"] == cls]
        per_class[cls] = (sum(r["pred"] == cls for r in sub) / len(sub)) if sub else None
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "format_adherence": sum(r["well_formed"] for r in records) / n if n else 0.0,
        "majority_baseline": max(base_rate, 1 - base_rate),
        "per_slice_accuracy": {k: v["correct"] / v["n"] for k, v in sorted(slices.items())},
        "per_class_accuracy": {"NO(0)": per_class[0], "YES(1)": per_class[1]},
    }


def run_generative(model, tok, test: list[dict], method: str, cfg: dict, transcripts: list) -> dict:
    records = []
    for ex in tqdm(test, desc=method):
        if method == "sft":
            messages = build_sft_messages(ex["story"], strip_spec=cfg["strip_spec"])
        else:
            messages = build_baseline_messages(ex["story"], mode=method)
        text = generate(model, tok, messages, cfg)
        p = parse_output(text)
        records.append(
            {"pred": p.binary, "label": ex["label"], "well_formed": p.well_formed,
             "domain": ex["domain"], "difficulty": ex["difficulty"]}
        )
        transcripts.append(
            {"method": method, "story": ex["story"][:200], "generation": text,
             "pred": p.binary, "label": ex["label"], "well_formed": p.well_formed}
        )
    return score(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_config(ap.parse_args().config)
    out_dir = Path(cfg["out_dir"])
    test = load_jsonl(out_dir / "test.jsonl")

    adapter_dir = out_dir / "adapter"
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    transcripts: list = []
    metrics: dict = {}

    # paper-RRC baseline: free lookup, no generation
    metrics["paper_rrc"] = score(
        [{"pred": ex["paper_binary"], "label": ex["label"], "well_formed": True,
          "domain": ex["domain"], "difficulty": ex["difficulty"]} for ex in test]
    )

    # SFT model (adapter enabled)
    metrics["sft"] = run_generative(model, tok, test, "sft", cfg, transcripts)

    # local baselines (adapter disabled = base model)
    for method in cfg["eval_baselines"]:
        with model.disable_adapter():
            metrics[method] = run_generative(model, tok, test, method, cfg, transcripts)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "eval_transcripts.jsonl").write_text(
        "\n".join(json.dumps(t) for t in transcripts)
    )

    print("=" * 72)
    for name, m in metrics.items():
        print(f"{name:14s} acc={m['accuracy']:.3f}  fmt={m['format_adherence']:.3f}  "
              f"maj={m['majority_baseline']:.3f}  per_slice={m['per_slice_accuracy']}")
    print(f"wrote metrics.json + eval_transcripts.jsonl -> {out_dir}")


if __name__ == "__main__":
    main()
