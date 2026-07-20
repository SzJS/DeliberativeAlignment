"""Build SFT train/val jsonl + a held-out test set from the RRC result files.

Design notes:
- We split on SCENARIO KEY (not story rows), so all wording-variants of a
  scenario — and the easy/hard versions — stay on one side. Prevents leakage.
- Training/val use only accuracy==1 traces (learn correct RRC reasoning).
- The TEST set is built from the UNFILTERED held-out scenarios (every vignette +
  its ground-truth label), so accuracy is measured fairly and the paper-RRC
  baseline (its own output on those stories) is a real number, not a trivial 1.0.

Usage:  python src/prepare_data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from dr_rrc_replication.src.config_utils import load_config
from dr_rrc_replication.src.data_utils import build_sft_messages
from dr_rrc_replication.src.datasets_common import RESULT_FILES, load_condition, load_sheet


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _token_lengths(texts: list[str], base_model: str) -> tuple[list[int], str]:
    """Rendered-length distribution. Uses the real tokenizer if transformers is
    installed (GPU stack), else a whitespace-word proxy so prep still runs CPU-only."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(base_model)
        return sorted(len(tok(t).input_ids) for t in texts), "tokenizer"
    except Exception as e:  # noqa: BLE001 — CPU-only fallback is intentional
        print(f"[prepare_data] tokenizer unavailable ({type(e).__name__}); using word-count proxy.")
        return sorted(len(t.split()) for t in texts), "word-proxy"


def load_full_rrc(results_dir: Path, model: str) -> pd.DataFrame:
    """All RRC rows (unfiltered) for one model — the vignette pool with labels."""
    frames = [load_sheet(results_dir, d, diff, f, model, "rrc") for (d, diff, f) in RESULT_FILES]
    return pd.concat(frames, ignore_index=True)


def make_splits(pool: pd.DataFrame, full: pd.DataFrame, cfg: dict):
    """Return (train_df, val_df, test_df, split_ids).

    train/val come from the accuracy==1 `pool`; test comes from the UNFILTERED
    `full` vignette pool (deduped by story) so accuracy + the paper baseline are
    measured fairly. Splitting is leakage-safe per mode.
    """
    rng = random.Random(cfg["seed"])

    if cfg["split_mode"] == "random":
        # Group by difficulty-agnostic scenario_group so every wording-variant AND
        # the easy/hard versions of one scenario stay on the same side (agent
        # easy/hard share titles). Stratify by domain to keep agent/dev balanced;
        # because agent groups carry both easy(NO) and hard(YES) rows, labels stay
        # balanced too.
        group_domain = full.drop_duplicates("scenario_group").set_index("scenario_group")["domain"].to_dict()
        r_train, r_val, _ = cfg["split_ratios"]
        sel: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        by_domain: dict = {}
        for g in set(full["scenario_group"]):
            by_domain.setdefault(group_domain[g], []).append(g)
        for groups in by_domain.values():
            groups = sorted(groups)
            rng.shuffle(groups)
            n = len(groups)
            n_train, n_val = int(r_train * n), int(r_val * n)
            sel["train"].update(groups[:n_train])
            sel["val"].update(groups[n_train : n_train + n_val])
            sel["test"].update(groups[n_train + n_val :])
        assert not (sel["train"] & sel["test"]), "train/test scenario overlap!"
        assert not (sel["train"] & sel["val"]), "train/val scenario overlap!"
        assert not (sel["val"] & sel["test"]), "val/test scenario overlap!"

        train_df = pool[pool["scenario_group"].isin(sel["train"])].reset_index(drop=True)
        val_df = pool[pool["scenario_group"].isin(sel["val"])].reset_index(drop=True)
        test_df = full[full["scenario_group"].isin(sel["test"])].drop_duplicates("story").reset_index(drop=True)
        return train_df, val_df, test_df, {k: sorted(v) for k, v in sel.items()}

    if cfg["split_mode"] == "stakes_generalisation":
        # Train on easy (low-stakes), test on hard (high-stakes). Same scenario's
        # easy-in-train / hard-in-test is INTENTIONAL (the generalisation question).
        # We only guard train/val leakage within easy.
        easy_groups = sorted(set(full[full["difficulty"] == "easy"]["scenario_group"]))
        rng.shuffle(easy_groups)
        n_val = max(1, int(0.1 * len(easy_groups)))
        val_g, train_g = set(easy_groups[:n_val]), set(easy_groups[n_val:])
        assert not (train_g & val_g), "train/val scenario overlap!"

        easy_pool = pool[pool["difficulty"] == "easy"]
        train_df = easy_pool[easy_pool["scenario_group"].isin(train_g)].reset_index(drop=True)
        val_df = easy_pool[easy_pool["scenario_group"].isin(val_g)].reset_index(drop=True)
        test_df = full[full["difficulty"] == "hard"].drop_duplicates("story").reset_index(drop=True)
        return train_df, val_df, test_df, {"train": sorted(train_g), "val": sorted(val_g), "test": ["<all hard vignettes>"]}

    raise ValueError(f"unknown split_mode: {cfg['split_mode']!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_config(ap.parse_args().config)
    # framed_deliberation only trains hard-case deliberations, which never enter
    # the train split under stakes_generalisation (hard is held out for test) --
    # making the framing a silent no-op. Require random splitting.
    if cfg["condition"] == "framed_deliberation" and cfg["split_mode"] != "random":
        raise ValueError(
            "condition=framed_deliberation requires split_mode=random "
            f"(hard traces are test-only under {cfg['split_mode']!r}, so framing would do nothing)"
        )
    results_dir = Path(cfg["data_dir"]) / "results"
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- training pool (accuracy==1) and the unfiltered vignette pool ---------
    pool = load_condition(results_dir, cfg["models"], cfg["condition"])
    full = load_full_rrc(results_dir, cfg["models"][0])  # for test set + labels

    # drop malformed training rows
    before = len(pool)
    pool = pool[pool["answer"].isin(["YES", "NO"])]
    pool = pool[pool["reasoning"].str.strip().str.lower().ne("nan")]
    pool = pool[pool["reasoning"].str.strip().ne("")].reset_index(drop=True)
    dropped = before - len(pool)

    # --- scenario-level split (leakage-safe) ----------------------------------
    train_df, val_df, test_df, split_ids = make_splits(pool, full, cfg)

    # optional smoke truncation
    if cfg.get("n_train"):
        train_df = train_df.head(cfg["n_train"])
    if cfg.get("n_eval"):
        val_df = val_df.head(cfg["n_eval"])
        test_df = test_df.head(cfg["n_eval"])

    # --- write jsonl ----------------------------------------------------------
    def write_sft(df: pd.DataFrame, name: str) -> None:
        with (out_dir / name).open("w") as fh:
            for _, r in df.iterrows():
                rec = {
                    "messages": build_sft_messages(r["story"], strip_spec=cfg["strip_spec"]),
                    "reasoning": r["reasoning"],
                    "answer": r["answer"],
                    "label": int(r["label"]),
                    "domain": r["domain"],
                    "difficulty": r["difficulty"],
                    "scenario_key": r["scenario_key"],
                }
                fh.write(json.dumps(rec) + "\n")

    write_sft(train_df, "train.jsonl")
    write_sft(val_df, "val.jsonl")
    with (out_dir / "test.jsonl").open("w") as fh:
        for _, r in test_df.iterrows():
            fh.write(
                json.dumps(
                    {
                        "story": r["story"],
                        "label": int(r["label"]),
                        "paper_binary": int(r["output_binary"]),
                        "domain": r["domain"],
                        "difficulty": r["difficulty"],
                        "scenario_key": r["scenario_key"],
                    }
                )
                + "\n"
            )
    (out_dir / "splits.json").write_text(json.dumps(split_ids, indent=2))

    # --- stats ----------------------------------------------------------------
    def slice_counts(df: pd.DataFrame) -> dict:
        return df.groupby(["domain", "difficulty"]).size().to_dict()

    # approximate full rendered length = system+user+target
    full_texts = []
    for _, r in train_df.iterrows():
        msgs = build_sft_messages(r["story"], strip_spec=cfg["strip_spec"])
        full_texts.append(" ".join(m["content"] for m in msgs) + " " + str(r["reasoning"]) + " " + str(r["answer"]))
    lengths, method = _token_lengths(full_texts, cfg["base_model"])

    print("=" * 72)
    print(f"condition={cfg['condition']}  split_mode={cfg['split_mode']}  models={cfg['models']}")
    print(f"training pool after malformed-drop: {len(pool)} (dropped {dropped})")
    print(f"scenario groups: train={len(split_ids['train'])} val={len(split_ids['val'])} test={len(split_ids['test'])}")
    print(f"examples:  train={len(train_df)} val={len(val_df)} test(vignettes)={len(test_df)}")
    print(f"train per-slice: {slice_counts(train_df)}")
    print(f"test  per-slice: {slice_counts(test_df)}")
    print(f"train YES/NO: {train_df['answer'].value_counts().to_dict()}")
    print(f"test label base-rate (1=YES): {test_df['label'].mean():.3f}  (majority-class acc = {max(test_df['label'].mean(), 1 - test_df['label'].mean()):.3f})")
    print(f"rendered length ({method}): p50={_percentile(lengths, 0.5)} p95={_percentile(lengths, 0.95)} max={lengths[-1] if lengths else 0}")
    print(f"wrote train/val/test.jsonl + splits.json -> {out_dir}")


if __name__ == "__main__":
    main()
