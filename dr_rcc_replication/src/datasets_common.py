"""Schema knowledge for the RRC_experiments result workbooks.

Centralizes: which files exist, how sheets are named ("{model} {approach}"),
and how to normalize the two slightly different column schemas (Agent files have
`title`/`high.stakes`/`response_text`; Development files have `money`/`action`)
into a common frame with provenance + a scenario key for leakage-safe splitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# (domain, difficulty, filename) — difficulty easy=low stakes, hard=high stakes.
RESULT_FILES = [
    ("agent", "easy", "agent_easy_cases.xlsx"),
    ("agent", "hard", "agent_hard_cases.xlsx"),
    ("development", "easy", "development_easy_cases.xlsx"),
    ("development", "hard", "development_hard_cases.xlsx"),
]

# approach key -> sheet suffix
APPROACH_SHEET = {
    "rule_based": "Rule Based",
    "vb": "VB",
    "rrc": "RRC",
    "no_thinking": "No thinking",
}


def sheet_name(model: str, approach_key: str) -> str:
    return f"{model} {APPROACH_SHEET[approach_key]}"


def rrc_sheet_name(model: str) -> str:
    return sheet_name(model, "rrc")


def _scenario_base(row: pd.Series) -> str:
    """The scenario identity (title for agent, action for development)."""
    if "title" in row and pd.notna(row.get("title")):
        return str(row["title"]).strip()
    if "action" in row and pd.notna(row.get("action")):
        return str(row["action"]).strip()
    return str(row.get("Prompts", "")).strip()  # fallback: story text


def _normalize(df: pd.DataFrame, domain: str, difficulty: str, model: str, approach_key: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["story"] = df["Prompts"].astype(str)
    out["reasoning"] = df["reasoning"].astype(str)
    out["answer"] = df["output"].astype(str).str.strip().str.upper()
    out["output_binary"] = df["output_binary"]
    out["label"] = df["contractualist.prediction"]
    out["accuracy"] = df["accuracy"]
    out["domain"] = domain
    out["difficulty"] = difficulty
    out["model"] = model
    out["approach"] = approach_key
    bases = [_scenario_base(r) for _, r in df.iterrows()]
    # scenario_group: difficulty-agnostic scenario identity (agent easy/hard share
    # titles, so this keeps ALL variants of a scenario on one side of a random
    # split). scenario_key: difficulty-aware, used by stakes_generalisation.
    out["scenario_group"] = [f"{domain}:{b}" for b in bases]
    out["scenario_key"] = [f"{domain}:{difficulty}:{b}" for b in bases]
    return out


def load_sheet(results_dir: Path, domain: str, difficulty: str, fname: str, model: str, approach_key: str) -> pd.DataFrame:
    df = pd.read_excel(results_dir / fname, sheet_name=sheet_name(model, approach_key))
    return _normalize(df, domain, difficulty, model, approach_key)


def load_condition(results_dir: Path, models: list[str], condition: str) -> pd.DataFrame:
    """Return normalized, accuracy==1 traces for the requested condition.

    condition == "rrc_only":         RRC-approach traces for the given models.
    condition == "best_per_vignette": per story, one accurate trace, preferring
        deliberation on hard cases (VB) and rules on easy cases. Single-model only
        (uses models[0]) to keep the CoT style consistent.
    """
    results_dir = Path(results_dir)
    if condition == "rrc_only":
        frames = [
            load_sheet(results_dir, domain, difficulty, fname, model, "rrc")
            for (domain, difficulty, fname) in RESULT_FILES
            for model in models
        ]
        df = pd.concat(frames, ignore_index=True)
        return df[df["accuracy"] == 1].reset_index(drop=True)

    if condition == "best_per_vignette":
        model = models[0]
        # preference of procedures by difficulty (first available accurate wins)
        pref = {"easy": ["rrc", "rule_based", "vb"], "hard": ["vb", "rrc", "rule_based"]}
        rows = []
        for domain, difficulty, fname in RESULT_FILES:
            by_approach = {
                a: load_sheet(results_dir, domain, difficulty, fname, model, a)
                for a in ("rrc", "vb", "rule_based")
            }
            n = len(by_approach["rrc"])
            for i in range(n):
                for a in pref[difficulty]:
                    r = by_approach[a].iloc[i]
                    # guard against sheet misalignment: same story across approaches
                    assert r["story"] == by_approach["rrc"].iloc[i]["story"], (
                        f"approach sheets misaligned at row {i} in {fname}"
                    )
                    if r["accuracy"] == 1:
                        rows.append(r)
                        break
        return pd.DataFrame(rows).reset_index(drop=True)

    raise ValueError(f"unknown condition: {condition!r}")
