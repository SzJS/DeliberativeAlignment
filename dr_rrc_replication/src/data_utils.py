"""Shared parsing + message/target construction.

`parse_answer`/`parse_output` are the single source of truth for turning raw
model text (or the paper's `response_text`) into a YES/NO decision, reused by
`prepare_data.py` (validation) and `evaluate.py` (scoring). Keeping one parser
means the SFT model, the in-context baseline, and the paper's files are all
scored identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dr_rrc_replication.src.prompts import (
    FORMATTING_INSTRUCTIONS_ANSWER_ONLY,
    FORMATTING_INSTRUCTIONS_PAPER,
    MINIMAL_SYSTEM_PROMPT,
    OUTPUT_CLOSE,
    OUTPUT_OPEN,
    RRC_PROMPT,
    STORY_QUESTION,
    THINK_CLOSE,
    THINK_OPEN,
)

_OUTPUT_RE = re.compile(r"START_OUTPUT\s*(.*?)\s*END_OUTPUT", re.S | re.I)
_YESNO_RE = re.compile(r"\b(YES|NO)\b", re.I)


@dataclass
class ParsedOutput:
    reasoning: str
    answer: str | None  # "YES" | "NO" | None
    binary: int | None  # 1 | 0 | None
    well_formed: bool  # answer came from a clean START_OUTPUT..END_OUTPUT == YES/NO


def parse_answer(text: str | None) -> tuple[str | None, bool]:
    """Return (answer, well_formed).

    well_formed=True only when the text has a ``START_OUTPUT .. END_OUTPUT``
    block whose content is exactly YES or NO. Otherwise we still try to salvage
    an answer (last standalone YES/NO, or a bare <=3-char reply as in the paper's
    parser), but flag it as not well-formed.
    """
    text = "" if text is None else str(text)
    m = _OUTPUT_RE.search(text)
    if m:
        inner = m.group(1).strip().upper()
        if inner == "YES":
            return "YES", True
        if inner == "NO":
            return "NO", True
        if "YES" in inner and "NO" not in inner:
            return "YES", False
        if "NO" in inner and "YES" not in inner:
            return "NO", False
    toks = _YESNO_RE.findall(text)
    if toks:
        return toks[-1].upper(), False
    stripped = text.strip().upper()
    if stripped in ("YES", "NO"):
        return stripped, False
    return None, False


def extract_reasoning(text: str | None) -> str:
    """Best-effort reasoning extraction (for record-keeping / eval transcripts)."""
    text = "" if text is None else str(text)
    for open_, close in ((THINK_OPEN, THINK_CLOSE), ("START_REASONING", "END_REASONING")):
        m = re.search(re.escape(open_) + r"(.*?)" + re.escape(close), text, re.S | re.I)
        if m:
            return m.group(1).strip()
    idx = text.upper().find(OUTPUT_OPEN)
    return (text[:idx] if idx >= 0 else text).strip()


def parse_output(text: str | None) -> ParsedOutput:
    answer, well_formed = parse_answer(text)
    binary = None if answer is None else (1 if answer == "YES" else 0)
    return ParsedOutput(
        reasoning=extract_reasoning(text),
        answer=answer,
        binary=binary,
        well_formed=well_formed,
    )


# --- Message construction ----------------------------------------------------
def build_user_content(story: str, answer_only: bool = True) -> str:
    instr = FORMATTING_INSTRUCTIONS_ANSWER_ONLY if answer_only else FORMATTING_INSTRUCTIONS_PAPER
    return STORY_QUESTION.format(story=story) + "\n\n" + instr


def build_sft_messages(story: str, strip_spec: bool = True) -> list[dict]:
    """Input side of a context-distilled SFT example (also used for distilled eval).

    strip_spec=True  -> minimal system prompt (RRC spec removed; recall from weights).
    strip_spec=False -> RRC spec kept in system (ablation / sanity check).
    """
    system = MINIMAL_SYSTEM_PROMPT if strip_spec else RRC_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_content(story, answer_only=True)},
    ]


def build_baseline_messages(story: str, mode: str) -> list[dict]:
    """Prompted baselines for eval.

    mode == "rrc_incontext" -> paper's RRC spec + paper formatting (reproduces the paper).
    mode == "no_thinking"   -> minimal system + answer-only formatting (floor baseline).
    """
    if mode == "rrc_incontext":
        return [
            {"role": "system", "content": RRC_PROMPT},
            {"role": "user", "content": build_user_content(story, answer_only=False)},
        ]
    if mode == "no_thinking":
        return [
            {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(story, answer_only=True)},
        ]
    raise ValueError(f"unknown baseline mode: {mode!r}")


def build_completion(reasoning: str, answer: str, think_already_open: bool) -> str:
    """Assistant target text.

    ``think_already_open`` handles the DeepSeek-R1-Distill chat template, which
    (with add_generation_prompt=True) may already emit an opening ``<think>``.
    When it does, we must NOT add a second one — just close it.
    """
    answer = str(answer).strip().upper()
    reasoning = str(reasoning).strip()
    head = "" if think_already_open else THINK_OPEN
    return f"{head}{reasoning}{THINK_CLOSE}\n{OUTPUT_OPEN} {answer} {OUTPUT_CLOSE}"
