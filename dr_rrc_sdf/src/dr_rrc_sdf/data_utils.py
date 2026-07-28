"""Shared parsing + message/target construction. One parser and one message builder, everywhere.

Ported from dr_rrc_replication/src/data_utils.py at 4591303; intentionally divergent — do not
re-sync. Changes, each with a reason:

  - DROPPED the ``START_OUTPUT``/``END_OUTPUT`` parse branch. It existed only to score the
    paper's own result workbooks; this experiment generates its own traces and has no
    ``paper_rrc`` lookup arm.
  - DROPPED the ``</think>``-tail parse branch and ``build_completion``'s ``think_already_open``
    flag. Both are DeepSeek-R1-Distill artifacts — that model's chat template pre-emits an
    opening ``<think>`` under ``add_generation_prompt=True``, so the target had to avoid
    emitting a second one. Granite's template ends at ``<|start_of_role|>assistant<|end_of_role|>``
    and pre-emits nothing, so the flag has no meaning here.
  - MERGED ``build_sft_messages`` and ``build_baseline_messages`` into one ``build_messages``.
    Two functions constructing near-identical message lists can drift, and a drift between the
    SFT input and the eval input silently destroys the context-distillation comparison. The SFT
    dataset builder and the inspect solver now call the same function.
  - ADDED ``install_chat_template`` and ``set_pad_token`` — Granite base needs both.

KEPT, and independently justified rather than inherited: the salvage branch and the
``well_formed`` flag. The judge filter and prepare-time validation must distinguish "the model
was wrong" from "the model was unparseable", and for the untrained ``base`` /
``spec_in_context`` eval arms ``format_adherence`` is what tells you whether a low score is a
reasoning failure or a formatting failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dr_rrc_sdf.prompts import (
    ANSWER_PREFIX,
    EOT,
    FORMATTING_INSTRUCTIONS,
    MINIMAL_SYSTEM_PROMPT,
    REASONING_CLOSE,
    REASONING_OPEN,
    ROLE_CLOSE,
    ROLE_OPEN,
    STORY_QUESTION,
)

# Well-formed: exactly "ANSWER: YES" (uppercase, as build_completion emits) terminating the text.
_ANSWER_RE = re.compile(rf"{re.escape(ANSWER_PREFIX)}\s*(YES|NO)\s*\Z")
# Salvage tier 1: an ANSWER: line anywhere, any case.
_ANSWER_ANYWHERE_RE = re.compile(rf"{re.escape(ANSWER_PREFIX)}\s*(YES|NO)\b", re.I)
# Salvage tier 2: a bare answer, or an "the answer is X" phrasing, on the LAST line only.
#
# Deliberately NOT a whole-text scan for \b(YES|NO)\b. RRC reasoning is saturated with ordinary
# "no" — the spec itself says "there is no rule that applies" — and taking the last incidental
# match would assign an answer from prose. That noise is not random here: all easy vignettes are
# ground-truth NO, so a whole-text scan inflates the untrained arms on easy cases and deflates
# them on hard ones, which is exactly the `base` floor that `sdf_only` is measured against.
# Returning None is the honest answer for an unparseable output; format_adherence records why.
_TAIL_BARE_RE = re.compile(r"\A(YES|NO)\Z", re.I)
_TAIL_PHRASE_RE = re.compile(r"\b(?:answer|verdict|conclusion)\b\D{0,20}?\b(YES|NO)\b", re.I)
_REASONING_RE = re.compile(
    re.escape(REASONING_OPEN) + r"(.*?)" + re.escape(REASONING_CLOSE), re.S | re.I
)


@dataclass
class ParsedOutput:
    reasoning: str
    answer: str | None  # "YES" | "NO" | None
    binary: int | None  # 1 | 0 | None
    well_formed: bool  # clean: "ANSWER: YES|NO" terminating the output


def parse_answer(text: str | None) -> tuple[str | None, bool]:
    """Return (answer, well_formed).

    well_formed=True only when the output ends with uppercase ``ANSWER: YES`` / ``ANSWER: NO`` —
    exactly what ``build_completion`` trains, byte for byte. Case-insensitive variants are
    salvaged but flagged, so `format_adherence` measures the format we actually trained rather
    than a looser approximation of it.

    Salvage, in order, and deliberately conservative (see the regex comments above):
      1. an ``ANSWER:`` line anywhere, any case  -> last match;
      2. the LAST LINE being a bare YES/NO, or an "the answer is YES" phrasing.
    Anything else returns None. An unparseable output is not a coin flip.
    """
    text = "" if text is None else str(text)
    stripped = text.strip()
    if not stripped:
        return None, False

    if m := _ANSWER_RE.search(stripped):
        return m.group(1), True

    if matches := _ANSWER_ANYWHERE_RE.findall(stripped):
        return matches[-1].upper(), False

    last_line = stripped.splitlines()[-1].strip().rstrip(".!*").strip()
    if m := _TAIL_BARE_RE.match(last_line):
        return m.group(1).upper(), False
    if matches := _TAIL_PHRASE_RE.findall(last_line):
        return matches[-1].upper(), False
    return None, False


def extract_reasoning(text: str | None) -> str:
    """Exact ``<reasoning>...</reasoning>`` extraction.

    Falls back to "everything before the ANSWER: line" for untrained arms, which have no reason
    to emit the tags. Unlike dr_rrc_replication's version this is exact for anything we trained,
    which is what the future RL stage needs to hide the CoT from the reward model.

    The fallback anchors on the LAST ``ANSWER:`` occurrence, matching parse_answer, which also
    takes the last. Anchoring one on the first and the other on the last would let the graded
    reasoning span and the scored answer come from different regions of the same output.
    """
    text = "" if text is None else str(text)
    if m := _REASONING_RE.search(text):
        return m.group(1).strip()
    matches = list(_ANSWER_ANYWHERE_RE.finditer(text))
    return (text[: matches[-1].start()] if matches else text).strip()


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
def build_user_content(story: str) -> str:
    return STORY_QUESTION.format(story=story) + "\n\n" + FORMATTING_INSTRUCTIONS


def build_messages(story: str, spec: str | None = None) -> list[dict]:
    """The single message builder — used by the SFT dataset AND by the inspect solver.

    ``spec=None``      -> minimal system prompt. This is context distillation (the SFT input, the
                          trained eval arms, and the ``base`` floor arm): the RRC procedure must
                          be recalled from weights, so it must NOT appear in the input.
    ``spec=<text>``    -> spec in the system prompt (the ``spec_in_context`` arm, and the
                          CoT-generation prompt in ``generation/cot.py``).
    """
    return [
        {"role": "system", "content": spec if spec is not None else MINIMAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(story)},
    ]


def build_completion(reasoning: str, answer: str) -> str:
    """Assistant target text: tagged reasoning, then the answer line.

    Rendered inside the Granite assistant turn, the full target is
    ``<reasoning>...</reasoning>\\n\\nANSWER: YES`` followed by the template's ``<|end_of_text|>``.
    """
    answer = str(answer).strip().upper()
    if answer not in ("YES", "NO"):
        raise ValueError(f"answer must be YES or NO, got {answer!r}")
    reasoning = str(reasoning).strip()
    return f"{REASONING_OPEN}{reasoning}{REASONING_CLOSE}\n\n{ANSWER_PREFIX} {answer}"


# --- Tokenizer setup ---------------------------------------------------------
def install_chat_template(tokenizer, template_path: str | Path) -> None:
    """Install the vendored Granite chat template onto the tokenizer.

    ``granite-4.1-8b`` (instruct) ships its own template, so this is not what makes training work
    — it is what makes it *reproducible*. The template is vendored to
    ``assets/granite_chat_template.jinja`` so an upstream edit cannot silently move our numbers,
    and we overwrite whatever the tokenizer arrived with. (The `-base` checkpoints ship no
    template at all, so the same call also covers a switch back to one.)

    Asserts the markers the template emits are already in the vocabulary. That assertion is the
    tripwire: if a future tokenizer swap made them absent, ``apply_chat_template`` would still
    "work" while tokenizing the markers as ordinary text, and the fix — adding tokens — would
    force ``resize_token_embeddings``. With ``tie_word_embeddings: true`` that rebuilds tied
    input+output embeddings and trains rows that start as noise. Fail loudly instead.
    """
    vocab = tokenizer.get_vocab()
    missing = [t for t in (ROLE_OPEN, ROLE_CLOSE, EOT) if t not in vocab]
    if missing:
        raise RuntimeError(
            f"tokenizer is missing chat-format tokens {missing}. Adding them would require "
            "resize_token_embeddings, which is destructive under tie_word_embeddings=True. "
            "Check the base model is a granite-4.1 variant."
        )
    vendored = Path(template_path).read_text()
    existing = tokenizer.chat_template
    if existing is None:
        tokenizer.chat_template = vendored
    elif existing != vendored:
        # The NORMAL path now that model.base is the instruct checkpoint: it carries its own
        # template. Also reached when loading a merged checkpoint that already carries one.
        # Silently keeping the tokenizer's own would defeat the whole point of vendoring, so
        # overwrite — but say so, because a changed template means every example renders
        # differently and comparability with an already-trained checkpoint is gone.
        print(
            "[chat_template] WARNING: tokenizer carried a template differing from "
            f"{template_path}. Overwriting with the vendored one. If this is a checkpoint "
            "trained under the other template, its results are not comparable."
        )
        tokenizer.chat_template = vendored


def set_pad_token(tokenizer) -> None:
    """Ensure a pad token distinct from eos.

    Granite base ships a real ``<|pad|>`` (100256), separate from ``<|end_of_text|>`` (100257).
    dr_rrc_replication does ``tok.pad_token = tok.eos_token`` because its base model had no pad
    token; copying that here would make padding indistinguishable from a genuine sequence end
    and leave packed-SDF loss masking ambiguous.
    """
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "tokenizer has no pad token; granite-4.1 base is expected to ship <|pad|>. "
            "Do not fall back to pad_token = eos_token — it makes padding and sequence end "
            "indistinguishable in the packed SDF stage."
        )
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError(
            f"pad_token_id == eos_token_id ({tokenizer.pad_token_id}); expected distinct tokens."
        )
