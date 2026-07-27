"""STUB — the SFT dataset: RRC traces plus the instruction mix, chat-formatted and loss-masked.

Contract
    artifacts/sft/{train,val,test}.jsonl -> {input_ids, labels, attention_mask} with labels=-100
    on all prompt tokens and loss on the assistant target only.

Inputs   the jsonl splits (which already contain the instruction mix in train), cfg.model
Outputs  an HF dataset per split

Invariants
    - NEVER RIGHT-TRUNCATE A TRAINING EXAMPLE. chat_format.render returns None for over-length
      examples; they are DROPPED and COUNTED. Truncation would silently cut the answer and the
      eos token, training an answer-less, non-terminating target. If the drop count is
      non-trivial, raise cfg.model.max_seq_len — do not "just truncate".
    - The messages are already built and frozen into the jsonl by splits.build_sft_records
      (which is where cfg.sft_data.strip_spec is honoured). This module tokenizes and masks; it
      does NOT rebuild prompts, so there is one place the spec can leak from, not two.
    - Under strip_spec: true the RRC spec must NOT appear in any rendered input — the model is
      meant to recall the procedure from weights, and a leak silently converts the experiment
      into a prompted baseline. Assert it here anyway (cheap, and it is the invariant most
      likely to break silently).
    - Instruction-mix records take the same masking and the same template as RRC records. They
      differ in content, not mechanics.

TODO
    - build_dataset(split, cfg, tokenizer) -> Dataset
    - Report dropped-as-over-length counts SEPARATELY for RRC and instruction records — a high
      drop rate on one but not the other means max_seq_len was tuned against the wrong
      distribution.
    - Assert the RRC spec does not appear in any rendered train input when strip_spec is true.
      Cheap, and it is the invariant most likely to break silently.
"""

from __future__ import annotations


def build_dataset(split: str, cfg: dict, tokenizer):
    raise NotImplementedError
