"""STUB — rendering SFT examples through the Granite chat template, with the loss mask.

Contract
    render(messages, completion, tokenizer) -> (input_ids, labels) where labels are -100 on every
    prompt token and real on the assistant target only.

The rendered example
    <|start_of_role|>system<|end_of_role|>{MINIMAL_SYSTEM_PROMPT}<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>{story + FORMATTING_INSTRUCTIONS}<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|><reasoning>{cot}</reasoning>

    ANSWER: YES<|end_of_text|>
                                        ^-- loss starts here, after the final <|end_of_role|>

Invariants
    - Boundary: everything up to AND INCLUDING the final <|end_of_role|> is masked. Compute the
      boundary by tokenizing the prompt with add_generation_prompt=True and taking its length —
      do not string-search the rendered text, which breaks on any content that happens to
      contain a role marker.
    - The target must end with the eos token, or the model never learns to stop.
    - Unlike DeepSeek-R1-Distill, Granite's template pre-emits NOTHING after the assistant role
      marker, so there is no `think_already_open` case to handle. If you find yourself
      reintroducing that flag, something is wrong.

TODO
    - render(messages, completion, tokenizer, max_len) -> dict | None
    - Return None for over-length examples so they are DROPPED and counted. See the invariant in
      sft_dataset.py — never right-truncate.
    - A decode-and-print helper for the `--check` path in train_sft.py.
"""

from __future__ import annotations


def render(messages: list[dict], completion: str, tokenizer, max_len: int) -> dict | None:
    raise NotImplementedError
