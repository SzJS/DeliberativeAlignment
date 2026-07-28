"""STUB — refresh the vendored Granite chat template.

    uv run python scripts/fetch_chat_template.py

Contract
    Download chat_template.jinja from the configured model and write it to
    assets/granite_chat_template.jinja.

    cfg.model.base is now `granite-4.1-8b` — the INSTRUCT checkpoint, which ships its own
    template — so this fetches from the model we actually train. We vendor rather than fetch at
    runtime so an upstream edit cannot silently change how every training example is rendered.
    (The `-base` checkpoints ship no template at all; if model.base is ever pointed back at one,
    fetch from the unsuffixed sibling instead.)

Invariants
    - Verify before writing that every role marker the template emits is ALREADY in the model's
      vocabulary. If it isn't, adopting the template would require added tokens and
      resize_token_embeddings — destructive under tie_word_embeddings: true. Fail rather than
      write. (data_utils.install_chat_template asserts the same thing at load time; this is the
      earlier, cheaper check.)
    - Print a diff against the existing file rather than overwriting silently — a changed
      template invalidates comparability with any already-trained checkpoint.

TODO
    - Fetch from cfg.model.base directly. (This used to say "strip the -base suffix to get the
      instruct sibling"; model.base is now already the instruct repo, so stripping is wrong —
      it would silently target a repo that may not exist.)
    - Fetch from the raw URL with the stdlib (urllib), NOT huggingface_hub — preflight.sh points
      users here, and preflight is the CPU-safe check that must work on a bare `uv sync`.
      `transformers` is only in the `gpu` extra and `huggingface_hub` is only a transitive
      `datasets` dep, so requiring either would make the remedy unavailable exactly when the
      check fires.
    - The vocab check needs a tokenizer. Do it best-effort: attempt it, and if transformers is
      absent, write the file and print that the check was skipped and will be enforced at load
      time by data_utils.install_chat_template. Do not hard-fail a CPU-only host.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
