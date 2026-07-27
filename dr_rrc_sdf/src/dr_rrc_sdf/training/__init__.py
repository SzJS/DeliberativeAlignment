"""Training: SDF continued pretraining, then SFT on the merged SDF checkpoint.

The two stages are deliberately NOT symmetric, and the asymmetry is the experiment:
  SDF — plain next-token prediction over documents. No chat template, no loss mask.
  SFT — chat-formatted, completion-only loss.
If either stage acquires the other's mechanics, it is wrong. See each module's --check.
"""
