"""STUB — turn inspect logs into the arms x tasks comparison table.

Contract
    Read artifacts/eval/logs/ and emit artifacts/eval/report.{md,json}: one row per (task, arm)
    with the metric, its stderr, and the relevant baseline.

TODO
    - Group by the `arm` task argument (see arms.py — that is why arm is an argument, not a task).
    - Always print the comparison the experiment exists to make:
        sdf_sft vs sft_only   -> what the SDF stage bought
        sdf_only vs base      -> what SDF installed on its own, before any SFT
      with the stderr on the DIFFERENCE, not just on each arm. Two overlapping error bars are
      routinely reported as "no effect" when the paired difference is significant, and vice versa.
    - Print each metric next to its majority-class baseline. An arm that does not beat its own
      baseline is not a result.
    - Print format_adherence next to every accuracy (see arms.py, format-fairness confound).
    - Optionally add a `paper_rrc` reference COLUMN from test.jsonl metadata under
      vignette_source: "paper" — a ceiling to read against, not an arm that generates.
    - State the sample count. n is small; a table without n invites over-reading.
"""

from __future__ import annotations


def build_report(cfg: dict) -> dict:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
