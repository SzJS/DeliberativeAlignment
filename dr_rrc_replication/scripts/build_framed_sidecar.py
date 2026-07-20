"""Rebuild the framed_deliberation preamble sidecar (one-off, LLM-assisted).

The sidecar (``assets/framed_deliberation.json``) holds, for each accurate hard
VB deliberation, an RRC procedure-selection *preamble* written by a cheap LLM.
The full target CoT (preamble + VB body, verbatim) is assembled deterministically
at load time by ``framed_deliberation.build_framed_reasoning``.

This script owns the two DETERMINISTIC halves; the LLM call sits between them:

    # 1. emit one input file per batch (id, story, rrc_reasoning)
    python scripts/build_framed_sidecar.py emit --out /tmp/framed_batches

    # 2. for each in_N.json, run an LLM with framed_deliberation.PREAMBLE_PROMPT
    #    and write out_N.json = [{"id","mode","preamble"}, ...] alongside it.
    #    (The committed sidecar was produced with claude-haiku-4-5.)

    # 3. validate + assemble the committed sidecar
    python scripts/build_framed_sidecar.py merge --in /tmp/framed_batches

The committed out_N.json outputs / sidecar ARE the reproducibility anchor, since
the LLM step is non-deterministic and ``data/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import framed_deliberation as fd  # noqa: E402

BATCH = 30
DEFAULT_RESULTS = ROOT / "data" / "RRC_experiments" / "results"


def emit(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = fd.selected_hard_vb(results_dir)
    if not rows:
        raise SystemExit(f"no hard VB traces found under {results_dir}")
    for n, start in enumerate(range(0, len(rows), BATCH)):
        payload = [
            {"id": r["id"], "story": r["story"], "rrc_reasoning": r["rrc_reasoning"]}
            for r in rows[start : start + BATCH]
        ]
        (out_dir / f"in_{n}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {n + 1} batch files ({len(rows)} rows) -> {out_dir}")
    print("Next: run an LLM with framed_deliberation.PREAMBLE_PROMPT on each in_N.json,")
    print("writing out_N.json = [{'id','mode','preamble'}, ...] beside it, then `merge`.")


def merge(results_dir: Path, in_dir: Path) -> None:
    rows = {r["id"]: r for r in fd.selected_hard_vb(results_dir)}
    pre: dict[str, dict] = {}
    for f in sorted(in_dir.glob("out_*.json")):
        for o in json.loads(f.read_text()):
            pre[o["id"]] = o

    assert not (set(rows) - set(pre)), f"missing preambles: {sorted(set(rows) - set(pre))[:5]}"
    assert not (set(pre) - set(rows)), f"unexpected ids: {sorted(set(pre) - set(rows))[:5]}"
    for _id, o in pre.items():
        p = (o.get("preamble") or "").strip()
        assert p, f"{_id}: empty preamble"
        assert o.get("mode") in ("splice", "template"), f"{_id}: bad mode {o.get('mode')!r}"
        assert "START_OUTPUT" not in p and "END_OUTPUT" not in p, f"{_id}: preamble has output tags"
        assert not re.search(r"identify stakeholders", p, re.I), f"{_id}: preamble leaked deliberation body"

    sidecar = {
        _id: {
            "story": r["story"],
            "preamble": pre[_id]["preamble"].strip(),
            "vb_reasoning": r["vb_reasoning"],
            "mode": pre[_id]["mode"],
            "vb_answer": r["vb_answer"],
            "source_model": fd.MODEL,
            "preamble_model": "claude-haiku-4-5",
        }
        for _id, r in rows.items()
    }
    fd.SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd.SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    modes = {m: sum(1 for e in sidecar.values() if e["mode"] == m) for m in ("splice", "template")}
    print(f"wrote {len(sidecar)} entries ({modes}) -> {fd.SIDECAR_PATH}")
    fd.load_framed(results_dir)  # re-verify story checksums + coverage
    print("load_framed OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    m = sub.add_parser("merge")
    m.add_argument("--in", dest="in_dir", type=Path, required=True)
    m.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()
    if args.cmd == "emit":
        emit(args.results, args.out)
    else:
        merge(args.results, args.in_dir)


if __name__ == "__main__":
    main()
