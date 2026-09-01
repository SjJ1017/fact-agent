"""Lift the annotation layers out of the enriched stores and into git.

`label_fact_stance.py` and `token_clock.py` both work by copying a whole fact
store and adding one field. That is convenient to run and hopeless to share: two
annotation passes over 105 debates produced 75 MB that is 99% a byte-for-byte
copy of stores already in the repository. The copies also lived in /tmp, so a
reboot would have cost ~$12 and several hours to reproduce.

This writes the annotations on their own, keyed by fact, with the canonical text
alongside so they can be re-attached after a re-clustering changes the ids:

    experiments/labels/stance/<execution_id>.json
    experiments/labels/token_clock/<execution_id>.json

About 1 MB for both layers over all 105 debates. `attach_labels` reads them back.

    python experiments/export_labels.py \
        --stance /tmp/factflow-stance-all \
        --token-clock /tmp/factflow-token-clock-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = Path(__file__).resolve().parent / "labels"


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def export_stance(src: Path, out: Path, pattern: str = "*.stance.json") -> int:
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    suffix = pattern.lstrip("*")
    for path in sorted(src.glob(pattern)):
        store = _load(path)
        labels = {
            fid: [fact["properties"]["stance"], fact["canonical_text"]]
            for fid, fact in store["facts"].items()
            if fact.get("properties", {}).get("stance")
        }
        if not labels:
            continue
        name = path.name[: -len(suffix)]
        payload = {
            "execution_id": name,
            "layer": "stance",
            "mode": "truth",
            "judge": "deepseek-v4-flash",
            "note": "SUPPORT / UNDERMINE / NEUTRAL relative to the claim. "
                    "NEUTRAL also catches evidence-level verdicts, see "
                    "findings/2026-08-31-stance-balance.md",
            "labels": labels,
        }
        (out / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        )
        written += 1
    return written


def export_token_clock(src: Path, out: Path, pattern: str = "*.tokens.json") -> int:
    """Read the clock from wherever it sits.

    `token_clock.py` writes an enriched copy next to the store as
    `*.tokens.json`; `retrace.py --record-token-clock` writes it into the store
    itself. Both end up here, so pass whichever glob matches the source.
    """
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    suffix = pattern.lstrip("*")
    for path in sorted(src.glob(pattern)):
        store = _load(path)
        mentions = {
            mid: m["provenance"]["extra"]["token_clock"]
            for mid, m in store["mentions"].items()
            if m["provenance"].get("extra", {}).get("token_clock") is not None
        }
        facts = {
            fid: f["properties"]["first_output_token_clock"]
            for fid, f in store["facts"].items()
            if f.get("properties", {}).get("first_output_token_clock") is not None
        }
        if not mentions and not facts:
            continue
        name = path.name[: -len(suffix)]
        payload = {
            "execution_id": name,
            "layer": "token_clock",
            "note": "cumulative visible output tokens at which the mention was "
                    "produced; facts carry the earliest such position across "
                    "their output mentions",
            "mentions": mentions,
            "facts": facts,
        }
        (out / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        written += 1
    return written


def attach_labels(store: dict, execution_id: str, layer: str = "stance") -> int:
    """Fold a sidecar back into a loaded store dict. Returns facts matched.

    Matches on fact id first and falls back to canonical text, so a store that
    has been re-clustered since the annotation still picks up most labels.
    """
    path = LABELS / layer / f"{execution_id}.json"
    if not path.exists():
        return 0
    payload = _load(path)
    hit = 0
    if layer == "stance":
        by_text = {text: stance for stance, text in payload["labels"].values()}
        for fid, fact in store["facts"].items():
            entry = payload["labels"].get(fid)
            value = entry[0] if entry else by_text.get(fact["canonical_text"])
            if value:
                fact.setdefault("properties", {})["stance"] = value
                hit += 1
    else:
        for mid, clock in payload["mentions"].items():
            if mid in store["mentions"]:
                store["mentions"][mid]["provenance"].setdefault("extra", {})[
                    "token_clock"
                ] = clock
        for fid, clock in payload["facts"].items():
            if fid in store["facts"]:
                store["facts"][fid].setdefault("properties", {})[
                    "first_output_token_clock"
                ] = clock
                hit += 1
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stance", type=Path, help="directory holding stance-annotated stores")
    ap.add_argument("--stance-glob", default="*.stance.json")
    ap.add_argument("--token-clock", type=Path,
                    help="directory holding clock-annotated stores; pass "
                         "--token-clock-glob '*.v2.json' for stores written by "
                         "retrace.py --record-token-clock")
    ap.add_argument("--token-clock-glob", default="*.tokens.json")
    ap.add_argument("--out", type=Path, default=LABELS)
    args = ap.parse_args()

    if args.stance:
        n = export_stance(args.stance, args.out / "stance", args.stance_glob)
        print(f"stance      {n:3d} debates -> {args.out / 'stance'}")
    if args.token_clock:
        n = export_token_clock(args.token_clock, args.out / "token_clock",
                               args.token_clock_glob)
        print(f"token_clock {n:3d} debates -> {args.out / 'token_clock'}")


if __name__ == "__main__":
    main()
