"""Rebuild fact clusters from stored mentions using the real matcher.

`run_perspectrum.py` traces with `light_match`, which merges only on stemmed
Jaccard >= .80 with top_k=1 and never calls an adjudicator. Its docstring is
honest about being a fast validation pass, and its stated reasoning - that a
false merge fabricates a transmission whereas a false split only undercounts
one - holds for counting transmissions. It does not hold for demography.

When A states a fact at r1 and B restates it in different words at r2, a failed
merge records A's fact as having DIED and B's wording as having been BORN. So
false splitting inflates births and deaths together and drives cohort survival
toward zero: it does not undercount the signal, it manufactures the high-churn
pattern that the persona comparison is looking for.

Extraction is not repeated here. Mentions are already in the stores, so this
costs adjudication calls only - roughly 150 for 21 stores at the default
threshold, against the ~500 extraction calls that produced them.

    python experiments/rematch.py experiments/perspectrum_pilot_full \
        --model glm-5.3-flash --provider opencode

Originals are left alone; results are written alongside with a `.rematch.json`
suffix so the two can be compared before anything is replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.blocking import candidate_pairs  # noqa: E402
from factflow.llm import LLM  # noqa: E402
from factflow.match import match  # noqa: E402
from factflow.types import FactMention, FactStore  # noqa: E402


def load_mentions(path: Path) -> list[FactMention]:
    raw = json.loads(path.read_text())
    return [FactMention.model_validate(m) for m in raw["mentions"].values()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*.store.json")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--provider", default="opencode", choices=["opencode", "deepseek", "openai"])
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--suffix", default=".rematch.json")
    ap.add_argument("--stack-dump", type=Path,
                    help="if the pass stalls, dump every thread's stack here every 45s")
    ap.add_argument("--dry-run", action="store_true",
                    help="report candidate-pair counts and cost, call nothing")
    a = ap.parse_args()

    if a.stack_dump:
        import faulthandler
        faulthandler.dump_traceback_later(45, repeat=True, exit=False,
                                          file=open(a.stack_dump, "w"))

    paths = sorted(a.out_dir.glob(a.glob))
    paths = [p for p in paths if not p.name.endswith(a.suffix)]
    if not paths:
        print(f"no stores matching {a.glob} under {a.out_dir}", file=sys.stderr)
        return 1

    if a.dry_run:
        total = 0
        for p in paths:
            ms = load_mentions(p)
            n = len(candidate_pairs(ms, threshold=a.threshold, top_k=a.top_k))
            total += n
            print(f"  {p.name:<62} {len(ms):>4} mentions  {n:>5} pairs")
        print(f"\n{len(paths)} stores, {total} candidate pairs, "
              f"~{total // a.batch_size + len(paths)} adjudication calls")
        return 0

    llm = getattr(LLM, a.provider)(a.model, max_concurrency=a.concurrency)
    # A gateway request that never returns will otherwise stall the whole pass:
    # the client default retries twice, so an unbounded call costs 3x the wait.
    llm.backend.client = llm.backend.client.with_options(
        timeout=a.timeout, max_retries=a.max_retries)

    for i, p in enumerate(paths, 1):
        target = p.with_name(p.name.replace(".store.json", "") + a.suffix)
        if target.exists():
            print(f"[{i}/{len(paths)}] {p.name} -> already done, skipping", flush=True)
            continue
        try:
            ms = load_mentions(p)
            before = len(json.loads(p.read_text())["facts"])
        except Exception as exc:
            # The tracer may still be writing this file; leave it for a later pass.
            print(f"[{i}/{len(paths)}] {p.name} -> unreadable ({exc}), skipping", flush=True)
            continue
        try:
            store = match(llm, ms, threshold=a.threshold, top_k=a.top_k, batch_size=a.batch_size)
        except Exception as exc:
            print(f"[{i}/{len(paths)}] {p.name} -> MATCH FAILED: {exc}", flush=True)
            continue
        store.save(str(target))
        after = len(store.facts)
        print(f"[{i}/{len(paths)}] {p.name}  {len(ms)} mentions  "
              f"facts {before} -> {after}  (merge {1 - after / max(len(ms), 1):.1%}, "
              f"was {1 - before / max(len(ms), 1):.1%})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
