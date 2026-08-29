"""Attach Perspectrum's annotated perspectives to a store as gold source facts.

Every claim in Perspectrum ships with human-written perspectives, each labelled
support or undermine. They were sitting unused in `gold_perspectives` while the
analysis reported that 100% of facts were `injected` - true, but only because
nothing had been marked otherwise.

With them in the store, a run can be asked a question it could not be asked
before, and one with an answer outside our own definitions: how much of the
annotated argument did this debate actually reach? Coverage is not a metric we
chose; it is a comparison against what people said the arguments were.

Runs as a separate additive pass so it can be applied to stores that already
exist, without re-extracting or re-matching them. Gold facts are registered as
SOURCE mentions and linked to whatever agents said via the ordinary matcher, so
"an agent reached this perspective" means the same thing it means everywhere
else in the pipeline.

    python experiments/add_gold.py experiments/perspectrum_pilot_full \
        --glob "*.v2.json" --model minimax-m2.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.atomize import atomize  # noqa: E402
from factflow.blocking import SbertBlocker  # noqa: E402
from factflow.extract import extract_facts  # noqa: E402
from factflow.llm import LLM  # noqa: E402
from factflow.match import match  # noqa: E402
from factflow.types import Channel, FactStore, Provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*.v2.json")
    ap.add_argument("--model", default="minimax-m2.5")
    ap.add_argument("--embed", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=200.0)
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()

    paths = sorted(a.out_dir.glob(a.glob))
    if not paths:
        print(f"no {a.glob} under {a.out_dir}", file=sys.stderr)
        return 1

    blocker = SbertBlocker(a.embed)
    llm = LLM.opencode(a.model, max_concurrency=a.concurrency, max_tokens=a.max_tokens)
    llm.backend.client = llm.backend.client.with_options(timeout=a.timeout, max_retries=1)

    seen = (0, 0, 0)
    for i, p in enumerate(paths, 1):
        debate_p = p.with_name(p.name.split(".")[0] + ".debate.json")
        if not debate_p.exists():
            print(f"[{i}/{len(paths)}] {p.name}: no debate file, skipping", flush=True)
            continue
        debate = json.loads(debate_p.read_text())
        gold = debate.get("gold_perspectives") or []
        if not gold:
            print(f"[{i}/{len(paths)}] {p.name}: no gold_perspectives", flush=True)
            continue
        store = FactStore.load(str(p))
        if any(m.provenance.extra.get("gold") for m in store.mentions.values()):
            print(f"[{i}/{len(paths)}] {p.name}: gold already present, skipping", flush=True)
            continue

        ex = debate["execution_id"]
        t0 = time.time()
        mentions = []
        for n, g in enumerate(gold):
            prov = Provenance(execution_id=ex, round=0, channel=Channel.SOURCE,
                              doc_id=f"gold-{n}",
                              extra={"gold": True, "stance": g.get("stance")})
            try:
                mentions.extend(extract_facts(llm, g["text"], prov))
            except Exception as exc:
                print(f"    gold-{n} extract failed: {type(exc).__name__}", flush=True)
        if not mentions:
            print(f"[{i}/{len(paths)}] {p.name}: gold extraction produced nothing", flush=True)
            continue
        mentions = atomize(llm, mentions, batch_size=20)

        before = len(store.facts)
        # Matching against the existing store is what links a perspective to the
        # agents that reached it: a gold mention joining an existing fact means
        # somebody said it, and a gold fact standing alone means nobody did.
        store = match(llm, mentions, store=store, blocker=blocker,
                      threshold=a.threshold, top_k=12, batch_size=16,
                      )
        store.save(str(p))
        reached = sum(1 for f in store.facts.values()
                      if any(store.mentions[m].provenance.extra.get("gold")
                             for m in f.mention_ids)
                      and any(store.mentions[m].provenance.channel == Channel.OUTPUT
                              for m in f.mention_ids))
        n_gold = sum(1 for f in store.facts.values()
                     if any(store.mentions[m].provenance.extra.get("gold")
                            for m in f.mention_ids))
        print(f"[{i}/{len(paths)}] {p.name}  +{len(mentions)} gold mentions  "
              f"facts {before} -> {len(store.facts)}  "
              f"reached {reached}/{n_gold} gold facts  {time.time()-t0:.0f}s  "
              f"{_delta(llm, seen)}", flush=True)
        seen = (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens)
    print(f"\ntotal: {llm.usage.report(a.model)}", flush=True)
    return 0


def _delta(llm, seen) -> str:
    """This store's share of the running total, so a per-store cost is measured
    rather than inferred from an average."""
    from factflow.backends import Usage
    d = Usage(input_tokens=llm.usage.input_tokens - seen[1],
              output_tokens=llm.usage.output_tokens - seen[2],
              calls=llm.usage.calls - seen[0])
    return d.report(llm.model)


if __name__ == "__main__":
    raise SystemExit(main())
