"""Re-extract, atomise, and match the Perspectrum debates properly.

Rebuilds the fact stores from the saved transcripts, replacing three choices
that made the first pass unmeasurable:

  extraction   run_perspectrum used its own four-line prompt, which asks for
               facts that are short and self-contained but never asks that they
               be atomic, and caps them at eight. On one turn that yielded 8
               joined facts against 23 atomic ones from EXTRACTION_SYSTEM.

  atomising    a safety net over extraction, since even the tuned prompt leaves
               the occasional conjunction. Two thirds of a store never reaches
               the model - a regex decides who is suspect.

  matching     SAME/DIFFERENT rather than the five-way relation. The typed
               relation is the better description and the wrong instrument:
               deciding a direction of entailment cost 15999 reasoning tokens
               and an empty reply, while sameness costs 3656 and answers. Once
               facts are atomic the question is mostly set membership anyway.

    python experiments/retrace.py experiments/perspectrum_pilot_full \
        --model minimax-m2.5 --suffix .v2.json

Stores are written beside the originals under a new suffix; nothing is
overwritten, so the two passes can be compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.atomize import atomize  # noqa: E402
from factflow.blocking import SbertBlocker, TfidfBlocker  # noqa: E402
from factflow.extract import extract_facts  # noqa: E402
from factflow.llm import LLM  # noqa: E402
from factflow.match import match  # noqa: E402
from factflow.types import Channel, Provenance  # noqa: E402


def slots(debate: dict) -> list[tuple[str, Provenance]]:
    """Every piece of text in a run, with where it came from.

    Evidence and the claim are SOURCE: they are what every agent was given, so
    a fact traceable to them is grounded rather than invented. Turns are OUTPUT.
    """
    ex = debate["execution_id"]
    out = [(debate["claim"], Provenance(execution_id=ex, round=0, channel=Channel.SOURCE,
                                        doc_id="claim"))]
    for e in debate.get("evidence", []):
        out.append((e["text"], Provenance(execution_id=ex, round=0, channel=Channel.SOURCE,
                                          doc_id=e["id"], extra={"stance": e.get("stance")})))
    for slot, text in debate["transcript"].items():
        agent, rnd = slot.split("|")
        out.append((text, Provenance(execution_id=ex, agent_id=agent, round=int(rnd),
                                     channel=Channel.OUTPUT)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*deepseek*.debate.json")
    ap.add_argument("--model", default="minimax-m2.5")
    ap.add_argument("--suffix", default=".v2.json")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="reasoning eats ~4000 of this before a character is written")
    ap.add_argument("--timeout", type=float, default=200.0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--embed", default="BAAI/bge-base-en-v1.5",
                    help="blocker model; '' falls back to TF-IDF")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="0.70 on bge-base loses no SAME and drops 94.9%% of the rest")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-atomize", action="store_true")
    a = ap.parse_args()

    paths = sorted(a.out_dir.glob(a.glob))
    if not paths:
        print(f"no {a.glob} under {a.out_dir}", file=sys.stderr)
        return 1

    blocker = SbertBlocker(a.embed) if a.embed else TfidfBlocker()
    llm = LLM.opencode(a.model, max_concurrency=a.concurrency, max_tokens=a.max_tokens)
    llm.backend.client = llm.backend.client.with_options(timeout=a.timeout, max_retries=1)

    for i, p in enumerate(paths, 1):
        target = p.with_name(p.name.replace(".debate.json", "") + a.suffix)
        if target.exists():
            print(f"[{i}/{len(paths)}] {p.name} -> done, skipping", flush=True)
            continue
        debate = json.loads(p.read_text())
        label = debate["execution_id"].replace("perspectrum-", "")
        t0 = time.time()

        pieces = slots(debate)

        # The fourteen slots of a run are independent, so extracting them one
        # after another spends fourteen round trips of wall clock on work that
        # fits in two. `llm.map` bounds the fan-out at the configured
        # concurrency and keeps input order, so mentions stay in slot order.
        def _extract(piece):
            text, prov = piece
            try:
                return extract_facts(llm, text, prov)
            except Exception as exc:
                print(f"    extract failed {prov.doc_id or prov.agent_id}|{prov.round}: "
                      f"{type(exc).__name__}", flush=True)
                return []

        mentions = [m for group in llm.map(_extract, pieces) for m in group]
        raw = len(mentions)
        if not raw:
            print(f"[{i}/{len(paths)}] {label}: no facts extracted, skipping", flush=True)
            continue

        if not a.no_atomize:
            mentions = atomize(llm, mentions, batch_size=20)
        split = len(mentions)

        store = match(llm, mentions, blocker=blocker, threshold=a.threshold,
                      top_k=a.top_k, batch_size=a.batch_size,
                      progress=label)
        store.save(str(target))
        print(f"[{i}/{len(paths)}] {label}  extract {raw} -> atomise {split} -> "
              f"{len(store.facts)} facts  (merge {1 - len(store.facts)/max(split,1):.0%})  "
              f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
