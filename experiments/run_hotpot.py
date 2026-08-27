"""Debate -> extract -> match, on a handful of HotpotQA questions.

Under full broadcast, CONTEXT is derived rather than extracted: a fact is
available to agent A at round r if it is in the source documents or was uttered
by any agent before round r.  That keeps the availability channel exact without
paying to extract the same paragraphs once per agent per round.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from mad import run_debate, save, to_trace

from factflow import LLM, Channel, FactStore, TraceRecord, extract_trace, match

OUT = Path(__file__).parent / "out"


def derive_context(store: FactStore, execution_id: str, agents: list[str], rounds: list[int]) -> dict:
    """Which facts were available to whom, when. Broadcast topology."""
    first_seen: dict[str, int] = {}
    for fid, fact in store.facts.items():
        for mid in fact.mention_ids:
            p = store.mentions[mid].provenance
            if p.execution_id != execution_id:
                continue
            r = 0 if p.channel == Channel.SOURCE else (p.round or 0)
            first_seen[fid] = min(first_seen.get(fid, 99), r)
    return {
        (fid, a, r): first_seen.get(fid, 99) < r
        for fid in store.facts
        for a in agents
        for r in rounds
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-questions", type=int, default=3)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    llm = LLM.deepseek(model=args.model, max_concurrency=args.concurrency)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = [r for _, r in zip(range(args.n_questions), ds)]

    summary, skipped = [], []
    for i, row in enumerate(rows):
        exec_id = f"hotpot-{i}"
        print(f"\n{'='*78}\n[{i+1}/{len(rows)}] {row['question']}\n  gold: {row['answer']}  ({row['type']}/{row['level']})")

        print("  debating ...", flush=True)
        try:
            result = run_debate(llm, row, n_agents=args.agents, n_rounds=args.rounds)
        except Exception as exc:  # noqa: BLE001
            # One unusable question should not cost the other eleven.
            print(f"  SKIPPED: {type(exc).__name__}: {exc}")
            skipped.append({"exec_id": exec_id, "question": row["question"], "error": str(exc)[:300]})
            continue
        save(result, OUT / f"{exec_id}.debate.json")
        print(f"  finals: {result.final}")

        records = [TraceRecord.model_validate(r) for r in to_trace(result, exec_id)]
        print(f"  extracting from {len(records)} records ...", flush=True)
        mentions = extract_trace(llm, records, focus=row["question"], include_discourse=False)
        by_ch = defaultdict(int)
        for m in mentions:
            by_ch[m.provenance.channel.value] += 1
        print(f"  {len(mentions)} mentions  {dict(by_ch)}")

        print("  matching ...", flush=True)
        store = match(llm, mentions)
        print(f"  -> {len(store.facts)} canonical facts")

        store.save(str(OUT / f"{exec_id}.store.json"))
        summary.append(
            {
                "exec_id": exec_id,
                "question": row["question"],
                "gold": row["answer"],
                "finals": result.final,
                "mentions": len(mentions),
                "facts": len(store.facts),
                "by_channel": dict(by_ch),
            }
        )

    (OUT / "summary.json").write_text(
        json.dumps({"runs": summary, "skipped": skipped}, indent=2, ensure_ascii=False)
    )
    print(f"\n{'='*78}\nwrote {len(summary)} runs to {OUT}")
    if skipped:
        print(f"SKIPPED {len(skipped)} question(s):")
        for s_ in skipped:
            print(f"   {s_['exec_id']}: {s_['error'][:120]}")
    if llm.failures:
        print(f"WARNING: {len(llm.failures)} LLM calls failed after repair:")
        for f in llm.failures[:5]:
            print("   ", f[:160])
    print("cache:", llm.cache.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
