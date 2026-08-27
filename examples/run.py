"""Full pipeline on a small trace. Requires ANTHROPIC_API_KEY or `ant auth login`.

    python examples/run.py

What to look for in the output: agent A states the trial dates and the transfer
amount in round 1, then in round 2 restates the evidence as "UBI reduces poverty
without hurting employment".  The specifics are gone.  Because the matcher
returns typed relations rather than a same/different bit, that shows up as an
entailment edge, not as a merge - the round-2 fact is a *different* canonical
fact that the round-1 facts entail.
"""

import json
from pathlib import Path

from factflow import (
    CRITICALITY,
    TRUTH,
    LLM,
    LLMConfig,
    TraceRecord,
    annotate_store,
    extract_trace,
    match,
)

HERE = Path(__file__).parent
QUESTION = "Should governments implement universal basic income?"

llm = LLM(LLMConfig(max_concurrency=8))

records = [TraceRecord.model_validate(r) for r in json.loads((HERE / "trace.json").read_text())]
source_text = next(r.text for r in records if r.provenance.channel.value == "source")

print(f"[1/3] extracting from {len(records)} records ...")
mentions = extract_trace(llm, records, focus=QUESTION)
print(f"      {len(mentions)} mentions")

print("[2/3] matching ...")
store = match(llm, mentions)
print(f"      {len(store.facts)} canonical facts")

print("[3/3] annotating ...")
store = annotate_store(llm, store, [TRUTH, CRITICALITY], instruction=QUESTION, reference=source_text)

store.save(str(HERE / "store.json"))
print(f"\nsaved -> {HERE / 'store.json'}\n")

print(f"{'fact':<62} {'crit':<5} {'truth':<12} where")
print("-" * 110)
for fact in store.facts.values():
    where = sorted(
        f"{m.provenance.agent_id}r{m.provenance.round}:{m.provenance.channel.value[:3]}"
        for m in (store.mentions[i] for i in fact.mention_ids)
    )
    crit = "yes" if fact.properties.get("critical") else "no"
    print(f"{fact.canonical_text[:60]:<62} {crit:<5} {str(fact.properties.get('truth')):<12} {' '.join(where)}")

degraded = [r for r in store.relations if r.relation in ("A_ENTAILS_B", "B_ENTAILS_A")]
print(f"\n{len(degraded)} entailment edges (candidate degradation events):")
for r in degraded[:10]:
    a, b = store.mentions[r.a].text, store.mentions[r.b].text
    specific, general = (a, b) if r.relation == "A_ENTAILS_B" else (b, a)
    print(f"  {specific[:52]!r}\n    -> weakened to -> {general[:52]!r}")
