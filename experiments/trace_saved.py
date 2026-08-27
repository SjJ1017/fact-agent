"""Extract and match facts over debates that were already run.

Kept separate from the sweep because tracing costs far more than the debate it
traces, and because the interesting comparison is between configurations that
scored the SAME. full/generalist and chain/generalist both reach 53% on the hard
MMLU-Pro subset while differing sharply in unanimity (44/47 vs 37/47): if flow
analysis is worth anything, it should distinguish two systems that a scoreboard
cannot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from concurrent.futures import ThreadPoolExecutor

from factflow import LLM, Channel, FactStore, TraceRecord, extract_trace, match


def records_from_debate(d: dict, execution_id: str) -> list[TraceRecord]:
    recs = [{"text": d["documents"],
             "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                            "channel": "source", "doc_id": "vignette", "extra": {"gold": True}}}]
    for k, v in sorted(d.get("options", {}).items()):
        recs.append({"text": f"Option {k}: {v}",
                     "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                                    "channel": "source", "doc_id": f"option-{k}",
                                    "extra": {"gold": k == d["gold_answer"]}}})
    for key, text in sorted(d["transcript"].items(), key=lambda kv: (kv[0].split("|")[1], kv[0])):
        a, r = key.split("|")
        recs.append({"text": text,
                     "provenance": {"execution_id": execution_id, "agent_id": a, "round": int(r),
                                    "channel": "output", "doc_id": d["qid"],
                                    "extra": {"role": d.get("roles", {}).get(a, "generalist")}}})
    return [TraceRecord.model_validate(r) for r in recs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--configs", default="full-generalist,chain-generalist,full-specialists")
    ap.add_argument("--limit", type=int, default=8, help="questions per config")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.dir)
    mk = {"openai": LLM.openai, "opencode": LLM.opencode}[args.provider]
    llm = mk(args.model, max_concurrency=args.concurrency)

    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    todo = []
    for cfg in wanted:
        files = sorted(root.glob(f"*-{cfg}.debate.json"))[: args.limit]
        todo += [(cfg, f) for f in files]
    print(f"tracing {len(todo)} runs with {args.model}\n", flush=True)

    def one(item):
        cfg, path = item
        exec_id = path.stem.replace(".debate", "")
        out = root / f"{exec_id}.store.json"
        if out.exists():
            return cfg, FactStore.load(str(out))
        d = json.loads(path.read_text())
        try:
            mentions = extract_trace(llm, records_from_debate(d, exec_id),
                                     focus=d["question"][:200], include_discourse=False)
            store = match(llm, mentions)
        except Exception as exc:  # noqa: BLE001
            print(f"  {exec_id}: FAILED {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            return cfg, None
        store.save(str(out))
        print(f"  {exec_id}: {len(mentions)} mentions -> {len(store.facts)} facts", flush=True)
        return cfg, store

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, todo))

    by_cfg: dict[str, list[FactStore]] = {}
    for cfg, st in results:
        if st is not None:
            by_cfg.setdefault(cfg, []).append(st)

    print("\n" + "=" * 78)
    print(f"{'config':<20}{'facts':>8}{'src kept':>10}{'novel':>8}{'transmit':>10}{'persist':>9}{'entail':>8}")
    rows = []
    for cfg, stores in by_cfg.items():
        agg = {"facts": 0, "src": 0, "src_kept": 0, "novel": 0, "trans": 0, "pers": 0, "ent": 0}
        for s in stores:
            agents = sorted({m.provenance.agent_id for m in s.mentions.values() if m.provenance.agent_id})
            rounds = sorted({m.provenance.round for m in s.mentions.values()
                             if m.provenance.round and m.provenance.channel == Channel.OUTPUT})
            said = {(a, r): set() for a in agents for r in rounds}
            src = set()
            for fid, f in s.facts.items():
                agg["facts"] += 1
                for mid in f.mention_ids:
                    p = s.mentions[mid].provenance
                    if p.channel == Channel.SOURCE:
                        src.add(fid)
                    elif (p.agent_id, p.round) in said:
                        said[(p.agent_id, p.round)].add(fid)
                if not any(s.mentions[m].provenance.channel == Channel.SOURCE for m in f.mention_ids):
                    agg["novel"] += 1
            agg["src"] += len(src)
            agg["src_kept"] += len({f for slot in said.values() for f in slot} & src)
            for r, nxt in zip(rounds, rounds[1:]):
                for a in agents:
                    for b in agents:
                        moved = said[(a, r)] & said[(b, nxt)]
                        if a == b:
                            agg["pers"] += len(moved)
                        else:
                            agg["trans"] += len(moved - said[(b, r)])
            agg["ent"] += sum(1 for x in s.relations if x.relation in ("A_ENTAILS_B", "B_ENTAILS_A"))
        n = len(stores)
        row = dict(config=cfg, n=n, facts=agg["facts"] / n,
                   src_kept=agg["src_kept"] / max(agg["src"], 1), novel=agg["novel"] / n,
                   transmit=agg["trans"] / n, persist=agg["pers"] / n, entail=agg["ent"] / n)
        rows.append(row)
        print(f"{cfg:<20}{row['facts']:>8.0f}{row['src_kept']:>9.0%}{row['novel']:>8.0f}"
              f"{row['transmit']:>10.0f}{row['persist']:>9.0f}{row['entail']:>8.0f}")
    (root / "flow_compare.json").write_text(json.dumps(rows, indent=2))
    print("\nper-run averages; transmit = facts reaching an agent that had not said them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
