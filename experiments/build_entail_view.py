"""Trace view with the entailment edges, not only the equality ones.

build_trace_view.py draws edges between occurrences of *one* fact: the same
proposition restated, carried forward, or taken up.  The NLI matcher produces
something that structure cannot hold -- an edge between *two different* facts,
one strictly weaker than the other.  That is the degradation signal: a fact
that survives with its number, its attribution or its condition removed.

Two edges are kept apart on purpose:

  entail      one statement entails the other, nothing more.  It says the two
              differ in strength; it says nothing about one coming from the
              other.
  degraded    the same, plus the evidence that it could have travelled: the
              weaker statement is later, and its speaker had the stronger turn
              visible.  Only these are claims about flow.

Everything else -- span placement, the equality links, the dossier panel --
comes from build_trace_view.py unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from build_trace_view import build, run_name  # noqa: E402


def entail_edges(store: dict, debate: dict, spans: dict) -> list[dict]:
    """Directed strong -> weak edges between two facts, with flow evidence."""
    m2f = store["mention_to_fact"]
    mentions = store["mentions"]
    if isinstance(mentions, list):
        mentions = {m["mention_id"]: m for m in mentions}
    facts = store["facts"]

    def slot_of(mid: str) -> str | None:
        p = (mentions.get(mid) or {}).get("provenance", {})
        return f"{p['agent_id']}|{p['round']}" if p.get("agent_id") else None

    placed = {slot: {g["f"] for s in items for g in s["fs"]}
              for slot, items in spans.items()}
    delivery = debate.get("delivery", {})

    out, seen = [], set()
    for r in store.get("relations", []):
        kind = r.get("relation")
        if kind not in ("A_ENTAILS_B", "B_ENTAILS_A"):
            continue
        strong_m, weak_m = ((r["a"], r["b"]) if kind == "A_ENTAILS_B"
                            else (r["b"], r["a"]))
        sf, wf = m2f.get(strong_m), m2f.get(weak_m)
        ss, ws = slot_of(strong_m), slot_of(weak_m)
        if not (sf and wf and ss and ws) or sf == wf:
            continue
        # both endpoints must be visible in the UI
        if sf not in placed.get(ss, ()) or wf not in placed.get(ws, ()):
            continue
        key = (sf, wf, ss, ws)
        if key in seen:
            continue
        seen.add(key)

        s_agent, s_round = ss.split("|")
        w_agent, w_round = ws.split("|")
        sr, wr = int(s_round), int(w_round)

        def could_see(target_slot: str, source_slot: str) -> bool:
            """Did the speaker of `target_slot` have `source_slot` in context?"""
            info = delivery.get(target_slot, {})
            vis = set(info.get("visible_peer_turns",
                               info.get("peer_turns", [])))
            vis |= set(info.get("visible_self_turns", []))
            src_agent = source_slot.split("|")[0]
            return source_slot in vis or src_agent == target_slot.split("|")[0]

        # Three situations that the earlier single `entail` label conflated.
        # Only the first is a claim about a fact losing content as it moves.
        if sr < wr:
            # the stronger statement came first: the weaker one may be a
            # degraded restatement, if its speaker could see the stronger turn
            kind = "degraded" if could_see(ws, ss) else "weaker_later"
        elif sr > wr:
            # the weaker came first.  This is refinement, not loss: a direction
            # stated and then given its parameters.  Same evidential test.
            kind = "refined" if could_see(ss, ws) else "stronger_later"
        else:
            # same round.  Two agents in one round generate in parallel and
            # cannot have transmitted; one agent's own two sentences differ in
            # wording, not in flow.
            kind = "concurrent"

        p = r.get("properties") or {}
        out.append({
            "kind": kind,
            "strong_fact": sf, "weak_fact": wf,
            "from": ss, "to": ws,
            "strong_text": facts.get(sf, {}).get("canonical_text", ""),
            "weak_text": facts.get(wf, {}).get("canonical_text", ""),
            "margin_strong": round(p.get("margin_ab" if kind == "A_ENTAILS_B"
                                         else "margin_ba", 0.0), 2),
            "margin_weak": round(p.get("margin_ba" if kind == "A_ENTAILS_B"
                                       else "margin_ab", 0.0), 2),
            "same_agent": s_agent == w_agent,
            "strong_round": sr, "weak_round": wr,
            "strong_agent": s_agent, "weak_agent": w_agent,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, action="append", required=True)
    ap.add_argument("--suffix", default=".store.json")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "findings" / "data" / "entail-view.json")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    debates, tally = [], defaultdict(int)
    for d in a.store_dir:
        for path in sorted(d.glob(f"*{a.suffix}")):
            view = build(path, suffix=a.suffix, use_stance=False)
            if not view:
                continue
            store = json.loads(path.read_text())
            deb = json.loads(
                path.with_name(path.name.replace(a.suffix, ".debate.json")).read_text())
            edges = entail_edges(store, deb, view["spans"])
            view["entail"] = edges
            for e in edges:
                tally[e["kind"]] += 1
            tally["debates"] += 1
            debates.append(view)
            if a.limit and len(debates) >= a.limit:
                break
        if a.limit and len(debates) >= a.limit:
            break

    payload = {"debates": debates,
               "counts": {k: v for k, v in tally.items() if k != "debates"},
               "n_degraded": tally["degraded"],
               "n_spans": sum(len(s) for d in debates for s in d["spans"].values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False,
                                separators=(",", ":")))
    print(f"{tally['debates']} 场 · {payload['n_spans']} span")
    for k in ("degraded", "refined", "weaker_later", "stronger_later",
              "concurrent"):
        if tally[k]:
            print(f"  {k:16s} {tally[k]:6,}")
    print(f"写入 {a.out}  ({a.out.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
