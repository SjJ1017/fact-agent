"""Build a self-contained interactive HTML view of matched debate traces.

Three views over the same data:

* **Flow** - canonical facts as rows, (agent, round) slots as columns. Where a
  fact was expressed, the cell is filled. Reading across a row is one fact's
  life; reading down a column is what an agent said at that point.
* **Rounds** - for each turn: the context the agent was given, what it said, and
  the atomic facts extracted from it, side by side. This is the view that makes
  extraction errors findable, because the source text sits next to its output.
* **Audit** - heuristic quality flags, worst first.

The page embeds its data as JSON and ships no external assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import audit
from .types import Channel, FactStore

TEMPLATE_CSS = Path(__file__).with_name("viewer.css")
TEMPLATE_JS = Path(__file__).with_name("viewer.js")


def _slot(prov) -> str:
    return "SRC" if prov.channel == Channel.SOURCE else f"{prov.agent_id}{prov.round}"


def build_run(store: FactStore, debate: dict[str, Any], run_id: str) -> dict[str, Any]:
    agents = sorted({p.agent_id for p in (m.provenance for m in store.mentions.values()) if p.agent_id})
    rounds = sorted({p.round for p in (m.provenance for m in store.mentions.values())
                     if p.round and p.channel == Channel.OUTPUT})

    gold_titles = set(debate.get("gold_titles", []))

    facts = []
    for fid, f in store.facts.items():
        slots, origin, doc = [], "introduced", None
        for mid in f.mention_ids:
            p = store.mentions[mid].provenance
            slots.append(_slot(p))
            if p.channel == Channel.SOURCE:
                doc = p.doc_id
                origin = "gold" if p.extra.get("gold") else "distractor"
        facts.append({
            "id": fid, "text": f.canonical_text, "origin": origin, "doc": doc,
            "slots": sorted(set(slots)), "n": len(f.mention_ids),
            "mentions": [{"slot": _slot(store.mentions[m].provenance),
                          "text": store.mentions[m].text} for m in f.mention_ids],
        })
    # Source facts first, then by when they first appear.
    order = {"gold": 0, "distractor": 1, "introduced": 2}
    facts.sort(key=lambda f: (order[f["origin"]], -f["n"], f["text"]))

    # Reconstruct each turn's context. Under full broadcast this is exact:
    # the documents, plus every peer's previous-round message.
    transcript = {tuple(k.split("|")): v for k, v in debate["transcript"].items()}
    turns = []
    for r in rounds:
        for a in agents:
            text = transcript.get((a, str(r)))
            if text is None:
                continue
            peers = [(b, transcript[(b, str(r - 1))]) for b in agents
                     if b != a and (b, str(r - 1)) in transcript]
            turn_facts = [
                {"id": store.mention_to_fact.get(mid), "text": m.text}
                for mid, m in store.mentions.items()
                if m.provenance.agent_id == a and m.provenance.round == r
                and m.provenance.channel == Channel.OUTPUT
            ]
            turns.append({"agent": a, "round": r, "text": text,
                          "peers": [{"agent": b, "text": t} for b, t in peers],
                          "facts": turn_facts})

    spacetime = build_spacetime(store, agents, rounds)

    flags = [{"severity": f.severity, "kind": f.kind, "fact_id": f.fact_id,
              "detail": f.detail, "partner_id": f.partner_id} for f in audit(store)]

    return {
        "id": run_id,
        "question": debate["question"],
        "gold_answer": debate["gold_answer"],
        "gold_titles": sorted(gold_titles),
        "finals": debate.get("final", {}),
        "agents": agents,
        "rounds": rounds,
        "columns": ["SRC"] + [f"{a}{r}" for r in rounds for a in agents],
        "facts": facts,
        "turns": turns,
        "documents": debate.get("documents", ""),
        "spacetime": spacetime,
        "flags": flags,
        "stats": {
            "mentions": len(store.mentions),
            "facts": len(store.facts),
            "gold": sum(1 for f in facts if f["origin"] == "gold"),
            "distractor": sum(1 for f in facts if f["origin"] == "distractor"),
            "introduced": sum(1 for f in facts if f["origin"] == "introduced"),
        },
    }


def build_spacetime(store: FactStore, agents: list[str], rounds: list[int]) -> dict[str, Any]:
    """Time-expanded interaction graph: one node per (agent, round).

    G^t is the agent graph at round t; the edges carry facts rather than
    messages, which is the whole point of the content-centric view. Three edge
    kinds, and they mean different things:

      origin       SRC -> (a, t_first)   a source fact first surfaces at (a, t)
      transmission (a, t) -> (b, t+1)    b now expresses a fact a expressed at t
                                          and b did not express at t - the fact
                                          moved between agents
      persistence  (a, t) -> (a, t+1)    a expresses the same fact again - the
                                          fact survived in place

    Transmission is inferred from co-expression under full broadcast, not
    observed: b could have read the same document. It is an upper bound on real
    transfer, and labelled as such in the UI.
    """
    said: dict[tuple[str, int], set[str]] = {(a, r): set() for a in agents for r in rounds}
    source_facts: set[str] = set()
    for fid, fact in store.facts.items():
        for mid in fact.mention_ids:
            p = store.mentions[mid].provenance
            if p.channel == Channel.SOURCE:
                source_facts.add(fid)
            elif p.channel == Channel.OUTPUT and p.agent_id and p.round in said and False:
                pass
        for mid in fact.mention_ids:
            p = store.mentions[mid].provenance
            if p.channel == Channel.OUTPUT and (p.agent_id, p.round) in said:
                said[(p.agent_id, p.round)].add(fid)

    nodes = [{"id": f"{a}{r}", "agent": a, "round": r, "n": len(said[(a, r)])}
             for r in rounds for a in agents]
    edges: list[dict[str, Any]] = []

    first_round = rounds[0] if rounds else None
    if first_round is not None:
        for a in agents:
            shared = said[(a, first_round)] & source_facts
            if shared:
                edges.append({"kind": "origin", "src": "SRC", "dst": f"{a}{first_round}",
                              "facts": sorted(shared)})

    for r, nxt in zip(rounds, rounds[1:]):
        for a in agents:
            for b in agents:
                moved = said[(a, r)] & said[(b, nxt)]
                if a == b:
                    if moved:
                        edges.append({"kind": "persistence", "src": f"{a}{r}", "dst": f"{b}{nxt}",
                                      "facts": sorted(moved)})
                else:
                    moved = moved - said[(b, r)]
                    if moved:
                        edges.append({"kind": "transmission", "src": f"{a}{r}", "dst": f"{b}{nxt}",
                                      "facts": sorted(moved)})
    return {"nodes": nodes, "edges": edges, "agents": agents, "rounds": rounds,
            "source_facts": sorted(source_facts)}


def render(runs: list[dict[str, Any]], title: str = "Fact Flow Explorer") -> str:
    payload = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    css = TEMPLATE_CSS.read_text(encoding="utf-8")
    js = TEMPLATE_JS.read_text(encoding="utf-8")
    return (
        f"<title>{title}</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&"
        'family=IBM+Plex+Serif:wght@500;600&display=swap">\n'
        f"<style>{css}</style>\n"
        '<div id="app"></div>\n'
        f'<script id="data" type="application/json">{payload}</script>\n'
        f"<script>{js}</script>\n"
    )


def build(store_dir: str | Path, out: str | Path, title: str = "Fact Flow Explorer") -> Path:
    store_dir, out = Path(store_dir), Path(out)
    runs = []
    for sp in sorted(store_dir.glob("*.store.json")):
        run_id = sp.name.replace(".store.json", "")
        dp = store_dir / f"{run_id}.debate.json"
        if not dp.exists():
            continue
        runs.append(build_run(FactStore.load(str(sp)), json.loads(dp.read_text()), run_id))
    if not runs:
        raise FileNotFoundError(f"no matched runs found in {store_dir}")
    out.write_text(render(runs, title), encoding="utf-8")
    return out
