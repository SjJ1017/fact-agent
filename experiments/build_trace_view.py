"""Extract everything a reader needs to inspect one debate, turn by turn.

Every number in the reports is a count over spans of text that a matcher
decided were the same proposition. Nothing published so far lets anyone look at
those spans. This writes the per-debate view data the trace viewer renders:
the transcript as it was produced, the character offsets of each extracted
fact inside it, which canonical fact each span belongs to, and the role prompt
each agent was given.

Span location is the fiddly part. The extractor records a `quote`, the verbatim
snippet it drew a fact from, but 15% of quotes do not appear literally in the
transcript — curly apostrophes, collapsed whitespace, an ellipsis. Folding both
sides (quote characters, case, runs of whitespace) while keeping an index map
back to the original recovers most of them, and a longest-prefix fallback
catches the rest, taking 85% located to 95%. Spans that still cannot be placed
are reported per debate rather than dropped silently, because an unplaced span
is exactly the kind of thing this viewer exists to make visible.

    python experiments/build_trace_view.py
    python experiments/build_trace_view.py \
        --store-dir experiments/clinicalbench_pilot \
        --suffix .store.json \
        --no-stance \
        --out findings/data/clinicalbench-trace-view.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from export_labels import attach_labels

ROOT = Path(__file__).resolve().parent
STORE_DIRS = [ROOT / "perspectrum_pilot_full", ROOT / "perspectrum_pilot_star_chain"]
OUT = ROOT.parent / "findings" / "data" / "trace-view.json"

FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                      "–": "-", "—": "-", "…": ".", " ": " "})
FINAL_RE = re.compile(
    r"^((?:FINAL(?: ANSWER| DIAGNOSIS)?|VERDICT|ANSWER))\s*[:：]\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)

ROLE_SUMMARY = {
    "neutral": {"A": "中立证据分析师", "B": "中立证据分析师", "C": "中立证据分析师"},
    "lenses": {"A": "因果与证据能确立什么", "B": "可行性与权衡",
               "C": "适用范围与不确定性"},
    "stance": {"A": "命题的支持方", "B": "命题的反对方", "C": "中立裁决者"},
}


def fold(text: str) -> tuple[str, list[int]]:
    """Case-folded, quote-normalised text plus a map back to original indices."""
    chars: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        c = ch.translate(FOLD)
        if c.isspace():
            if prev_space:
                continue
            c, prev_space = " ", True
        else:
            prev_space = False
        chars.append(c.lower())
        index.append(i)
    return "".join(chars), index


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """Character span of `quote` inside `text`, tolerating the usual drift."""
    folded, index = fold(text)
    needle, _ = fold(quote or "")
    needle = needle.strip()
    if not needle:
        return None
    at = folded.find(needle)
    if at >= 0:
        end = min(at + len(needle) - 1, len(index) - 1)
        return index[at], index[end] + 1
    for cut in (0.8, 0.6, 0.45):
        head = needle[: max(12, int(len(needle) * cut))]
        at = folded.find(head)
        if at >= 0:
            end = min(at + len(head) - 1, len(index) - 1)
            return index[at], index[end] + 1
    return None


def short_role(prompt: str) -> str:
    first = (prompt or "").strip().split(".", 1)[0]
    first = re.sub(r"^You are an?\s+", "", first, flags=re.IGNORECASE)
    return first[:56] + ("…" if len(first) > 56 else "")


def run_name(store_path: Path, suffix: str) -> str:
    if not store_path.name.endswith(suffix):
        raise ValueError(f"{store_path} does not end with {suffix}")
    return store_path.name[: -len(suffix)]


def build_links(debate: dict, spans: dict[str, list[dict]],
                source_docs: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Build drawable fact edges from recorded visibility and source ownership.

    The browser should not infer flow from DOM order. In particular, a peer
    restating a fact the receiver already said is persistence, not uptake; and
    a source document cannot connect directly to an agent that was not dealt it.
    Only placed spans are included because every endpoint must exist in the UI.
    """
    said = {
        slot: {group["f"] for span in items for group in span["fs"]}
        for slot, items in spans.items()
    }
    agents = list(debate.get("roles", {}))
    rounds = sorted({int(slot.split("|")[1]) for slot in debate["transcript"]})
    delivery = debate.get("delivery", {})
    evidence_ids = {item["id"] for item in debate.get("evidence", [])}
    links: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()

    def add(fact_id: str, kind: str, source: str, target: str) -> None:
        key = fact_id, kind, source, target
        if key not in seen:
            seen.add(key)
            links[fact_id].append(
                {"kind": kind, "from": source, "to": target})

    # A source edge is an activation path, not a generic lexical association.
    # Draw it only when this agent held the source document and first expressed
    # the fact. With legacy traces that lack per-turn source_ids, fall back to
    # the full evidence set rather than deleting every source edge.
    for agent in agents:
        expressed_before: set[str] = set()
        for rnd in rounds:
            slot = f"{agent}|{rnd}"
            current = said.get(slot, set())
            info = delivery.get(slot, {})
            held = set(info.get("source_ids", evidence_ids))
            for fact_id in current - expressed_before:
                for doc_id in source_docs.get(fact_id, []):
                    if doc_id in held:
                        add(fact_id, "origin", doc_id, slot)
            expressed_before |= current

    # Same-agent continuation is distinct from peer uptake.
    for agent in agents:
        for previous, current in zip(rounds, rounds[1:]):
            source, target = f"{agent}|{previous}", f"{agent}|{current}"
            for fact_id in said.get(source, set()) & said.get(target, set()):
                add(fact_id, "persistence", source, target)

    # A peer edge is possible uptake only when the fact is new to the receiver
    # and an earlier visible peer turn expressed it. Keep at most the latest
    # matching turn per source agent under cumulative memory.
    for receiver in agents:
        expressed_before: set[str] = set()
        for rnd in rounds:
            target = f"{receiver}|{rnd}"
            new_facts = said.get(target, set()) - expressed_before
            info = delivery.get(target, {})
            visible = info.get("visible_peer_turns", info.get("peer_turns", []))
            by_agent: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for source in visible:
                source_agent, source_round = source.split("|", 1)
                if int(source_round) < rnd:
                    by_agent[source_agent].append((int(source_round), source))
            for fact_id in new_facts:
                for candidates in by_agent.values():
                    matching = [item for item in candidates
                                if fact_id in said.get(item[1], set())]
                    if matching:
                        add(fact_id, "transmission", max(matching)[1], target)
            expressed_before |= said.get(target, set())

    return {fact_id: edges for fact_id, edges in links.items()}


def build(store_path: Path, suffix: str = ".v2.json", use_stance: bool = True) -> dict | None:
    name = run_name(store_path, suffix)
    debate_path = store_path.with_name(f"{name}.debate.json")
    if not debate_path.exists():
        return None
    with store_path.open() as fh:
        store = json.load(fh)
    with debate_path.open() as fh:
        debate = json.load(fh)
    if use_stance:
        attach_labels(store, name, "stance")

    stance_of = (
        {fid: f.get("properties", {}).get("stance")
         for fid, f in store["facts"].items()}
        if use_stance else {}
    )
    # Which evidence documents a fact's cluster also touches. A cluster holding
    # both a source mention from E2 and an output mention is a fact the agent
    # took from the dossier rather than introduced, and the viewer can draw
    # that edge. Only 12.7% of source facts merge with anything an agent said,
    # so this is a floor: absence of an edge is not evidence of novelty.
    doc_of = {mid: m["provenance"].get("doc_id")
              for mid, m in store["mentions"].items()
              if m["provenance"]["channel"] == "source"}
    src_of = {}
    for fid, fact in store["facts"].items():
        docs = sorted({doc_of[m] for m in fact["mention_ids"]
                       if m in doc_of and doc_of[m] not in (None, "claim")})
        if docs:
            src_of[fid] = docs
    canonical = {fid: f["canonical_text"] for fid, f in store["facts"].items()}

    # spans, grouped by the slot they sit in
    spans: dict[str, list[dict]] = {}
    unplaced = 0
    for mid, mention in store["mentions"].items():
        prov = mention["provenance"]
        if prov["channel"] != "output" or not prov.get("agent_id"):
            continue
        slot = f"{prov['agent_id']}|{prov['round']}"
        text = debate["transcript"].get(slot, "")
        where = locate(text, mention.get("quote") or "")
        fid = store["mention_to_fact"].get(mid)
        if where is None or not fid:
            unplaced += 1
            continue
        spans.setdefault(slot, []).append(
            {"a": where[0], "b": where[1], "f": fid, "t": mention["text"]})


    # Overlapping spans cannot both be marked in the DOM, but throwing the
    # shorter one away throws away a *fact*. Extraction routinely draws several
    # atoms from one sentence — "E2 concerns language and multiculturalism, not
    # religious symbols" yields three, all quoting the same clause — and the
    # earlier version kept the longest and silently dropped the rest, so those
    # facts looked unextracted in the viewer. Now one region carries the list.
    for slot, items in spans.items():
        items.sort(key=lambda x: (x["a"], -(x["b"] - x["a"])))
        kept: list[dict] = []
        for item in items:
            if kept and item["a"] < kept[-1]["b"]:
                host = kept[-1]
                if item["f"] not in {g["f"] for g in host["fs"]}:
                    host["fs"].append({"f": item["f"], "t": item["t"]})
                if item["b"] > host["b"]:
                    host["b"] = item["b"]
                continue
            kept.append({"a": item["a"], "b": item["b"],
                         "f": item["f"], "t": item["t"],
                         "fs": [{"f": item["f"], "t": item["t"]}]})
        spans[slot] = kept

    order = sorted(debate["transcript"],
                   key=lambda s: (int(s.split("|")[1]), s.split("|")[0]))
    used = {g["f"] for items in spans.values() for s in items for g in s["fs"]}
    links = build_links(debate, spans, src_of)

    verdicts = {}
    for slot, text in debate["transcript"].items():
        hit = None
        for hit in FINAL_RE.finditer(text):
            pass
        if hit:
            at = hit.start()
            verdicts[slot] = {"a": at, "b": len(text),
                              "l": hit.group(1).strip().upper(),
                              "v": hit.group(2).strip().upper()}

    role_short = ROLE_SUMMARY.get(debate.get("panel"), {})
    if not role_short:
        role_short = {agent: short_role(prompt)
                      for agent, prompt in debate.get("roles", {}).items()}

    return {
        "id": name,
        "model": debate.get("model", "?"),
        "claim_id": debate.get("claim_id", debate.get("case_id", name)),
        "claim": debate["claim"],
        "topology": debate["topology"],
        "panel": debate["panel"],
        "order": order,
        "text": debate["transcript"],
        "spans": spans,
        "verdicts": verdicts,
        "facts": {fid: {"c": canonical[fid], "s": stance_of.get(fid),
                        **({"src": src_of[fid]} if fid in src_of else {})}
                  for fid in used},
        "links": {fid: links.get(fid, []) for fid in used},
        "roles": debate["roles"],
        "role_short": role_short,
        "delivery": {slot: info.get("peer_turns", [])
                     for slot, info in debate.get("delivery", {}).items()},
        # Round 1 was labelled "no input", which is false and hid the thing
        # that matters most for reading any of this: the dossier is in context
        # on every single turn, all three rounds. Nobody has to receive a
        # peer's message to restate E1.
        "sources": {slot: info.get("source_ids", [])
                    for slot, info in debate.get("delivery", {}).items()},
        # Present only under self-last memory. The viewer has to distinguish
        # "this agent said it again having seen itself say it" from "this agent
        # produced it again with no record of having done so", and those look
        # identical without this.
        "self_prior": {slot: info["self_turn"]
                       for slot, info in debate.get("delivery", {}).items()
                       if info.get("self_turn")},
        "visible": {slot: {"peers": info.get("visible_peer_turns",
                                             info.get("peer_turns", [])),
                           "self": info.get("visible_self_turns", [])}
                    for slot, info in debate.get("delivery", {}).items()},
        # Run-level, not inferred from whether some turn happens to have a
        # prior: round 1 never does, under any condition.
        "memory": debate.get("memory",
                             "self-last" if debate.get("self_history") else "peer-only"),
        "evidence": [{"id": e["id"], "stance": e.get("stance"), "text": e["text"]}
                     for e in debate.get("evidence", [])],
        "unplaced": unplaced,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every trace, not just the model the reports analyse. The 33 GLM debates
    # have no star/chain counterpart so they are excluded from every cross-
    # topology number, but they are real runs and this page is for looking at
    # runs, not for comparing conditions.
    ap.add_argument("--store-dir", type=Path, action="append", default=None,
                    help="directory containing matched store/debate files; repeatable")
    ap.add_argument("--suffix", default=".v2.json",
                    help="store filename suffix to strip before .debate.json lookup")
    ap.add_argument("--model", default="", help="留空 = 渲染全部模型")
    ap.add_argument("--no-stance", action="store_true",
                    help="do not attach stance labels; use for clinicalbench")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    debates = []
    for directory in args.store_dir or STORE_DIRS:
        pattern = f"*{args.model}*{args.suffix}" if args.model else f"*{args.suffix}"
        for path in sorted(directory.glob(pattern)):
            view = build(path, suffix=args.suffix, use_stance=not args.no_stance)
            if view:
                debates.append(view)

    total_spans = sum(len(s) for d in debates for s in d["spans"].values())
    total_unplaced = sum(d["unplaced"] for d in debates)
    payload = {"model": args.model or "all", "stance": not args.no_stance,
               "debates": debates,
               "n_spans": total_spans, "n_unplaced": total_unplaced}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False,
                                   separators=(",", ":")))
    size = args.out.stat().st_size
    from collections import Counter
    by_model = Counter(d["model"] for d in debates)
    print("按模型：" + "　".join(f"{k} {v}" for k, v in sorted(by_model.items())))
    print(f"{len(debates)} 场 · {total_spans} 个已定位 span · "
          f"{total_unplaced} 个未定位 "
          f"({total_spans / (total_spans + total_unplaced):.1%} 命中)")
    print(f"wrote {args.out}  ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
