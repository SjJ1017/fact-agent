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
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from export_labels import attach_labels

ROOT = Path(__file__).resolve().parent
STORE_DIRS = [ROOT / "perspectrum_pilot_full", ROOT / "perspectrum_pilot_star_chain"]
OUT = ROOT.parent / "findings" / "data" / "trace-view.json"

FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                      "–": "-", "—": "-", "…": ".", " ": " "})
VERDICT_RE = re.compile(r"VERDICT\s*[:：]\s*([A-Za-z_ ]+)")

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


def build(store_path: Path) -> dict | None:
    name = store_path.name[: -len(".v2.json")]
    debate_path = store_path.with_name(f"{name}.debate.json")
    if not debate_path.exists():
        return None
    with store_path.open() as fh:
        store = json.load(fh)
    with debate_path.open() as fh:
        debate = json.load(fh)
    attach_labels(store, name, "stance")

    stance_of = {fid: f.get("properties", {}).get("stance")
                 for fid, f in store["facts"].items()}
    # Which evidence documents a fact's cluster also touches. A cluster holding
    # both a source mention from E2 and an output mention is a fact the agent
    # took from the dossier rather than introduced, and the viewer can draw
    # that edge. Only 12.7% of source facts merge with anything an agent said,
    # so this is a floor: absence of an edge is not evidence of novelty.
    doc_of = {mid: m["provenance"].get("doc_id")
              for mid, m in store["mentions"].items()
              if m["provenance"]["channel"] == "source"}
    # Two different links from a document to a fact, and they mean different
    # things. `src` is a cluster that holds both a mention from Ex and an
    # output mention — the fact was taken from that document. `ev` is a fact
    # whose text names Ex — the agent is talking *about* the document. The
    # first is what we would like to measure and it only fires 4.4% of the
    # time; the second is lexical and fires far more often, and it is what
    # actually moves between agents.
    ev_re = re.compile(r"\bE([1-4])\b")
    src_of = {}
    ev_of = {}
    for fid, fact in store["facts"].items():
        named = sorted({"E" + m for m in ev_re.findall(fact["canonical_text"])})
        if named:
            ev_of[fid] = named
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

    verdicts = {}
    for slot, text in debate["transcript"].items():
        hit = VERDICT_RE.search(text)
        if hit:
            at = hit.start()
            verdicts[slot] = {"a": at, "b": len(text),
                              "v": hit.group(1).strip().upper()}

    return {
        "id": name,
        "model": debate.get("model", "?"),
        "claim_id": debate["claim_id"],
        "claim": debate["claim"],
        "topology": debate["topology"],
        "panel": debate["panel"],
        "order": order,
        "text": debate["transcript"],
        "spans": spans,
        "verdicts": verdicts,
        "facts": {fid: {"c": canonical[fid], "s": stance_of.get(fid),
                        **({"src": src_of[fid]} if fid in src_of else {}),
                        **({"ev": ev_of[fid]} if fid in ev_of else {})}
                  for fid in used},
        "roles": debate["roles"],
        "role_short": ROLE_SUMMARY.get(debate["panel"], {}),
        "delivery": {slot: info.get("peer_turns", [])
                     for slot, info in debate.get("delivery", {}).items()},
        # Round 1 was labelled "no input", which is false and hid the thing
        # that matters most for reading any of this: the dossier is in context
        # on every single turn, all three rounds. Nobody has to receive a
        # peer's message to restate E1.
        "sources": {slot: info.get("source_ids", [])
                    for slot, info in debate.get("delivery", {}).items()},
        "evidence": [{"id": e["id"], "stance": e["stance"], "text": e["text"]}
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
    ap.add_argument("--model", default="", help="留空 = 渲染全部模型")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    debates = []
    for directory in STORE_DIRS:
        pattern = f"*{args.model}*.v2.json" if args.model else "*.v2.json"
        for path in sorted(directory.glob(pattern)):
            view = build(path)
            if view:
                debates.append(view)

    total_spans = sum(len(s) for d in debates for s in d["spans"].values())
    total_unplaced = sum(d["unplaced"] for d in debates)
    payload = {"model": args.model or "all", "debates": debates,
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
