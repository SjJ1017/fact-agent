"""Command line entry point.

    factflow extract  trace.json -o mentions.json
    factflow match    mentions.json -o store.json [--store prior.json]
    factflow annotate store.json --props truth,critical --reference ref.txt -o store.json
    factflow stats    store.json
    factflow view     experiments/out -o explorer.html

The trace format is a JSON list of records:

    [{"text": "...", "provenance": {"execution_id": "run-0", "agent_id": "A",
      "round": 1, "channel": "output"}}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import TraceRecord, extract_trace
from .llm import LLM, LLMConfig
from .match import match
from .properties import ABSTRACTION, CRITICALITY, POLARITY_KIND, RELEVANCE, TRUTH, annotate_store
from .types import FactMention, FactStore
from .bench import run_bench
from .viewer import build as build_viewer

BUILTIN_PROPS = {
    "truth": TRUTH,
    "critical": CRITICALITY,
    "relevant": RELEVANCE,
    "abstraction": ABSTRACTION,
    "finding_kind": POLARITY_KIND,
}


def _llm(args) -> LLM:
    cfg = dict(max_concurrency=args.concurrency, cache_enabled=not args.no_cache)
    provider = getattr(args, "provider", "anthropic")
    if provider == "openai":
        return LLM.openai(args.model, **cfg)
    if provider == "deepseek":
        return LLM.deepseek(args.model, **cfg)
    return LLM(LLMConfig(model=args.model, **cfg))


def cmd_extract(args) -> int:
    records = [TraceRecord.model_validate(r) for r in json.loads(Path(args.trace).read_text())]
    mentions = extract_trace(_llm(args), records, focus=args.focus)
    Path(args.out).write_text(
        json.dumps([m.model_dump(mode="json") for m in mentions], indent=2, ensure_ascii=False)
    )
    print(f"{len(mentions)} mentions from {len(records)} records -> {args.out}")
    return 0


def cmd_match(args) -> int:
    mentions = [FactMention.model_validate(m) for m in json.loads(Path(args.mentions).read_text())]
    prior = FactStore.load(args.store) if args.store else None
    store = match(_llm(args), mentions, store=prior, threshold=args.threshold, top_k=args.top_k)
    store.save(args.out)
    print(f"{len(mentions)} mentions -> {len(store.facts)} canonical facts -> {args.out}")
    return 0


def cmd_annotate(args) -> int:
    store = FactStore.load(args.store_path)
    names = [n.strip() for n in args.props.split(",") if n.strip()]
    unknown = [n for n in names if n not in BUILTIN_PROPS]
    if unknown:
        print(f"unknown properties: {unknown}; available: {sorted(BUILTIN_PROPS)}", file=sys.stderr)
        return 2
    reference = Path(args.reference).read_text() if args.reference else None
    store = annotate_store(
        _llm(args),
        store,
        [BUILTIN_PROPS[n] for n in names],
        instruction=args.instruction,
        reference=reference,
    )
    store.save(args.out)
    print(f"annotated {len(store.facts)} facts with {names} -> {args.out}")
    return 0


def cmd_bench(args) -> int:
    llm = _llm(args)
    r = run_bench(llm)
    print(f"model      {r.model}")
    print(f"extraction {r.extraction_score:6.1%}  ({r.extraction_passed}/{r.extraction_total} checks)")
    print(f"relations  {r.relation_score:6.1%}  ({r.relation_correct}/{r.relation_total})")
    print(f"false EQUIVALENT (the costly error): {r.false_equivalent}")
    print(f"tokens {r.input_tokens} in / {r.output_tokens} out   {r.seconds:.0f}s")
    if r.failures:
        print(f"\n{len(r.failures)} failed checks:")
        for f in r.failures:
            print(f"   {f}")
    return 0


def cmd_view(args) -> int:
    out = build_viewer(args.store_dir, args.out, title=args.title)
    size = out.stat().st_size
    print(f"wrote {out} ({size/1024:.0f} KB) - open it in a browser")
    return 0


def cmd_stats(args) -> int:
    store = FactStore.load(args.store_path)
    print(f"executions       : {len(store.executions())}")
    print(f"agents           : {store.agents()}")
    print(f"rounds           : {store.rounds()}")
    print(f"mentions         : {len(store.mentions)}")
    print(f"canonical facts  : {len(store.facts)}")
    sizes = sorted((len(f.mention_ids) for f in store.facts.values()), reverse=True)
    if sizes:
        print(f"cluster sizes    : max={sizes[0]} median={sizes[len(sizes) // 2]} singletons={sizes.count(1)}")
    by_rel: dict[str, int] = {}
    for r in store.relations:
        by_rel[r.relation] = by_rel.get(r.relation, 0) + 1
    if by_rel:
        print(f"relations        : {by_rel}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="factflow", description=__doc__)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "deepseek"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--no-cache", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="text -> atomic facts")
    e.add_argument("trace")
    e.add_argument("-o", "--out", default="mentions.json")
    e.add_argument("--focus", default=None, help="task question, to steer phrasing")
    e.set_defaults(func=cmd_extract)

    m = sub.add_parser("match", help="mentions -> canonical facts")
    m.add_argument("mentions")
    m.add_argument("-o", "--out", default="store.json")
    m.add_argument("--store", default=None, help="prior store to match against (keeps ids stable)")
    m.add_argument("--threshold", type=float, default=0.50)
    m.add_argument("--top-k", type=int, default=12)
    m.set_defaults(func=cmd_match)

    a = sub.add_parser("annotate", help="add properties to canonical facts")
    a.add_argument("store_path")
    a.add_argument("--props", required=True, help=f"comma-separated: {','.join(sorted(BUILTIN_PROPS))}")
    a.add_argument("--reference", default=None, help="file with reference material")
    a.add_argument("--instruction", default=None, help="task context, e.g. the question")
    a.add_argument("-o", "--out", default="store.json")
    a.set_defaults(func=cmd_annotate)

    b = sub.add_parser("bench", help="score this model on the gold probe set")
    b.set_defaults(func=cmd_bench)

    v = sub.add_parser("view", help="build an interactive HTML explorer")
    v.add_argument("store_dir", help="directory holding *.store.json and *.debate.json")
    v.add_argument("-o", "--out", default="explorer.html")
    v.add_argument("--title", default="Fact Flow Explorer")
    v.set_defaults(func=cmd_view)

    s = sub.add_parser("stats", help="summarise a store")
    s.add_argument("store_path")
    s.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
