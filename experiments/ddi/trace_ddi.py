"""Extract and match facts over completed DDI runs.

Separate from the runner because tracing costs far more than the debate it
traces: roughly 20 extraction calls plus 100+ adjudication calls per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from factflow import LLM, TraceRecord, extract_trace, match

OUT = Path(__file__).parent / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--provider", default="opencode")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dir)
    mk = {"opencode": LLM.opencode, "openai": LLM.openai, "deepseek": LLM.deepseek}[args.provider]
    llm = mk(args.model, max_concurrency=args.concurrency)
    llm.backend.client = llm.backend.client.with_options(timeout=args.timeout, max_retries=1)

    todo = sorted(root.glob("*.trace.json"))
    if args.limit:
        todo = todo[: args.limit]
    print(f"tracing {len(todo)} runs with {args.model}\n", flush=True)

    def one(tp: Path):
        exec_id = tp.name.replace(".trace.json", "")
        out = root / f"{exec_id}.store.json"
        if out.exists():
            return exec_id, True
        recs = [TraceRecord.model_validate(r) for r in json.loads(tp.read_text())]
        try:
            mentions = extract_trace(llm, recs, include_discourse=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  {exec_id}: FAILED {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            return exec_id, False
        if not any(m.provenance.channel.value == "output" for m in mentions):
            # An exhausted quota yields an empty but plausible-looking store that
            # would report zero joins for every condition. Refuse to write it.
            print(f"  {exec_id}: NO AGENT MENTIONS - not saving", flush=True)
            return exec_id, False
        match(llm, mentions).save(str(out))
        print(f"  {exec_id}: {len(mentions)} mentions", flush=True)
        return exec_id, True

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(one, todo))

    ok = sum(1 for _, good in results if good)
    print(f"\n{ok}/{len(results)} traced")
    if llm.failures:
        print(f"WARNING: {len(llm.failures)} LLM calls failed; first: {llm.failures[0][:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
