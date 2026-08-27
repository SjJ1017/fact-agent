"""Sweep topology x roles over MedQA, tracing facts through each configuration."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from frameworks import ROLES, run_framework, save, to_trace

from factflow import LLM, TraceRecord, extract_trace, match

OUT = Path(__file__).parent / "medqa_out"
LETTERS = "ABCDEFGHIJ"


def load_rows(dataset: str, n: int, categories: str) -> list[dict]:
    """Normalise both datasets to {question, options: {letter: text}, answer_idx}."""
    if dataset == "medqa":
            return [dict(r, id=f"medqa-{i}") for i, r in zip(range(n), ds)]

    wanted = {c.strip().lower() for c in categories.split(",") if c.strip()}
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
    rows = []
    for r in ds:
        if wanted and r["category"].lower() not in wanted:
            continue
        opts = {LETTERS[i]: o for i, o in enumerate(r["options"]) if o and o != "N/A"}
        rows.append({
            "id": f"mmlu-{r['question_id']}",
            "question": r["question"],
            "options": opts,
            "answer_idx": r["answer"],
            "category": r["category"],
        })
        if len(rows) >= n:
            break
    return rows
PANELS = {
    "generalist": None,
    # Domain roles: differ in what they know.
    "specialists": ["internist", "pharmacologist", "pathophysiologist"],
    "sciexperts": ["theorist", "calculator", "skeptic"],
    # Functional roles: differ in what they DO to facts. The pipeline shape most
    # multi-agent frameworks actually ship.
    "pipeline": ["decomposer", "analyzer", "summarizer"],
    "critique": ["analyzer", "critic", "verifier"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-questions", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--provider", default="opencode")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--configs", default="full/generalist,chain/generalist,star/generalist,full/specialists")
    ap.add_argument("--trace", action="store_true", help="also extract+match facts (slow)")
    ap.add_argument("--parallel", type=int, default=6, help="questions in flight at once")
    ap.add_argument("--screen", type=int, default=0,
                    help="screen this many questions single-agent first and keep only the "
                         "failures (plus an equal control sample of successes)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request seconds; a wedged gateway call otherwise stalls a sweep")
    ap.add_argument("--dataset", default="medqa", choices=["medqa", "mmlu-pro"])
    ap.add_argument("--outdir", default=None, help="override the output directory name")
    ap.add_argument("--categories", default="physics,chemistry,biology,health",
                    help="mmlu-pro only: comma-separated categories to sample from")
    args = ap.parse_args()

    global OUT
    OUT = Path(__file__).parent / (args.outdir or f"{args.dataset}_out")
    OUT.mkdir(exist_ok=True)
    mk = {"opencode": LLM.opencode, "openai": LLM.openai}[args.provider]
    llm = mk(args.model, max_concurrency=args.concurrency)
    llm.backend.client = llm.backend.client.with_options(timeout=args.timeout, max_retries=1)

    rows = load_rows(args.dataset, args.screen or args.n_questions, args.categories)

    if args.screen:
        # Every benchmark tried so far is saturated: the single agent already
        # scores 87-92%, so a committee has almost nothing to add and the agents
        # simply agree. Screening isolates the questions where collaboration
        # could matter at all, and keeps a matched control of questions it got
        # right so the comparison is not purely a hard-case sample.
        print(f"=== screening {len(rows)} questions single-agent ===", flush=True)

        def screen_one(row):
            try:
                return row, run_framework(llm, row, topology="full", n_agents=1, n_rounds=1).correct
            except Exception:  # noqa: BLE001
                return row, None

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            screened = list(pool.map(screen_one, rows))
        failed = [r for r, ok in screened if ok is False]
        passed = [r for r, ok in screened if ok is True]
        keep_control = min(len(failed), args.n_questions // 2)
        rows = failed[: args.n_questions] + passed[:keep_control]
        print(f"  single-agent failed {len(failed)}/{len(screened)} "
              f"-> testing {len(failed[:args.n_questions])} failures "
              f"+ {keep_control} controls\n", flush=True)
        for r in rows:
            r["_screened_correct"] = r in passed

    configs = [c.strip().split("/") for c in args.configs.split(",") if c.strip()]
    summary = []

    # Single-agent baseline: the number multi-agent has to beat.
    # Questions are independent; running them serially made a sweep take an hour
    # of wall clock on a slow gateway for no reason.
    # One flaky gateway call must not void a whole sweep, but a question that
    # never answered must not silently count as wrong either - it is dropped from
    # the denominator and reported.
    def solo_one(row):
        try:
            return run_framework(llm, row, topology="full", n_agents=1, n_rounds=1).correct
        except Exception as exc:  # noqa: BLE001
            print(f"  {row['id']}: dropped ({type(exc).__name__})", flush=True)
            return None

    print("=== single agent (baseline) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        got = [r for r in pool.map(solo_one, rows) if r is not None]
    solo, n_solo = sum(got), len(got)
    print(f"  {solo}/{n_solo} = {solo/max(n_solo,1):.0%}"
          + (f"   ({len(rows)-n_solo} dropped)" if n_solo < len(rows) else "") + "\n", flush=True)
    summary.append({"config": "single-agent", "correct": solo, "n": n_solo,
                    "accuracy": solo / max(n_solo, 1)})

    for topo, panel in configs:
        label = f"{topo}/{panel}"
        print(f"=== {label} ===", flush=True)
        def one_question(row):
            exec_id = f"{row['id']}-{topo}-{panel}"
            try:
                r = run_framework(llm, row, topology=topo, role_names=PANELS[panel],
                                  n_agents=3, n_rounds=args.rounds)
            except Exception as exc:  # noqa: BLE001
                print(f"  {row['id']}: SKIPPED {type(exc).__name__}: {str(exc)[:110]}", flush=True)
                return None
            save(r, OUT / f"{exec_id}.debate.json")
            if args.trace:
                recs = [TraceRecord.model_validate(x) for x in to_trace(r, exec_id)]
                mentions = extract_trace(llm, recs, focus=row["question"][:200],
                                         include_discourse=False)
                match(llm, mentions).save(str(OUT / f"{exec_id}.store.json"))
            return r

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = [r for r in pool.map(one_question, rows) if r is not None]
        correct = sum(r.correct for r in results)
        unanimous = sum(len(set(r.final.values())) == 1 for r in results)
        n = len(results)
        dropped = len(rows) - n
        print(f"  {correct}/{n} = {correct/max(n,1):.0%}   unanimous {unanimous}/{max(n,1)}"
              + (f"   ({dropped} dropped)" if dropped else "") + "\n", flush=True)
        summary.append({"config": label, "topology": topo, "panel": panel,
                        "correct": correct, "n": n, "accuracy": correct / max(n, 1),
                        "unanimous": unanimous, "dropped": dropped})

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 66)
    print(f"{'config':<24}{'accuracy':>10}{'unanimous':>12}")
    for s in summary:
        u = f"{s['unanimous']}/{s['n']}" if "unanimous" in s else "-"
        print(f"{s['config']:<24}{s['accuracy']:>9.0%}{u:>12}")
    if llm.failures:
        print(f"\n{len(llm.failures)} LLM failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
