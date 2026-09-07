#!/usr/bin/env python3
"""Extract and atomize Avalon traces without running any fact matching."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from factflow.atomize import atomize  # noqa: E402
from factflow.extract import extract_facts  # noqa: E402
from factflow.llm import LLM  # noqa: E402
from factflow.types import Channel, Provenance  # noqa: E402
from run_perspectrum import load_opencode_key  # noqa: E402


FOCUS = ("Claims, observations, decisions, and beliefs about player identities, "
         "trust, proposed teams, votes, quest outcomes, and strategic deductions "
         "in this five-player Avalon game")
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def slots(trace: dict[str, Any]) -> list[tuple[str, Provenance]]:
    execution = trace["execution_id"]
    records = [(
        trace["rules"]["rules_text"],
        Provenance(execution_id=execution, round=0, channel=Channel.SOURCE,
                   doc_id="game-rules", extra={"visibility": "public"}),
    )]
    for agent, text in trace["private_observations"].items():
        records.append((
            text,
            Provenance(execution_id=execution, agent_id=agent, round=0,
                       channel=Channel.CONTEXT, doc_id=f"{agent}-role-card",
                       extra={"visibility": "private", "phase": "initial-role"}),
        ))
    for event in trace["events"]:
        extra = {
            "event_id": event["event_id"],
            "quest": event["quest"],
            "proposal": event["proposal"],
            "phase": event["phase"],
            "visibility": event["visibility"],
            "recipients": [f"P{x}" for x in event["recipients"]],
            "action": event.get("action"),
        }
        if event["kind"] == "agent":
            agent = f"P{event['actor']}"
            provenance = Provenance(
                execution_id=execution, agent_id=agent,
                round=int(event["event_id"]), channel=Channel.OUTPUT,
                doc_id=f"event-{event['event_id']}", extra=extra,
            )
        else:
            provenance = Provenance(
                execution_id=execution, agent_id=None,
                round=int(event["event_id"]), channel=Channel.CONTEXT,
                doc_id=f"event-{event['event_id']}", extra=extra,
            )
        records.append((event["extraction_text"], provenance))
    return records


def process_one(llm: LLM, path: Path, redo: bool) -> dict[str, Any]:
    target = path.with_name(path.name.replace(".trace.json", ".atomized.json"))
    if target.exists() and not redo:
        return {"trace": path.name, "status": "skipped", "target": target.name}
    trace = json.loads(path.read_text())
    failures: list[dict[str, Any]] = []
    started = time.time()

    def extract_one(piece: tuple[str, Provenance]):
        text, provenance = piece
        try:
            return extract_facts(
                llm, text, provenance, focus=FOCUS, include_discourse=True, attempts=3)
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "doc_id": provenance.doc_id,
                "agent_id": provenance.agent_id,
                "round": provenance.round,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return []

    mentions = [mention for batch in llm.map(extract_one, slots(trace))
                for mention in batch]
    raw_count = len(mentions)
    if failures:
        failure_path = target.with_suffix(".failures.json")
        write_json_atomic(failure_path, {"trace": path.name, "failures": failures})
        return {"trace": path.name, "status": "failed", "failures": len(failures),
                "raw_mentions": raw_count, "elapsed_seconds": time.time() - started}

    mentions = atomize(
        llm, mentions, batch_size=20, prefilter=True, progress=trace["execution_id"])
    payload = {
        "mentions": {
            mention.mention_id: json.loads(mention.model_dump_json())
            for mention in mentions
        },
        "facts": {},
        "mention_to_fact": {},
        "relations": [],
        "derived_from": {
            "file": path.name,
            "stage": "extract+atomize",
            "model": llm.model,
            "atomized": True,
            "matched": False,
            "focus": FOCUS,
        },
    }
    write_json_atomic(target, payload)
    return {"trace": path.name, "status": "complete", "target": target.name,
            "raw_mentions": raw_count, "atomized_mentions": len(mentions),
            "elapsed_seconds": time.time() - started}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--parallel-games", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    load_opencode_key()
    paths = sorted(args.outdir.glob("avalon-5p-*.trace.json"))
    if not paths:
        raise SystemExit(f"no Avalon trace files under {args.outdir}")
    llm = LLM.opencode(args.model, max_concurrency=args.concurrency,
                       cache_dir=str(args.outdir / ".extract-cache"))
    llm.backend.client = llm.backend.client.with_options(
        timeout=args.timeout, max_retries=2)
    log(f"extract+atomize: {len(paths)} traces; model {args.model}; "
        f"parallel games {args.parallel_games}; NO MATCHING")
    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel_games) as pool:
        futures = {pool.submit(process_one, llm, path, args.redo): path for path in paths}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            detail = (f"{result.get('raw_mentions', 0)} raw -> "
                      f"{result.get('atomized_mentions', 0)} atomized")
            log(f"[{completed}/{len(paths)}] {result['trace']}: "
                f"{result['status']} {detail}")

    summary = {
        "model": args.model,
        "matched": False,
        "results": sorted(results, key=lambda x: x["trace"]),
        "accounting": {
            "calls": llm.usage.calls,
            "input_tokens": llm.usage.input_tokens,
            "output_tokens": llm.usage.output_tokens,
            "cost_usd": llm.usage.cost(args.model),
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }
    write_json_atomic(args.outdir / "extraction-manifest.json", summary)
    failed = sum(result["status"] == "failed" for result in results)
    log(f"done: {len(results) - failed}/{len(results)} complete; {failed} failed; "
        f"{llm.usage.report(args.model)}; NO MATCHING")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
