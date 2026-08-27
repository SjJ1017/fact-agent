"""Score several models on the probe set and report accuracy against cost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from factflow import LLM
from factflow.bench import run_bench

# USD per 1M tokens. Edit to match your account; token counts are measured, so
# only this table needs changing if prices move.
PRICES = {
    "gpt-5.5":       (1.25, 10.00),
    "gpt-5.4":       (1.25, 10.00),
    "gpt-5.4-mini":  (0.25,  2.00),
    "gpt-5.4-nano":  (0.05,  0.40),
    "gpt-4.1":       (2.00,  8.00),
    "gpt-4.1-mini":  (0.40,  1.60),
    "gpt-4.1-nano":  (0.10,  0.40),
    "deepseek-chat": (0.27,  1.10),
}

DEFAULT = ["gpt-5.4-nano", "gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]


def cost(model: str, i: int, o: int) -> float | None:
    p = PRICES.get(model)
    return None if p is None else i / 1e6 * p[0] + o / 1e6 * p[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT))
    ap.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    ap.add_argument("-o", "--out", default="experiments/out/bench.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model} ===", flush=True)
        mk = LLM.deepseek if args.provider == "deepseek" else LLM.openai
        try:
            llm = mk(model, max_concurrency=6, cache_enabled=not args.no_cache)
            r = run_bench(llm)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        c = cost(model, r.input_tokens, r.output_tokens)
        print(f"  extraction {r.extraction_score:6.1%}  ({r.extraction_passed}/{r.extraction_total} checks)")
        print(f"  relations  {r.relation_score:6.1%}  ({r.relation_correct}/{r.relation_total})"
              f"   false-EQUIVALENT: {r.false_equivalent}")
        print(f"  tokens {r.input_tokens}/{r.output_tokens}"
              + (f"   ~${c:.4f} per bench run" if c is not None else "")
              + f"   {r.seconds:.0f}s")
        if r.failures:
            print(f"  {len(r.failures)} failed checks, first few:")
            for f in r.failures[:4]:
                print(f"     {f[:150]}")
        rows.append({
            "model": model, "extraction": r.extraction_score, "relations": r.relation_score,
            "extraction_passed": r.extraction_passed, "extraction_total": r.extraction_total,
            "relation_correct": r.relation_correct, "relation_total": r.relation_total,
            "false_equivalent": r.false_equivalent,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "usd": c, "seconds": r.seconds,
            "failures": r.failures, "per_relation": r.per_relation,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n{'='*92}")
    print(f"{'model':<16}{'extract':>9}{'relations':>11}{'falseEQ':>9}{'tokens':>14}{'USD':>10}{'sec':>7}")
    for r in rows:
        tok = f"{r['input_tokens']}/{r['output_tokens']}"
        usd = f"${r['usd']:.4f}" if r["usd"] is not None else "-"
        print(f"{r['model']:<16}{r['extraction']:>8.1%}{r['relations']:>11.1%}"
              f"{r['false_equivalent']:>9}{tok:>14}{usd:>10}{r['seconds']:>7.0f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
