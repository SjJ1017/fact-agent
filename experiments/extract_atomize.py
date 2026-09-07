#!/usr/bin/env python3
"""Extract and atomise debates into matching inputs, and stop there.

`retrace.py` runs the old SAME/DIFF matcher inline before saving, which is
both the slow part and work we throw away: matching now happens on the GPU box
with the NLI judge. This runs the same `extract_facts` and `atomize` on the
same slots, then writes the `.atomized.json` that `match_traces.py` consumes,
so the mentions are identical to what retrace would have produced while the
matching step is left to the judge that will actually be used.

Both fan-outs are widened: slots within a debate go through `llm.map`, and
several debates run at once, since one debate has about eleven slots and
cannot keep a large concurrency busy on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from factflow.atomize import atomize  # noqa: E402
from factflow.extract import extract_facts  # noqa: E402
from factflow.llm import LLM  # noqa: E402
from retrace import slots  # noqa: E402


def one(llm, path: Path, suffix: str, no_atomize: bool) -> tuple[str, int, int, float]:
    debate = json.loads(path.read_text())
    label = debate["execution_id"]
    t0 = time.time()

    def _extract(piece):
        text, prov = piece
        try:
            return extract_facts(llm, text, prov)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    抽取失败 {prov.doc_id or prov.agent_id}|{prov.round}: "
                  f"{type(exc).__name__}", flush=True)
            return []

    mentions = [m for g in llm.map(_extract, slots(debate)) for m in g]
    raw = len(mentions)
    if not raw:
        return label, 0, 0, time.time() - t0
    if not no_atomize:
        mentions = atomize(llm, mentions, batch_size=20)

    out = path.with_name(path.name.replace(".debate.json", "") + suffix)
    out.write_text(json.dumps(
        {"mentions": {m.mention_id: json.loads(m.model_dump_json()) for m in mentions},
         "facts": {}, "mention_to_fact": {}, "relations": [],
         "derived_from": {"file": path.name, "stage": "extract+atomize",
                          "model": llm.model, "atomized": not no_atomize}},
        ensure_ascii=False), encoding="utf-8")
    return label, raw, len(mentions), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*.debate.json")
    ap.add_argument("--model", default="minimax-m2.5")
    ap.add_argument("--suffix", default=".atomized.json")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--parallel-debates", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--no-atomize", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    paths = sorted(a.out_dir.glob(a.glob))
    if not a.redo:
        paths = [p for p in paths
                 if not p.with_name(p.name.replace(".debate.json", "")
                                    + a.suffix).exists()]
    if not paths:
        print("没有待处理的 debate")
        return 0

    llm = LLM.opencode(a.model, max_concurrency=a.concurrency, max_tokens=a.max_tokens)
    llm.backend.client = llm.backend.client.with_options(timeout=a.timeout, max_retries=1)
    print(f"{len(paths)} 场，模型 {a.model}，并发 {a.concurrency}，"
          f"同时处理 {a.parallel_debates} 场", flush=True)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.parallel_debates) as pool:
        for label, raw, split, dt in pool.map(
                lambda p: one(llm, p, a.suffix, a.no_atomize), paths):
            done += 1
            tag = "无事实，跳过" if not raw else f"抽取 {raw} → 原子化 {split}"
            print(f"[{done}/{len(paths)}] {label[-46:]}  {tag}  {dt:.0f}s", flush=True)
    print(f"\n用时 {time.time()-t0:.0f}s   {llm.usage.report(a.model)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
