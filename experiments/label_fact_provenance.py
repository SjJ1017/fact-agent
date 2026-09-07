#!/usr/bin/env python3
"""Label each fact by which source paper it belongs to, and by discipline.

Written for IDRBench, where two papers from different fields are handed to a
panel that must integrate them.  Two questions per fact:

  paper      which of the two source papers this fact is about -- B, C, both,
             or neither.  "Neither" is the interesting class: it is the panel's
             own contribution, and it is most of the output.
  field      the discipline the fact speaks in.  A fact can belong to paper B
             and still be phrased in the vocabulary of C's field, which is what
             integration looks like from the inside.

The labeller is shown both papers' titles and abstracts because neither
question is answerable from the fact alone: "the cGAN predicts steady-state
fields" belongs to whichever paper contributed the cGAN.

Labels never touch the traces.  They are written to a sidecar keyed by
execution id, the way stance labels are, so a mislabelled run can be discarded
without regenerating anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.llm import LLM  # noqa: E402

SYSTEM = """\
You label atomic facts drawn from a panel discussion about integrating two
research papers.

For each fact give two labels.

`paper` — which source paper the fact is about:
  B        it states something the first paper contributes, describes or claims
  C        the same for the second paper
  BOTH     it states a relation between the two, or something both contain
  PANEL    neither paper says it: the panel's own proposal, plan, evaluation
           design, judgement about the papers, or statement about the task

`field` — the discipline the fact speaks in, as one lowercase phrase of one to
three words drawn from the fact itself (for example "cell biology",
"deep learning", "numerical methods", "clinical trials", "cryo-em",
"quantum algorithms", "discussion" for talk about the task rather than about
subject matter). Use the same phrase for facts in the same field; do not
invent a new phrasing for each one.

Judge only what the fact says. A fact mentioning a paper's method belongs to
that paper even when the sentence is a criticism of it. A proposal that
combines both methods is PANEL, not BOTH: BOTH is for describing the papers,
PANEL for what the discussion adds."""


class FactLabel(BaseModel):
    fact_id: str
    paper: Literal["B", "C", "BOTH", "PANEL"]
    field: str = Field(max_length=40)


class FactLabels(BaseModel):
    labels: list[FactLabel] = Field(default_factory=list)


def load_opencode_key() -> None:
    if os.environ.get("OPENCODE_API_KEY"):
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == "OPENCODE_API_KEY":
            os.environ["OPENCODE_API_KEY"] = value.strip().strip("'\"")
            return


def label_batch(llm: LLM, papers: str, facts: list[tuple[str, str]]) -> dict:
    entries = "\n".join(f"- id={fid}: {text}" for fid, text in facts)
    res = llm.parse(
        system=SYSTEM,
        user=f"<papers>\n{papers}\n</papers>\n\n<facts>\n{entries}\n</facts>",
        output_format=FactLabels,
        cache_if=lambda r: bool(r.labels),
    )
    want = {fid for fid, _ in facts}
    got = {l.fact_id: {"paper": l.paper, "field": l.field.strip().lower()}
           for l in res.labels if l.fact_id in want}
    missing = want - set(got)
    if missing:
        if len(facts) == 1:
            raise ValueError("response omitted its only fact id")
        look = dict(facts)
        for fid in sorted(missing):
            got.update(label_batch(llm, papers, [(fid, look[fid])]))
    return got


def label_store(llm: LLM, store_path: Path, debate_path: Path, out_dir: Path,
                batch_size: int, redo: bool) -> tuple[str, int]:
    name = debate_path.name.replace(".debate.json", "")
    out = out_dir / f"{name}.json"
    if out.exists() and not redo:
        return name, 0
    store = json.loads(store_path.read_text())
    debate = json.loads(debate_path.read_text())
    papers = "\n\n".join(
        f"[{e['id']}] {e.get('text','')[:1400]}" for e in debate.get("evidence", []))
    facts = [(fid, f["canonical_text"]) for fid, f in store["facts"].items()
             if f.get("canonical_text")]
    labels: dict = {}
    for i in range(0, len(facts), batch_size):
        labels.update(label_batch(llm, papers, facts[i:i + batch_size]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"execution_id": name, "model": llm.model, "n": len(labels),
         "labels": labels}, ensure_ascii=False, indent=1))
    return name, len(labels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "experiments" / "idrbench_generation_10x5_r3")
    ap.add_argument("--suffix", default=".nli.store.json")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "labels" / "provenance")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--parallel-stores", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    load_opencode_key()
    llm = LLM.opencode(a.model, max_concurrency=a.concurrency)
    stores = sorted(a.dir.glob(f"*{a.suffix}"))
    if a.limit:
        stores = stores[:a.limit]
    pairs = [(p, p.with_name(p.name.replace(a.suffix, ".debate.json")))
             for p in stores]
    pairs = [(s, d) for s, d in pairs if d.exists()]
    print(f"{len(pairs)} 个 store，模型 {a.model}")

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.parallel_stores) as pool:
        futs = {pool.submit(label_store, llm, s, d, a.out_dir,
                            a.batch_size, a.redo): d.name for s, d in pairs}
        for fut in as_completed(futs):
            name, n = fut.result()
            done += 1
            print(f"  [{done}/{len(pairs)}] {name[:52]} {n} 条"
                  f"{'（已有，跳过）' if n == 0 else ''}", flush=True)
    print(f"\n用时 {time.time() - t0:.0f}s   {llm.usage.report(a.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
