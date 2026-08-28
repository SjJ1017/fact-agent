"""Inspect MAST-Data: what is in it, and is it usable for fact-lifecycle analysis.

MAST (Cemri et al., arXiv:2503.13657) released 1642 multi-agent execution traces
with binary human annotations for 14 failure modes, inter-annotator kappa 0.88.
It is the only public corpus that labels the failure modes a fact-lifecycle
decomposition claims to separate, so it is the cheapest possible validation
target: no new data collection, and someone else already paid for the labels.

The paper states the motivating problem directly:

    "Diagnosing FC2 failures can be complex, as similar surface behaviors
     (e.g., missing information) can stem from different root causes like
     withholding (FM-2.4), ignoring input (FM-2.5), or context mismanagement
     (FM-1.4)"

Those three are exactly what InContext/InOutput separates:

    FM-1.4 context loss     never in the agent's context      -> not in output
    FM-2.4 withholding      in its context, from its own work -> not in output
    FM-2.5 ignoring input   in its context, from a PEER       -> not in output

This script answers the prerequisite question: are the traces structured enough
to reconstruct who held which fact when.

    python experiments/mast/inspect.py --download
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
FULL = HERE / "MAD_full_dataset.json"

# The three modes the paper calls hard to tell apart.
TARGET = {"1.4": "context loss", "2.4": "withholding", "2.5": "ignoring input"}
NAMES = {
    "1.1": "disobey task spec", "1.2": "disobey role spec", "1.3": "step repetition",
    "1.4": "context loss", "1.5": "unaware of completion",
    "2.1": "conversation reset", "2.2": "wrong assumption", "2.3": "task derailment",
    "2.4": "withholding info", "2.5": "ignoring input", "2.6": "reasoning-action mismatch",
    "3.1": "premature termination", "3.2": "no/incomplete verification", "3.3": "incorrect verification",
}
# Per-framework speaker patterns. Each framework logs in its own format; there is
# no shared message schema, which is the main cost of using this corpus.
SPEAKER = {
    "AG2": r"^\s*name:\s*(\S+)",
    "MetaGPT": r"FROM:\s*(\S+)\s+TO:|^(\w[\w ]*?):\s*$",
    "ChatDev": r"\*\*\[(\w[^\]]*)\]\*\*",
    "Magentic": r"-{6,}\s*(\w+)\s*-{6,}",
    "OpenManus": r"\|\s*(\w+)\s*\|",
    "AppWorld": r"^\s*(\w+):\s",
    "HyperAgent": r"HyperAgent[_-](\w+)",
}


def download() -> None:
    from huggingface_hub import hf_hub_download

    for f in ("MAD_full_dataset.json", "MAD_human_labelled_dataset.json"):
        hf_hub_download("mcemri/MAST-Data", f, repo_type="dataset", local_dir=str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download or not FULL.exists():
        download()

    d = json.loads(FULL.read_text())
    print(f"{len(d)} traces\n")

    print("=== framework x benchmark ===")
    for (m, b), n in Counter((r["mas_name"], r["benchmark_name"]) for r in d).most_common():
        print(f"   {m:<12}{b:<20}{n:>5}")

    print("\n=== label distribution (share of traces carrying the mode) ===")
    cnt = Counter()
    per = []
    for r in d:
        on = [k for k, v in r["mast_annotation"].items() if v]
        per.append(len(on))
        cnt.update(on)
    print(f"   mean {sum(per)/len(per):.2f} modes per trace; {per.count(0)} traces carry none")
    for k in sorted(NAMES):
        star = "  <- hard to tell apart" if k in TARGET else ""
        print(f"   FM-{k}  {NAMES[k]:<26}{cnt[k]:>5}  ({cnt[k]/len(d):>5.1%}){star}")

    print("\n=== is a trace parseable into per-agent turns? ===")
    print(f"   {'framework':<12}{'traces':>7}{'median chars':>14}{'distinct speakers':>19}")
    for mas in SPEAKER:
        rs = [r for r in d if r["mas_name"] == mas]
        if not rs:
            continue
        lens = sorted(len(r["trace"]["trajectory"]) for r in rs)
        pat = re.compile(SPEAKER[mas], re.M)
        spk = []
        for r in rs[:40]:
            hits = {g for m in pat.finditer(r["trace"]["trajectory"]) for g in m.groups() if g}
            spk.append(len(hits))
        print(f"   {mas:<12}{len(rs):>7}{lens[len(lens)//2]:>14,}{sum(spk)/len(spk):>19.1f}")

    print("\n=== feasibility for the three target modes ===")
    for k, name in TARGET.items():
        pos = cnt[k]
        print(f"   FM-{k} {name:<16} {pos:>5} positives"
              + ("   too rare to evaluate alone" if pos < 50 else "   workable"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
