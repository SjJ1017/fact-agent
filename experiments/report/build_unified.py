"""Render the unified fact-flow report.

Three separate analyses had accumulated: the flow profile, the effective-
structure page, and a markdown note on transmission funnels. Each assumed the
reader had already read the others. This merges them into one article that
assumes nothing, and computes nothing of its own — every number is read out of
the four JSON files the analysis scripts write.

    python experiments/analyze_flow_profile.py --topologies full star chain
    python experiments/analyze_stance_flow.py
    python experiments/analyze_flow_extras.py
    python experiments/analyze_effective_structure.py
    python experiments/analyze_rq_extensions.py
    python experiments/report/build_unified.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrap import write_both

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "findings" / "data"

CELLS = [f"{t}/{p}" for t in ("full", "star", "chain")
         for p in ("neutral", "lenses", "stance")]

# (source, key, label, format, note shown in the definition list)
ROWS = [
    ("profile", "output_tokens",      "输出 token 数",   "{:.0f}"),
    ("profile", "n_facts",            "事实数",          "{:.1f}"),
    ("profile", "facts_per_k",        "事实密度",        "{:.1f}"),
    ("profile", "novel_per_k",        "新事实密度",      "{:.1f}"),
    ("profile", "reception",          "接触率",          "{:.1%}"),
    ("profile", "adopted_share",      "采纳占比",        "{:.1%}"),
    ("profile", "adopted_share_recv", "条件采纳率",      "{:.1%}"),
    ("profile", "held_share",         "自持占比",        "{:.1%}"),
    ("profile", "nestedness",         "嵌套度",          "{:.3f}"),
    ("profile", "reach",              "事实覆盖",        "{:.2f}"),
    ("profile", "flow_gini",          "流量集中度",      "{:.3f}"),
    ("profile", "meta_share",         "元陈述占比",      "{:.0%}"),
    ("profile", "balance",            "正反平衡",        "{:.3f}"),
    ("funnel",  "uptake",             "接触→采纳",       "{:.1%}"),
    ("funnel",  "post_adoption_retention", "采纳→留存", "{:.1%}"),
    ("funnel",  "end_to_end",         "接触→留存",       "{:.1%}"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--template", type=Path, default=HERE / "unified.tpl.html")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "findings"
                            / "2026-09-02-fact-flow-report.html")
    args = ap.parse_args()

    def load(name):
        with (args.data / name).open() as fh:
            return json.load(fh)

    profile = load("flow-profile.json")
    stance = load("stance-flow.json")
    extras = load("flow-extras.json")
    rq = load("rq-extensions.json")

    funnel = rq["round2_transmission_funnel"]["cells"]

    trs = []
    for source, key, label, fmt in ROWS:
        tds = []
        for i, cell in enumerate(CELLS):
            if source == "profile":
                value = profile["cells"][cell].get(key)
            else:
                # per-run mean, not the pooled ratio: the intervals in
                # rq-extensions describe the mean over runs, and quoting a
                # pooled number next to a per-run interval invites the reader
                # to check one against the other and find they disagree.
                value = funnel.get(cell, {}).get("per_run", {}).get(key, {}).get("mean")
            text = "—" if value is None else fmt.format(value)
            sep = i > 0 and cell.split("/")[0] != CELLS[i - 1].split("/")[0]
            tds.append(('<td class="sep">' if sep else "<td>") + text + "</td>")
        trs.append(f'<tr><th scope="row">{label}</th>' + "".join(tds) + "</tr>")

    page = args.template.read_text()
    for token, payload in (("PROFILE", profile), ("STANCE", stance),
                           ("EXTRA", extras), ("RQ", rq)):
        page = page.replace(f"/*__{token}__*/null",
                            json.dumps(payload, ensure_ascii=False,
                                       separators=(",", ":")))
    page = page.replace("<!--__TABLE__-->", "\n".join(trs))
    out, fragment = write_both(args.out, page)
    print(f"wrote {out}  ({len(page):,} bytes)\n      {fragment}  (发布用片段)")


if __name__ == "__main__":
    main()
