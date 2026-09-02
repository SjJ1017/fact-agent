"""Render the flow-profile report from findings/data/flow-profile.json.

Nothing here computes anything. `analyze_flow_profile.py` writes the numbers,
this fills them into a template so no figure or table cell is ever typed by
hand. Publish the output as an artifact.

    python experiments/analyze_flow_profile.py     # numbers
    python experiments/report/build_flow_profile.py  # page
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "findings" / "data" / "flow-profile.json"
FLOW = HERE.parent.parent / "findings" / "data" / "stance-flow.json"
EXTRA = HERE.parent.parent / "findings" / "data" / "flow-extras.json"

ROWS = [
    ("output_tokens",      "输出 token",      "{:.0f}"),
    ("n_facts",            "事实数",          "{:.1f}"),
    ("facts_per_k",        "事实密度 /kTok",  "{:.1f}"),
    ("novel_per_k",        "新事实密度",      "{:.1f}"),
    ("adopted_per_k",      "采纳密度",        "{:.1f}"),
    ("reception",          "接收率",          "{:.1%}"),
    ("adopted_share",      "采纳占比",        "{:.1%}"),
    ("adopted_share_recv", "条件采纳率",      "{:.1%}"),
    ("held_share",         "自持占比",        "{:.1%}"),
    ("nestedness",         "嵌套度",          "{:.3f}"),
    ("reach",              "事实覆盖",        "{:.2f}"),
    ("flow_gini",          "流量基尼",        "{:.3f}"),
    ("delivery_use",       "投递利用率",      "{:.0%}"),
    ("meta_share",         "元陈述占比",      "{:.0%}"),
    ("balance",            "正反平衡",        "{:.3f}"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--stance-flow", type=Path, default=FLOW)
    ap.add_argument("--extras", type=Path, default=EXTRA)
    ap.add_argument("--template", type=Path, default=HERE / "flow-profile.tpl.html")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "findings"
                            / "2026-09-01-flow-profile.html")
    args = ap.parse_args()

    with args.data.open() as fh:
        data = json.load(fh)
    with args.stance_flow.open() as fh:
        flow = json.load(fh)
    with args.extras.open() as fh:
        extras = json.load(fh)

    cell_names = [f"{topology}/{panel}"
                  for topology in data["topologies"]
                  for panel in ("neutral", "lenses", "stance")
                  if f"{topology}/{panel}" in data["cells"]]

    trs = []
    for key, label, fmt in ROWS:
        cells = []
        for i, name in enumerate(cell_names):
            value = data["cells"][name].get(key)
            text = "—" if value is None else fmt.format(value)
            # rule between topology blocks, wherever they start
            sep = i > 0 and name.split("/")[0] != cell_names[i - 1].split("/")[0]
            cells.append(('<td class="sep">' if sep else "<td>") + text + "</td>")
        trs.append(f'<tr><th scope="row">{label}</th>' + "".join(cells) + "</tr>")

    page = args.template.read_text()
    page = page.replace("/*__DATA__*/null",
                        json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("<!--__TABLE__-->", "\n".join(trs))
    page = page.replace("/*__FLOW__*/null",
                        json.dumps(flow, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("/*__EXTRA__*/null",
                        json.dumps(extras, ensure_ascii=False, separators=(",", ":")))
    args.out.write_text(page)
    print(f"wrote {args.out}  ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
