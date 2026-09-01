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

CELLS = ["full/neutral", "full/lenses", "full/stance",
         "star/neutral", "star/lenses", "star/stance"]
ROWS = [
    ("n_facts",       "事实数",     "{:.1f}"),
    ("novel",         "新事实",     "{:.1f}"),
    ("adopted_share", "采纳占比",   "{:.1%}"),
    ("held_share",    "自持占比",   "{:.1%}"),
    ("nestedness",    "嵌套度",     "{:.3f}"),
    ("reach",         "事实覆盖",   "{:.2f}"),
    ("flow_gini",     "流量基尼",   "{:.3f}"),
    ("delivery_use",  "投递利用率", "{:.0%}"),
    ("meta_share",    "元陈述占比", "{:.0%}"),
    ("balance",       "正反平衡",   "{:.3f}"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--template", type=Path, default=HERE / "flow-profile.tpl.html")
    ap.add_argument("--out", type=Path, default=HERE / "flow-profile.html")
    args = ap.parse_args()

    with args.data.open() as fh:
        data = json.load(fh)

    trs = []
    for key, label, fmt in ROWS:
        cells = []
        for i, name in enumerate(CELLS):
            value = data["cells"][name].get(key)
            text = "—" if value is None else fmt.format(value)
            cells.append(('<td class="sep">' if i == 3 else "<td>") + text + "</td>")
        trs.append(f'<tr><th scope="row">{label}</th>' + "".join(cells) + "</tr>")

    page = args.template.read_text()
    page = page.replace("/*__DATA__*/null",
                        json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("<!--__TABLE__-->", "\n".join(trs))
    args.out.write_text(page)
    print(f"wrote {args.out}  ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
