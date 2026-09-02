"""Render the effective-structure report from findings/data/effective-structure.json.

Nothing here computes anything. `analyze_effective_structure.py` writes the
numbers; this fills them into the template so no figure or table cell is typed
by hand. Publish the output as an artifact.

    python experiments/analyze_effective_structure.py            # numbers
    python experiments/report/build_effective_structure.py       # page
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "findings" / "data" / "effective-structure.json"

CELLS = ["full/neutral", "full/lenses", "full/stance",
         "star/neutral", "star/lenses", "star/stance"]
ROWS = [
    ("n_facts",        "事实数",           "{:.1f}"),
    ("novel_share",    "新事实占比",       "{:.1%}"),
    ("adopted_share",  "采纳占比",         "{:.1%}"),
    ("held_share",     "自持占比",         "{:.1%}"),
    ("reach",          "事实覆盖",         "{:.2f}"),
    ("nestedness",     "嵌套度",           "{:.3f}"),
    ("n_eff",          "有效独立数 n_eff", "{:.2f}"),
    ("echo",           "唯一来源回声率",   "{:.2f}"),
    ("r1_surv",        "r1→r3 存活",       "{:.1%}"),
    ("own_peak_attr",  "峰值保留",         "{:.1%}"),
    ("retain_adopted", "采纳保留收益",     "{:.1%}"),
    ("final_first_round|1", "终局·首现于r1", "{:.0%}"),
    ("final_first_round|3", "终局·首现于r3", "{:.0%}"),
    ("verdict|unanimous", "终局三人一致",  "{:.0%}"),
    ("verdict|r1same",     "r1多数=终局多数", "{:.0%}"),
]


def lookup(cell_row: dict, key: str):
    cur = cell_row
    for part in key.split("|"):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--template", type=Path, default=HERE / "effective-structure.tpl.html")
    ap.add_argument("--out", type=Path, default=HERE / "effective-structure.html")
    args = ap.parse_args()

    data = json.loads(args.data.read_text())

    trs = []
    for key, label, fmt in ROWS:
        cell_td = []
        for i, name in enumerate(CELLS):
            row = data["cells"][name]
            value = lookup(row, key)
            if value is None:
                text = "—"
            elif key.startswith("verdict|"):
                text = fmt.format(value / (row["verdict"]["runs"] or 1))
            else:
                text = fmt.format(value)
            cell_td.append(('<td class="sep">' if i == 3 else "<td>") + text + "</td>")
        trs.append(f'<tr><th scope="row">{label}</th>' + "".join(cell_td) + "</tr>")

    page = args.template.read_text()
    page = page.replace("/*__DATA__*/null",
                        json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("<!--__TABLE__-->", "\n".join(trs))
    args.out.write_text(page)
    print(f"wrote {args.out}  ({len(page):,} bytes)")


if __name__ == "__main__":
    main()