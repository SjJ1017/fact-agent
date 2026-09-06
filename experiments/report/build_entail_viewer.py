"""Render the entailment trace view as one standalone HTML file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "findings" / "data" / "entail-view.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--template", type=Path, default=HERE / "entail-viewer.tpl.html")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "findings" / "entail-traces.html")
    a = ap.parse_args()

    payload = a.data.read_text()
    html = a.template.read_text().replace("__DATA__", payload)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    print(f"wrote {a.out}  ({a.out.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
