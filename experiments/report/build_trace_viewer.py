"""Render the trace viewer: one page, all 108 debates, no server.

    python experiments/build_trace_view.py          # extract the view data
    python experiments/report/build_trace_viewer.py # render the page
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrap import write_both

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "findings" / "data" / "trace-view.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--template", type=Path, default=HERE / "trace-viewer.tpl.html")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "findings" / "trace-viewer.html")
    args = ap.parse_args()

    with args.data.open() as fh:
        payload = json.load(fh)
    page = args.template.read_text().replace(
        "/*__TRACE__*/null",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out, fragment = write_both(args.out, page)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)\n      {fragment}")


if __name__ == "__main__":
    main()
