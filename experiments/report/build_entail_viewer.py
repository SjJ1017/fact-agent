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

    import json

    full = json.loads(a.data.read_text())
    debates = full["debates"]
    # One file per debate, fetched on demand.  Inlining all forty meant the
    # page parsed 9 MB of JSON before it could show the first one, which is
    # where the wait was -- drawing takes 5 ms and a hover half of that.
    side = a.out.with_suffix("")
    side = side.parent / (side.name + "-data")
    side.mkdir(parents=True, exist_ok=True)
    index = []
    for i, d in enumerate(debates):
        (side / f"{i}.json").write_text(
            json.dumps(d, ensure_ascii=False, separators=(",", ":")))
        index.append({"i": i, "id": d["id"],
                      "n_entail": len(d.get("entail", [])),
                      "n_degraded": sum(1 for e in d.get("entail", [])
                                        if e["kind"] == "degraded")})
    stub = {"index": index, "dir": side.name,
            "n_entail": full.get("n_entail"), "n_degraded": full.get("n_degraded")}
    html = a.template.read_text().replace(
        "__DATA__", json.dumps(stub, ensure_ascii=False, separators=(",", ":")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    big = max((side / f"{i}.json").stat().st_size for i in range(len(debates)))
    print(f"wrote {a.out}  ({a.out.stat().st_size / 2**10:.0f} KB)")
    print(f"      {len(debates)} 场分文件在 {side.name}/，最大 {big / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
