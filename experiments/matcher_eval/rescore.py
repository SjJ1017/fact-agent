"""Re-score saved runs against the gold as it stands now, calling no model.

Result files keep every pair's id, gold and prediction, so a label correction
only needs the scores recomputed.  Prints the same table run.sh prints at the
end, so it drops straight in where those numbers were.

What it cannot do is refit a threshold: raw margins are not stored, so a
scoring model is judged at the threshold it chose under the old labels, which
makes it slightly conservative.  oneword / cot / entail have no threshold and
are exact.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = {
    "strict": {"equivalent"},
    "near": {"equivalent", "near_match"},
    "entail": {"equivalent", "near_match", "entail_ab", "entail_ba"},
}


def gold_map(path: Path, policy: str) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if policy == "file":
            out[r["id"]] = r["gold"]
        else:
            rel = r.get("relation") or (
                "equivalent" if r["gold"] == "SAME" else "different")
            out[r["id"]] = "SAME" if rel in POLICY[policy] else "DIFFERENT"
    return out


def score(rows) -> dict:
    tp = sum(1 for g, p in rows if g == "SAME" and p == "SAME")
    fp = sum(1 for g, p in rows if g != "SAME" and p == "SAME")
    fn = sum(1 for g, p in rows if g == "SAME" and p != "SAME")
    tn = len(rows) - tp - fp - fn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(n=len(rows), acc=(tp + tn) / len(rows) if rows else 0.0,
                prec=pr, rec=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--policy", default="auto",
                    choices=["auto", "file", "strict", "near", "entail"],
                    help="auto(默认)=按每次运行当时用的口径比，差异只来自标注"
                         "修正。指定一个口径则是换口径重算，那会连没修正过的"
                         "标签一起动")
    ap.add_argument("--set", dest="which", default="all",
                    choices=["all", "main", "idrbench"],
                    help="只看某个 oracle 的结果")
    ap.add_argument("--diff", action="store_true", help="额外列出新旧对比")
    a = ap.parse_args()

    files = {n: HERE / n for n in ("pairs.jsonl", "pairs_idrbench.jsonl")
             if (HERE / n).exists()}
    cache: dict[tuple[str, str], dict[str, str]] = {}

    def golds_for(which: str, policy: str) -> dict[str, str]:
        key = (which, policy)
        if key not in cache:
            cache[key] = gold_map(files[which], policy)
        return cache[key]

    def run_policy(d, name) -> str:
        if d.get("policy"):
            return d["policy"]
        m = re.match(r".+\.(keep|drop|only)\.(.+)\.json$", name)
        tail = (m.group(2).rsplit(".", 1)[-1] if m else "")
        return tail if tail in ("strict", "near", "entail") else "file"

    rows, flips = [], {}
    for f in sorted(a.results.glob("*.json")):
        if f.name == "summary.json":
            continue
        d = json.loads(f.read_text())
        preds = d.get("predictions")
        if not preds:
            continue
        idr = preds[0]["id"].startswith("idr-")
        if a.which == "main" and idr:
            continue
        if a.which == "idrbench" and not idr:
            continue
        which = "pairs_idrbench.jsonl" if idr else "pairs.jsonl"
        if which not in files:
            continue
        pol = run_policy(d, f.name) if a.policy == "auto" else a.policy
        gm = golds_for(which, pol)

        new = [(gm[p["id"]], p["pred"]) for p in preds if p["id"] in gm]
        old = [(p["gold"], p["pred"]) for p in preds if p["id"] in gm]
        if not new:
            continue
        for p in preds:
            if p["id"] in gm and gm[p["id"]] != p["gold"]:
                flips[p["id"]] = (p["gold"], gm[p["id"]],
                                  "模型本来就对" if p["pred"] == gm[p["id"]]
                                  else "模型仍错")

        # 难度标签在文件名里：<model>.<mode>.<contested>.<tag>.json
        m = re.match(r".+\.(keep|drop|only)\.(.+)\.json$", f.name)
        tag = m.group(2) if m else "all"
        s_new, s_old = score(new), score(old)
        rows.append(dict(model=d["model"], kind=f'{d["kind"]}/{d.get("mode", "-")}',
                         contested=d["contested"], tag=tag, policy=pol,
                         new=s_new, old=s_old,
                         ms=round(d["seconds"] / max(1, s_new["n"]) * 1000)))

    if not rows:
        print(f"{a.results} 里没有可重算的结果")
        return 1

    print(f'{"模型":<38}{"类型/模式":<16}{"集合":<6}{"难度":<22}'
          f'{"F1":>7}{"精确":>7}{"召回":>7}{"准确":>7}{"ms/对":>8}')
    for r in sorted(rows, key=lambda r: -r["new"]["f1"]):
        n = r["new"]
        print(f'{r["model"][:37]:<38}{r["kind"]:<16}{r["contested"]:<6}{r["tag"]:<22}'
              f'{n["f1"]:>7.3f}{n["prec"]:>7.3f}{n["rec"]:>7.3f}{n["acc"]:>7.3f}'
              f'{r["ms"]:>8}')

    if a.diff:
        print(f'\n{"模型":<38}{"模式":<16}{"集合":<6}{"难度":<22}'
              f'{"F1旧":>8}{"F1新":>8}{"Δ":>8}')
        for r in sorted(rows, key=lambda r: -r["new"]["f1"]):
            d_ = r["new"]["f1"] - r["old"]["f1"]
            print(f'{r["model"][:37]:<38}{r["kind"]:<16}{r["contested"]:<6}'
                  f'{r["tag"]:<22}{r["old"]["f1"]:>8.3f}{r["new"]["f1"]:>8.3f}'
                  f'{d_:>+8.3f}')

    if flips:
        label = ("gold 被修正的" if a.policy in ("auto", "file")
                 else f"在 --policy {a.policy} 下与运行时 gold 不同的")
        print(f'\n{label} {len(flips)} 条：')
        for pid, (o, n, ok) in sorted(flips.items()):
            print(f'  {pid}: {o} -> {n}   ({ok})')
    else:
        print('\n没有标签变动，所有数字与原来一致')

    print('\n阈值沿用各次运行拟合的值（原始分数没存），打分模型的数字偏保守；'
          'oneword/cot/entail 没有阈值，是精确的。')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
