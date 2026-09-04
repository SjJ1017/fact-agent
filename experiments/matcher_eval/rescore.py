"""Re-score saved runs against the current gold, without calling any model.

Result files keep the per-pair prediction, so a corrected label only needs the
scores recomputed.  What this cannot do is refit a threshold: the raw margins
are not stored, so the numbers here are what each run achieved at the
threshold it chose, judged by the labels as they stand now.  For a scoring
model that makes them slightly pessimistic; for an LLM in oneword/cot/entail
mode there is no threshold and they are exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

POLICY = {
    "strict": {"equivalent"},
    "near": {"equivalent", "near_match"},
    "entail": {"equivalent", "near_match", "entail_ab", "entail_ba"},
}


def gold_map(path: Path, policy: str) -> dict[str, tuple[str, bool, str]]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rel = r.get("relation") or ("equivalent" if r["gold"] == "SAME" else "different")
        gold = r["gold"] if policy == "file" else (
            "SAME" if rel in POLICY[policy] else "DIFFERENT")
        out[r["id"]] = (gold, r.get("contested", False), r.get("difficulty", "?"))
    return out


def score(rows):
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
    ap.add_argument("--policy", default=None,
                    choices=["file", "strict", "near", "entail"],
                    help="默认沿用每个结果文件自己的口径")
    ap.add_argument("--only-changed", action="store_true",
                    help="只列出分数变了的")
    a = ap.parse_args()

    golds = {}
    for name in ("pairs.jsonl", "pairs_idrbench.jsonl"):
        p = HERE / name
        if p.exists():
            golds[name] = p

    out = []
    for f in sorted(a.results.glob("*.json")):
        if f.name == "summary.json":
            continue
        d = json.loads(f.read_text())
        preds = d.get("predictions")
        if not preds:
            continue
        # which oracle: the id prefix says it
        which = ("pairs_idrbench.jsonl" if preds[0]["id"].startswith("idr-")
                 else "pairs.jsonl")
        if which not in golds:
            continue
        pol = a.policy or ("entail" if ".entail" in f.name else "near")
        gm = gold_map(golds[which], pol)

        missing = [p["id"] for p in preds if p["id"] not in gm]
        rows_new, rows_old, flips = [], [], []
        for p in preds:
            if p["id"] not in gm:
                continue
            g_new = gm[p["id"]][0]
            rows_new.append((g_new, p["pred"]))
            rows_old.append((p["gold"], p["pred"]))
            if g_new != p["gold"]:
                flips.append((p["id"], p["gold"], g_new, p["pred"]))
        if not rows_new:
            continue
        old, new = score(rows_old), score(rows_new)
        out.append(dict(file=f.name, model=d["model"],
                        kind=f'{d["kind"]}/{d.get("mode","-")}',
                        contested=d["contested"], policy=pol,
                        old=old, new=new, flips=flips, missing=len(missing)))

    if not out:
        return print("results/ 里没有可重算的结果") or 1

    changed = [r for r in out if abs(r["new"]["f1"] - r["old"]["f1"]) > 1e-9]
    show = changed if a.only_changed else out
    print(f'{"模型":<28}{"模式":<16}{"集合":<6}{"口径":<8}'
          f'{"F1旧":>8}{"F1新":>8}{"Δ":>8}{"准旧":>7}{"准新":>7}{"翻转":>5}')
    for r in sorted(show, key=lambda r: -r["new"]["f1"]):
        d = r["new"]["f1"] - r["old"]["f1"]
        print(f'{r["model"][:27]:<28}{r["kind"]:<16}{r["contested"]:<6}{r["policy"]:<8}'
              f'{r["old"]["f1"]:>8.3f}{r["new"]["f1"]:>8.3f}{d:>+8.3f}'
              f'{r["old"]["acc"]:>7.3f}{r["new"]["acc"]:>7.3f}{len(r["flips"]):>5}')

    tot = sorted({f[0] for r in out for f in r["flips"]})
    if tot:
        print(f'\n改动的 gold 共 {len(tot)} 条：')
        seen = set()
        for r in out:
            for pid, g_old, g_new, pred in r["flips"]:
                if pid in seen:
                    continue
                seen.add(pid)
                ok = "模型本来就对" if pred == g_new else "模型仍错"
                print(f'  {pid}: {g_old} -> {g_new}   ({ok})')
    miss = sum(r["missing"] for r in out)
    if miss:
        print(f'\n注意：{miss} 条预测在当前 oracle 里找不到对应 id，已跳过')
    print(f'\n分数有变化的 {len(changed)}/{len(out)} 个结果文件')
    print('阈值沿用各自跑时拟合的值（原始分数没存），所以打分模型的新数字偏保守；'
          'oneword/cot/entail 没有阈值，是精确的。')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
