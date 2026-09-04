"""The hosted judge's score on the same 152 pairs the local candidates see.

The recorded calibration number (minimax-m2.5, accuracy 0.967) is over 30
in-domain clinical pairs chosen for calibration.  This set is deliberately
harder -- mined from the blocker's near-miss band and seeded with 24 merges
the pipeline gets wrong -- so that number is not a threshold a local model has
to clear.  This produces one that is.

Two protocols, because the local harness and production do not ask the same
way and the difference is worth seeing:

  production  the real IDENTITY_SYSTEM, batched JSON, `diff` written before
              `same`.  What the pipeline actually runs.
  oneword     the local harness's prompt, one pair at a time, answer in one
              word.  Comparable to what run_local_matcher.py measures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from factflow.llm import LLM  # noqa: E402
from factflow.match import IDENTITY_SYSTEM, IdentityResult  # noqa: E402
from run_local_matcher import (SYSTEM as ONEWORD_SYSTEM, USER, breakdown,  # noqa: E402
                               load, score)
from run_perspectrum import load_opencode_key  # noqa: E402


def run_production(llm, rows, batch):
    # pair_id is an int in the production schema and the model echoes it as
    # one; passing the string ids made every lookup miss.
    preds = {}
    chunks = [list(enumerate(rows[i:i + batch], start=i))
              for i in range(0, len(rows), batch)]
    for n, chunk in enumerate(chunks, 1):
        payload = [{"pair_id": k, "a": r["a"], "b": r["b"]} for k, r in chunk]
        res = llm.parse(system=IDENTITY_SYSTEM,
                        user=json.dumps(payload, ensure_ascii=False),
                        output_format=IdentityResult)
        for j in res.judgements:
            preds[int(j.pair_id)] = "SAME" if j.same else "DIFFERENT"
        print(f"    批 {n}/{len(chunks)}  已判 {len(preds)}/{len(rows)}", flush=True)

    # Defaulting a missing pair_id to DIFFERENT produced a run with TP 0 and
    # FP 0 that still printed an accuracy -- every id had failed to round-trip
    # and the fallback quietly answered for all 152.  A measurement that can
    # be manufactured by a lookup miss is not a measurement.
    missing = [k for k in range(len(rows)) if k not in preds]
    if missing:
        raise SystemExit(
            f"模型只回了 {len(preds)}/{len(rows)} 对，缺 {len(missing)} 个 pair_id。\n"
            f"前几个: {missing[:5]}\n"
            f"模型返回的 id 例: {list(preds)[:5]}\n"
            "pair_id 没有原样返回，换更小的 --batch 或改用 --protocol oneword。")
    return [preds[k] for k in range(len(rows))]


def run_oneword(llm, rows):
    out = []
    for n, r in enumerate(rows, 1):
        text = llm.chat(system=ONEWORD_SYSTEM,
                        user=USER.format(a=r["a"], b=r["b"]),
                        temperature=0.0, max_tokens=8,
                        sample_id=f"judge-oneword:{r['id']}")
        out.append("SAME" if "same" in text.strip().lower()[:8] else "DIFFERENT")
        if n % 20 == 0 or n == len(rows):
            print(f"    {n}/{len(rows)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="minimax-m2.5")
    ap.add_argument("--protocol", default="production",
                    choices=["production", "oneword"])
    ap.add_argument("--batch", type=int, default=16, help="production 协议的批大小")
    ap.add_argument("--pairs", type=Path, default=None,
                    help="换一个 oracle 文件，默认 pairs.jsonl")
    ap.add_argument("--policy", default="file",
                    choices=["file", "strict", "near", "entail"])
    ap.add_argument("--contested", default="keep", choices=["keep", "drop", "only"])
    ap.add_argument("--difficulty", nargs="+",
                    choices=["trivial", "easy", "medium", "hard"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    load_opencode_key()
    llm = LLM.opencode(a.model, max_concurrency=1)
    rows = load(a.limit, a.contested, a.difficulty, False, a.pairs, a.policy)

    print(f">>> {a.model}  [{a.protocol}]  {len(rows)} 对", flush=True)
    t0 = time.time()
    preds = (run_production(llm, rows, a.batch) if a.protocol == "production"
             else run_oneword(llm, rows))
    secs = time.time() - t0
    best = score(rows, preds)

    print(f"\n{a.model}   [hosted, {a.protocol}]   contested={a.contested}")
    print(f"  n={best['n']}   {secs:.0f}s")
    print(f"  准确率 {best['acc']:.3f}   SAME 精确率 {best['same_precision']:.3f}   "
          f"SAME 召回 {best['same_recall']:.3f}   F1 {best['f1']:.3f}")
    print(f"  TP {best['tp']}  FP {best['fp']}  FN {best['fn']}  TN {best['tn']}")

    breakdown(rows, preds)

    wrong = [(r, p) for r, p in zip(rows, preds)
             if (p == "SAME") != (r["gold"] == "SAME")]
    print(f"\n  错判 {len(wrong)} 条"
          f"（其中有争议 {sum(1 for r, _ in wrong if r['contested'])} 条），前 6 条：")
    for r, p in wrong[:6]:
        print(f"   [{r['id']}] gold={r['gold']} pred={p}"
              f"{'  有争议' if r['contested'] else ''}")
        print(f"      A: {r['a'][:86]}")
        print(f"      B: {r['b'][:86]}")

    tag = "-".join(a.difficulty) if a.difficulty else "all"
    if a.pairs:
        tag = f"{a.pairs.stem.replace('pairs_', '')}.{tag}"
    out = (HERE / "results" /
           f"hosted__{a.model}.{a.protocol}.{a.contested}.{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": f"{a.model} (hosted)", "kind": f"hosted/{a.protocol}",
         "contested": a.contested, "policy": a.policy, "seconds": secs, "best": best, "curve": None,
         "predictions": [{"id": r["id"], "gold": r["gold"], "pred": p,
                          "contested": r["contested"]}
                         for r, p in zip(rows, preds)]},
        ensure_ascii=False, indent=1, sort_keys=True))
    print(f"\n  写入 {out}")

    summary = HERE / "results" / "summary.json"
    rows = json.loads(summary.read_text()) if summary.exists() else []
    key = (f"{a.model} (hosted)", a.protocol, a.contested,
           "-".join(a.difficulty) if a.difficulty else "all")
    rows = [r for r in rows
            if (r["model"], r["mode"], r["contested"], r["difficulty"]) != key]
    rows.append({"model": key[0], "kind": "hosted", "mode": a.protocol,
                 "contested": a.contested, "difficulty": key[3],
                 "four_bit": False, "seconds": round(secs, 1),
                 "ms_per_pair": round(secs / max(1, best["n"]) * 1000), **best})
    rows.sort(key=lambda r: -r["f1"])
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"  汇总 {summary}  （共 {len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
