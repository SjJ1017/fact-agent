"""Score a local model on the atomic-fact SAME/DIFFERENT decision.

Usage
  # cross-encoder / reranker (no generation, needs a threshold sweep)
  python run_local_matcher.py --kind reranker --model BAAI/bge-reranker-v2-m3
  # bi-encoder cosine, the current pipeline's blocker as a baseline
  python run_local_matcher.py --kind biencoder --model BAAI/bge-base-en-v1.5
  # instruct LLM, constrained to one word
  python run_local_matcher.py --kind llm --model Qwen/Qwen3-14B-Instruct --dtype bfloat16

Everything runs on one GPU.  --limit trims the set for a smoke test; --contested
drop|keep|only controls the 26 pairs where a careful reader could disagree.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "pairs.jsonl"

# The wording the production judge uses, so a local model is measured on the
# same task and not on a friendlier one.
SYSTEM = (
    "You compare two atomic factual statements drawn from a discussion.\n"
    "Answer SAME if they assert the same proposition about the same entities, "
    "even if worded differently.\n"
    "Answer DIFFERENT if either states something the other does not: a "
    "different entity, a different quantity, an added or dropped condition, or "
    "a claim that merely follows from the other.\n"
    "Reply with exactly one word: SAME or DIFFERENT."
)
USER = "A: {a}\nB: {b}"


def load(limit: int | None, contested: str) -> list[dict]:
    rows = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    if contested == "drop":
        rows = [r for r in rows if not r["contested"]]
    elif contested == "only":
        rows = [r for r in rows if r["contested"]]
    return rows[:limit] if limit else rows


def score(rows, pred) -> dict:
    tp = sum(1 for r, p in zip(rows, pred) if r["gold"] == "SAME" and p == "SAME")
    fp = sum(1 for r, p in zip(rows, pred) if r["gold"] != "SAME" and p == "SAME")
    fn = sum(1 for r, p in zip(rows, pred) if r["gold"] == "SAME" and p != "SAME")
    tn = len(rows) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(rows), "acc": (tp + tn) / len(rows),
            "same_precision": prec, "same_recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_biencoder(rows, model, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device=device)
    a = m.encode([r["a"] for r in rows], normalize_embeddings=True)
    b = m.encode([r["b"] for r in rows], normalize_embeddings=True)
    return [float((x * y).sum()) for x, y in zip(a, b)]


def run_reranker(rows, model, device):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(model, device=device)
    return [float(s) for s in m.predict([(r["a"], r["b"]) for r in rows])]


def run_llm(rows, model, device, dtype, max_new_tokens=4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=getattr(torch, dtype), device_map=device)
    net.eval()
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(a=r["a"], b=r["b"])}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(net.device)
        with torch.no_grad():
            gen = net.generate(ids, max_new_tokens=max_new_tokens,
                               do_sample=False,
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)
        out.append("SAME" if "same" in text.strip().lower()[:8] else "DIFFERENT")
    return out


def sweep(rows, scores):
    """Best threshold, and the curve, for a model that returns a number."""
    best, curve = None, []
    lo, hi = min(scores), max(scores)
    for i in range(41):
        t = lo + (hi - lo) * i / 40
        s = score(rows, ["SAME" if v >= t else "DIFFERENT" for v in scores])
        curve.append({"threshold": round(t, 4), **s})
        if best is None or s["f1"] > best["f1"]:
            best = {"threshold": round(t, 4), **s}
    return best, curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", required=True, choices=["biencoder", "reranker", "llm"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--contested", default="keep", choices=["keep", "drop", "only"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = load(a.limit, a.contested)
    t0 = time.time()
    if a.kind == "llm":
        pred = run_llm(rows, a.model, a.device, a.dtype)
        result = {"best": score(rows, pred), "curve": None}
        preds = pred
    else:
        fn = run_biencoder if a.kind == "biencoder" else run_reranker
        scores = fn(rows, a.model, a.device)
        best, curve = sweep(rows, scores)
        result = {"best": best, "curve": curve}
        preds = ["SAME" if v >= best["threshold"] else "DIFFERENT" for v in scores]
    secs = time.time() - t0

    b = result["best"]
    print(f"\n{a.model}  ({a.kind}, {a.contested} contested)")
    print(f"  n={b['n']}  {secs:.0f}s  {secs / max(1, b['n']) * 1000:.0f} ms/对")
    if "threshold" in b:
        print(f"  最佳阈值 {b['threshold']}")
    print(f"  准确率 {b['acc']:.3f}   SAME 精确率 {b['same_precision']:.3f}   "
          f"SAME 召回 {b['same_recall']:.3f}   F1 {b['f1']:.3f}")
    print(f"  TP {b['tp']}  FP {b['fp']}  FN {b['fn']}  TN {b['tn']}")

    wrong = [(r, p) for r, p in zip(rows, preds) if (p == "SAME") != (r["gold"] == "SAME")]
    print(f"\n  错判 {len(wrong)} 条，前 8 条：")
    for r, p in wrong[:8]:
        print(f"   [{r['id']}] gold={r['gold']} pred={p}"
              f"{' (有争议)' if r['contested'] else ''}")
        print(f"      A: {r['a'][:88]}")
        print(f"      B: {r['b'][:88]}")

    if a.out:
        a.out.write_text(json.dumps(
            {"model": a.model, "kind": a.kind, "contested": a.contested,
             "seconds": secs, **result,
             "predictions": [{"id": r["id"], "gold": r["gold"], "pred": p}
                             for r, p in zip(rows, preds)]},
            ensure_ascii=False, indent=1, sort_keys=True))
        print(f"\n  写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
