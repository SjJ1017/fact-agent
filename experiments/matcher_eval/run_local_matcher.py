"""Score a local model on the atomic-fact SAME/DIFFERENT decision.

Three model families, auto-detected from the name unless --kind says otherwise:

  biencoder  sentence embeddings, cosine.  The current blocker; the floor.
  reranker   cross-encoder over the pair.  Small, fast, needs a threshold.
  llm        instruct model.  By default it does NOT generate: it reads the
             logits of the first answer token and compares SAME against
             DIFFERENT.  That is one forward pass instead of a decode loop,
             it cannot wander off format, and it yields a margin, so the same
             threshold sweep applies as for the scoring models.

Everything below runs in one process on one GPU.  See run.sh for the wrapper
that sets the caches and pins the device.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "pairs.jsonl"

# The production judge's wording, so a local model is measured on the same
# task and not on a friendlier one.
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


def detect_kind(model: str) -> str:
    low = model.lower()
    if "rerank" in low or "cross-encoder" in low or "nli-" in low:
        return "reranker"
    if any(t in low for t in ("bge-base", "bge-large", "bge-small", "gte-",
                              "e5-", "all-minilm", "all-mpnet", "sentence-t")):
        return "biencoder"
    return "llm"


def load(limit, contested):
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


def sweep(rows, scores):
    best, curve = None, []
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    for i in range(81):
        t = lo + span * i / 80
        s = score(rows, ["SAME" if v >= t else "DIFFERENT" for v in scores])
        curve.append({"threshold": round(t, 4), **s})
        if best is None or s["f1"] > best["f1"]:
            best = {"threshold": round(t, 4), **s}
    return best, curve


# --------------------------------------------------------------------------


def run_biencoder(rows, model, batch, **_):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device=os.environ.get("FF_DEVICE", "cuda"))
    a = m.encode([r["a"] for r in rows], normalize_embeddings=True,
                 batch_size=batch, show_progress_bar=False)
    b = m.encode([r["b"] for r in rows], normalize_embeddings=True,
                 batch_size=batch, show_progress_bar=False)
    return [float((x * y).sum()) for x, y in zip(a, b)]


def run_reranker(rows, model, batch, **_):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(model, device=os.environ.get("FF_DEVICE", "cuda"))
    return [float(s) for s in m.predict([(r["a"], r["b"]) for r in rows],
                                        batch_size=batch,
                                        show_progress_bar=False)]


def _answer_token_ids(tok):
    """First-token ids for SAME and DIFFERENT, in the shapes a chat model emits."""
    out = {"SAME": set(), "DIFFERENT": set()}
    for word in out:
        for variant in (word, " " + word, word.capitalize(), word.lower(),
                        " " + word.capitalize(), " " + word.lower()):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                out[word].add(ids[0])
    # a token claimed by both words is useless as evidence for either
    both = out["SAME"] & out["DIFFERENT"]
    return {k: sorted(v - both) for k, v in out.items()}


def run_llm(rows, model, batch, dtype="bfloat16", load_4bit=False,
            generate=False, **_):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = os.environ.get("FF_DEVICE", "cuda")
    kw = {"dtype": getattr(torch, dtype), "device_map": dev}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw = {"device_map": dev, "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)}
    net = AutoModelForCausalLM.from_pretrained(model, **kw)
    net.eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": USER.format(a=r["a"], b=r["b"])}],
        add_generation_prompt=True, tokenize=False) for r in rows]

    if generate:
        out = []
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(net.device)
            with torch.no_grad():
                gen = net.generate(**enc, max_new_tokens=4, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for j, seq in enumerate(gen):
                text = tok.decode(seq[enc["input_ids"].shape[-1]:],
                                  skip_special_tokens=True)
                out.append("SAME" if "same" in text.strip().lower()[:8]
                           else "DIFFERENT")
        return out, None

    ids = _answer_token_ids(tok)
    if not ids["SAME"] or not ids["DIFFERENT"]:
        raise SystemExit(
            f"{model}: could not isolate first tokens for SAME/DIFFERENT; "
            "rerun with --generate")
    margins = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(net.device)
        with torch.no_grad():
            logits = net(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        s = torch.logsumexp(lp[:, ids["SAME"]], dim=-1)
        d = torch.logsumexp(lp[:, ids["DIFFERENT"]], dim=-1)
        margins += (s - d).tolist()
    return margins, "margin"


RUNNERS = {"biencoder": run_biencoder, "reranker": run_reranker, "llm": run_llm}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["biencoder", "reranker", "llm"],
                    help="default: guessed from the model name")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--4bit", dest="load_4bit", action="store_true",
                    help="needed for anything over ~20B on a 46GB card")
    ap.add_argument("--generate", action="store_true",
                    help="decode a word instead of reading the first-token logits")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--contested", default="keep", choices=["keep", "drop", "only"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    kind = a.kind or detect_kind(a.model)
    rows = load(a.limit, a.contested)
    t0 = time.time()
    res = RUNNERS[kind](rows, a.model, a.batch, dtype=a.dtype,
                        load_4bit=a.load_4bit, generate=a.generate)
    secs = time.time() - t0

    if kind == "llm" and a.generate:
        preds, curve = res[0], None
        best = score(rows, preds)
    else:
        scores = res[0] if isinstance(res, tuple) else res
        best, curve = sweep(rows, scores)
        preds = ["SAME" if v >= best["threshold"] else "DIFFERENT" for v in scores]

    print(f"\n{a.model}   [{kind}{', 4bit' if a.load_4bit else ''}"
          f"{', generate' if a.generate else ''}]   contested={a.contested}")
    print(f"  n={best['n']}   {secs:.0f}s   {secs / max(1, best['n']) * 1000:.0f} ms/对")
    if "threshold" in best:
        print(f"  最佳阈值 {best['threshold']}   (在本集合上拟合，是乐观上界)")
    print(f"  准确率 {best['acc']:.3f}   SAME 精确率 {best['same_precision']:.3f}   "
          f"SAME 召回 {best['same_recall']:.3f}   F1 {best['f1']:.3f}")
    print(f"  TP {best['tp']}  FP {best['fp']}  FN {best['fn']}  TN {best['tn']}")

    wrong = [(r, p) for r, p in zip(rows, preds)
             if (p == "SAME") != (r["gold"] == "SAME")]
    print(f"\n  错判 {len(wrong)} 条"
          f"（其中有争议 {sum(1 for r, _ in wrong if r['contested'])} 条），前 6 条：")
    for r, p in wrong[:6]:
        print(f"   [{r['id']}] gold={r['gold']} pred={p}"
              f"{'  有争议' if r['contested'] else ''}")
        print(f"      A: {r['a'][:86]}")
        print(f"      B: {r['b'][:86]}")

    out = a.out or (HERE / "results" /
                    f"{a.model.replace('/', '__')}.{a.contested}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": a.model, "kind": kind, "contested": a.contested,
         "four_bit": a.load_4bit, "generate": a.generate,
         "seconds": secs, "best": best, "curve": curve,
         "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
         "predictions": [{"id": r["id"], "gold": r["gold"], "pred": p,
                          "contested": r["contested"]}
                         for r, p in zip(rows, preds)]},
        ensure_ascii=False, indent=1, sort_keys=True))
    print(f"\n  写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
