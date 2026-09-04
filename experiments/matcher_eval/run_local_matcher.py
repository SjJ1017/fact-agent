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


class Progress:
    """One line per batch, flushed, with an ETA.

    A 14B over 152 pairs is minutes of silence otherwise, and a first run also
    spends much longer than that downloading; without this there is no way to
    tell a slow model from a hung one.
    """

    def __init__(self, total: int, label: str, quiet: bool = False):
        self.total, self.label, self.quiet = total, label, quiet
        self.t0 = time.time()

    def __call__(self, done: int) -> None:
        if self.quiet:
            return
        el = time.time() - self.t0
        rate = done / el if el else 0
        eta = (self.total - done) / rate if rate else 0
        print(f"    {self.label} {done}/{self.total}  "
              f"{el:5.0f}s 已用  {eta:5.0f}s 剩余  {rate:5.1f} 对/秒",
              flush=True)


def detect_kind(model: str) -> str:
    low = model.lower()
    if "rerank" in low or "cross-encoder" in low or "nli-" in low:
        return "reranker"
    if any(t in low for t in ("bge-base", "bge-large", "bge-small", "gte-",
                              "e5-", "all-minilm", "all-mpnet", "sentence-t")):
        return "biencoder"
    return "llm"


def load(limit, contested, difficulty=None):
    rows = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    if contested == "drop":
        rows = [r for r in rows if not r["contested"]]
    elif contested == "only":
        rows = [r for r in rows if r["contested"]]
    if difficulty:
        want = set(difficulty)
        rows = [r for r in rows if r.get("difficulty") in want]
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


def breakdown(rows, preds) -> None:
    """Scores by difficulty band and by source.

    Difficulty is the bge cosine band and nothing else, so it is reproducible
    and independent of who labelled the pair.  Source says where the pair came
    from.  They are not the same cut: the probes are constructed minimal pairs
    that land all over the cosine range, and the trivial band holds two
    DIFFERENT pairs at cosine 0.98 that exist to catch a model doing nothing
    but thresholding similarity.
    """
    import collections

    def table(key, order, title):
        groups = collections.defaultdict(list)
        for r, p in zip(rows, preds):
            groups[r.get(key, "?")].append((r, p))
        names = [n for n in order if n in groups] + \
                [n for n in sorted(groups) if n not in order]
        if len(names) < 2:
            return
        print(f"\n  {title:<10}{'n':>5}{'准确':>8}{'精确':>8}{'召回':>8}"
              f"{'F1':>8}{'SAME占比':>10}")
        for n in names:
            g = groups[n]
            sub = score([r for r, _ in g], [p for _, p in g])
            share = sum(1 for r, _ in g if r["gold"] == "SAME") / len(g)
            prec = f"{sub['same_precision']:>8.3f}" if share else f"{'-':>8}"
            rec = f"{sub['same_recall']:>8.3f}" if share else f"{'-':>8}"
            f1 = f"{sub['f1']:>8.3f}" if share else f"{'-':>8}"
            print(f"  {n:<10}{sub['n']:>5}{sub['acc']:>8.3f}{prec}{rec}{f1}"
                  f"{share:>10.0%}")

    table("difficulty", ["trivial", "easy", "medium", "hard"], "难度")
    table("source", ["probe", "clin", "near", "merge", "trivial", "easy",
                     "medium"], "来源")


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


def run_biencoder(rows, model, batch, quiet=False, **_):
    from sentence_transformers import SentenceTransformer
    print("  载入模型…", flush=True)
    m = SentenceTransformer(model, device=os.environ.get("FF_DEVICE", "cuda"))
    print("  编码中…", flush=True)
    a = m.encode([r["a"] for r in rows], normalize_embeddings=True,
                 batch_size=batch, show_progress_bar=not quiet)
    b = m.encode([r["b"] for r in rows], normalize_embeddings=True,
                 batch_size=batch, show_progress_bar=not quiet)
    return [float((x * y).sum()) for x, y in zip(a, b)]


def run_reranker(rows, model, batch, quiet=False, **_):
    """Cross-encoder score per pair.

    A reranker emits one number.  An NLI model emits three (contradiction /
    entailment / neutral), and float() on that row is what raised "only
    0-dimensional arrays can be converted to Python scalars".  For NLI both
    directions are scored and the smaller entailment probability kept, so the
    score is bidirectional entailment.
    """
    import numpy as np
    from sentence_transformers import CrossEncoder
    print("  载入模型…", flush=True)
    m = CrossEncoder(model, device=os.environ.get("FF_DEVICE", "cuda"))

    def predict(pairs):
        return np.asarray(m.predict(pairs, batch_size=batch,
                                    show_progress_bar=not quiet))

    print("  打分中…", flush=True)
    ab = predict([(r["a"], r["b"]) for r in rows])
    if ab.ndim == 1:
        return [float(v) for v in ab]

    labels = getattr(getattr(m, "config", None), "id2label", None) or {}
    idx = next((i for i, n in labels.items() if "entail" in str(n).lower()), None)
    if idx is None:
        idx = 1 if ab.shape[1] == 3 else ab.shape[1] - 1
        print(f"  ! {ab.shape[1]} 类输出且无 id2label，按第 {idx} 列当 entailment",
              flush=True)

    def entail(raw):
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, int(idx)]

    print("  反向打分中…", flush=True)
    ba = predict([(r["b"], r["a"]) for r in rows])
    return [float(v) for v in np.minimum(entail(ab), entail(ba))]


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
            generate=False, quiet=False, **_):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  载入权重…（首次会先下载，几十 GB 时这一步最久）", flush=True)
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = os.environ.get("FF_DEVICE", "cuda")
    # Triton JIT-compiles a driver shim with gcc on first use, and that build
    # fails on boxes without the CUDA headers it wants -- or on a scratch
    # TMPDIR mounted noexec.  Nothing here needs a fused kernel: one forward
    # pass over 400 short prompts is not where the time goes.
    kw = {"dtype": getattr(torch, dtype), "device_map": dev,
          "attn_implementation": os.environ.get("FF_ATTN", "eager")}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw = {"device_map": dev,
              "attn_implementation": os.environ.get("FF_ATTN", "eager"),
              "quantization_config": BitsAndBytesConfig(
                  load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)}
    net = AutoModelForCausalLM.from_pretrained(model, **kw)
    net.eval()
    if torch.cuda.is_available():
        print(f"  显存占用 {torch.cuda.memory_allocated() / 2**30:.1f} GB",
              flush=True)

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": USER.format(a=r["a"], b=r["b"])}],
        add_generation_prompt=True, tokenize=False) for r in rows]

    if generate:
        out = []
        tick = Progress(len(prompts), "生成", quiet)
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
            tick(len(out))
        return out, None

    ids = _answer_token_ids(tok)
    if not ids["SAME"] or not ids["DIFFERENT"]:
        raise SystemExit(
            f"{model}: could not isolate first tokens for SAME/DIFFERENT; "
            "rerun with --generate")
    margins = []
    tick = Progress(len(prompts), "打分", quiet)
    i, bs = 0, batch
    while i < len(prompts):
        try:
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(net.device)
            with torch.no_grad():
                logits = net(**enc).logits[:, -1, :].float()
        except torch.OutOfMemoryError:
            # The peak is batch x sequence x vocab for the logits alone, which
            # for a 150k vocab is gigabytes before the model has done anything.
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"    显存不足，batch 降到 {bs}", flush=True)
            continue
        lp = torch.log_softmax(logits, dim=-1)
        sm = torch.logsumexp(lp[:, ids["SAME"]], dim=-1)
        df = torch.logsumexp(lp[:, ids["DIFFERENT"]], dim=-1)
        margins += (sm - df).tolist()
        i += bs
        tick(len(margins))
    return margins, "margin"


# A decoding prompt, for when latency does not matter.  Three things the
# one-word prompt leaves to chance:
#
#   * the rules are spelled out, including the numeric tolerance -- production
#     wants "12.3%" and "roughly 12%" recorded as one fact, and a bare
#     instruction to compare "the same proposition" says the opposite;
#   * the model must name the single difference before ruling, which is the
#     trick the hosted judge's schema uses: a dozen tokens of deliberation at
#     a length we choose, instead of an unbounded hidden budget;
#   * the answer sits on its own final line, so parsing does not depend on the
#     model stopping in the right place.
COT_SYSTEM = """\
You decide whether two atomic facts state the same thing.

SAME: they assert the same thing about the same subject. Paraphrase, word
order, synonyms, name variants and abbreviations, added or dropped hedges, and
a scope worded more or less precisely are all SAME. Numbers that agree to the
precision either side states are SAME: "12.3%" and "roughly 12%" are SAME,
"41.5" and "elevated" are SAME when both describe the same measurement.
If both would be recorded as one row in a table of who-claimed-what, they are
SAME.

DIFFERENT: they assert different things, contradict each other, or concern
different subjects. A qualifier that changes WHICH cases the claim covers
makes them DIFFERENT ("reduces mortality in patients over 65" vs "reduces
mortality"). A different entity, measurement, or document id makes them
DIFFERENT. A claim that merely follows from the other is DIFFERENT.

Answer in exactly this form and nothing else:

DIFF: <the single difference between a and b, at most eight words, or "none">
ANSWER: <SAME or DIFFERENT>"""

COT_USER = "a: {a}\nb: {b}"


def _parse_cot(text: str) -> tuple[str, str]:
    """The ANSWER line, and the note the model wrote before it."""
    diff, ans = "", ""
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("diff:"):
            diff = line.split(":", 1)[1].strip()
        elif low.startswith("answer:"):
            ans = line.split(":", 1)[1].strip().upper()
    if not ans:  # the model ignored the format; fall back to the last mention
        tail = text.strip().upper()
        i_s, i_d = tail.rfind("SAME"), tail.rfind("DIFFERENT")
        ans = "SAME" if i_s > i_d else "DIFFERENT"
    return ("SAME" if ans.startswith("SAME") else "DIFFERENT"), diff


def run_cot(rows, model, batch, dtype="bfloat16", load_4bit=False,
            quiet=False, max_new_tokens=48, **_):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  载入权重…（首次会先下载）", flush=True)
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = os.environ.get("FF_DEVICE", "cuda")
    # Triton JIT-compiles a driver shim with gcc on first use, and that build
    # fails on boxes without the CUDA headers it wants -- or on a scratch
    # TMPDIR mounted noexec.  Nothing here needs a fused kernel: one forward
    # pass over 400 short prompts is not where the time goes.
    kw = {"dtype": getattr(torch, dtype), "device_map": dev,
          "attn_implementation": os.environ.get("FF_ATTN", "eager")}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw = {"device_map": dev, "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)}
    net = AutoModelForCausalLM.from_pretrained(model, **kw)
    net.eval()
    if torch.cuda.is_available():
        print(f"  显存占用 {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": COT_SYSTEM},
         {"role": "user", "content": COT_USER.format(a=r["a"], b=r["b"])}],
        add_generation_prompt=True, tokenize=False) for r in rows]

    preds, notes = [], []
    tick = Progress(len(prompts), "解码", quiet)
    i, bs = 0, batch
    while i < len(prompts):
        try:
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(net.device)
            with torch.no_grad():
                gen = net.generate(**enc, max_new_tokens=max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"    显存不足，batch 降到 {bs}", flush=True)
            continue
        for seq in gen:
            text = tok.decode(seq[enc["input_ids"].shape[-1]:],
                              skip_special_tokens=True)
            ans, diff = _parse_cot(text)
            preds.append(ans)
            notes.append(diff)
        i += bs
        tick(len(preds))
    return preds, notes


RUNNERS = {"biencoder": run_biencoder, "reranker": run_reranker,
           "llm": run_llm, "cot": run_cot}


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
    ap.add_argument("--mode", default="logit", choices=["logit", "oneword", "cot"],
                    help="logit=读首 token 概率(最快); oneword=解码一个词; "
                         "cot=先写一句差别再判(最慢，质量最好)")
    ap.add_argument("--max-new-tokens", type=int, default=48,
                    help="cot 模式每对生成上限")
    ap.add_argument("--generate", action="store_true",
                    help="等同 --mode oneword（保留旧写法）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--contested", default="keep", choices=["keep", "drop", "only"])
    ap.add_argument("--difficulty", nargs="+",
                    choices=["trivial", "easy", "medium", "hard"],
                    help="只跑这些难度带，默认全部")
    ap.add_argument("--quiet", action="store_true", help="不打进度")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    kind = a.kind or detect_kind(a.model)
    mode = "oneword" if a.generate else a.mode
    if kind == "llm" and mode == "cot":
        kind = "cot"
    rows = load(a.limit, a.contested, a.difficulty)
    t0 = time.time()
    print(f"\n>>> {a.model}  [{kind}]  {len(rows)} 对", flush=True)
    res = RUNNERS[kind](rows, a.model, a.batch, dtype=a.dtype,
                        load_4bit=a.load_4bit, generate=(mode == "oneword"),
                        quiet=a.quiet, max_new_tokens=a.max_new_tokens)
    secs = time.time() - t0

    notes = None
    if kind == "cot":
        preds, notes = res
        curve, best = None, score(rows, preds)
    elif kind == "llm" and mode == "oneword":
        preds, curve = res[0], None
        best = score(rows, preds)
    else:
        scores = res[0] if isinstance(res, tuple) else res
        best, curve = sweep(rows, scores)
        preds = ["SAME" if v >= best["threshold"] else "DIFFERENT" for v in scores]

    shown = "llm" if kind == "cot" else kind
    print(f"\n{a.model}   [{shown}/{mode}{', 4bit' if a.load_4bit else ''}]"
          f"   contested={a.contested}")
    print(f"  n={best['n']}   {secs:.0f}s   {secs / max(1, best['n']) * 1000:.0f} ms/对")
    if "threshold" in best:
        print(f"  最佳阈值 {best['threshold']}   (在本集合上拟合，是乐观上界)")
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
        if notes:
            k = rows.index(r)
            if notes[k]:
                print(f"      模型写的差别: {notes[k][:86]}")

    tag = "-".join(a.difficulty) if a.difficulty else "all"
    out = a.out or (HERE / "results" /
                    f"{a.model.replace('/', '__')}.{mode}.{a.contested}.{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": a.model, "kind": shown, "contested": a.contested,
         "four_bit": a.load_4bit, "mode": mode,
         "seconds": secs, "best": best, "curve": curve,
         "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
         "predictions": [{"id": r["id"], "gold": r["gold"], "pred": p,
                          "contested": r["contested"],
                          **({"note": n} if notes else {})}
                         for r, p, n in zip(rows, preds,
                                            notes or [None] * len(preds))]},
        ensure_ascii=False, indent=1, sort_keys=True))
    print(f"\n  写入 {out}")

    # One aggregate row per finished run, upserted on (model, mode, contested,
    # difficulty).  The per-run file keeps the predictions, which is what any
    # error analysis needs; this is the table, so a rerun of one model does not
    # mean rebuilding it from a directory scan.
    summary = HERE / "results" / "summary.json"
    rows = json.loads(summary.read_text()) if summary.exists() else []
    key = (a.model, mode, a.contested, tag)
    rows = [r for r in rows
            if (r["model"], r["mode"], r["contested"], r["difficulty"]) != key]
    rows.append({"model": a.model, "kind": shown, "mode": mode,
                 "contested": a.contested, "difficulty": tag,
                 "four_bit": a.load_4bit, "seconds": round(secs, 1),
                 "ms_per_pair": round(secs / max(1, best["n"]) * 1000),
                 **{k: v for k, v in best.items()}})
    rows.sort(key=lambda r: -r["f1"])
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"  汇总 {summary}  （共 {len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
