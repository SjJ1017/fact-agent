"""Match already-atomized traces with a local model, in two entailment passes.

The traces in experiments/idrbench_* were extracted and atomized but never
matched: their `facts` and `mention_to_fact` are empty.  This fills them in
without the hosted judge.

Two passes per candidate pair -- does a entail b, does b entail a -- and the
identity rule is applied afterwards in code rather than asked of the model.
The direction survives into the store: Relation already carries A_ENTAILS_B
and B_ENTAILS_A, and its own docstring calls them the signal for degradation,
a fact still present but weaker.  Only EQUIVALENT edges build clusters.

Thresholds are not guessed.  --calibrate runs the same two passes over a
labelled oracle and fits the pair of cutoffs jointly, since the decision is
their conjunction; --match then reuses what was fitted.

  ./run.sh --match-calibrate                      # fit, ~2 min
  ./run.sh --match experiments/idrbench_generation_10x5_r3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "matcher_eval"))

from factflow.blocking import SbertBlocker, candidate_pairs  # noqa: E402
from factflow.match import cluster  # noqa: E402
from factflow.types import CanonicalFact, FactMention, Relation  # noqa: E402

from run_local_matcher import (ENTAIL_SYSTEM, ENTAIL_USER, Progress,  # noqa: E402
                               _yes_no_ids, chat_prompt, load, score)

CAL_DIR = HERE / "matcher_eval"
CAL = CAL_DIR / "entail_thresholds.json"


def cal_path(name) -> Path:
    """A bare name lands next to the oracle; a path is used as given."""
    if name is None:
        return CAL
    q = Path(name)
    return q if q.parent != Path(".") else CAL_DIR / q


def build(model: str, dtype: str, load_4bit: bool):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"载入 {model} …", flush=True)
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = os.environ.get("FF_DEVICE", "cuda")
    kw = {"dtype": getattr(torch, dtype), "device_map": dev,
          "attn_implementation": os.environ.get("FF_ATTN", "eager")}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw = {"device_map": dev,
              "attn_implementation": os.environ.get("FF_ATTN", "eager"),
              "quantization_config": BitsAndBytesConfig(
                  load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)}
    net = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
    if torch.cuda.is_available():
        print(f"  显存 {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    return tok, net, _yes_no_ids(tok)


def margins(tok, net, ids, pairs, batch, label, quiet=False):
    """log P(YES) - log P(NO) for the first answer token, one pass per pair."""
    import torch

    prompts = [chat_prompt(tok, ENTAIL_SYSTEM, ENTAIL_USER.format(a=x, b=y))
               for x, y in pairs]
    out: list[float] = []
    tick = Progress(len(prompts), label, quiet)
    i, bs = 0, batch
    while i < len(prompts):
        try:
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(net.device)
            with torch.no_grad():
                logits = net(**enc).logits[:, -1, :].float()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"    显存不足，batch 降到 {bs}", flush=True)
            continue
        lp = torch.log_softmax(logits, dim=-1)
        y = torch.logsumexp(lp[:, ids["YES"]], dim=-1)
        n = torch.logsumexp(lp[:, ids["NO"]], dim=-1)
        out += (y - n).tolist()
        i += bs
        if len(out) % max(bs, 200) < bs:
            tick(len(out))
    tick(len(out))
    return out


def calibrate(a) -> int:
    out = cal_path(a.cal)
    rows = load(None, a.contested, None, False, a.pairs, a.policy)
    tok, net, ids = build(a.model, a.dtype, a.load_4bit)
    ab = margins(tok, net, ids, [(r["a"], r["b"]) for r in rows], a.batch, "A⊨B")
    ba = margins(tok, net, ids, [(r["b"], r["a"]) for r in rows], a.batch, "B⊨A")

    # The margin is log P(YES) - log P(NO), so 0 is the point where the model
    # stops saying yes.  Sweeping below it lets the joint optimum drift to a
    # threshold that accepts pairs the model rejected: the first fit put the
    # reverse cutoff under -8, which made the second pass vacuous and turned
    # the two-pass design back into a one-directional test at twice the cost.
    def grid(v, floor):
        lo, hi = max(min(v), floor), max(v)
        if hi <= lo:
            return [lo]
        return [lo + (hi - lo) * i / 24 for i in range(25)]

    best = None
    for ta in grid(ab, a.margin_floor):
        for tb in grid(ba, a.margin_floor):
            pred = ["SAME" if (x >= ta and y >= tb) else "DIFFERENT"
                    for x, y in zip(ab, ba)]
            s = score(rows, pred)
            if best is None or s["f1"] > best[0]["f1"]:
                best = (s, ta, tb)
    s, ta, tb = best
    frac = sum(1 for x, y in zip(ab, ba) if x >= ta and y >= tb and min(x, y) < 0)
    if frac:
        print(f"! {frac} 对被判等价但至少一个方向的 margin 为负", file=sys.stderr)
    out.write_text(json.dumps(
        {"model": a.model, "pairs": str(a.pairs or "pairs.jsonl"),
         "policy": a.policy, "contested": a.contested,
         "threshold_ab": ta, "threshold_ba": tb, "fit": s,
         "margin_floor": a.margin_floor},
        ensure_ascii=False, indent=1))
    print(f"\n阈值 A⊨B {ta:.4f}   B⊨A {tb:.4f}")
    print(f"拟合于 {s['n']} 对：F1 {s['f1']:.3f}  精确 {s['same_precision']:.3f}  "
          f"召回 {s['same_recall']:.3f}  准确 {s['acc']:.3f}")
    print(f"写入 {out}")
    print("注意这是在评测集上拟合的，是乐观上界；真要报数需要独立的调阈子集。")
    return 0


def match_dir(a) -> int:
    cfile = cal_path(a.cal)
    if not cfile.exists():
        raise SystemExit(f"没有 {cfile}，先跑 --match-calibrate")
    cal = json.loads(cfile.read_text())
    ta, tb = cal["threshold_ab"], cal["threshold_ba"]
    if cal["model"] != a.model:
        print(f"! 阈值是用 {cal['model']} 拟合的，现在跑的是 {a.model}", file=sys.stderr)
    print(f"阈值 A⊨B {ta:.4f}  B⊨A {tb:.4f}  (来自 {cfile.name}，"
          f"policy={cal.get('policy')}，floor={cal.get('margin_floor', '未记录')})")
    if tb < 0 or ta < 0:
        print("! 有一个方向的阈值为负，那一遍等于恒真：两遍法退化成单向判断",
              file=sys.stderr)

    files = sorted(a.indir.glob(f"*{a.suffix}"))
    if not files:
        raise SystemExit(f"{a.indir} 下没有 *{a.suffix}")
    tok, net, ids = build(a.model, a.dtype, a.load_4bit)
    blk = SbertBlocker(model_name=a.embed)

    for n, f in enumerate(files, 1):
        out = f.with_name(f.name.replace(a.suffix, a.out_suffix))
        if out.exists() and not a.redo:
            print(f"[{n}/{len(files)}] {f.name} -> 已有，跳过", flush=True)
            continue
        d = json.loads(f.read_text())
        mentions = [FactMention(**m) for m in d["mentions"].values()]
        t0 = time.time()
        pairs = candidate_pairs(mentions, blocker=blk, threshold=a.threshold,
                                top_k=a.top_k)
        print(f"[{n}/{len(files)}] {f.name}  {len(mentions)} mention  "
              f"{len(pairs)} 候选对", flush=True)
        if pairs:
            ab = margins(tok, net, ids,
                         [(mentions[i].text, mentions[j].text) for i, j, _ in pairs],
                         a.batch, "  A⊨B", a.quiet)
            ba = margins(tok, net, ids,
                         [(mentions[j].text, mentions[i].text) for i, j, _ in pairs],
                         a.batch, "  B⊨A", a.quiet)
        else:
            ab = ba = []

        rels: list[Relation] = []
        tally = {"EQUIVALENT": 0, "A_ENTAILS_B": 0, "B_ENTAILS_A": 0, "UNRELATED": 0}
        for (i, j, sim), x, y in zip(pairs, ab, ba):
            fwd, rev = x >= ta, y >= tb
            kind = ("EQUIVALENT" if fwd and rev else
                    "A_ENTAILS_B" if fwd else
                    "B_ENTAILS_A" if rev else "UNRELATED")
            tally[kind] += 1
            rels.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                 relation=kind, confidence=float(sim),
                                 rationale=f"entail a->b {x:.2f} b->a {y:.2f}"))
        facts = cluster(mentions, rels, union_min_similarity=a.union_min)
        m2f = {mid: fa.fact_id for fa in facts for mid in fa.mention_ids}
        out.write_text(json.dumps(
            {"mentions": {m.mention_id: json.loads(m.model_dump_json())
                          for m in mentions},
             "facts": {fa.fact_id: json.loads(fa.model_dump_json()) for fa in facts},
             "mention_to_fact": m2f,
             "relations": [json.loads(r.model_dump_json()) for r in rels],
             "matching": {"model": a.model, "mode": "entail",
                          "threshold_ab": ta, "threshold_ba": tb,
                          "blocker": a.embed, "block_threshold": a.threshold,
                          "top_k": a.top_k, "union_min": a.union_min,
                          "counts": tally, "seconds": round(time.time() - t0, 1)}},
            ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        merged = len(mentions) - len(facts)
        print(f"      {len(facts)} 簇（合并 {merged}，{merged / max(1, len(mentions)):.0%}）"
              f"  等价 {tally['EQUIVALENT']}  单向 "
              f"{tally['A_ENTAILS_B'] + tally['B_ENTAILS_A']}"
              f"  {time.time() - t0:.0f}s  -> {out.name}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", type=Path, nargs="?")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--4bit", dest="load_4bit", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--suffix", default=".atomized.json")
    ap.add_argument("--out-suffix", default=".store.json")
    ap.add_argument("--embed", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.62,
                    help="blocking 的余弦下限。评测里 0.62 以下没有出现过真正的 "
                         "SAME，再低只是成倍增加候选对")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--union-min", type=float, default=0.85,
                    help="低于此相似度的 SAME 边只信它自己那一对，不用来串簇")
    ap.add_argument("--cal", default=None,
                    help="阈值文件。裸文件名放在 matcher_eval/ 下。"
                         "换配置时给不同的名字，避免覆盖上一次的结果")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    # 只在 --calibrate 时用
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--policy", default="file",
                    choices=["file", "strict", "near", "entail"])
    ap.add_argument("--contested", default="keep", choices=["keep", "drop", "only"])
    ap.add_argument("--margin-floor", type=float, default=0.0,
                    help="阈值搜索的下界。margin 是 log P(YES)-log P(NO)，"
                         "0 以下等于接受模型说 NO 的对；设 -inf 可关闭")
    a = ap.parse_args()

    if a.calibrate:
        return calibrate(a)
    if not a.indir:
        ap.error("要么 --calibrate，要么给一个目录")
    return match_dir(a)


if __name__ == "__main__":
    raise SystemExit(main())
