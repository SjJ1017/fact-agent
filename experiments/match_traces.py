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
                               _yes_no_ids, binary_scores, chat_prompt,
                               directional_gold, load, score)

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

    # Fit on one half, report on the other.  Sweeping a threshold on the same
    # pairs it is then scored against reports the best achievable number, not
    # the one production will see; every earlier calibration here carried that
    # caveat in a print statement instead of doing something about it.
    import collections
    import random as _rnd
    import statistics as _st

    g_ab, g_ba = directional_gold(rows, not a.strict_entail)
    idx = list(range(len(rows)))
    by_rel = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_rel[r.get("relation") or ("equivalent" if r["gold"] == "SAME"
                                     else "different")].append(i)
    rng = _rnd.Random(a.split_seed)
    test: set[int] = set()
    for rel, group in sorted(by_rel.items()):
        g = list(group)
        rng.shuffle(g)
        test.update(g[:max(1, round(len(g) * a.holdout))])
    train = [i for i in idx if i not in test]
    test = sorted(test)

    def pooled(ix):
        return ([g_ab[i] for i in ix] + [g_ba[i] for i in ix],
                [ab[i] for i in ix] + [ba[i] for i in ix])

    lab_tr, mar_tr = pooled(train)
    lo, hi = max(min(mar_tr), a.margin_floor), max(mar_tr)
    grid = [lo + (hi - lo) * i / 200 for i in range(201)] if hi > lo else [lo]
    t, best_f1 = lo, -1.0
    for cand in grid:
        f1 = binary_scores(lab_tr, mar_tr, cand)["f1"]
        if f1 > best_f1:
            best_f1, t = f1, cand

    lab_te, mar_te = pooled(test)
    f_tr = binary_scores(lab_tr, mar_tr, t)
    f_te = binary_scores(lab_te, mar_te, t)

    def induced(ix):
        sub = [rows[i] for i in ix]
        pred = ["SAME" if (ab[i] >= t and ba[i] >= t) else "DIFFERENT"
                for i in ix]
        return score(sub, pred), pred

    eq_tr, _ = induced(train)
    eq_te, _ = induced(test)
    cells = [(x >= t, y >= t) for x, y in zip(ab, ba)]
    tally = collections.Counter(
        "equivalent" if c == (True, True) else
        "a_entails_b" if c[0] else "b_entails_a" if c[1] else "neither"
        for c in cells)
    gap = _st.mean(x - y for x, y in zip(ab, ba))

    out.write_text(json.dumps(
        {"model": a.model, "pairs": str(a.pairs or "pairs.jsonl"),
         "policy": a.policy, "contested": a.contested,
         "threshold": t, "threshold_ab": t, "threshold_ba": t,
         "strict_entail": a.strict_entail, "margin_floor": a.margin_floor,
         "holdout": a.holdout, "split_seed": a.split_seed,
         "n_train": len(train), "n_test": len(test),
         "f_entail_train": f_tr, "f_entail_holdout": f_te,
         "induced_equivalence_train": eq_tr,
         "induced_equivalence_holdout": eq_te,
         "cells": dict(tally),
         "position_bias_mean_ab_minus_ba": round(gap, 3),
         "margins": [{"id": r["id"], "gold": r["gold"],
                      "relation": r.get("relation"),
                      "split": "test" if i in set(test) else "train",
                      "ab": round(x, 3), "ba": round(y, 3)}
                     for i, (r, x, y) in enumerate(zip(rows, ab, ba))]},
        ensure_ascii=False, indent=1))

    print(f"\n 阈值 {t:.4f}，在 {len(train)} 对（{len(lab_tr)} 个方向标注）上拟合")
    print(f"\n{'':18}{'训练':>22}{'留出':>22}")
    print(f" {'f 准确':<16}{f_tr['acc']:>22.3f}{f_te['acc']:>22.3f}")
    print(f" {'f 精确':<16}{f_tr['precision']:>22.3f}{f_te['precision']:>22.3f}")
    print(f" {'f 召回':<16}{f_tr['recall']:>22.3f}{f_te['recall']:>22.3f}")
    print(f" {'f F1':<16}{f_tr['f1']:>22.3f}{f_te['f1']:>22.3f}")
    print(f" {'等价 F1':<16}{eq_tr['f1']:>22.3f}{eq_te['f1']:>22.3f}")
    print(f" {'等价 精确':<15}{eq_tr['same_precision']:>22.3f}"
          f"{eq_te['same_precision']:>22.3f}")
    print(f"\n 留出集 {len(test)} 对，是可以报的数；训练那列是乐观上界。")
    print(f" 四象限（全集）{dict(tally)}   位置偏差 {gap:+.2f}")
    if abs(t - lo) < 1e-9:
        print("! 阈值贴在搜索下界上，等于没有约束", file=sys.stderr)
    print(f"写入 {out}")
    return 0


def match_dir(a) -> int:
    cfile = cal_path(a.cal)
    if not cfile.exists():
        raise SystemExit(f"没有 {cfile}，先跑 --match-calibrate")
    cal = json.loads(cfile.read_text())
    ta = tb = cal.get("threshold", cal["threshold_ab"])
    if cal["model"] != a.model:
        print(f"! 阈值是用 {cal['model']} 拟合的，现在跑的是 {a.model}", file=sys.stderr)
    print(f"阈值 A⊨B {ta:.4f}  B⊨A {tb:.4f}  (来自 {cfile.name}，"
          f"policy={cal.get('policy')}，floor={cal.get('margin_floor', '未记录')})")
    if tb < 0 or ta < 0:
        print("! 有一个方向的阈值为负，那一遍等于恒真：两遍法退化成单向判断",
              file=sys.stderr)

    files = sorted(f for d in a.indir for f in d.glob(f"*{a.suffix}"))
    if not files:
        raise SystemExit(
            "这些目录下没有 *" + a.suffix + "：" +
            ", ".join(str(d) for d in a.indir))
    if len(a.indir) > 1:
        print(f"{len(a.indir)} 个目录，共 {len(files)} 个文件")
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
            rels.append(Relation(
                a=mentions[i].mention_id, b=mentions[j].mention_id,
                relation=kind, confidence=float(sim),
                rationale=f"f(a,b)={x:.3f} f(b,a)={y:.3f} @{ta:.3f}",
                properties={"margin_ab": x, "margin_ba": y, "threshold": ta,
                            "f_ab": fwd, "f_ba": rev, "blocker_cosine": float(sim)}))
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
    ap.add_argument("indir", type=Path, nargs="*",
                    help="一个或多个目录；perspectrum 的语料按拓扑分在两个目录里")
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
    ap.add_argument("--holdout", type=float, default=0.4,
                    help="留出比例。阈值只在训练那半拟合，报的数来自留出那半")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--strict-entail", action="store_true",
                    help="拟合 f 时不把 near_match 当作双向蕴含")
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
