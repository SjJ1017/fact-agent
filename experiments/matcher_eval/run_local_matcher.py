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
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "pairs.jsonl"

# Blocking triton's import stops the failing gcc build, but it also breaks any
# model whose code imports triton without catching ImportError -- phi-4 does.
# So it is opt-in, not the default: FF_BLOCK_TRITON=1 for a model that only
# fails inside the shim build.  The real fix is whatever `run.sh --triton`
# reports, since the traceback never says why gcc returned 1.
if os.environ.get("FF_BLOCK_TRITON"):
    import importlib.abc

    class _NoTriton(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "triton" or name.startswith("triton."):
                raise ImportError(f"{name} blocked by FF_BLOCK_TRITON=1")
            return None

    sys.meta_path.insert(0, _NoTriton())


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


# Which relations count as one fact.  The binary gold in the file is `near`;
# the others re-derive gold from the relation tag, so a policy change is a flag
# rather than a relabelling and a full re-measurement.
POLICY = {
    "strict": {"equivalent"},
    "near": {"equivalent", "near_match"},
    "entail": {"equivalent", "near_match", "entail_ab", "entail_ba"},
}


def load(limit, contested, difficulty=None, drop_pre_atomization=False,
         pairs=None, policy="near"):
    src = pairs or PAIRS
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if policy != "file":
        keep = POLICY[policy]
        for r in rows:
            rel = r.get("relation") or ("equivalent" if r["gold"] == "SAME"
                                        else "different")
            r["gold"] = "SAME" if rel in keep else "DIFFERENT"
    if contested == "drop":
        rows = [r for r in rows if not r["contested"]]
    elif contested == "only":
        rows = [r for r in rows if r["contested"]]
    if difficulty:
        want = set(difficulty)
        rows = [r for r in rows if r.get("difficulty") in want]
    if drop_pre_atomization:
        rows = [r for r in rows if not r.get("pre_atomization")]
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


def directional_gold(rows, near_as_entail: bool = True):
    """Each pair is two examples for one entailment classifier.

    The model is used as an NLI system: a single f(x, y) -> entails or not,
    applied once each way.  Equivalence is then f(a,b) and f(b,a), not a thing
    the model is asked about, so f's threshold belongs on entailment labels
    rather than on the equivalence label it induces.

    `near_match` covers pairs recorded as one fact despite not being strict
    both-ways entailment -- a dropped hedge, a compatible rounding.  Counting
    it as entailing in both directions keeps the induced equivalence class
    equal to the binary gold; excluding it makes f strictly about entailment
    and shrinks that class.
    """
    ab, ba = [], []
    for r in rows:
        rel = r.get("relation") or ("equivalent" if r["gold"] == "SAME"
                                    else "different")
        both = rel == "equivalent" or (near_as_entail and rel == "near_match")
        ab.append(both or rel == "entail_ab")
        ba.append(both or rel == "entail_ba")
    return ab, ba


def binary_scores(labels, margins, t):
    tp = sum(1 for g, m in zip(labels, margins) if g and m >= t)
    fp = sum(1 for g, m in zip(labels, margins) if not g and m >= t)
    fn = sum(1 for g, m in zip(labels, margins) if g and m < t)
    tn = len(labels) - tp - fp - fn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(n=len(labels), acc=(tp + tn) / len(labels) if labels else 0.0,
                precision=pr, recall=rc,
                f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)


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
    table("relation", ["equivalent", "near_match", "entail_ab", "entail_ba",
                       "different"], "关系")
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


def chat_prompt(tok, system: str, user: str) -> str:
    """Chat template with thinking suppressed where the model has it.

    Qwen3's default template ends at `assistant\n` and the model then emits
    `<think>` itself, so the first generated token is the opening of a
    reasoning block and the SAME/DIFFERENT logits at that position are noise.
    Measured: Qwen3-14B scored 0.505 accuracy, calling nearly everything SAME,
    including random pairs at cosine 0.2.  `enable_thinking=False` pre-fills an
    empty think block so the next token is the answer.  Models without the
    argument ignore it.
    """
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)


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

    prompts = [chat_prompt(tok, SYSTEM, USER.format(a=r["a"], b=r["b"]))
               for r in rows]

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

    prompts = [chat_prompt(tok, COT_SYSTEM, COT_USER.format(a=r["a"], b=r["b"]))
               for r in rows]

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


# Two directional entailment questions instead of one identity question.
#
# Each question is crisp on its own: "poodle" entails "dog", "dog" does not
# entail "poodle", and the model never has to know what this project counts as
# one fact.  That policy moves out of the prompt and into ENTAIL_POLICY below,
# where it can be changed per analysis instead of being tuned by rewording.
#
# The direction is also worth keeping.  A one-way pair is a generalisation, not
# a coincidence, and an edge that says "B is a weaker form of A" is a different
# thing in the fact graph from "B repeats A".
ENTAIL_SYSTEM = """\
You judge logical entailment between two statements.

Question: if A is true, must B also be true?

Answer YES only when B follows necessarily from A. Answer NO when B could be
false while A is true, when B adds information A does not contain, or when the
two concern different subjects.

Do not consider whether either statement is actually true, and do not consider
whether they are about the same topic. Judge entailment only.

Reply with exactly one word: YES or NO."""

ENTAIL_USER = "A: {a}\nB: {b}"

# What the two directions mean.  Both ways is equivalence, which is the
# definition this project uses for one fact.
ENTAIL_POLICY = {
    (True, True): "SAME",        # equivalent
    (True, False): "DIFFERENT",  # a is stronger; b is a generalisation of a
    (False, True): "DIFFERENT",  # b is stronger
    (False, False): "DIFFERENT",  # unrelated, or contradictory
}
ENTAIL_LABEL = {
    (True, True): "equivalent",
    (True, False): "a_entails_b",
    (False, True): "b_entails_a",
    (False, False): "neither",
}


def _yes_no_ids(tok):
    out = {"YES": set(), "NO": set()}
    for word in out:
        for v in (word, " " + word, word.capitalize(), word.lower(),
                  " " + word.capitalize(), " " + word.lower()):
            ids = tok.encode(v, add_special_tokens=False)
            if ids:
                out[word].add(ids[0])
    both = out["YES"] & out["NO"]
    return {k: sorted(v - both) for k, v in out.items()}


def run_entail(rows, model, batch, dtype="bfloat16", load_4bit=False,
               quiet=False, **_):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  载入权重…（首次会先下载）", flush=True)
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
    net = AutoModelForCausalLM.from_pretrained(model, **kw)
    net.eval()
    if torch.cuda.is_available():
        print(f"  显存占用 {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    ids = _yes_no_ids(tok)
    if not ids["YES"] or not ids["NO"]:
        raise SystemExit(f"{model}: 无法分离 YES / NO 的首 token")

    def margins(pairs, label):
        prompts = [chat_prompt(tok, ENTAIL_SYSTEM, ENTAIL_USER.format(a=x, b=y))
                   for x, y in pairs]
        out = []
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
            tick(len(out))
        return out

    ab = margins([(r["a"], r["b"]) for r in rows], "A⊨B")
    ba = margins([(r["b"], r["a"]) for r in rows], "B⊨A")
    return ab, ba


RUNNERS = {"biencoder": run_biencoder, "reranker": run_reranker,
           "llm": run_llm, "cot": run_cot, "entail": run_entail}


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
    ap.add_argument("--mode", default="logit",
                    choices=["logit", "oneword", "cot", "entail"],
                    help="logit=读首 token 概率(最快); oneword=解码一个词; "
                         "cot=先写一句差别再判; entail=两遍单向蕴含，"
                         "等价性由 ENTAIL_POLICY 决定而不是靠 prompt 措辞")
    ap.add_argument("--strict-entail", action="store_true",
                    help="拟合 f 时不把 near_match 当作双向蕴含")
    ap.add_argument("--max-new-tokens", type=int, default=48,
                    help="cot 模式每对生成上限")
    ap.add_argument("--generate", action="store_true",
                    help="等同 --mode oneword（保留旧写法）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs", type=Path, default=None,
                    help="换一个 oracle 文件，默认 pairs.jsonl")
    ap.add_argument("--policy", default="file",
                    choices=["file", "strict", "near", "entail"],
                    help="哪些关系算同一条。file(默认)=原样用文件里的二值 "
                         "gold；strict=只要严格等价；near=再加近似匹配；"
                         "entail=再加单向蕴含。注意 near 并不等同文件里的 "
                         "gold —— relation 标注比二值标签细，两者有出入")
    ap.add_argument("--contested", default="keep", choices=["keep", "drop", "only"])
    ap.add_argument("--atomized-only", action="store_true",
                    help="去掉抽取器会拆开的那几条：管线不会产生那种输入")
    ap.add_argument("--difficulty", nargs="+",
                    choices=["trivial", "easy", "medium", "hard"],
                    help="只跑这些难度带，默认全部")
    ap.add_argument("--quiet", action="store_true", help="不打进度")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    kind = a.kind or detect_kind(a.model)
    mode = "oneword" if a.generate else a.mode
    if kind == "llm" and mode in ("cot", "entail"):
        kind = mode
    rows = load(a.limit, a.contested, a.difficulty, a.atomized_only,
                a.pairs, a.policy)
    t0 = time.time()
    label = f"{'llm' if kind == 'cot' else kind}"
    if kind in ("llm", "cot"):
        label += f"/{mode}"
    print(f"\n>>> {a.model}  [{label}]  {len(rows)} 对", flush=True)
    res = RUNNERS[kind](rows, a.model, a.batch, dtype=a.dtype,
                        load_4bit=a.load_4bit, generate=(mode == "oneword"),
                        quiet=a.quiet, max_new_tokens=a.max_new_tokens)
    secs = time.time() - t0

    notes = None
    if kind == "entail":
        ab, ba = res
        import collections
        import statistics as _st

        # One classifier, fitted once on the 2N directional labels, then
        # applied both ways.  Fitting it on the equivalence label instead would
        # tune f for a decision it does not make.
        g_ab, g_ba = directional_gold(rows, not a.strict_entail)
        labels = list(g_ab) + list(g_ba)
        marg = list(ab) + list(ba)
        lo, hi = min(marg), max(marg)
        grid = [lo + (hi - lo) * i / 200 for i in range(201)] if hi > lo else [lo]
        best_t, best_f1 = lo, -1.0
        for t in grid:
            f1 = binary_scores(labels, marg, t)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        fstat = binary_scores(labels, marg, best_t)

        cells = [(x >= best_t, y >= best_t) for x, y in zip(ab, ba)]
        preds = [ENTAIL_POLICY[c] for c in cells]
        notes = [ENTAIL_LABEL[c] for c in cells]
        best = score(rows, preds)
        best["threshold"] = round(best_t, 4)
        best["f_entail"] = {k: round(v, 4) for k, v in fstat.items()}
        curve = None

        gap = _st.mean(x - y for x, y in zip(ab, ba))
        print(f"\n  f 的阈值 {best_t:.3f}，在 {len(labels)} 个方向标注上拟合")
        print(f"  f 本身：准确 {fstat['acc']:.3f}  精确 {fstat['precision']:.3f}  "
              f"召回 {fstat['recall']:.3f}  F1 {fstat['f1']:.3f}")
        print(f"  位置偏差 mean(ab-ba) {gap:+.2f}")

        gold_cell = [ENTAIL_LABEL[(x, y)] for x, y in zip(g_ab, g_ba)]
        conf = collections.Counter(zip(gold_cell, notes))
        names = ["equivalent", "a_entails_b", "b_entails_a", "neither"]
        print(f"\n  四象限混淆（行=标注，列=预测）")
        print("  " + " " * 14 + "".join(f"{n[:11]:>13}" for n in names))
        for g in names:
            row = "".join(f"{conf.get((g, pnm), 0):>13}" for pnm in names)
            print(f"  {g:<14}{row}")
    elif kind == "cot":
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
    print(f"\n{a.model}   [{label}{', 4bit' if a.load_4bit else ''}]"
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
                print(f"      {'方向' if kind == 'entail' else '模型写的差别'}: "
                      f"{notes[k][:86]}")

    tag = "-".join(a.difficulty) if a.difficulty else "all"
    if a.pairs:
        tag = f"{a.pairs.stem.replace('pairs_', '')}.{tag}"
    if a.policy != "file":
        tag = f"{tag}.{a.policy}"
    if a.atomized_only:
        tag += "-atomized"
    out = a.out or (HERE / "results" /
                    f"{a.model.replace('/', '__')}.{mode}.{a.contested}.{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": a.model, "kind": shown, "contested": a.contested,
         "four_bit": a.load_4bit, "mode": mode, "policy": a.policy,
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
