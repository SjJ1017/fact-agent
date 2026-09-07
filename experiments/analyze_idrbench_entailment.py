"""Offline census and delivery-aware directional analysis of the 40 IDRBench traces.

Run from any directory: python3 experiments/analyze_idrbench_entailment.py
No model calls, re-extraction, relabeling of stores, or equivalence propagation.
Relations are measured on their scored mention endpoints. Cluster IDs are used
only for deduplication and conservative diagnostics, never to invent scored edges.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/idrbench_generation_10x5_r3"
OUT = ROOT / "findings/data/idrbench-entailment"
CELLS = ("full-generic", "full-specialist", "split-generic", "split-specialist-aligned")
LABELS = ("EQUIVALENT", "A_ENTAILS_B", "B_ENTAILS_A", "UNRELATED")
SEED = 20260906


def label(ab, ba, threshold):
    f, r = ab >= threshold, ba >= threshold
    return "EQUIVALENT" if f and r else "A_ENTAILS_B" if f else "B_ENTAILS_A" if r else "UNRELATED"


def slot(m):
    p = m["provenance"]
    return f"{p['agent_id']}|{p['round']}" if p["channel"] == "output" else None


def temporal_edge(a, b, relation, delivery):
    """Semantic orientation is independent of chronology and explicit visibility."""
    if slot(a) is None or slot(b) is None:
        return "source_involved", None, None
    pa, pb = a["provenance"], b["provenance"]
    if pa["round"] == pb["round"]:
        return "parallel", None, None
    early, late = (a, b) if pa["round"] < pb["round"] else (b, a)
    scope = "self" if early["provenance"]["agent_id"] == late["provenance"]["agent_id"] else "peer"
    if slot(early) not in delivery[slot(late)][f"visible_{scope}_turns"]:
        return "invisible", early, late
    if relation == "EQUIVALENT":
        kind = "equivalent"
    elif relation == "UNRELATED":
        kind = "unrelated"
    else:
        strong = a if relation == "A_ENTAILS_B" else b
        kind = "weaken" if early is strong else "strengthen"
    return f"{scope}_{kind}", early, late


def ratio(n, d):
    return n / d if d else None


def dist(xs):
    xs = sorted(xs)
    def q(p):
        i = (len(xs) - 1) * p
        lo = int(i)
        return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)
    return dict(n=len(xs), mean=st.mean(xs), sd=st.stdev(xs) if len(xs) > 1 else 0,
                min=xs[0], q25=q(.25), median=q(.5), q75=q(.75), max=xs[-1])


def paired(deltas):
    rng = random.Random(SEED)
    means = sorted(st.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    obs = abs(st.mean(deltas))
    p = sum(abs(st.mean(s * d for s, d in zip(signs, deltas))) >= obs - 1e-12
            for signs in itertools.product((-1, 1), repeat=len(deltas))) / 2 ** len(deltas)
    return dict(n=len(deltas), mean=st.mean(deltas), ci95=[means[249], means[9749]],
                positive=sum(d > 0 for d in deltas), negative=sum(d < 0 for d in deltas),
                exact_signflip_p=p, deltas=deltas)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def analyze(store, debate, threshold=None):
    mentions, m2f, delivery = store["mentions"], store["mention_to_fact"], debate["delivery"]
    t = store["matching"]["threshold_ab"] if threshold is None else threshold
    assert store["matching"]["threshold_ab"] == store["matching"]["threshold_ba"]
    out, sources = defaultdict(set), defaultdict(set)
    target_mentions = defaultdict(set)
    for mid, m in mentions.items():
        if slot(m):
            out[slot(m)].add(m2f[mid])
            target_mentions[slot(m)].add(mid)
        elif m["provenance"]["channel"] == "source":
            sources[m["provenance"]["doc_id"]].add(m2f[mid])
    contexts = {}
    for key, info in delivery.items():
        assert key in debate["transcript"]
        assert set(info["peer_turns"]) <= set(info["visible_peer_turns"])
        peers = set().union(*(out[p] for p in info["visible_peer_turns"]))
        own = set().union(*(out[p] for p in info["visible_self_turns"]))
        src = set().union(*(sources[p] for p in info["source_ids"]))
        contexts[key] = peers, own, src
    raw, temporal, within = Counter(), Counter(), Counter()
    sets = defaultdict(set)
    target_flags = defaultdict(set)
    events = []
    seen = set()
    for i, r in enumerate(store["relations"]):
        assert r["a"] != r["b"]
        pair = tuple(sorted((r["a"], r["b"])))
        assert pair not in seen, "duplicate scored mention pair"
        seen.add(pair)
        prop = r["properties"]
        k = label(prop["margin_ab"], prop["margin_ba"], t)
        if threshold is None:
            assert k == r["relation"]
            assert prop["threshold"] == t
        raw[k] += 1
        if m2f[r["a"]] == m2f[r["b"]]:
            within[k] += 1
        category, early, late = temporal_edge(mentions[r["a"]], mentions[r["b"]], k, delivery)
        if "ENTAILS" in k:
            temporal[category] += 1
        if early is None or category in ("invisible", "source_involved"):
            continue
        if k == "UNRELATED":
            continue
        em, lm = early["mention_id"], late["mention_id"]
        ef, lf = m2f[em], m2f[lm]
        ek, lk = slot(early), slot(late)
        target_flags[lm].add(category)
        sets[category + "_fact_pairs"].add((ef, lf))
        sets[category + "_text_pairs"].add((early["text"], early["polarity"], late["text"], late["polarity"]))
        sets[category + "_target_facts"].add((lk, lf))
        if category != "peer_weaken":
            continue
        peers, own, src = contexts[lk]
        info = delivery[lk]
        flags = dict(same_cluster=ef == lf, weak_in_peer=lf in peers,
                     weak_in_self=lf in own, weak_in_source=lf in src,
                     strong_in_self=ef in own, strong_in_source=ef in src,
                     strong_retained=ef in out[lk], latest_delivery=ek in info["peer_turns"])
        flags["new_weak"] = not (flags["weak_in_peer"] or flags["weak_in_self"] or flags["weak_in_source"])
        flags["conservative_candidate"] = flags["new_weak"] and not any(flags[x] for x in
            ("same_cluster", "strong_in_self", "strong_in_source", "strong_retained"))
        events.append(dict(execution_id=debate["execution_id"], case_id=debate["case_id"],
                           condition=debate["condition"], relation_index=i, early_mid=em,
                           late_mid=lm, early_fact=ef, late_fact=lf, early_slot=ek, late_slot=lk,
                           strong_text=early["text"], weak_text=late["text"],
                           strong_quote=early.get("quote"), weak_quote=late.get("quote"),
                           margin_ab=prop["margin_ab"], margin_ba=prop["margin_ba"], **flags))
    if threshold is None:
        assert raw == Counter(store["matching"]["counts"])
    row = dict(execution_id=debate["execution_id"], case_id=debate["case_id"],
               condition=debate["condition"], topology=debate["topology"], memory=debate["memory"],
               mentions=len(mentions), facts=len(store["facts"]), candidates=sum(raw.values()),
               **{k: raw[k] for k in LABELS})
    row["oneway"] = raw["A_ENTAILS_B"] + raw["B_ENTAILS_A"]
    row["oneway_share_related"] = ratio(row["oneway"], row["oneway"] + raw["EQUIVALENT"])
    row.update({"oneway_" + k: temporal[k] for k in
                ("source_involved", "parallel", "invisible", "peer_weaken", "peer_strengthen", "self_weaken", "self_strengthen")})
    row["oneway_same_cluster"] = within["A_ENTAILS_B"] + within["B_ENTAILS_A"]
    row["unrelated_same_cluster"] = within["UNRELATED"]
    row.update({k: len(v) for k, v in sets.items()})
    row["peer_weaken_distinct_cluster_pairs"] = len({(e["early_fact"], e["late_fact"]) for e in events if not e["same_cluster"]})
    for f in ("same_cluster", "latest_delivery", "new_weak", "conservative_candidate", "strong_retained", "weak_in_self", "weak_in_peer", "weak_in_source"):
        ev = [e for e in events if e[f]]
        row["weaken_" + f + "_edges"] = len(ev)
        row["weaken_" + f + "_target_facts"] = len({(e["late_slot"], e["late_fact"]) for e in ev})
    turns = []
    for key in sorted(out):
        agent, rr = key.split("|")
        flags = [target_flags[m] for m in target_mentions[key]]
        c = dict(execution_id=debate["execution_id"], case_id=debate["case_id"],
                 condition=debate["condition"], agent=agent, round=int(rr), output_mentions=len(flags),
                 output_facts=len(out[key]), empty=int(not flags))
        for name in ("peer_weaken", "peer_strengthen", "peer_equivalent", "self_weaken", "self_strengthen"):
            c[name + "_targets"] = sum(name in f for f in flags)
        c["peer_related_targets"] = sum(bool(f & {"peer_weaken", "peer_strengthen", "peer_equivalent"}) for f in flags)
        c["peer_weaken_without_peer_equivalence_targets"] = sum("peer_weaken" in f and "peer_equivalent" not in f for f in flags)
        c["conservative_target_facts"] = len({e["late_fact"] for e in events if e["late_slot"] == key and e["conservative_candidate"]})
        turns.append(c)
    later = [x for x in turns if x["round"] > 1]
    for name in ("output_mentions", "peer_weaken_targets", "peer_strengthen_targets", "peer_related_targets", "peer_weaken_without_peer_equivalence_targets"):
        row["later_" + name] = sum(x[name] for x in later)
    row["weaken_target_rate"] = ratio(row["later_peer_weaken_targets"], row["later_output_mentions"])
    row["strengthen_target_rate"] = ratio(row["later_peer_strengthen_targets"], row["later_output_mentions"])
    row["conservative_target_rate"] = ratio(row["weaken_conservative_candidate_target_facts"], sum(x["output_facts"] for x in later))
    # Paths use exact scored mention endpoints: no assumed links through clusters.
    successors = defaultdict(list)
    for e in events:
        successors[e["early_mid"]].append(e)
    chains = [(e, f) for e in events for f in successors[e["late_mid"]]]
    row["two_step_weaken_paths"] = len(chains)
    row["two_step_distinct_fact_paths"] = len({(e["early_fact"], e["late_fact"], f["late_fact"])
        for e, f in chains if len({e["early_fact"], e["late_fact"], f["late_fact"]}) == 3})
    return row, turns, events, chains


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    files = sorted(args.input.glob("*.nli.store.json"))
    assert len(files) == 40, f"Expected complete 40-trace corpus, found {len(files)}"
    rows, turns, events, audits, chain_examples, manifest, sensitivity = [], [], [], [], [], [], []
    configs = Counter()
    for path in files:
        base = path.name.removesuffix(".nli.store.json")
        dbpath = path.with_name(base + ".debate.json")
        atpath = path.with_name(base + ".atomized.json")
        s, d, atom = [json.loads(p.read_text()) for p in (path, dbpath, atpath)]
        assert s["mentions"] == atom["mentions"], "Store does not match frozen atomized input"
        assert len(d["transcript"]) == 9 and d["rounds"] == 3
        assert d["condition"] in CELLS and d["topology"] == "full" and d["memory"] == "cumulative"
        assert len(set(d["transcript"][f"{a}|1"] for a in "ABC")) == 3, "Identical round-one outputs"
        cfg = {k: v for k, v in s["matching"].items() if k not in ("seconds", "counts")}
        configs[json.dumps(cfg, sort_keys=True)] += 1
        manifest.append(dict(execution_id=base, files={str(p.relative_to(ROOT)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, dbpath, atpath)}))
        row, tr, ev, ch = analyze(s, d)
        rows.append(row); turns.extend(tr); events.extend(ev)
        # One fixed random event per trace, uniform over distinct scored text pairs.
        by_text = {}
        for e in sorted(ev, key=lambda x: (x["strong_text"], x["weak_text"], x["early_mid"], x["late_mid"])):
            by_text.setdefault((e["strong_text"], e["weak_text"]), e)
        rng = random.Random(f"{SEED}:{base}")
        selected = rng.choice(list(by_text.values()))
        audits.append(dict(audit_id=len(audits) + 1, **selected,
                           strong_turn=d["transcript"][selected["early_slot"]],
                           weak_turn=d["transcript"][selected["late_slot"]],
                           delivery=d["delivery"][selected["late_slot"]]))
        clean_ch = [(e, f) for e, f in ch if len({e["early_fact"], e["late_fact"], f["late_fact"]}) == 3]
        if clean_ch:
            e, f = clean_ch[0]
            chain_examples.append(dict(first=e, second=f))
        for t in (0.0, 2.64, 5.28, 7.92, 10.56):
            sr, _, _, _ = analyze(s, d, threshold=t)
            # Fixed-cluster diagnostics deliberately not used for threshold sensitivity.
            sensitivity.append({k: sr[k] for k in ("execution_id", "case_id", "condition", *LABELS,
                "oneway_share_related", "oneway_peer_weaken", "oneway_peer_strengthen", "weaken_target_rate", "strengthen_target_rate")} | {"threshold": t})
    assert len(configs) == 1, "Mixed matcher configurations"
    cases = sorted({r["case_id"] for r in rows})
    assert len(cases) == 10 and all(sum(r["case_id"] == q and r["condition"] == c for r in rows) == 1 for q in cases for c in CELLS)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "per-trace.csv", rows)
    write_csv(args.out / "per-turn.csv", turns)
    write_csv(args.out / "sensitivity.csv", sensitivity)
    with (args.out / "peer-weakening-events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (args.out / "audit-sample.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
    (args.out / "chain-examples.json").write_text(json.dumps(chain_examples, ensure_ascii=False, indent=2) + "\n")
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    effects = {}
    for metric in ("oneway_share_related", "weaken_target_rate", "strengthen_target_rate", "conservative_target_rate"):
        effects[metric] = {}
        for name, left, right in (("roles_full", "full-generic", "full-specialist"),
                                  ("roles_split", "split-generic", "split-specialist-aligned"),
                                  ("split_generic", "full-generic", "split-generic"),
                                  ("split_specialist", "full-specialist", "split-specialist-aligned")):
            effects[metric][name] = paired([lookup[q, right][metric] - lookup[q, left][metric] for q in cases])
        effects[metric]["interaction"] = paired([
            (lookup[q, "split-specialist-aligned"][metric] - lookup[q, "split-generic"][metric]) -
            (lookup[q, "full-specialist"][metric] - lookup[q, "full-generic"][metric]) for q in cases])
    totals = {k: sum(r.get(k, 0) for r in rows) for k, v in rows[0].items() if isinstance(v, int)}
    summary = dict(corpus=dict(traces=len(rows), cases=cases, unmatched_debates=sorted(
        p.name for p in args.input.glob("*.debate.json") if p.name.removesuffix(".debate.json") not in {r["execution_id"] for r in rows})),
        matcher=json.loads(next(iter(configs))), counts=totals,
        one_way_pooled=ratio(totals["oneway"], totals["oneway"] + totals["EQUIVALENT"]),
        one_way_trace_distribution=dist([r["oneway_share_related"] for r in rows]),
        cells={c: {"counts": {k: sum(r.get(k, 0) for r in rows if r["condition"] == c) for k in totals},
                   "mean_rates": {k: st.mean(r[k] for r in rows if r["condition"] == c) for k in effects}}
               for c in CELLS}, effects=effects,
        threshold_sensitivity={str(t): {k: sum(r[k] for r in sensitivity if r["threshold"] == t)
            for k in (*LABELS, "oneway_peer_weaken", "oneway_peer_strengthen")} for t in (0.0, 2.64, 5.28, 7.92, 10.56)},
        audit_design="One distinct scored text pair per trace, uniform deterministic selection; one endpoint event per pair. Agent review, not independent human gold.",
        calibration_provenance="Store records shared t=5.28 but no calibration file/hash/split; no entail_thresholds file present locally. In-sample historical accuracy is optimistic; current calibration code has holdout support, which does not prove these stores used it.",
        manifest=manifest)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("one_way_pooled", "one_way_trace_distribution", "counts")}, indent=2))


if __name__ == "__main__":
    main()
