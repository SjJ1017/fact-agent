"""Effective-structure analysis: budget, verification of position effects,
relay, verdicts.  All offline and deterministic; no LLM calls.

Reads the 72 deepseek-v4-flash stores (12 claims x 3 personas x full|star), the
stance/token labels, and the delivery graphs recorded in each debate file.

The first version of this analysis reported a "primacy/position effect": agent A
(= first speaker) appeared to seed ~2x as many final facts as C.  Verification
killed that claim, and this module implements the verified decomposition instead
of the original headline:

  * `obs_credit`  - "first utterer" under an alphabetical tie-break among
                    same-round staters.  This is the artifact we measured first.
  * `exp_credit`  - credit given uniformly among the agents that stated the
                    fact at its first round (no tie-break).  The gap between
                    obs and exp is pure tie-breaking.
  * `gen`         - did the agent state the fact at round 1 at all (parallel,
                    no visibility)?  Round-1 generation is symmetric.
  * `uniq`        - facts only one agent ever stated, by first round.
  * `adopt_r1uniq`- share of an agent's round-1-unique facts that another
                    agent later stated: position-symmetric.

Round 1 is parallel (prompts say "before seeing other panelists"), so there is
no such thing as a "first" speaker in round 1 - only a set of parallel staters.
Only round-2/3 utterances can be influenced, and only via a turn the receiving
agent actually saw (the recorded delivery graph).

Relay (star only) separates the same effect: a spoke's round-1 fact that the
other spoke also said at round 1 is *coincidence* (both read the dossier);
one the other spoke says only at round 2/3 is *possibly* relayed through hub A
(upper bound); one where A demonstrably preceded the other spoke is *via_a*.

Writes findings/data/effective-structure.json.

    python experiments/analyze_effective_structure.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "findings" / "data" / "effective-structure.json"

DIRS = {"full": "perspectrum_pilot_full", "star": "perspectrum_pilot_star_chain"}
CELLS = [f"{t}/{p}" for t in ("full", "star") for p in ("neutral", "lenses", "stance")]
AGENTS = ("A", "B", "C")
SEED = 20260901

NUM = re.compile(r"\d[\d,.]*(?:%|k|K|mg|μg|g|kg|ml|cm|mm|nm|\$|€|£|bn|m)?")
METRICS = ("n_facts", "novel_share", "adopted_share", "held_share", "reach",
           "nestedness", "n_eff", "echo", "r1_surv", "own_peak_attr",
           "numeric_ret", "spec3", "retain_adopted")


def boot(deltas: list[float], n: int = 20000) -> dict:
    rng = np.random.default_rng(SEED)
    d = np.array(deltas, float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    return {
        "delta": round(float(d.mean()), 5),
        "lo": round(float(lo), 5),
        "hi": round(float(hi), 5),
        "n": len(deltas),
        "positive": sum(1 for x in deltas if x > 0),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def load_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for topo, d in DIRS.items():
        for p in sorted(Path(d).glob(f"perspectrum-*-deepseek-v4-flash-{topo}-*.v2.json")):
            ex = p.name.rsplit(".v2.json", 1)[0]
            runs[ex] = {"store": json.loads(p.read_text()), "topo": topo}
        for ex in [e for e, r in runs.items() if r["topo"] == topo]:
            dbf = Path(d) / f"{ex}.debate.json"
            if dbf.exists():
                runs[ex]["debate"] = json.loads(dbf.read_text())
    return runs


def load_labels() -> tuple[dict, dict]:
    stance, tok = {}, {}
    for p in sorted((HERE / "labels" / "stance").glob("*.json")):
        lab = json.loads(p.read_text())
        stance[lab["execution_id"]] = {fid: v[0] for fid, v in lab["labels"].items()}
    for p in sorted((HERE / "labels" / "token_clock").glob("*.json")):
        t = json.loads(p.read_text())
        tok[t["execution_id"]] = t
    return stance, tok


def compute(ex: str, r: dict, stance_lab: dict, tok: dict) -> dict:
    s, d = r["store"], r.get("debate") or {}
    claim = ex.split("-")[1]
    topo, persona = r["topo"], ex.split("-")[-1]
    facts = s["facts"]

    out: dict[tuple, set] = defaultdict(set)
    staters: dict[str, set] = defaultdict(set)
    for mid, m in s["mentions"].items():
        p = m["provenance"]
        if p.get("channel") != "output" or not p.get("agent_id"):
            continue
        fid = s["mention_to_fact"].get(mid)
        if not fid:
            continue
        a, rr = p["agent_id"], p.get("round")
        out[(a, rr)].add(fid)
        staters[fid].add(a)
    dist = set(staters)

    delivered: dict[tuple, set] = {}
    for key, info in (d.get("delivery") or {}).items():
        a, rr = key.split("|")
        rr = int(rr)
        df: set = set()
        for pt in info.get("peer_turns", []):
            pa, prr = pt.split("|")
            df |= out.get((pa, int(prr)), set())
        delivered[(a, rr)] = df

    # ---- token budget decomposition ----
    novel = held = adopted = 0
    adopted_facts: set = set()
    T: Counter = Counter()
    for (a, rr), fs in out.items():
        prior: set = set()
        for r2 in range(1, rr):
            prior |= out.get((a, r2), set())
        dlv = delivered.get((a, rr), set())
        for fid in fs:
            if fid in prior:
                held += 1
            elif fid in dlv:
                adopted += 1
                adopted_facts.add(fid)
                for pa in AGENTS:
                    if pa != a and any(fid in out.get((pa, r3), set()) for r3 in range(1, rr)):
                        T[(pa, a)] += 1
            else:
                novel += 1
    tt = novel + held + adopted

    sets = {a: set().union(*[out.get((a, rr), set()) for rr in range(1, 4)]) for a in AGENTS}
    nest = [len(sets[x] & sets[y]) / min(len(sets[x]), len(sets[y]))
            for x in AGENTS for y in AGENTS if x < y and sets[x] and sets[y]]
    nestedness = st.mean(nest) if nest else 0.0

    J = np.zeros((3, 3))
    for i, x in enumerate(AGENTS):
        for j, y in enumerate(AGENTS):
            u = sets[x] | sets[y]
            J[i, j] = len(sets[x] & sets[y]) / len(u) if u else 0.0
    ev = np.linalg.eigvalsh(J)
    n_eff = (ev.sum() ** 2) / ((ev**2).sum() or 1.0)

    fround = {fid: min(rr for rr in (1, 2, 3) if any(fid in out.get((a, rr), set())
                                                     for a in AGENTS)) for fid in dist}
    multi = [f for f in dist if len(staters[f]) >= 2]
    echo = 0.0
    if multi:
        indep = [sum(1 for a in staters[f] if f in out.get((a, fround[f]), set()))
                 for f in multi]
        echo = sum(1 for k in indep if k == 1) / len(multi)
    reach = st.mean([len(staters[f]) for f in dist]) if dist else 0.0

    def nums(fid: str) -> set:
        return set(NUM.findall(facts[fid]["canonical_text"]))

    n_by_first: dict[int, set] = defaultdict(set)
    for f in dist:
        n_by_first[fround[f]] |= nums(f)
    numeric_ret = (len(n_by_first[1] & n_by_first[3]) / len(n_by_first[1])
                   if n_by_first[1] else 0.0)
    fs3 = set().union(*[out.get((a, 3), set()) for a in AGENTS])
    spec3 = sum(1 for f in fs3 if nums(f)) / len(fs3) if fs3 else 0.0

    alive = {rr: set().union(*[out.get((a, rr), set()) for a in AGENTS]) for rr in (1, 2, 3)}
    r1 = set(f for f in dist if fround[f] == 1)
    r1_surv = len(r1 & alive[3]) / len(r1) if r1 else 0.0
    peak = max(len(alive[rr]) for rr in (1, 2, 3))
    own_peak_attr = len(alive[3]) / peak if peak else 0.0
    retain_adopted = len(adopted_facts & alive[3]) / len(adopted_facts) if adopted_facts else 0.0

    final = alive[3]
    r1_staters = {fid: set(a for a in AGENTS if fid in out.get((a, 1), set())) for fid in final}
    nf = len(final)

    # ---- origin decomposition (no first-utterer claim; the verification) ----
    obs = {a: 0 for a in AGENTS}
    exp = {a: 0.0 for a in AGENTS}
    for fid in final:
        r0 = fround[fid]
        s0 = set(a for a in AGENTS if fid in out.get((a, r0), set()))
        if not s0:
            continue
        obs[min((r0, a) for a in s0)[1]] += 1      # alphabetical tie-break (the artifact)
        for a in s0:
            exp[a] += 1.0 / len(s0)                 # uniform among same-round staters
    gen = {a: sum(1 for fid in final if a in r1_staters[fid]) for a in AGENTS}
    uniq = {a: sum(1 for fid in final if staters[fid] == {a}) for a in AGENTS}
    uniq_by_round = {a: {rr: sum(1 for fid in final if staters[fid] == {a}
                                 and fround[fid] == rr) for rr in (1, 2, 3)} for a in AGENTS}
    adopt_r1uniq = {a: [0, 0] for a in AGENTS}
    for fid in final:
        if len(r1_staters[fid]) != 1:
            continue
        a0 = next(iter(r1_staters[fid]))
        adopt_r1uniq[a0][1] += 1
        if len(staters[fid]) > 1:
            adopt_r1uniq[a0][0] += 1
    final_first_round = {rr: len([f for f in final if fround[f] == rr]) / nf
                         for rr in (1, 2, 3)} if nf else {}

    # ---- relay (star): coincidence vs forced vs via A ----
    relay: dict[str, dict] = {}
    if topo == "star":
        r1_all = {a: set().union(*[out.get((a, 1), set())]) for a in AGENTS}
        for org, tgt in (("C", "B"), ("B", "C")):
            never = coincide = forced = via_a = 0
            for fid in r1_all[org]:
                tgt_rounds = [rr for rr in (1, 2, 3) if fid in out.get((tgt, rr), set())]
                if not tgt_rounds:
                    never += 1
                    continue
                if 1 in tgt_rounds:
                    coincide += 1
                    continue
                forced += 1
                first_tgt = min(tgt_rounds)
                if any(fid in out.get(("A", r3), set()) for r3 in range(1, first_tgt)):
                    via_a += 1
            relay[f"{org}_to_{tgt}"] = {"total": len(r1_all[org]),
                                        "never": never, "coincide": coincide,
                                        "forced": forced, "via_a": via_a}

    verdicts: dict[str, str] = {}
    for key, text in (d.get("transcript") or {}).items():
        m = re.search(r"VERDICT:\s*(\w+)", text or "")
        if m:
            verdicts[key] = m.group(1)

    prod: dict = {}
    if ex in tok:
        pos = {fid: ent.get("cumulative_visible_output_tokens", 0)
               for fid, ent in tok[ex]["facts"].items()}
        prod = {t0: sum(1 for f in dist if pos.get(f, 10**9) <= t0)
                for t0 in (500, 1000, 1500, 2000, 2500)}
        prod["total"] = max(pos.values()) if pos else 0

    return dict(
        claim=claim, topo=topo, persona=persona, execution=ex,
        n_facts=len(dist), novel_share=novel / max(1, tt),
        adopted_share=adopted / max(1, tt), held_share=held / max(1, tt),
        reach=reach, nestedness=nestedness, n_eff=n_eff, echo=echo,
        r1_surv=r1_surv, own_peak_attr=own_peak_attr,
        numeric_ret=numeric_ret, spec3=spec3, retain_adopted=retain_adopted,
        final=final, fround=fround, staters=staters, final_first_round=final_first_round,
        obs_credit={k: v / nf for k, v in obs.items()},
        exp_credit={k: v / nf for k, v in exp.items()},
        gen={k: v / nf for k, v in gen.items()},
        uniq_share={k: v / nf for k, v in uniq.items()},
        uniq_by_round=uniq_by_round,
        adopt_r1uniq={a: (adopt_r1uniq[a][0] / adopt_r1uniq[a][1]
                          if adopt_r1uniq[a][1] else None) for a in AGENTS},
        relay=relay, T=dict(T), verdicts=verdicts, prod=prod,
        labels=stance_lab.get(ex, {}), delivered=delivered, out=out,
    )


def main() -> int:
    runs = load_runs()
    if len(runs) != 72:
        print(f"expected 72 runs, found {len(runs)}", file=__import__("sys").stderr)
        return 1
    stance_lab, tok = load_labels()
    R = {ex: compute(ex, r, stance_lab, tok) for ex, r in runs.items()}

    # ---- cells ----
    cells: dict[str, dict] = {}
    for cell in CELLS:
        t, p = cell.split("/")
        vals = [r for r in R.values() if r["topo"] == t and r["persona"] == p]
        row: dict = {m: round(st.mean([r[m] for r in vals]), 5) for m in METRICS}
        row["final_first_round"] = {k: round(st.mean([r["final_first_round"][k]
                                                      for r in vals]), 5)
                                    for k in (1, 2, 3)}
        row["obs_credit"] = {a: round(st.mean([r["obs_credit"][a] for r in vals]), 5)
                             for a in AGENTS}
        row["exp_credit"] = {a: round(st.mean([r["exp_credit"][a] for r in vals]), 5)
                             for a in AGENTS}
        row["gen"] = {a: round(st.mean([r["gen"][a] for r in vals]), 5) for a in AGENTS}
        row["uniq_share"] = {a: round(st.mean([r["uniq_share"][a] for r in vals]), 5)
                             for a in AGENTS}
        row["uniq_by_round"] = {a: {rr: round(st.mean([r["uniq_by_round"][a][rr]
                                                       for r in vals]), 3)
                                    for rr in (1, 2, 3)} for a in AGENTS}
        row["adopt_r1uniq"] = {a: round(st.mean([x for r in vals
                                                 if (x := r["adopt_r1uniq"][a]) is not None]), 3)
                              for a in AGENTS}
        row["n"] = len(vals)
        unan = r1same = nv = 0
        maj: Counter = Counter()
        for r in vals:
            r3 = [r["verdicts"].get(f"{a}|3") for a in AGENTS]
            r3 = [v for v in r3 if v]
            if len(r3) < 2:
                continue
            nv += 1
            if len(set(r3)) == 1:
                unan += 1
            m3 = Counter(r3).most_common(1)[0][0]
            maj[m3] += 1
            r1 = [r["verdicts"].get(f"{a}|1") for a in AGENTS]
            r1 = [v for v in r1 if v]
            if r1 and Counter(r1).most_common(1)[0][0] == m3:
                r1same += 1
        row["verdict"] = {"runs": nv, "unanimous": unan, "r1same": r1same,
                          "majority": dict(maj)}
        cells[cell] = row

    # ---- paired persona ----
    paired_persona: dict[str, dict] = {}
    for topo in ("full", "star"):
        by_claim: dict[str, dict] = defaultdict(dict)
        for r in R.values():
            if r["topo"] == topo:
                by_claim[r["claim"]][r["persona"]] = r
        for panel in ("lenses", "stance"):
            for m in METRICS:
                deltas = [by_claim[c][panel][m] - by_claim[c]["neutral"][m]
                          for c in by_claim if panel in by_claim[c]
                          and "neutral" in by_claim[c]]
                paired_persona[f"{topo}|{panel}-neutral|{m}"] = boot(deltas)

    # ---- paired topology ----
    paired_topo: dict[str, dict] = {}
    by_key: dict[tuple, dict] = {}
    for r in R.values():
        by_key.setdefault((r["claim"], r["persona"]), {})[r["topo"]] = r
    for m in METRICS:
        deltas = [by_key[k]["star"][m] - by_key[k]["full"][m]
                  for k in by_key if len(by_key[k]) == 2]
        paired_topo[m] = boot(deltas)

    # ---- relay aggregate ----
    relay: dict[str, dict] = {}
    for persona in ("neutral", "lenses", "stance"):
        agg: dict[str, dict] = {}
        for k in ("C_to_B", "B_to_C"):
            tot = sum(r["relay"][k]["total"] for r in R.values()
                      if r["topo"] == "star" and r["persona"] == persona and r["relay"])
            s = {kk: sum(r["relay"][k][kk] for r in R.values()
                         if r["topo"] == "star" and r["persona"] == persona and r["relay"])
                 for kk in ("never", "coincide", "forced", "via_a")}
            s["total"] = tot
            agg[k] = s
        relay[persona] = agg

    # ---- stance-conditioned adoption ----
    side = {"A": "SUPPORT", "B": "UNDERMINE", "C": "NEUTRAL"}
    stance_adoption: dict[str, dict] = {}
    for topo in ("full", "star"):
        per = {a: {"n": 0, "same": 0, "ns": 0, "nu": 0} for a in AGENTS}
        for r in R.values():
            if r["topo"] != topo or r["persona"] != "stance":
                continue
            lab = r["labels"]
            for (a, rr), fs in r["out"].items():
                if rr == 1:
                    continue
                dlv = r["delivered"].get((a, rr), set())
                prior = set().union(*[r["out"].get((a, r2), set()) for r2 in range(1, rr)])
                for fid in fs:
                    lv = lab.get(fid)
                    if lv not in ("SUPPORT", "UNDERMINE"):
                        continue
                    if fid in prior or fid not in dlv:
                        continue
                    per[a]["n"] += 1
                    if lv == side[a]:
                        per[a]["same"] += 1
            for (a, rr), dlv in r["delivered"].items():
                for fid in dlv:
                    lv = lab.get(fid)
                    if lv == "SUPPORT":
                        per[a]["ns"] += 1
                    elif lv == "UNDERMINE":
                        per[a]["nu"] += 1
        stance_adoption[topo] = {}
        for a in AGENTS:
            ns, nu = per[a]["ns"], per[a]["nu"]
            stance_adoption[topo][a] = {
                "side": side[a], "n": per[a]["n"],
                "same": (per[a]["same"] / per[a]["n"]) if per[a]["n"] else None,
                "base_support": (ns / (ns + nu)) if ns + nu else None,
                "expected_same": (ns / (ns + nu)) if side[a] == "SUPPORT"
                                 else ((nu / (ns + nu)) if ns + nu else None),
            }

    # ---- transmission ----
    transmission: dict[str, dict] = {}
    for cell in CELLS:
        t, p = cell.split("/")
        vals = [r for r in R.values() if r["topo"] == t and r["persona"] == p]
        agg: Counter = Counter()
        n = len(vals)
        for r in vals:
            for (frm, to), c in r["T"].items():
                agg[f"{frm}->{to}"] += c
        transmission[cell] = {k: round(v / n, 4) for k, v in agg.items()}

    # ---- eta^2 ----
    eta2: dict[str, float] = {}
    rows = list(R.values())
    for m in METRICS:
        gm = st.mean([r[m] for r in rows])
        sst = sum((r[m] - gm) ** 2 for r in rows)
        if not sst:
            eta2[m] = 0.0
            continue
        ssb = 0.0
        for cell in CELLS:
            t, p = cell.split("/")
            vals = [r[m] for r in rows if r["topo"] == t and r["persona"] == p]
            ssb += len(vals) * (st.mean(vals) - gm) ** 2
        eta2[m] = round(ssb / sst, 4)

    per_debate = {
        f"{r['topo']}|{r['claim']}|{r['persona']}": {
            "n_facts": r["n_facts"], "novel_share": r["novel_share"],
            "adopted_share": r["adopted_share"], "held_share": r["held_share"],
            "reach": r["reach"], "nestedness": r["nestedness"], "n_eff": r["n_eff"],
            "echo": r["echo"], "r1_surv": r["r1_surv"],
            "obs_credit_A": r["obs_credit"]["A"], "obs_credit_C": r["obs_credit"]["C"],
            "exp_credit_A": r["exp_credit"]["A"], "exp_credit_C": r["exp_credit"]["C"],
            "gen_A": r["gen"]["A"], "gen_C": r["gen"]["C"],
            "uniq_A": r["uniq_share"]["A"], "uniq_C": r["uniq_share"]["C"],
            "final_r1": r["final_first_round"][1], "final_r3": r["final_first_round"][3],
        }
        for r in R.values()
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": "2026-09-01", "model": "deepseek-v4-flash",
        "n_runs": len(R), "n_claims": 12,
        "cells": cells, "paired_persona": paired_persona,
        "paired_topo": {m: {"delta": v["delta"], "lo": v["lo"], "hi": v["hi"],
                            "n": v["n"], "positive": v["positive"],
                            "excludes_zero": v["excludes_zero"]}
                        for m, v in paired_topo.items()},
        "relay": relay, "stance_adoption": stance_adoption,
        "transmission": transmission, "eta2": eta2, "per_debate": per_debate,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())