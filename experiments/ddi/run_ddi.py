"""Run the DDI join experiment.

Four conditions. The comparison between them, not any single accuracy, is the
experiment:

    solo-both     one agent, both dossiers.  Upper bound: no join needed, the
                  two mechanism facts are already in one context.
    solo-half     one agent, ONE dossier.    Prior-knowledge floor: whatever it
                  scores here it knew without being told.
    split         A gets drug A, B gets drug B, they discuss.  The real test --
                  the answer is only derivable if the mechanism facts meet.
    broadcast     both agents get both dossiers.  Control for "two agents" as
                  distinct from "distributed evidence".

`split` minus `solo-half` is the information the collaboration actually
contributed. `solo-both` minus `split` is what the split costs. Neither is
visible from a single accuracy number, and neither says whether a correct answer
came from joining evidence or from the model already knowing the pair -- that
question needs the trace, and analyze_ddi.py answers it.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cases import DDICase, load_cases

from factflow import LLM

OUT = Path(__file__).parent / "out"
CONDITIONS = ("solo-both", "solo-half", "split", "broadcast")

SYSTEM = """\
You are assessing whether two co-prescribed drugs interact.

Reason only from the drug information you have been given plus what you can \
derive from it. Name the specific enzyme or mechanism you are relying on. If \
the information you hold is not sufficient to decide, say so explicitly rather \
than guessing.

Keep your answer under 130 words. End with exactly two lines:
MECHANISM: <the enzyme and direction, or "insufficient information">
FINAL ANSWER: <YES or NO>"""

ASK = """\
Question: when {a} and {b} are taken together, is there a clinically \
significant pharmacokinetic interaction between them?

Drug information available to you:

{dossiers}"""

PEER = """\
Question: when {a} and {b} are taken together, is there a clinically \
significant pharmacokinetic interaction between them?

Drug information available to you:

{dossiers}

{peer_header}

{peers}

Consider what your colleague reported, then give your own updated assessment."""


@dataclass
class DDIRun:
    case_id: str
    condition: str
    drug_a: str
    drug_b: str
    gold: bool
    enzyme: str
    critical_pair: tuple[str, str]
    transcript: dict[str, str] = field(default_factory=dict)
    final: dict[str, str] = field(default_factory=dict)
    mechanism: dict[str, str] = field(default_factory=dict)
    verdict: str = ""
    correct: bool = False
    named_enzyme: bool = False


def _parse(text: str) -> tuple[str, str]:
    ans, mech = "", ""
    for line in (text or "").splitlines():
        u = line.strip().upper()
        if u.startswith("FINAL ANSWER:"):
            ans = "YES" if "YES" in u.split(":", 1)[1] else ("NO" if "NO" in u.split(":", 1)[1] else "")
        elif u.startswith("MECHANISM:"):
            mech = line.split(":", 1)[1].strip()
    return ans, mech


class EmptyResponse(RuntimeError):
    """A reasoning model returned no content.

    Reasoning models spend the output budget on reasoning before emitting any
    text, so a cap sized for the visible answer yields an empty string with no
    error. That is indistinguishable from a refusal downstream, and it scales
    with prompt length -- which means the condition with the longest prompts
    looks like the one that reasons worst. A first run of this experiment
    reported 6% for the split condition on exactly that artifact.
    """


def _require_text(text: str, agent: str, rnd: int) -> str:
    if not (text or "").strip():
        raise EmptyResponse(
            f"agent {agent} round {rnd} returned no content - raise --max-tokens "
            f"(reasoning models need headroom well beyond the visible answer)")
    return text


def run_case(llm: LLM, case: DDICase, condition: str, rounds: int = 2, seed: int = 0,
             max_tokens: int = 6000) -> DDIRun:
    a, b = case.drug_a, case.drug_b
    doss_a = a.dossier(shuffle_seed=seed)
    doss_b = b.dossier(shuffle_seed=seed + 1)
    both = f"{doss_a}\n\n{doss_b}"

    run = DDIRun(case.case_id, condition, a.name, b.name, case.interacts,
                 case.enzyme, case.critical_pair)

    def ask(agent: str, dossiers: str) -> str:
        return _require_text(llm.chat(
            system=SYSTEM, user=ASK.format(a=a.name, b=b.name, dossiers=dossiers),
            temperature=0.3, max_tokens=max_tokens), agent, 1)

    if condition == "solo-both":
        run.transcript["A|1"] = ask("A", both)
    elif condition == "solo-half":
        # Only drug A's dossier: anything correct here was already known.
        run.transcript["A|1"] = ask("A", doss_a)
    else:
        view = {"A": doss_a, "B": doss_b} if condition == "split" else {"A": both, "B": both}
        for rnd in range(1, rounds + 1):
            if rnd == 1:
                out = llm.map(lambda ag: (ag, ask(ag, view[ag])), ["A", "B"],
                              tolerate_failures=False)
            else:
                def later(ag):
                    other = "B" if ag == "A" else "A"
                    prev = run.transcript.get(f"{other}|{rnd - 1}", "")
                    return ag, llm.chat(
                        system=SYSTEM,
                        user=PEER.format(a=a.name, b=b.name, dossiers=view[ag],
                                         peer_header=f"Your colleague reviewing the other drug reported:",
                                         peers=prev),
                        temperature=0.3, max_tokens=max_tokens)

                out = [(ag, _require_text(t, ag, rnd)) for ag, t in
                       llm.map(later, ["A", "B"], tolerate_failures=False)]
            for ag, text in out:
                run.transcript[f"{ag}|{rnd}"] = text

    last = max((k for k in run.transcript), key=lambda k: (int(k.split("|")[1]), k))
    agents = sorted({k.split("|")[0] for k in run.transcript})
    final_round = max(int(k.split("|")[1]) for k in run.transcript)
    for ag in agents:
        txt = run.transcript.get(f"{ag}|{final_round}", "")
        ans, mech = _parse(txt)
        run.final[ag] = ans
        run.mechanism[ag] = mech
    votes = [v for v in run.final.values() if v]
    run.verdict = max(set(votes), key=votes.count) if votes else ""
    run.correct = (run.verdict == "YES") == case.interacts and bool(run.verdict)
    # Did anyone name the enzyme that carries the mechanism?
    if case.enzyme != "none shared":
        blob = " ".join(run.mechanism.values()).upper()
        run.named_enzyme = case.enzyme.upper() in blob
    return run


def to_trace(run: DDIRun, case: DDICase, execution_id: str) -> list[dict]:
    """Trace records. Every dossier fact is a SOURCE fact, tagged for relevance."""
    recs = []
    for drug in (case.drug_a, case.drug_b):
        for fact in drug.all_facts():
            recs.append({
                "text": fact,
                "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                               "channel": "source", "doc_id": drug.name,
                               "extra": {"gold": fact == drug.mechanism,
                                         "critical": fact in case.critical_pair}},
            })
    for key, text in sorted(run.transcript.items(), key=lambda kv: (kv[0].split("|")[1], kv[0])):
        ag, rnd = key.split("|")
        recs.append({
            "text": text,
            "provenance": {"execution_id": execution_id, "agent_id": ag, "round": int(rnd),
                           "channel": "output", "doc_id": run.case_id},
        })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="opencode")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1, help="runs per case, for variance")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="output budget. Reasoning models consume most of it before "
                         "emitting text; too low yields silent empty responses.")
    ap.add_argument("--only", default=None, help="'positive', 'negative', or a case id")
    ap.add_argument("--trace", action="store_true", help="also extract+match facts")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    mk = {"opencode": LLM.opencode, "openai": LLM.openai, "deepseek": LLM.deepseek}[args.provider]
    llm = mk(args.model, max_concurrency=args.parallel * 2)
    llm.backend.client = llm.backend.client.with_options(timeout=args.timeout, max_retries=1)

    cases = load_cases(args.only)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"{len(cases)} cases x {len(conditions)} conditions x {args.repeats} repeats "
          f"on {args.model}\n", flush=True)

    summary = []
    for cond in conditions:
        jobs = [(c, r) for c in cases for r in range(args.repeats)]

        def one(job):
            case, rep = job
            exec_id = f"{case.case_id}-{cond}" + (f"-r{rep}" if args.repeats > 1 else "")
            try:
                run = run_case(llm, case, cond, rounds=args.rounds, seed=rep,
                               max_tokens=args.max_tokens)
            except Exception as exc:  # noqa: BLE001
                print(f"  {exec_id}: SKIPPED {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                return None
            (OUT / f"{exec_id}.run.json").write_text(
                json.dumps({**asdict(run), "exec_id": exec_id}, indent=2, ensure_ascii=False))
            (OUT / f"{exec_id}.trace.json").write_text(
                json.dumps(to_trace(run, case, exec_id), indent=2, ensure_ascii=False))
            return run

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            runs = [r for r in pool.map(one, jobs) if r is not None]

        blank = sum(1 for r in runs if not r.verdict)
        if blank:
            print(f"  WARNING: {blank}/{len(runs)} runs produced no parsable verdict", flush=True)
        n = len(runs)
        if not n:
            print(f"{cond}: no runs completed\n", flush=True)
            continue
        acc = sum(r.correct for r in runs) / n
        pos = [r for r in runs if r.gold]
        neg = [r for r in runs if not r.gold]
        named = sum(r.named_enzyme for r in pos) / max(len(pos), 1)
        print(f"{cond:<12} acc {acc:>5.0%}  ({n} runs)   "
              f"positives {sum(r.correct for r in pos)}/{len(pos)}  "
              f"negatives {sum(r.correct for r in neg)}/{len(neg)}  "
              f"named the enzyme {named:>4.0%}", flush=True)
        summary.append({"condition": cond, "n": n, "accuracy": acc,
                        "pos_correct": sum(r.correct for r in pos), "pos_n": len(pos),
                        "neg_correct": sum(r.correct for r in neg), "neg_n": len(neg),
                        "named_enzyme_rate": named})

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    if llm.failures:
        print(f"\nWARNING: {len(llm.failures)} LLM calls failed; first: {llm.failures[0][:140]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
