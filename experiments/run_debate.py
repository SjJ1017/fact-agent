"""Run one dataset-task's debates from a YAML spec.

Generalises run_perspectrum.py without replacing it: that script produced the
published corpus and stays byte-frozen, and datasets/perspectrum.yaml exists
so this runner can be checked against it.

What is new here beyond being dataset-agnostic is that `source_ids` in the
delivery record is per agent -- the items that agent was actually dealt --
rather than the whole dossier.  While every agent holds everything, a fact
appearing in B's turn is never attributable to transmission (fix.md defect
five); once the deal is uneven, "B stated a figure only A was given" is an
observable directed edge, which is what a flow-and-graph reading needs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factflow.llm import LLM  # noqa: E402
from factflow.spec import TaskSpec, load_spec, render_context  # noqa: E402
from factflow.tasks import Case, Item, coverage, deal  # noqa: E402
from frameworks import neighbours  # noqa: E402
from loaders import load_cases  # noqa: E402
from run_perspectrum import load_opencode_key  # noqa: E402

# Defaults phrased for any panel; a dataset overrides them under task.turn.
# perspectrum.yaml overrides all three with its original wording, which is what
# keeps the frozen corpus reproducible from a config.
TURN = {
    "first": "Give your independent assessment before seeing the others.",
    "no_peers": ("You have already given an assessment above. No other participant's "
                 "statement is available to you.\n\nUpdate your assessment. Revise it "
                 "only when your own information warrants it."),
    "with_peers": ("These are the other participants' latest statements:\n\n{peers}\n\n"
                   "{own}Update your assessment. Address concrete claims from the others "
                   "and revise your position only when the information warrants it."),
    "peer_header": "--- Participant {agent} ---",
    "own_prefix": "Your own earlier assessment is above. ",
}


def turn_text(spec: TaskSpec, key: str) -> str:
    return (getattr(spec, "turn", None) or {}).get(key, TURN[key])


def build_prompt(spec: TaskSpec, case: Case, items: Sequence[Item],
                 peers: Sequence[tuple[str, str]], has_own_prior: bool) -> str:
    """The user turn for one agent in one round.

    Three shapes, so the instruction matches what is actually in context.
    Telling an agent to answer "before seeing the others" while its own
    previous answer sits in the conversation is a contradiction the model has
    to resolve on its own.
    """
    base = render_context(spec, case, items)
    if not peers:
        return base + "\n\n" + turn_text(spec, "first" if not has_own_prior else "no_peers")
    header = turn_text(spec, "peer_header")
    block = "\n\n".join(f"{header.format(agent=a)}\n{t}" for a, t in peers)
    own = turn_text(spec, "own_prefix") if has_own_prior else ""
    return base + "\n\n" + turn_text(spec, "with_peers").format(peers=block, own=own)


def history_prompt(spec: TaskSpec, case: Case, items: Sequence[Item]) -> str:
    """What sits in front of an agent's own previous answer under self-last.

    It carries the agent's context and nothing else. It must not be the real
    prior prompt, which would smuggle the round-before's peer output in with
    it, and it must not claim the answer was given before seeing the others,
    which is false from round 2 on.
    """
    return render_context(spec, case, items) + "\n\nGive your assessment."


def run_case(llm: LLM, spec: TaskSpec, case: Case, model: str, topology: str,
             condition: str, memory: str, max_agent_tokens: int,
             sample_tag: str, seed: int, progress: bool = False) -> dict[str, Any]:
    agents = list(spec.agents)
    persona = spec.persona_for(condition)
    roles = spec.roles(condition)
    disclosure = spec.disclosure_for(condition)
    dealt = deal(case, agents, disclosure, seed=seed)
    cov = coverage(dealt, case)

    transcript: dict[str, str] = {}
    prompts: dict[str, str] = {}
    threads: dict[str, list[tuple[str, str]]] = {a: [] for a in agents}
    delivery: dict[str, dict[str, Any]] = {}

    for rnd in range(1, spec.rounds + 1):
        if progress:
            print(f"    {case.id} / {condition} / round {rnd}/{spec.rounds}: "
                  f"requesting {len(agents)} agents", flush=True)
        requests: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        for agent in agents:
            peers = [(p, transcript[f"{p}|{rnd - 1}"])
                     for p in neighbours(topology, agents, agent)
                     if f"{p}|{rnd - 1}" in transcript]
            key, prior = f"{agent}|{rnd}", f"{agent}|{rnd - 1}"
            history: list[tuple[str, str]] = list(threads[agent])
            has_own_prior = memory != "peer-only" and prior in transcript

            user = build_prompt(spec, case, dealt[agent], peers, has_own_prior)
            prompts[key] = user
            if memory == "cumulative":
                threads[agent] = history + [("user", user)]
            elif memory == "self-last" and has_own_prior:
                history = [("user", history_prompt(spec, case, dealt[agent])),
                           ("assistant", transcript[prior])]
                threads[agent] = []

            fresh = [f"{p}|{rnd - 1}" for p, _ in peers]
            if memory == "cumulative":
                vis_peers = [t for r in range(1, rnd)
                             for t in (f"{n}|{r}" for n in neighbours(topology, agents, agent))
                             if t in transcript]
                vis_self = [f"{agent}|{r}" for r in range(1, rnd)
                            if f"{agent}|{r}" in transcript]
            else:
                vis_peers = list(fresh)
                vis_self = [prior] if has_own_prior else []

            delivery[key] = {
                # per agent, not the whole case: this is the difference that
                # makes transmission identifiable at all
                "source_ids": [i.id for i in dealt[agent]],
                "peer_turns": fresh,
                "self_turn": prior if has_own_prior else None,
                "visible_peer_turns": vis_peers,
                "visible_self_turns": vis_self,
            }
            requests[agent] = (user, history)

        def ask(agent: str) -> tuple[str, str]:
            user, history = requests[agent]
            system = roles[agent] + "\n\n" + spec.system
            base = (f"{spec.name}:{sample_tag}:{case.id}:{model}:{topology}:"
                    f"{condition}:{agent}:{rnd}")
            for attempt, cap in enumerate((max_agent_tokens, max_agent_tokens * 2,
                                           max_agent_tokens * 4), 1):
                text = llm.chat(system=system, user=user, temperature=0.7, max_tokens=cap,
                                history=history, sample_id=f"{base}:attempt={attempt}")
                if text.strip():
                    return agent, text
            raise RuntimeError(
                f"empty model response after retry for {model} case={case.id} "
                f"agent={agent} round={rnd}")

        for agent, text in llm.map(ask, agents, tolerate_failures=False):
            transcript[f"{agent}|{rnd}"] = text
            if memory == "cumulative":
                threads[agent] = threads[agent] + [("assistant", text)]
        if progress:
            print(f"    completed round {rnd}/{spec.rounds}; "
                  f"{llm.usage.report(model)}", flush=True)

    suffix = "" if memory == "peer-only" else f"-{memory}"
    return {
        "execution_id": f"{spec.name}-{case.id}-{model}-{topology}-{condition}{suffix}",
        "dataset": spec.name,
        "case_id": case.id,
        "claim": case.question,
        "public": case.public,
        "model": model,
        "topology": topology,
        # `panel` stays for old analysis code; new cross-dataset code should
        # read condition/persona, which do not conflate roles with disclosure.
        "panel": condition,
        "condition": condition,
        "persona": persona,
        "memory": memory,
        "roles": roles,
        "rounds": spec.rounds,
        "evidence": [{"id": i.id, "text": i.text, "stance": i.side,
                      "tags": list(i.tags)} for i in case.items],
        "disclosure": {**disclosure, **cov},
        "meta": {k: v for k, v in case.meta.items() if k != "held"},
        "transcript": transcript,
        "prompts": prompts,
        "delivery": delivery,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("-n", "--n-cases", type=int, default=1)
    ap.add_argument(
        "--case-filter",
        action="append",
        default=[],
        help=("shell-style pattern matched against case id or meta.task_type; "
              "repeat for multiple patterns"),
    )
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--provider", default="opencode")
    ap.add_argument("--topologies", nargs="+", default=["full"])
    ap.add_argument("--conditions", "--panels", dest="conditions", nargs="+",
                    help="named YAML conditions; defaults to the spec's condition list")
    ap.add_argument("--memory", default="cumulative",
                    choices=["cumulative", "self-last", "peer-only"])
    ap.add_argument("--max-agent-tokens", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-tag", default="v1")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the round-1 prompts, call no model")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    conditions = args.conditions or list(spec.condition_names)
    unknown = sorted(set(conditions) - set(spec.condition_names))
    if unknown:
        raise SystemExit(
            f"{spec.name}: unknown conditions {unknown}; have {list(spec.condition_names)}")
    cases = load_cases(spec, n=args.n_cases, seed=args.seed)
    if args.case_filter:
        cases = [
            case for case in cases
            if any(
                fnmatch.fnmatch(case.id, pattern)
                or fnmatch.fnmatch(str(case.meta.get("task_type", "")), pattern)
                for pattern in args.case_filter
            )
        ]
    if not cases:
        detail = f" after filters {args.case_filter}" if args.case_filter else ""
        raise SystemExit(f"{spec.name}: loader returned no cases{detail}")

    bad = [t for t in args.topologies
           if spec.n_agents < 3 and t in ("star", "chain")]
    if bad:
        print(f"note: {spec.name} has {spec.n_agents} agents, so {bad} is the same "
              f"graph as 'full'. Running it anyway would report a topology effect "
              f"that cannot exist.", file=sys.stderr)

    if args.dry_run:
        for case in cases:
            for condition in conditions:
                dealt = deal(case, spec.agents, spec.disclosure_for(condition), seed=args.seed)
                cov = coverage(dealt, case)
                print(f"\n{'=' * 72}\n{spec.name} / case {case.id} / condition {condition}")
                print(f"每 agent 均 {cov['mean_per_agent']:.1f} 条，"
                      f"独占 {len(cov['exclusive'])}/{cov['n_items']}，"
                      f"未覆盖 {cov['uncovered']}")
                a = spec.agents[0]
                print(f"--- {a} 的 round-1 prompt ---")
                print(build_prompt(spec, case, dealt[a], [], False))
        return 0

    if args.provider == "opencode":
        load_opencode_key()
    makers = {"opencode": LLM.opencode, "openai": LLM.openai,
              "deepseek": LLM.deepseek}
    if args.provider not in makers:
        raise SystemExit(f"unknown provider {args.provider!r}; have {sorted(makers)}")
    llm = makers[args.provider](args.model, max_concurrency=args.concurrency)
    args.outdir.mkdir(parents=True, exist_ok=True)

    planned = len(cases) * len(args.topologies) * len(conditions)
    print(
        f"plan: {len(cases)} cases x {len(args.topologies)} topologies x "
        f"{len(conditions)} conditions = {planned} debates",
        flush=True,
    )
    completed = 0
    skipped = 0
    for case in cases:
        for topology in args.topologies:
            for condition in conditions:
                suffix = "" if args.memory == "peer-only" else f"-{args.memory}"
                exec_id = (f"{spec.name}-{case.id}-{args.model}-{topology}-"
                           f"{condition}{suffix}")
                out = args.outdir / f"{exec_id}.debate.json"
                if out.exists():
                    print(f"[skip] {out}", flush=True)
                    skipped += 1
                    continue
                run = run_case(llm, spec, case, args.model, topology, condition,
                               args.memory, args.max_agent_tokens, args.sample_tag,
                               args.seed, progress=True)
                tmp = out.with_suffix(out.suffix + ".tmp")
                tmp.write_text(json.dumps(run, ensure_ascii=False, indent=1,
                                          sort_keys=True) + "\n", encoding="utf-8")
                tmp.replace(out)
                completed += 1
                print(f"[{completed + skipped}/{planned}] {out}", flush=True)
    print(
        f"complete: wrote {completed}, skipped {skipped}; "
        f"{llm.usage.report(args.model)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
