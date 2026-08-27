"""Multi-agent frameworks: topology x role assignment, over MedQA.

HotpotQA turned out to be the wrong substrate. Three agents read the same ten
paragraphs, agreed in round 1, and spent rounds 2-3 restating each other - the
task is retrieval, and retrieval does not need a committee. Selectivity was
measurable but every run was unanimous and correct, so nothing about
collaboration was under test.

MedQA-USMLE is harder in the way that matters here: one clinical vignette, four
plausible options, and reasoning that spans pharmacology, physiology and
clinical judgement. No agent can look anything up, so what an agent contributes
is what it knows, which is what makes role assignment and topology do work.

Three topologies, differing in who reads whom:

    full   every agent reads every other      (the Du et al. debate default)
    chain  A -> B -> C, each reads only its predecessor
    star   spokes read the hub, the hub reads everyone
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from factflow import LLM

TOPOLOGIES = ("full", "chain", "star")

BASE_SYSTEM = """\
You are one participant in a panel answering a hard exam question.
Reason from the specifics given. Name the concrete fact, mechanism, value or
principle you are relying on rather than gesturing at it.
Keep your answer under 140 words. End with exactly one line:
FINAL ANSWER: <letter>"""

# Two different things get called "roles" in the literature, and they predict
# different fact-flow signatures.
#
#   DOMAIN roles (internist, pharmacologist) differ in what they KNOW, so they
#   should decorrelate the evidence pool - which is what they did: round-1
#   pairwise fact overlap fell 0.83 -> 0.28.
#
#   FUNCTIONAL roles (decomposer, analyzer, summarizer, critic) differ in what
#   they DO to facts. An analyst should derive new facts; a summarizer should
#   drop irrelevant ones and relay the rest; a critic should contradict. If the
#   role prompt is doing any work at all, each should leave a distinct signature
#   in the trace, and a summarizer that only relays is the redundant node that
#   topology-refinement papers try to prune.
FUNCTIONAL_ROLES = {
    "decomposer": "You are the decomposer. Break the question into the specific sub-questions "
                  "that must be settled, and state what each one hinges on. Do not attempt the "
                  "final answer until the parts are laid out.",
    "analyzer": "You are the analyzer. Work each sub-question through: derive the consequences, "
                "compute what can be computed, and state the intermediate results explicitly.",
    "summarizer": "You are the summarizer. Consolidate what the panel has established. Keep what "
                  "bears on the answer, drop what does not, and say plainly which points are "
                  "settled and which are still open.",
    "critic": "You are the critic. Find the specific step where an argument fails. Name the "
              "option a mistaken reading would lead to, and say what would have to be true for "
              "the current answer to be wrong.",
    "verifier": "You are the verifier. Re-derive the proposed answer independently and check it "
                "against every constraint in the question. Say explicitly what you checked.",
}

ROLES = {
    "internist": "You are a general internist. You weigh the whole clinical picture and the "
                 "differential, and you are the one who notices when a finding does not fit.",
    "pharmacologist": "You are a clinical pharmacologist. You reason about drug mechanisms, "
                      "interactions, contraindications and adverse effects.",
    "pathophysiologist": "You are a pathophysiologist. You reason from underlying mechanism - "
                         "what process would produce exactly this constellation of findings.",
    "surgeon": "You are a surgeon. You weigh procedural indications, operative risk and "
               "anatomy.",
    "ethicist": "You are a clinical ethicist. You weigh disclosure, consent, capacity and "
                "professional obligation.",
    "theorist": "You are a theoretician. You reason from first principles and check whether "
                "the governing equations or laws actually apply to this case.",
    "calculator": "You are careful with quantities. You check units, magnitudes, signs and "
                  "arithmetic, and you recompute anything that looks convenient.",
    "skeptic": "You are the panel's skeptic. Your job is to find the specific step where an "
               "argument breaks, and to say which option a mistaken reading would lead to.",
}
ROLES.update(FUNCTIONAL_ROLES)

ROUND1 = """{vignette}

Options:
{options}

Give your reasoning and your answer."""

LATER = """{vignette}

Options:
{options}

{peer_header}

{peers}

Consider these, then give your own updated reasoning and answer. If you disagree, say
which specific finding or mechanism supports your position."""


def neighbours(topology: str, agents: Sequence[str], agent: str) -> list[str]:
    """Who this agent reads at the end of each round."""
    i = agents.index(agent)
    if topology == "full":
        return [a for a in agents if a != agent]
    if topology == "chain":
        return [agents[i - 1]] if i > 0 else []
    if topology == "star":
        hub = agents[0]
        return [a for a in agents if a != hub] if agent == hub else [hub]
    raise ValueError(f"unknown topology {topology!r}")


@dataclass
class RunResult:
    qid: str
    question: str
    vignette: str
    options: dict[str, str]
    gold: str
    topology: str
    roles: dict[str, str] = field(default_factory=dict)
    transcript: dict[tuple[str, int], str] = field(default_factory=dict)
    final: dict[str, str] = field(default_factory=dict)
    majority: str = ""
    correct: bool = False


LETTER = re.compile(r"FINAL ANSWER:\s*\(?([A-J])\)?", re.I)


def extract_letter(text: str) -> str:
    hits = LETTER.findall(text or "")
    return hits[-1].upper() if hits else ""


def run_framework(
    llm: LLM,
    row: dict,
    topology: str = "full",
    role_names: Sequence[str] | None = None,
    n_agents: int = 3,
    n_rounds: int = 3,
) -> RunResult:
    agents = [chr(ord("A") + i) for i in range(n_agents)]
    roles = dict(zip(agents, role_names or [])) if role_names else {}
    options = dict(row["options"])
    opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    res = RunResult(
        qid=str(row.get("id", row["question"][:40])),
        question=row["question"],
        vignette=row["question"],
        options=options,
        gold=row["answer_idx"],
        topology=topology,
        roles={a: (role_names[i] if role_names else "generalist") for i, a in enumerate(agents)},
    )

    for rnd in range(1, n_rounds + 1):
        prompts = {}
        for a in agents:
            peers = [(b, res.transcript[(b, rnd - 1)]) for b in neighbours(topology, agents, a)
                     if (b, rnd - 1) in res.transcript]
            if rnd == 1 or not peers:
                prompts[a] = ROUND1.format(vignette=res.vignette, options=opt_text)
            else:
                header = ("These are the other panelists' latest responses:" if len(peers) > 1
                          else f"This is panelist {peers[0][0]}'s latest response:")
                prompts[a] = LATER.format(
                    vignette=res.vignette, options=opt_text, peer_header=header,
                    peers="\n\n".join(f"--- Panelist {b} ({res.roles[b]}) ---\n{t}" for b, t in peers),
                )

        def one(a):
            system = BASE_SYSTEM
            if roles:
                system = ROLES[roles[a]] + "\n\n" + BASE_SYSTEM
            return a, llm.chat(system=system, user=prompts[a], temperature=0.7, max_tokens=700,
                               sample_id=f"{res.qid}:{topology}:{a}:{rnd}")

        for a, text in llm.map(one, agents, tolerate_failures=False):
            res.transcript[(a, rnd)] = text
        missing = [a for a in agents if (a, rnd) not in res.transcript]
        if missing:
            raise RuntimeError(f"round {rnd}: no output for {missing}")

    for a in agents:
        res.final[a] = extract_letter(res.transcript[(a, n_rounds)])
    votes = [v for v in res.final.values() if v]
    res.majority = max(set(votes), key=votes.count) if votes else ""
    res.correct = res.majority == res.gold
    return res


def to_trace(res: RunResult, execution_id: str) -> list[dict[str, Any]]:
    """SOURCE is the vignette plus options - in MedQA there is no corpus to cite."""
    records = [{
        "text": res.vignette,
        "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                       "channel": "source", "doc_id": "vignette", "extra": {"gold": True}},
    }]
    for k, v in sorted(res.options.items()):
        records.append({
            "text": f"Option {k}: {v}",
            "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                           "channel": "source", "doc_id": f"option-{k}",
                           "extra": {"gold": k == res.gold}},
        })
    for (agent, rnd), text in sorted(res.transcript.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        records.append({
            "text": text,
            "provenance": {"execution_id": execution_id, "agent_id": agent, "round": rnd,
                           "channel": "output", "doc_id": res.qid,
                           "extra": {"role": res.roles.get(agent, "generalist")}},
        })
    return records


def save(res: RunResult, path: Path) -> None:
    path.write_text(json.dumps({
        "qid": res.qid, "question": res.question, "gold_answer": res.gold,
        "gold_titles": [f"option-{res.gold}"], "documents": res.vignette,
        "options": res.options, "topology": res.topology, "roles": res.roles,
        "transcript": {f"{a}|{r}": t for (a, r), t in res.transcript.items()},
        "final": res.final, "majority": res.majority, "correct": res.correct,
    }, indent=2, ensure_ascii=False))
