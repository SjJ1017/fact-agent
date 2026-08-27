"""Multi-agent debate baseline (Du et al. 2023 shape) over HotpotQA.

Deliberately the plain version: N agents, fully connected, R rounds, every agent
sees the same documents.  No role prompts, no distributed evidence.  Agents
differ only in what they choose to say, which is the point - it isolates
*selection* from *access*.

Emits a trace in factflow's TraceRecord format.  CONTEXT records are computed
rather than re-extracted: under full broadcast, agent A's context at round r is
exactly the source documents plus every peer output from rounds < r, so
re-running extraction over it would pay many times for the same propositions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from factflow import LLM

SYSTEM = """\
You are one participant in a small panel answering a factual question.
Ground every claim in the provided documents. Cite the concrete detail you are \
relying on - names, dates, numbers - rather than gesturing at it.
Keep your answer under 120 words. End with exactly one line:
FINAL ANSWER: <your answer>"""

ROUND1 = """\
{docs}

Question: {question}

Give your reasoning and your answer."""

LATER = """\
{docs}

Question: {question}

These are the other panelists' latest responses:

{peers}

Consider them, then give your own updated reasoning and answer. If you disagree, \
say what specifically in the documents supports your position."""


@dataclass
class DebateResult:
    qid: str
    question: str
    gold_answer: str
    gold_titles: list[str]
    documents: str
    paragraphs: list[tuple[str, str]] = field(default_factory=list)
    transcript: dict[tuple[str, int], str] = field(default_factory=dict)
    final: dict[str, str] = field(default_factory=dict)


def format_docs(context: dict) -> str:
    parts = []
    for title, sents in zip(context["title"], context["sentences"]):
        parts.append(f"[{title}] " + " ".join(s.strip() for s in sents))
    return "\n\n".join(parts)


def extract_final(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip().upper().startswith("FINAL ANSWER:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[-1][:120] if text.strip() else ""


def run_debate(llm: LLM, row: dict, n_agents: int = 3, n_rounds: int = 3) -> DebateResult:
    agents = [chr(ord("A") + i) for i in range(n_agents)]
    docs = format_docs(row["context"])
    paragraphs = [
        (title, f"[{title}] " + " ".join(x.strip() for x in sents))
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"])
    ]
    result = DebateResult(
        qid=row["id"],
        question=row["question"],
        gold_answer=row["answer"],
        gold_titles=list(row["supporting_facts"]["title"]),
        documents=docs,
        paragraphs=paragraphs,
    )

    for rnd in range(1, n_rounds + 1):
        if rnd == 1:
            prompts = {a: ROUND1.format(docs=docs, question=row["question"]) for a in agents}
        else:
            prompts = {}
            for a in agents:
                peers = "\n\n".join(
                    f"--- Panelist {b} ---\n{result.transcript[(b, rnd - 1)]}"
                    for b in agents
                    if b != a
                )
                prompts[a] = LATER.format(docs=docs, question=row["question"], peers=peers)

        # Agents within a round answer independently; they only see round r-1.
        # tolerate_failures=False on purpose: a dropped turn is not a missing
        # data point, it is a hole the next round indexes into. Better to fail
        # here with the real error than to corrupt the trace silently.
        outputs = llm.map(
            lambda a: (a, llm.chat(system=SYSTEM, user=prompts[a], temperature=0.7, max_tokens=600)),
            agents,
            tolerate_failures=False,
        )
        for a, text in outputs:
            result.transcript[(a, rnd)] = text
        missing = [a for a in agents if (a, rnd) not in result.transcript]
        if missing:
            raise RuntimeError(f"round {rnd}: no output for agent(s) {missing}")

    for a in agents:
        result.final[a] = extract_final(result.transcript[(a, n_rounds)])
    return result


def to_trace(result: DebateResult, execution_id: str) -> list[dict[str, Any]]:
    """Trace records for factflow. SOURCE = one record per paragraph; OUTPUT = each agent turn.

    Source paragraphs are kept separate rather than concatenated so each source
    fact carries the title it came from.  HotpotQA labels which two titles are
    the gold supporting ones, so that provenance is what lets retention be split
    into gold vs distractor - without it, "90% of source facts were dropped"
    cannot be distinguished from correct distractor filtering.
    """
    records = [
        {
            "text": para,
            "provenance": {
                "execution_id": execution_id,
                "agent_id": None,
                "round": 0,
                "channel": "source",
                "doc_id": title,
                "extra": {"gold": title in result.gold_titles},
            },
        }
        for title, para in result.paragraphs
    ]
    for (agent, rnd), text in sorted(result.transcript.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        records.append(
            {
                "text": text,
                "provenance": {
                    "execution_id": execution_id,
                    "agent_id": agent,
                    "round": rnd,
                    "channel": "output",
                    "doc_id": result.qid,
                },
            }
        )
    return records


def save(result: DebateResult, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "qid": result.qid,
                "question": result.question,
                "gold_answer": result.gold_answer,
                "gold_titles": result.gold_titles,
                "documents": result.documents,
                "transcript": {f"{a}|{r}": t for (a, r), t in result.transcript.items()},
                "final": result.final,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
