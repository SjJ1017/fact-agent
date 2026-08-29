"""Run reproducible fact-flow experiments on Perspectrum's real debate evidence.

The task is deliberately evidence-grounded.  All panelists receive a balanced
dossier of human-curated support and undermine evidence for one real claim.
Persona prompts can therefore change what gets selected, challenged, repeated,
and retained without a synthetic allocation of atomic facts.

Every run saves the exact prompt delivery graph.  This makes ``Exposed`` a
lookup from the trace rather than an assumption about a topology.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from frameworks import neighbours
from factflow import CanonicalFact, Channel, FactMention, FactStore, LLM, Polarity, Relation, TraceRecord
from factflow.aggregate import build_view, signature, trajectory
from factflow.blocking import candidate_pairs, tokenize


DATA_FILES = (
    "perspectrum_with_answers_v1.0.json",
    "perspective_pool_v1.0.json",
    "evidence_pool_v1.0.json",
    "dataset_split_v1.0.json",
)

BASE_SYSTEM = """\
You are one member of an evidence review panel assessing a contested public
claim. Work only from the supplied dossier. State concrete empirical claims
and distinguish evidence from value judgments. Do not invent sources,
statistics, or quotations.
Keep your response under 180 words. End with one line in exactly this form:
VERDICT: SUPPORT | UNDERMINE | UNCERTAIN"""

PANEL_ROLES = {
    "neutral": [
        "You are a neutral evidence analyst. Seek the strongest relevant facts on both sides.",
        "You are a neutral evidence analyst. Seek the strongest relevant facts on both sides.",
        "You are a neutral evidence analyst. Seek the strongest relevant facts on both sides.",
    ],
    "lenses": [
        "You are a causal-evidence analyst. Focus on mechanisms, causal links, and what the evidence can actually establish.",
        "You are an implementation and trade-off analyst. Focus on feasibility, unintended consequences, scope, and affected populations.",
        "You are a scope and uncertainty analyst. Focus on exceptions, missing conditions, external validity, and overgeneralization.",
    ],
    "stance": [
        "You are an advocate for the claim. Find its strongest dossier-supported case, but explicitly acknowledge material counterevidence.",
        "You are a critic of the claim. Find its strongest dossier-supported countercase, but explicitly acknowledge material supporting evidence.",
        "You are an impartial adjudicator. Compare the evidence on both sides and identify which asserted links are not established.",
    ],
    "functional": [
        "You are an evidence curator. Surface the concrete facts in the dossier that the panel must not omit.",
        "You are a falsification critic. Test every proposed conclusion against counterevidence and identify unsupported leaps.",
        "You are a synthesis analyst. Preserve relevant support and counterevidence, resolving contradictions only when the dossier warrants it.",
    ],
}

LIGHT_EXTRACTION_SYSTEM = """\
Return JSON only: {"facts":[{"text":string,"polarity":"affirm"|"negate"}]}.
Extract at most 8 short, self-contained, verifiable facts from the supplied text.
Preserve names, numbers, and conditions. Do not infer, repeat a fact, or emit
claims about the conversation. Use "negate" only for an explicit denial."""

class LightFact(BaseModel):
    text: str
    polarity: Literal["affirm", "negate"] = "affirm"


class LightExtraction(BaseModel):
    facts: list[LightFact]


@dataclass(frozen=True)
class Case:
    claim_id: int
    claim: str
    evidence: tuple[dict[str, str], ...]
    perspectives: tuple[dict[str, str], ...]


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def shorten(text: str, limit: int) -> str:
    """Keep the longest complete sentence available under the evidence budget."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    boundary = max(text.rfind(". ", 0, limit), text.rfind("; ", 0, limit))
    if boundary >= limit // 2:
        return text[:boundary + 1]
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def load_cases(
    data_dir: Path,
    n: int,
    seed: int,
    split: str,
    max_evidence_per_side: int,
    min_evidence_per_side: int,
    evidence_chars: int,
) -> list[Case]:
    missing = [name for name in DATA_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing Perspectrum data files under {data_dir}: {missing}")

    claims = _load_json(data_dir / "perspectrum_with_answers_v1.0.json")
    perspectives = {row["pId"]: row for row in _load_json(data_dir / "perspective_pool_v1.0.json")}
    evidence = {row["eId"]: row for row in _load_json(data_dir / "evidence_pool_v1.0.json")}
    splits = _load_json(data_dir / "dataset_split_v1.0.json")
    usable: list[Case] = []

    for row in claims:
        if split != "all" and splits.get(str(row["cId"])) != split:
            continue
        by_stance: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cluster in row["perspectives"]:
            stance = str(cluster["stance_label_3"]).lower()
            if stance in {"support", "undermine"} and cluster.get("evidence"):
                by_stance[stance].append(cluster)
        if not by_stance["support"] or not by_stance["undermine"]:
            continue

        selected: list[dict[str, str]] = []
        seen_evidence: set[int] = set()
        for stance in ("support", "undermine"):
            count = 0
            for cluster in by_stance[stance]:
                for eid in cluster["evidence"]:
                    item = evidence.get(eid)
                    if item is None or eid in seen_evidence or not item.get("text", "").strip():
                        continue
                    seen_evidence.add(eid)
                    selected.append({"id": f"E{len(selected) + 1}", "stance": stance,
                                     "text": shorten(item["text"], evidence_chars),
                                     "source": item.get("source", "")})
                    count += 1
                    if count >= max_evidence_per_side:
                        break
                if count >= max_evidence_per_side:
                    break
        if sum(item["stance"] == "support" for item in selected) < min_evidence_per_side:
            continue
        if sum(item["stance"] == "undermine" for item in selected) < min_evidence_per_side:
            continue

        gold_perspectives: list[dict[str, str]] = []
        for stance in ("support", "undermine"):
            for cluster in by_stance[stance]:
                for pid in cluster["pids"][:1]:
                    item = perspectives.get(pid)
                    if item and item.get("text", "").strip():
                        gold_perspectives.append({"stance": stance, "text": item["text"].strip()})
                        break
        usable.append(Case(int(row["cId"]), row["text"].strip(), tuple(selected), tuple(gold_perspectives)))

    if len(usable) < n:
        raise ValueError(f"only {len(usable)} eligible {split} claims; requested {n}")
    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:n]


def dossier(case: Case) -> str:
    entries = "\n\n".join(f"[{item['id']}] {item['text']}" for item in case.evidence)
    return f"Claim under review:\n{case.claim}\n\nEvidence dossier:\n{entries}"


def prompt_for(case: Case, peer_text: Sequence[tuple[str, str]], role: str) -> str:
    base = dossier(case)
    if not peer_text:
        return base + "\n\nGive your independent assessment before seeing other panelists."
    peer_block = "\n\n".join(f"--- Panelist {agent} ---\n{text}" for agent, text in peer_text)
    return (base + "\n\nThese are the panelists' latest assessments:\n\n" + peer_block +
            "\n\nUpdate your assessment. Address concrete claims from the other panelists and revise "
            "your position only when the dossier or their reasoning warrants it.")


def run_case(
    llm: LLM,
    case: Case,
    model: str,
    topology: str,
    panel: str,
    rounds: int,
    max_agent_tokens: int,
    sample_tag: str,
) -> dict[str, Any]:
    agents = ["A", "B", "C"]
    roles = dict(zip(agents, PANEL_ROLES[panel]))
    transcript: dict[str, str] = {}
    prompts: dict[str, str] = {}
    delivery: dict[str, dict[str, Any]] = {}

    for rnd in range(1, rounds + 1):
        requests: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        for agent in agents:
            peers = [(peer, transcript[f"{peer}|{rnd - 1}"])
                     for peer in neighbours(topology, agents, agent)
                     if f"{peer}|{rnd - 1}" in transcript]
            key = f"{agent}|{rnd}"
            user = prompt_for(case, peers, roles[agent])
            prompts[key] = user
            delivery[key] = {
                "source_ids": ["claim", *[item["id"] for item in case.evidence]],
                "peer_turns": [f"{peer}|{rnd - 1}" for peer, _ in peers],
            }
            requests[agent] = (user, peers)

        def ask(agent: str) -> tuple[str, str]:
            user, _ = requests[agent]
            system = roles[agent] + "\n\n" + BASE_SYSTEM
            sample_base = (f"perspectrum:{sample_tag}:{case.claim_id}:{model}:{topology}:"
                           f"{panel}:{agent}:{rnd}")
            # Some reasoning-capable flash endpoints can exhaust a short budget before
            # emitting a visible answer. Retries preserve the prompt and sampling setup.
            for attempt, cap in enumerate((max_agent_tokens, max_agent_tokens * 2,
                                           max_agent_tokens * 4), 1):
                text = llm.chat(system=system, user=user, temperature=0.7, max_tokens=cap,
                                sample_id=f"{sample_base}:attempt={attempt}")
                if text.strip():
                    return agent, text
            raise RuntimeError(f"empty model response after retry for {model} claim={case.claim_id} agent={agent} round={rnd}")

        for agent, text in llm.map(ask, agents, tolerate_failures=False):
            transcript[f"{agent}|{rnd}"] = text

    execution_id = f"perspectrum-{case.claim_id}-{model}-{topology}-{panel}"
    return {
        "execution_id": execution_id,
        "claim_id": case.claim_id,
        "claim": case.claim,
        "model": model,
        "topology": topology,
        "panel": panel,
        "roles": roles,
        "rounds": rounds,
        "evidence": list(case.evidence),
        "gold_perspectives": list(case.perspectives),
        "transcript": transcript,
        "prompts": prompts,
        "delivery": delivery,
    }


def records_for(run: dict[str, Any]) -> list[TraceRecord]:
    execution_id = run["execution_id"]
    records = [TraceRecord.model_validate({
        "text": run["claim"],
        "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                       "channel": "source", "doc_id": "claim", "extra": {"kind": "claim"}},
    })]
    for item in run["evidence"]:
        records.append(TraceRecord.model_validate({
            "text": item["text"],
            "provenance": {"execution_id": execution_id, "agent_id": None, "round": 0,
                           "channel": "source", "doc_id": item["id"],
                           "extra": {"gold": True, "kind": "evidence", "stance": item["stance"]}},
        }))
    for key, text in sorted(run["transcript"].items(), key=lambda item: (int(item[0].split("|")[1]), item[0])):
        agent, rnd = key.split("|")
        records.append(TraceRecord.model_validate({
            "text": text,
            "provenance": {"execution_id": execution_id, "agent_id": agent, "round": int(rnd),
                           "channel": "output", "doc_id": f"turn-{key}",
                           "extra": {"role": run["roles"][agent]}},
        }))
    return records


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _light_chat_json(llm: LLM, system: str, user: str, schema: type[BaseModel], max_tokens: int) -> BaseModel:
    error = ""
    for attempt, cap in enumerate((max_tokens, max_tokens * 2, max_tokens * 3), 1):
        retry = user if attempt == 1 else user + "\n\nReturn valid JSON only. Previous parse error: " + error[:160]
        text = llm.chat(system=system, user=retry, temperature=0, max_tokens=cap)
        try:
            return schema.model_validate(_json_object(text))
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
    raise ValueError(f"light JSON parse failed: {error}")


def extract_with_progress(llm: LLM, records: Sequence[TraceRecord], label: str,
                          max_tokens: int) -> tuple[list[FactMention], list[dict[str, str]]]:
    """Bounded extraction for the validation study, with per-record progress."""
    started = time.monotonic()
    results: list[list[FactMention] | None] = [None] * len(records)
    failures: list[dict[str, str]] = []

    def one(index: int, record: TraceRecord) -> list[FactMention]:
        parsed = _light_chat_json(llm, LIGHT_EXTRACTION_SYSTEM, record.text, LightExtraction, max_tokens)
        mentions: list[FactMention] = []
        seen: set[str] = set()
        for fact in parsed.facts:
            text = " ".join(fact.text.split()).strip()
            norm = text.lower()
            if not text or norm in seen:
                continue
            seen.add(norm)
            mentions.append(FactMention(mention_id=FactMention.make_id(text, record.provenance), text=text,
                                        polarity=Polarity(fact.polarity), provenance=record.provenance))
        return mentions

    with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as pool:
        futures = {pool.submit(one, index, record): index for index, record in enumerate(records)}
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            record = records[index]
            error = ""
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = []
                error = f"{type(exc).__name__}: {exc}"
                failures.append({
                    "channel": record.provenance.channel.value,
                    "doc_id": record.provenance.doc_id or "",
                    "agent_id": record.provenance.agent_id or "",
                    "round": str(record.provenance.round),
                    "error": error,
                })
            elapsed = time.monotonic() - started
            print(f"[extract {label} {completed}/{len(records)}] channel={record.provenance.channel.value} "
                  f"slot={record.provenance.doc_id or record.provenance.agent_id} "
                  f"facts={len(results[index] or [])}" + (" FAILED" if error else "") +
                  f" elapsed={elapsed / 60:.1f}m", flush=True)
    return [mention for batch in results if batch for mention in batch], failures


def light_match(mentions: Sequence[FactMention], label: str) -> FactStore:
    """Conservative local matching for a fast validation trace.

    At this stage, false merging would fabricate a transmission event, whereas
    false splitting only undercounts one. We therefore merge only when the
    stemmed-token Jaccard overlap is at least .80. Full LLM relation matching is
    deliberately reserved for later, audited traces.
    """
    mentions = list(mentions)
    pairs = candidate_pairs(mentions, threshold=0.80, top_k=1)
    relations: list[Relation] = []
    for left, right, _ in pairs:
        a, b = mentions[left], mentions[right]
        ta, tb = tokenize(a.text), tokenize(b.text)
        jaccard = len(ta & tb) / max(len(ta | tb), 1)
        if jaccard < 0.80:
            continue
        relation = "CONTRADICTS" if a.polarity != b.polarity else "EQUIVALENT"
        relations.append(Relation(a=a.mention_id, b=b.mention_id, relation=relation, confidence=jaccard))
    print(f"[match {label}] mentions={len(mentions)} lexical_candidates={len(pairs)} "
          f"accepted={len(relations)}", flush=True)

    parent = list(range(len(mentions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    lookup = {mention.mention_id: index for index, mention in enumerate(mentions)}
    for relation in relations:
        if relation.relation == "EQUIVALENT":
            union(lookup[relation.a], lookup[relation.b])
    groups: dict[int, list[FactMention]] = defaultdict(list)
    for index, mention in enumerate(mentions):
        groups[find(index)].append(mention)

    store = FactStore()
    store.add_mentions(mentions)
    store.relations.extend(relations)
    for members in groups.values():
        source = [mention for mention in members if mention.provenance.channel == Channel.SOURCE]
        canonical = max(source or members, key=lambda mention: len(mention.text))
        fact = CanonicalFact(fact_id=CanonicalFact.make_id(canonical.text), canonical_text=canonical.text,
                             polarity=canonical.polarity, mention_ids=[mention.mention_id for mention in members])
        store.assign(fact)
    return store


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def grouped(values: Iterable[tuple[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, row in values:
        for metric, value in row.items():
            acc[key][metric].append(float(value))
    return {key: {metric: statistics.mean(values) for metric, values in metrics.items()}
            for key, metrics in acc.items()}


def summarize(outdir: Path) -> dict[str, Any]:
    rows: list[tuple[str, dict[str, float]]] = []
    trajectories: dict[str, list[dict[str, float]]] = defaultdict(list)
    for debate_path in sorted(outdir.glob("*.debate.json")):
        store_path = debate_path.with_suffix("").with_suffix(".store.json")
        if not store_path.exists():
            continue
        run = _load_json(debate_path)
        store = FactStore.load(str(store_path))
        view = build_view(store, run["execution_id"], roles=run["roles"])
        key = f"{run['model']}|{run['topology']}|{run['panel']}"
        metrics = signature(view)
        rows.append((key, metrics))
        trajectories[key].append(trajectory(view))

    summary = grouped(rows)
    for key, runs in trajectories.items():
        if not runs:
            continue
        for index in range(max(len(run) for run in runs)):
            aligned = [run[index] for run in runs if index < len(run)]
            for metric in ("alive", "born", "carried", "died", "cumulative", "born_contested", "born_survives"):
                summary[key][f"r{index + 1}:{metric}"] = statistics.mean(float(row[metric]) for row in aligned)
        summary[key]["n_runs"] = float(len(runs))
    result = {"configs": summary, "n_traced_runs": len(rows)}
    save_json(outdir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "perspectrum_out")
    parser.add_argument("--models", default="deepseek-v4-flash,glm-5.3-flash")
    parser.add_argument("--trace-model", default="glm-5.3-flash")
    parser.add_argument("--panels", default="neutral,lenses,stance")
    parser.add_argument("--topologies", default="full")
    parser.add_argument("-n", "--n-claims", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--evidence-per-side", type=int, default=2)
    parser.add_argument("--min-evidence-per-side", type=int, default=2)
    parser.add_argument("--evidence-chars", type=int, default=700)
    parser.add_argument("--max-agent-tokens", type=int, default=800)
    parser.add_argument("--trace-max-tokens", type=int, default=1600,
                        help="completion cap for fact extraction and relation adjudication")
    parser.add_argument("--parallel", type=int, default=2, help="debates in flight; each debate asks its panel in parallel")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trace", action="store_true", help="extract and match atomic facts after the debates")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    topologies = [item.strip() for item in args.topologies.split(",") if item.strip()]
    unknown = set(panels) - set(PANEL_ROLES)
    if unknown:
        raise ValueError(f"unknown panels: {sorted(unknown)}")
    for topology in topologies:
        neighbours(topology, ["A", "B", "C"], "A")

    if args.min_evidence_per_side > args.evidence_per_side:
        raise ValueError("--min-evidence-per-side cannot exceed --evidence-per-side")
    cases = load_cases(args.data_dir, args.n_claims, args.seed, args.split,
                       args.evidence_per_side, args.min_evidence_per_side, args.evidence_chars)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_dir": str(args.data_dir.resolve()), "claim_ids": [case.claim_id for case in cases],
        "models": models, "trace_model": args.trace_model, "panels": panels, "topologies": topologies,
        "rounds": args.rounds, "seed": args.seed, "split": args.split,
        "evidence_per_side": args.evidence_per_side, "min_evidence_per_side": args.min_evidence_per_side,
        "evidence_chars": args.evidence_chars,
        "max_agent_tokens": args.max_agent_tokens, "trace_max_tokens": args.trace_max_tokens,
    }
    save_json(args.outdir / "manifest.json", manifest)

    jobs = [(model, topology, panel, case) for model in models for topology in topologies
            for panel in panels for case in cases]
    print(f"[debate] {len(jobs)} runs = {len(models)} models x {len(topologies)} topologies x "
          f"{len(panels)} panels x {len(cases)} real claims", flush=True)
    started = time.monotonic()
    completed: list[Path] = []

    def run_one(job: tuple[str, str, str, Case]) -> Path:
        model, topology, panel, case = job
        path = args.outdir / f"perspectrum-{case.claim_id}-{model}-{topology}-{panel}.debate.json"
        if path.exists() and not args.overwrite:
            return path
        llm = LLM.opencode(model, max_concurrency=args.concurrency)
        llm.backend.client = llm.backend.client.with_options(timeout=args.timeout, max_retries=1)
        save_json(path, run_case(llm, case, model, topology, panel, args.rounds,
                                 args.max_agent_tokens, args.outdir.name))
        return path

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            model, topology, panel, case = futures[future]
            try:
                path = future.result()
                completed.append(path)
                elapsed = time.monotonic() - started
                eta = elapsed / index * (len(jobs) - index)
                print(f"[debate {index}/{len(jobs)}] claim={case.claim_id} model={model} "
                      f"topology={topology} panel={panel} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[debate {index}/{len(jobs)}] FAILED claim={case.claim_id} model={model} "
                      f"topology={topology} panel={panel}: {type(exc).__name__}: {exc}", flush=True)

    if args.trace:
        debate_paths = sorted(set(completed))
        print(f"[trace] {len(debate_paths)} saved debates; extractor/matcher={args.trace_model}", flush=True)
        tracer = LLM.opencode(args.trace_model, max_concurrency=args.concurrency,
                              max_tokens=args.trace_max_tokens)
        tracer.backend.client = tracer.backend.client.with_options(timeout=args.timeout, max_retries=1)
        trace_started = time.monotonic()
        trace_failures: list[dict[str, str]] = []
        for index, debate_path in enumerate(debate_paths, 1):
            store_path = debate_path.with_suffix("").with_suffix(".store.json")
            run = _load_json(debate_path)
            label = f"{index}/{len(debate_paths)} claim={run['claim_id']}"
            try:
                if not store_path.exists() or args.overwrite:
                    records = records_for(run)
                    mentions, extraction_failures = extract_with_progress(
                        tracer, records, label, min(args.trace_max_tokens, 800)
                    )
                    light_match(mentions, label).save(str(store_path))
                    save_json(
                        debate_path.with_suffix("").with_suffix(".extraction_failures.json"),
                        extraction_failures,
                    )
                    print(f"[match {label}] complete", flush=True)
            except Exception as exc:  # noqa: BLE001
                trace_failures.append({"execution_id": run["execution_id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[trace {label}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            elapsed = time.monotonic() - trace_started
            eta = elapsed / index * (len(debate_paths) - index)
            print(f"[trace {index}/{len(debate_paths)}] claim={run['claim_id']} model={run['model']} "
                  f"topology={run['topology']} panel={run['panel']} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
        save_json(args.outdir / "trace_failures.json", trace_failures)
        summary = summarize(args.outdir)
        print(f"[summary] traced={summary['n_traced_runs']} failures={len(trace_failures)} -> "
              f"{args.outdir / 'summary.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
