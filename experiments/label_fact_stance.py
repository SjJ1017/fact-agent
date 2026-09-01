#!/usr/bin/env python3
"""Post-hoc label each output canonical fact's argumentative stance.

This script never feeds labels, fact ids, or extracted facts back to debate
agents. A label means how the proposition would bear on the debate claim *if it
were true*, not whether the proposition is actually true or well supported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.llm import LLM  # noqa: E402
from factflow.types import FactStore  # noqa: E402


# Two readings of "stance", and they answer different questions.
#
# TRUTH reads the fact against the world: is it a reason the claim is correct?
# Under it, "E1 provides no data on actual effects" is NEUTRAL - absence of
# evidence for a claim is not evidence against it.
#
# ARGUMENTATIVE reads the fact against the debate: which side does saying this
# put you on? Attacking a supporting exhibit is arguing against the claim, so
# the same sentence is UNDERMINE. This needs the evidence line-up in the prompt,
# because the direction of "E1 provides no data" depends entirely on whether E1
# was a support or an undermine exhibit.
#
# Roughly 29% of the facts that TRUTH calls NEUTRAL name exactly one exhibit and
# are therefore recoverable under ARGUMENTATIVE - and they split almost evenly
# (167 support-side, 164 undermine-side), so B± barely moves. The rest are
# statements about the debate itself: "the verdict is UNCERTAIN", "the claim
# rests on value judgments". Those stay NEUTRAL under both readings, and that is
# the right outcome for this pipeline: they are the *force* acting on the fact
# population - why an agent drops or keeps a claim - while what we measure is
# its *motion*. A force is not a position.

STANCE_SYSTEM_TRUTH = """\
You label the argumentative direction of atomic propositions in a debate.

For each fact, label its direction relative to the claim under review, assuming
the fact is true. Do not judge whether it is true, credible, or explicitly
cited. Return SUPPORT if it is a reason the claim is correct; UNDERMINE if it
is a reason the claim is incorrect; NEUTRAL if it is irrelevant, purely
definitional/meta, or its direction cannot be determined from the fact alone.

The claim may be phrased negatively or normatively. Read it literally. A fact
about a cause, effect, risk, counterexample, limitation, or cost can bear on a
claim even when it does not repeat the claim's wording. Do not infer a stance
from the identity of the speaker or the panel condition.
"""

STANCE_SYSTEM_ARGUMENTATIVE = """\
You label which side of a debate a proposition puts its speaker on.

For each fact, label its direction relative to the claim under review, assuming
the fact is true. Do not judge whether it is true or credible. Return SUPPORT if
saying it argues for the claim; UNDERMINE if saying it argues against the claim;
NEUTRAL only if it does neither.

A fact can argue a side in two ways, and both count:

  DIRECTLY - it is a reason the claim is correct or incorrect.

  BY ATTACKING AN EXHIBIT - the dossier's exhibits are listed below with the
  side each one argues for. Saying an exhibit is weak, unsupported, irrelevant,
  or off-topic argues AGAINST the side that exhibit was arguing for. "E1 gives
  no data" is UNDERMINE when E1 supports the claim, and SUPPORT when E1
  undermines it. Describing what an exhibit says, without praise or criticism,
  takes the side the exhibit takes.

NEUTRAL is for statements about the debate rather than the claim: that the
evidence is mixed overall, that the question is normative, that no verdict is
possible. These name no side.

The claim may be phrased negatively or normatively. Read it literally. Do not
infer a stance from the identity of the speaker or the panel condition.
"""

STANCE_SYSTEM = STANCE_SYSTEM_TRUTH   # default; --stance-mode selects


class StanceLabel(BaseModel):
    fact_id: str
    stance: Literal["SUPPORT", "UNDERMINE", "NEUTRAL"]


class StanceLabels(BaseModel):
    labels: list[StanceLabel] = Field(default_factory=list)


def load_opencode_key() -> None:
    """Load only the existing local OpenCode key for non-interactive runs."""
    if os.environ.get("OPENCODE_API_KEY"):
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == "OPENCODE_API_KEY":
            os.environ["OPENCODE_API_KEY"] = value.strip().strip("'\"")
            return


def output_fact_ids(store: FactStore) -> list[str]:
    return sorted(
        fact_id for fact_id, fact in store.facts.items()
        if "first_output_token_clock" in fact.properties
    )


def label_batch(llm: LLM, claim: str, facts: list[tuple[str, str]],
                system: str = STANCE_SYSTEM_TRUTH,
                exhibits: list[dict] | None = None) -> dict[str, str]:
    entries = "\n".join(f"- id={fact_id}: {text}" for fact_id, text in facts)
    if exhibits:
        # Which side each exhibit argues for. Without it the argumentative
        # reading cannot direct a criticism, since "E1 gives no data" flips
        # meaning with E1's own side.
        listing = "\n".join(f"- {e['id']} argues {e['stance'].upper()}" for e in exhibits)
        claim = f"{claim}\n</claim>\n\n<exhibits>\n{listing}"
    result = llm.parse(
        system=system,
        user=f"<claim>\n{claim}\n</claim>\n\n<facts>\n{entries}\n</facts>",
        output_format=StanceLabels,
        cache_if=lambda response: bool(response.labels),
    )
    expected = {fact_id for fact_id, _ in facts}
    labels = {label.fact_id: label.stance for label in result.labels if label.fact_id in expected}
    missing = expected - set(labels)
    if missing:
        if len(facts) == 1:
            raise ValueError("response omitted its only fact id")
        # A rare truncated list must not discard a nearly complete batch or be
        # silently assigned a stance. Re-ask just the omitted facts instead.
        lookup = dict(facts)
        for fact_id in sorted(missing):
            labels.update(label_batch(llm, claim, [(fact_id, lookup[fact_id])]))
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--debate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.tokens.json")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--parallel-stores", type=int, default=1,
                        help="Number of independent stores to label concurrently.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--stance-mode", choices=("truth", "argumentative"),
                        default="truth",
                        help="truth: is the fact a reason the claim is correct. "
                             "argumentative: which side does saying it put you on - "
                             "criticising an exhibit argues against that exhibit's side. "
                             "See the module comment; they answer different questions and "
                             "should not be mixed in one analysis.")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(args.glob))
    if not paths:
        parser.error(f"no {args.glob} under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_opencode_key()
    system = (STANCE_SYSTEM_ARGUMENTATIVE if args.stance_mode == "argumentative"
              else STANCE_SYSTEM_TRUTH)
    llm = LLM.opencode(args.model, max_concurrency=args.concurrency, max_tokens=args.max_tokens)
    llm.backend.client = llm.backend.client.with_options(timeout=args.timeout, max_retries=1)
    started = time.monotonic()

    def process(index: int, path: Path) -> str:
        target = args.output_dir / path.name.removesuffix(".tokens.json").replace(".v2", "")
        target = target.with_name(target.name + ".stance.json")
        if target.exists():
            return f"[stance {index}/{len(paths)}] {path.name}: done, skipping"
        debate_path = args.debate_dir / (path.name.removesuffix(".tokens.json") + ".debate.json")
        if not debate_path.exists():
            return f"[stance {index}/{len(paths)}] {path.name}: missing debate, skipping"
        store = FactStore.load(str(path))
        debate = json.loads(debate_path.read_text())
        fact_ids = output_fact_ids(store)
        pairs = [(fact_id, store.facts[fact_id].canonical_text) for fact_id in fact_ids]
        labels: dict[str, str] = {}
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            labels.update(label_batch(
                llm, debate["claim"], batch, system=system,
                exhibits=debate.get("evidence") if args.stance_mode == "argumentative" else None))
            print(f"[stance {index}/{len(paths)}] {path.name} {min(start + len(batch), len(pairs))}/{len(pairs)} facts", flush=True)
        # Recorded per fact: two readings must never be silently mixed.
        for fact_id, stance in labels.items():
            store.facts[fact_id].properties["stance"] = stance
            store.facts[fact_id].properties["stance_label_model"] = args.model
            store.facts[fact_id].properties["stance_mode"] = args.stance_mode
        store.save(str(target))
        elapsed = time.monotonic() - started
        return (f"[stance {index}/{len(paths)}] {path.name} -> {target.name} "
                f"facts={len(labels)} elapsed={elapsed / 60:.1f}m {llm.usage.report(args.model)}")

    with ThreadPoolExecutor(max_workers=args.parallel_stores) as pool:
        futures = {pool.submit(process, index, path): (index, path)
                   for index, path in enumerate(paths, 1)}
        for future in as_completed(futures):
            index, path = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[stance {index}/{len(paths)}] {path.name}: FAILED "
                      f"{type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
