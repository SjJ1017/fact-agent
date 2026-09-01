"""Attach a reproducible visible-token clock to extracted fact mentions.

The original debate traces save text but not provider usage, so this module does
not claim to recover billed or hidden-reasoning tokens.  It counts the visible
prompt and completion text with a fixed locally cached tokenizer instead.  A
trace clock is linearized by (round, agent) solely for accounting: panelists in
the same round were generated concurrently.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from factflow.types import Channel, FactMention, FactStore


TOKENIZER_ID = "BAAI/bge-base-en-v1.5"
CLOCK_ORDER = "round_then_agent"


@dataclass(frozen=True)
class TurnClock:
    """Visible-token accounting for one generated panelist turn."""

    prompt_tokens: int
    output_tokens: int
    output_before_turn: int
    total_before_turn: int


class VisibleTokenCounter:
    """A fixed tokenizer used only for comparable trace-length estimates."""

    def __init__(self, tokenizer_id: str = TOKENIZER_ID) -> None:
        self.tokenizer_id = tokenizer_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
        # We only encode text; the encoder's sequence length warning is about
        # model inference and would otherwise be noisy for long dossiers.
        self.tokenizer.model_max_length = 10**30

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))


def ordered_turns(debate: dict[str, Any]) -> list[str]:
    return sorted(
        debate["transcript"],
        key=lambda key: (int(key.split("|", 1)[1]), key.split("|", 1)[0]),
    )


def turn_clocks(debate: dict[str, Any], counter: VisibleTokenCounter) -> dict[str, TurnClock]:
    """Return deterministic clock positions for every saved output turn."""
    output_total = 0
    total = 0
    clocks: dict[str, TurnClock] = {}
    for key in ordered_turns(debate):
        prompt_tokens = counter.count(debate.get("prompts", {}).get(key, ""))
        output_tokens = counter.count(debate["transcript"][key])
        clocks[key] = TurnClock(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_before_turn=output_total,
            total_before_turn=total,
        )
        output_total += output_tokens
        total += prompt_tokens + output_tokens
    return clocks


def _quote_start(text: str, quote: str | None) -> int | None:
    if not quote:
        return None
    exact = text.find(quote)
    if exact >= 0:
        return exact
    quote = quote.strip()
    if not quote:
        return None
    whitespace_tolerant = re.escape(quote).replace(r"\ ", r"\s+")
    match = re.search(whitespace_tolerant, text)
    if match is None:
        return None
    return match.start()


def mention_clock(
    mention: FactMention,
    debate: dict[str, Any],
    clocks: dict[str, TurnClock],
    counter: VisibleTokenCounter,
) -> dict[str, Any] | None:
    """Token position at a mention's supporting quote, or end-of-turn fallback."""
    provenance = mention.provenance
    if provenance.channel != Channel.OUTPUT or provenance.agent_id is None or provenance.round is None:
        return None
    key = f"{provenance.agent_id}|{provenance.round}"
    turn_text = debate["transcript"].get(key)
    clock = clocks.get(key)
    if turn_text is None or clock is None:
        return None
    start = _quote_start(turn_text, mention.quote)
    quote_found = start is not None
    output_prefix_tokens = counter.count(turn_text[:start]) if quote_found else clock.output_tokens
    return {
        "tokenizer": counter.tokenizer_id,
        "clock_order": CLOCK_ORDER,
        "measurement": "estimated_visible_tokens",
        "quote_found": quote_found,
        "turn_prompt_tokens": clock.prompt_tokens,
        "turn_output_tokens": clock.output_tokens,
        "output_prefix_tokens": output_prefix_tokens,
        "cumulative_visible_output_tokens": clock.output_before_turn + output_prefix_tokens,
        "cumulative_visible_total_tokens": clock.total_before_turn + clock.prompt_tokens + output_prefix_tokens,
    }


def annotate_mentions(
    mentions: Iterable[FactMention], debate: dict[str, Any], counter: VisibleTokenCounter
) -> list[FactMention]:
    """Copy output mentions with per-fact token-clock metadata in provenance.extra."""
    clocks = turn_clocks(debate, counter)
    annotated: list[FactMention] = []
    for mention in mentions:
        clock = mention_clock(mention, debate, clocks, counter)
        if clock is None:
            annotated.append(mention)
            continue
        extra = dict(mention.provenance.extra)
        extra["token_clock"] = clock
        annotated.append(mention.model_copy(update={
            "provenance": mention.provenance.model_copy(update={"extra": extra}),
        }))
    return annotated


def annotate_store(store: FactStore, debate: dict[str, Any], counter: VisibleTokenCounter) -> FactStore:
    """Annotate an existing store and record each canonical fact's first output occurrence."""
    annotated = annotate_mentions(store.mentions.values(), debate, counter)
    store.mentions = {mention.mention_id: mention for mention in annotated}
    for fact in store.facts.values():
        candidates = [
            store.mentions[mention_id]
            for mention_id in fact.mention_ids
            if "token_clock" in store.mentions[mention_id].provenance.extra
        ]
        if not candidates:
            continue
        first = min(
            candidates,
            key=lambda mention: (
                mention.provenance.extra["token_clock"]["cumulative_visible_output_tokens"],
                mention.provenance.round or 0,
                mention.provenance.agent_id or "",
                mention.mention_id,
            ),
        )
        fact.properties["first_output_token_clock"] = {
            "mention_id": first.mention_id,
            "agent_id": first.provenance.agent_id,
            "round": first.provenance.round,
            **first.provenance.extra["token_clock"],
        }
    return store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--glob", default="*.v2.json")
    parser.add_argument("--suffix", default=".tokens.json")
    parser.add_argument("--output-dir", type=Path,
                        help="Write annotated stores here rather than beside the input stores.")
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    args = parser.parse_args()

    paths = sorted(args.out_dir.glob(args.glob))
    if not paths:
        parser.error(f"no {args.glob} under {args.out_dir}")
    counter = VisibleTokenCounter(args.tokenizer)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        debate_path = path.with_name(path.name.removesuffix(".v2.json") + ".debate.json")
        if not debate_path.exists():
            print(f"[{index}/{len(paths)}] {path.name}: missing {debate_path.name}, skipping")
            continue
        store = FactStore.load(str(path))
        debate = json.loads(debate_path.read_text())
        target_name = path.name.removesuffix(".v2.json") + args.suffix
        target = (args.output_dir / target_name) if args.output_dir is not None else path.with_name(target_name)
        annotate_store(store, debate, counter).save(str(target))
        print(f"[{index}/{len(paths)}] {path.name} -> {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
