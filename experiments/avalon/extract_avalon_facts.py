#!/usr/bin/env python3
"""Extract atomic propositions from Avalon traces, decontextualised and typed.

Avalon breaks three assumptions the debate extractor was built on.

*Everything is indexical.* "I'll go on this one" is not a proposition until it
says which player and which quest, and neither is recoverable from the sentence.
So each utterance is extracted against a reconstructed state block -- speaker,
quest, proposal, score, leader, team on the table, quest history -- and the
prompt requires every pronoun, "this/last quest", and "the team" to be resolved
to explicit player numbers and quest numbers.

*Speakers lie on purpose.* Deception is the phenomenon, so the extractor must
never repair a false claim, and must never be told who is actually evil. The
roles are in the trace and a separate offline pass can score truth afterwards;
handing them to the extractor would silently delete the thing worth studying.

*"Is a fact" is ambiguous here.* "Quest 3 failed" and "Player 3 is evil" are
both stated flatly, but the first is public record and the second is one
player's claim. Every proposition therefore carries a modality, which is what
lets an analysis separate what the game established from what an agent asserted,
believed, or guessed.

`scope` is recorded but deliberately not used to partition matching. A state
claim restated at a later quest is worth matching precisely because the match
is the observation -- the same sentence about a world that has moved on -- and
provenance carries the quest index, so an analysis can always tell the two
apart. `match_traces.py --partition scope_key` exists for the case where that
is not wanted.

Public speech, private-until-reveal rationales and secret votes are all kept,
each tagged with its visibility, because they answer different questions -- but
they must never be pooled: a flow analysis that reads secret rationales is using
information no other agent could see.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from factflow.llm import LLM  # noqa: E402
from factflow.types import Channel, FactMention, Polarity, Provenance  # noqa: E402

MODALITY = Literal["record", "assertion", "belief", "speculation",
                   "intention", "directive", "strategy"]
SCOPE = Literal["game", "quest", "state"]

SYSTEM = """\
You decompose one utterance from a five-player Avalon game into atomic
propositions for a fact-tracking system.

You are given the game state at the moment the utterance was made. Use it only
to resolve references. You are NOT told anyone's secret role, and you must not
guess it.

RULES

1. ATOMIC. One predicate per proposition. Split every conjunction and list.

2. FULLY DECONTEXTUALISED. Each proposition must stand alone with no access to
   the utterance, the speaker, or the game state. Resolve every reference:
     "I think he's clean"            -> Player 2 is not Evil.
     "I'll go on this one"           -> Player 2 will join the Quest 3 team.
     "that failed"                   -> Quest 2 failed.
     "the team looks fine"           -> The proposed Quest 3 team [0, 2, 4] is acceptable to Player 2.
     "last round they voted no"      -> Player 4 voted to reject the Quest 2 team.
   Write player numbers as "Player N" and quests as "Quest K". Never leave
   "I", "you", "he", "they", "this quest", "last round", or "the team".

   Bare definite descriptions are also unresolved. "the team", "the quest",
   "the leader", "the mission" must become "the Quest K team", the explicit
   roster "[0, 2]", or "Player N". If the state block does not say which, the
   proposition is not extractable -- drop it rather than leaving it vague.

3. NEVER JUDGE TRUTH. Players lie as a matter of play. Extract a false claim
   exactly as stated. Do not correct it, soften it, or mark it as a lie. "I am
   a loyal Servant" is extracted as "Player 3 is a loyal Servant" regardless of
   whether that is true.

4. MODALITY. Label each proposition with how it was put forward:
     record      - restates a public game event that the state block confirms
                   (a quest outcome, a vote tally, who proposed which team)
     assertion   - stated flatly as fact about the world, roles, or players,
                   but not confirmed by the record ("Player 3 is Evil")
     belief      - explicitly framed as the speaker's own mental state
                   ("I trust Player 2", "I am confident Player 3 is Evil")
     speculation - hedged ("probably", "I suspect", "should be", "might")
     intention   - the speaker's own future action ("I will approve this team")
     directive   - a recommendation or request aimed at others
                   ("we should send Player 1", "vote no")
     strategy    - a general claim about how Avalon is best played, naming no
                   specific player and no specific quest ("small teams are
                   safer early", "vote patterns reveal more than speech").
                   Use this whenever the claim would still make sense in a
                   different game; it keeps play maxims out of `assertion`.
   When an utterance both asserts and hedges the same content, use the weaker
   label. A claim about the speaker's own role is an `assertion`, not a
   `belief`: it is a claim about the world.

5. ABOUT. List the player numbers each proposition concerns, [] if none.
   Resolve descriptive references to players into numbers as well: "two
   untested players" -> name them, "a proven success player" -> name them,
   "three players with clean records" -> one proposition per named player. If
   the state block does not identify who is meant, drop the proposition.

6. SCOPE. Say whether the proposition can be the same proposition in a later
   quest:
     game  - one truth for the whole game: roles, alignment, partnerships
             ("Player 3 is Evil", "Player 0 is Merlin"). Restating it later is
             the same proposition.
     quest - tied to one quest or proposal and meaningless outside it ("the
             Quest 3 team [0,1,2] should be approved", "Player 2 will play
             SUCCESS on Quest 3").
     state - describes the game situation as it stands and can flip as the
             game moves ("Player 2 has a clean record", "Player 4 has rejected
             twice"). The same sentence at a later quest is a NEW claim about
             a changed world, not a restatement.
   When unsure between game and state, choose state.

7. SKIP the mechanical action line (TEAM: [...], MISSION: SUCCESS, VOTE:
   APPROVE) -- the game record already holds it. Extract the reasoning around
   it. Skip questions and pure pleasantries.

8. POLARITY. Use "deny" when the proposition is asserted as not holding, and
   write the proposition itself affirmatively.

Return JSON: {"propositions": [{"text": ..., "modality": ..., "scope": ...,
"about": [ints], "polarity": "affirm"|"deny", "quote": "supporting span"}]}. Return an empty
list if the utterance carries no proposition.
"""


class Prop(BaseModel):
    text: str
    modality: MODALITY = "assertion"
    scope: SCOPE = "state"
    about: list[int] = Field(default_factory=list)
    polarity: Literal["affirm", "deny"] = "affirm"
    quote: str = ""


class Props(BaseModel):
    propositions: list[Prop] = Field(default_factory=list)


def state_block(events: list[dict], i: int) -> str:
    """What was publicly true when event i was produced."""
    ev = events[i]
    good = evil = 0
    history: list[str] = []
    team: list[int] | None = None
    leader: int | None = None
    for e in events[:i]:
        if e["phase"] == "mission-result":
            ok = "succeeded" in (e.get("output") or "")
            good, evil = (good + 1, evil) if ok else (good, evil + 1)
            history.append(f"  Quest {e['quest']} {'succeeded' if ok else 'failed'}")
        elif e["phase"] == "proposal":
            team = e.get("action")
            leader = int(e["actor"])
        elif e["phase"] == "team-vote-result" and "rejected" in (e.get("output") or ""):
            team = None
    lines = [f"Game: five players, Player 0 through Player 4.",
             f"Now: Quest {ev['quest']}, proposal attempt {ev['proposal']}.",
             f"Score so far: Good {good}, Evil {evil}."]
    if history:
        lines.append("Completed quests:")
        lines += history
    if leader is not None:
        lines.append(f"Current quest leader: Player {leader}.")
    if team is not None:
        lines.append(f"Team currently on the table: {team}.")
    if ev.get("actor") is not None:
        lines.append(f"The utterance below was made by Player {ev['actor']}.")
    lines.append(f"Its visibility: {ev.get('visibility')}.")
    return "\n".join(lines)


PRINT_LOCK = threading.Lock()


def extract_event(llm: LLM, events: list[dict], i: int, execution_id: str
                  ) -> list[FactMention]:
    ev = events[i]
    text = (ev.get("output") or "").strip()
    if not text:
        return []
    if ev["kind"] == "environment":
        # Environment lines are the record itself; they need no model call and
        # must not be re-interpreted as somebody's claim.
        prov = Provenance(execution_id=execution_id, agent_id=None,
                          round=int(ev["quest"]), channel=Channel.SOURCE,
                          doc_id=f"env-{ev['event_id']}",
                          extra={"modality": "record", "scope": "quest",
                                 "scope_key": f"q{ev['quest']}", "about": [],
                                 "visibility": ev.get("visibility"),
                                 "phase": ev["phase"], "quest": int(ev["quest"]),
                                 "proposal": int(ev["proposal"])})
        m = FactMention(mention_id="", text=ev.get("extraction_text") or text,
                        quote=text, provenance=prov)
        return [FactMention(mention_id=FactMention.make_id(m.text, prov),
                            text=m.text, quote=text, provenance=prov)]

    user = (f"<state>\n{state_block(events, i)}\n</state>\n\n"
            f"<utterance>\n{text}\n</utterance>")
    # An occasional empty completion should cost one utterance, not the game:
    # raising here aborted a whole trace after four hundred good extractions.
    res = None
    for budget in (None, 8000):
        try:
            res = llm.parse(system=SYSTEM, user=user, output_format=Props,
                            max_tokens=budget, cache_if=lambda r: True)
            break
        except Exception as exc:                                  # noqa: BLE001
            with PRINT_LOCK:
                print(f"    事件 {ev['event_id']} 抽取失败（{type(exc).__name__}）"
                      f"{'，加大预算重试' if budget is None else '，跳过'}", flush=True)
    if res is None:
        return []
    out = []
    for p in res.propositions:
        if not p.text.strip():
            continue
        prov = Provenance(execution_id=execution_id, agent_id=f"P{ev['actor']}",
                          round=int(ev["quest"]), channel=Channel.OUTPUT,
                          extra={"modality": p.modality, "scope": p.scope,
                                 "scope_key": ("game" if p.scope == "game"
                                               else f"q{ev['quest']}"),
                                 "about": p.about,
                                 "visibility": ev.get("visibility"),
                                 "phase": ev["phase"], "quest": int(ev["quest"]),
                                 "proposal": int(ev["proposal"]),
                                 "event_id": ev["event_id"]})
        out.append(FactMention(
            mention_id=FactMention.make_id(p.text.strip(), prov),
            text=p.text.strip(), quote=p.quote or text[:200],
            polarity=Polarity.NEGATE if p.polarity == "deny" else Polarity.AFFIRM,
            provenance=prov))
    return out


def one_trace(llm: LLM, path: Path, suffix: str) -> tuple[str, int, float]:
    d = json.loads(path.read_text())
    events = d["events"]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        groups = list(pool.map(
            lambda i: extract_event(llm, events, i, d["execution_id"]),
            range(len(events))))
    mentions = [m for g in groups for m in g]
    out = path.with_name(path.name.replace(".trace.json", "") + suffix)
    out.write_text(json.dumps(
        {"mentions": {m.mention_id: json.loads(m.model_dump_json()) for m in mentions},
         "facts": {}, "mention_to_fact": {}, "relations": [],
         "derived_from": {"file": path.name, "stage": "avalon-extract",
                          "model": llm.model,
                          "note": "roles withheld from the extractor by design"}},
        ensure_ascii=False), encoding="utf-8")
    return d["execution_id"], len(mentions), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "avalon_5p_deepseek_v4_flash")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--suffix", default=".atomized.json")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--parallel-traces", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--thinking", default="disabled",
                    help="deepseek-v4-flash 默认高推理，会把预算耗光后返回空串")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    paths = sorted(a.dir.glob("*.trace.json"))
    if not a.redo:
        paths = [p for p in paths
                 if not p.with_name(p.name.replace(".trace.json", "")
                                    + a.suffix).exists()]
    if not paths:
        print("没有待处理的 trace")
        return 0
    llm = LLM.opencode(a.model, max_concurrency=a.concurrency,
                       max_tokens=a.max_tokens)
    llm.backend.thinking = a.thinking or None
    print(f"{len(paths)} 局，模型 {a.model}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.parallel_traces) as pool:
        for eid, n, dt in pool.map(lambda p: one_trace(llm, p, a.suffix), paths):
            print(f"  {eid}  {n} 条命题  {dt:.0f}s", flush=True)
    print(f"\n用时 {time.time()-t0:.0f}s   {llm.usage.report(a.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
