#!/usr/bin/env python3
"""Run reproducible five-player Avalon self-play and save channel-aware traces."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from factflow.backends import Usage  # noqa: E402
from factflow.llm import LLM  # noqa: E402
from run_perspectrum import load_opencode_key  # noqa: E402


LABEL = r"\*{{0,2}}{name}\*{{0,2}}\s*:\*{{0,2}}"
TEAM_RE = re.compile(LABEL.format(name="TEAM") + r"\s*\[([^\]]*)\]", re.IGNORECASE)
VOTE_RE = re.compile(
    LABEL.format(name="VOTE") + r"\s*\*{0,2}(APPROVE|REJECT|YES|NO)\b",
    re.IGNORECASE,
)
MISSION_RE = re.compile(
    LABEL.format(name="MISSION") + r"\s*\*{0,2}(SUCCESS|PASS|FAIL)\b",
    re.IGNORECASE,
)
TARGET_RE = re.compile(
    LABEL.format(name="TARGET") + r"\s*\*{0,2}(?:PLAYER\s*|P)?(\d+)\b",
    re.IGNORECASE,
)
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_team(text: str, team_size: int) -> list[int] | None:
    matches = TEAM_RE.findall(text or "")
    if not matches:
        return None
    team = [int(x) for x in re.findall(r"\d+", matches[-1])]
    if len(team) != team_size or len(set(team)) != team_size:
        return None
    return team if all(0 <= player < 5 for player in team) else None


def parsed_team_size(text: str) -> int:
    matches = TEAM_RE.findall(text or "")
    return len(re.findall(r"\d+", matches[-1])) if matches else 0


def parse_discussion(text: str, word_limit: int) -> str | None:
    text = (text or "").strip()
    return text if text and len(text.split()) <= word_limit else None


def parse_team_vote(text: str) -> str | None:
    matches = VOTE_RE.findall(text or "")
    if not matches:
        return None
    return {"YES": "APPROVE", "NO": "REJECT"}.get(matches[-1].upper(),
                                                         matches[-1].upper())


def parse_mission_vote(text: str, good_player: bool) -> str | None:
    matches = MISSION_RE.findall(text or "")
    if not matches:
        return None
    action = "SUCCESS" if matches[-1].upper() == "PASS" else matches[-1].upper()
    if good_player and action != "SUCCESS":
        return None
    return action


def parse_target(text: str, assassin: int) -> int | None:
    matches = TARGET_RE.findall(text or "")
    if not matches:
        return None
    target = int(matches[-1])
    return target if 0 <= target < 5 and target != assassin else None


def role_assignment(index: int, roles: list[str]) -> dict[int, str]:
    """Two balanced five-game blocks with ten distinct role arrangements."""
    if len(roles) != 5:
        raise ValueError("five-player Avalon requires exactly five role slots")
    if index // 5 % 2 == 0:
        base = list(roles)
    else:
        # Servants are adjacent in the first block and separated in the second,
        # so the second set of rotations cannot duplicate the first set.
        base = ["Merlin", "Servant", "Minion", "Servant", "Assassin"]
    shift = index % 5
    return {player: base[(player + shift) % 5] for player in range(5)}


def private_observation(player: int, roles: dict[int, str]) -> str:
    role = roles[player]
    evil = [p for p, r in roles.items() if r in {"Minion", "Assassin"}]
    if role == "Merlin":
        return (f"Your hidden role is Merlin on the Good side. Privately, you know "
                f"that Players {evil[0]} and {evil[1]} are Evil, but this knowledge "
                "must not be stated as privileged knowledge in public.")
    if role in {"Minion", "Assassin"}:
        partner = next(p for p in evil if p != player)
        return (f"Your hidden role is {role} on the Evil side. Privately, you know "
                f"that Player {partner} is your Evil partner ({roles[partner]}).")
    return ("Your hidden role is Servant on the Good side. You have no private "
            "knowledge of any other player's alignment.")


def validate_config(raw: dict[str, Any]) -> None:
    game = raw["game"]
    if len(game["players"]) != 5:
        raise ValueError("this runner supports exactly five players")
    expected = sorted(["Merlin", "Servant", "Servant", "Minion", "Assassin"])
    if sorted(game["roles"]) != expected:
        raise ValueError(f"roles must be {expected}")
    if game["quest_team_sizes"] != [2, 3, 2, 3, 3]:
        raise ValueError("five-player quest sizes must be 2,3,2,3,3")
    if game["fails_required"] != [1, 1, 1, 1, 1]:
        raise ValueError("all five-player quests require one fail card")


@dataclass
class Answer:
    text: str
    action: Any
    invalid_attempts: list[dict[str, str]]


class ActionValidationError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, str]]):
        super().__init__(message)
        self.attempts = attempts


class AvalonGame:
    def __init__(self, config: dict[str, Any], game_index: int, outdir: Path):
        self.config = config
        self.game_index = game_index
        self.game_number = game_index + 1
        self.game_id = f"avalon-5p-{self.game_number:02d}"
        self.seed = int(config["seed"]) + game_index
        self.rng = random.Random(self.seed)
        self.roles = role_assignment(game_index, list(config["game"]["roles"]))
        self.initial_leader = (2 * game_index + game_index // 5) % 5
        self.leader = self.initial_leader
        self.quest = 1
        self.proposal = 1
        self.successes = 0
        self.failures = 0
        self.quest_outcomes: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.public_history: list[str] = []
        self.private_history: dict[int, list[str]] = {p: [] for p in range(5)}
        self.seq = 0
        self.outdir = outdir
        self.partial_path = outdir / f"{self.game_id}.partial.json"
        self.final_path = outdir / f"{self.game_id}.trace.json"
        self.started = time.time()
        self.llm = self._make_llm()

    def _make_llm(self) -> LLM:
        model = self.config["model"]
        provider = self.config.get("provider", "opencode")
        makers = {"opencode": LLM.opencode, "deepseek": LLM.deepseek,
                  "openai": LLM.openai}
        if provider not in makers:
            raise ValueError(f"unsupported provider {provider!r}")
        llm = makers[provider](model, max_concurrency=5,
                               cache_dir=str(self.outdir / ".cache"))
        llm.backend.reasoning_effort = self.config.get("reasoning_effort")
        llm.backend.thinking = self.config.get("thinking")
        llm.backend.client = llm.backend.client.with_options(
            timeout=float(self.config.get("timeout_seconds", 180)), max_retries=2)
        return llm

    def system_prompt(self, player: int) -> str:
        prompts = self.config["prompts"]
        role = self.roles[player]
        return "\n\n".join([
            prompts["rules"],
            prompts["common_system"],
            f"You are Player {player}.",
            private_observation(player, self.roles),
            prompts["role_strategies"][role],
        ])

    def state_text(self) -> str:
        outcomes = ", ".join(
            f"Q{x['quest']}={'SUCCESS' if x['success'] else 'FAIL'}"
            for x in self.quest_outcomes) or "none yet"
        return (f"Current state: Quest {self.quest}; proposal {self.proposal}/5; "
                f"leader Player {self.leader}; score Good {self.successes}, "
                f"Evil {self.failures}; completed quests: {outcomes}.")

    def view_for(self, player: int) -> str:
        public = "\n".join(self.public_history) or "No public events yet."
        private = "\n".join(self.private_history[player]) or "No prior private actions."
        return (f"{self.state_text()}\n\nPUBLIC HISTORY (seen by every player):\n"
                f"{public}\n\nYOUR PRIVATE ACTION HISTORY (seen only by you):\n{private}")

    def ask(self, player: int, phase: str, instruction: str,
            parser: Callable[[str], Any] | None = None,
            repair_hint: Callable[[str], str] | None = None) -> Answer:
        system = self.system_prompt(player)
        base_user = self.view_for(player) + "\n\nCURRENT ACTION:\n" + instruction
        invalid: list[dict[str, str]] = []
        user = base_user
        cap = int(self.config.get("max_tokens", 1200))
        for attempt in range(1, 4):
            text = self.llm.chat(
                system=system,
                user=user,
                model=self.config["model"],
                max_tokens=cap,
                temperature=float(self.config.get("temperature", 0.95)),
                sample_id=f"{self.game_id}:{phase}:p{player}:q{self.quest}:"
                          f"a{self.proposal}:try{attempt}",
            ).strip()
            action = text if parser is None and text else (parser(text) if text else None)
            if action is not None:
                return Answer(text=text, action=action, invalid_attempts=invalid)
            reason = "empty response" if not text else "missing or invalid required action line"
            invalid.append({"output": text, "reason": reason})
            if not text:
                cap = min(cap * 2, 16000)
            detail = repair_hint(text) if repair_hint is not None else reason
            user = (base_user + "\n\nYour previous response was invalid because it had a "
                    f"missing or illegal action line. {detail} Return a legal move using "
                    "the exact final-line format requested. Previous response:\n" + text[:1500])
        raise ActionValidationError(
            f"{self.game_id} Player {player} {phase}: invalid after 3 attempts", invalid)

    def ask_many(self, requests: list[tuple[int, str, str, Callable[[str], Any] | None,
                                             Callable[[str], str] | None]]) -> dict[int, Answer]:
        """Run actions that are simultaneous under the game rules in parallel."""
        def one(request):
            player, phase, instruction, parser, repair_hint = request
            return player, self.ask(player, phase, instruction, parser, repair_hint)

        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            return dict(pool.map(one, requests))

    def record_environment(self, phase: str, text: str,
                           action: dict[str, Any] | None = None) -> None:
        self.seq += 1
        event = {
            "event_id": self.seq,
            "kind": "environment",
            "quest": self.quest,
            "proposal": self.proposal,
            "phase": phase,
            "actor": None,
            "visibility": "public",
            "recipients": list(range(5)),
            "output": text,
            "extraction_text": text,
            "action": action,
        }
        self.events.append(event)
        self.public_history.append(f"[E{self.seq}] {text}")
        self.checkpoint()

    def record_agent(self, player: int, phase: str, answer: Answer,
                     visibility: str, public_text: str | None,
                     extraction_text: str) -> None:
        self.seq += 1
        recipients = list(range(5)) if visibility == "public" else [player]
        event = {
            "event_id": self.seq,
            "kind": "agent",
            "quest": self.quest,
            "proposal": self.proposal,
            "phase": phase,
            "actor": player,
            "role": self.roles[player],
            "visibility": visibility,
            "recipients": recipients,
            "output": answer.text,
            "extraction_text": extraction_text,
            "action": answer.action,
            "invalid_attempts": answer.invalid_attempts,
        }
        self.events.append(event)
        if visibility == "public" and public_text:
            self.public_history.append(f"[E{self.seq}] {public_text}")
        else:
            self.private_history[player].append(
                f"[E{self.seq} {phase}] {answer.text}")
        self.checkpoint()

    def trace(self, complete: bool = False, outcome: str | None = None) -> dict[str, Any]:
        transcript = {
            f"P{event['actor']}|{event['event_id']}": event["output"]
            for event in self.events if event["kind"] == "agent"
        }
        delivery = {
            f"P{event['actor']}|{event['event_id']}": {
                "visibility": event["visibility"],
                "recipients": [f"P{x}" for x in event["recipients"]],
                "quest": event["quest"],
                "proposal": event["proposal"],
                "phase": event["phase"],
            }
            for event in self.events if event["kind"] == "agent"
        }
        return {
            "execution_id": self.game_id,
            "dataset": "avalon-5p",
            "game_number": self.game_number,
            "seed": self.seed,
            "model": self.config["model"],
            "provider": self.config.get("provider", "opencode"),
            "temperature": self.config.get("temperature", 0.95),
            "thinking": self.config.get("thinking"),
            "reasoning_effort": self.config.get("reasoning_effort"),
            "complete": complete,
            "roles": {f"P{p}": role for p, role in self.roles.items()},
            "initial_leader": self.initial_leader,
            "private_observations": {
                f"P{p}": private_observation(p, self.roles) for p in range(5)
            },
            "rules": {
                "quest_team_sizes": self.config["game"]["quest_team_sizes"],
                "fails_required": self.config["game"]["fails_required"],
                "max_proposals": self.config["game"]["max_proposals"],
                "rules_text": self.config["prompts"]["rules"],
            },
            "events": self.events,
            "transcript": transcript,
            "delivery": delivery,
            "quest_outcomes": self.quest_outcomes,
            "score": {"good": self.successes, "evil": self.failures},
            "outcome": outcome,
            "elapsed_seconds": round(time.time() - self.started, 3),
            "accounting": {
                "generation": {
                    "calls": self.llm.usage.calls,
                    "input_tokens": self.llm.usage.input_tokens,
                    "output_tokens": self.llm.usage.output_tokens,
                    "cost_usd": self.llm.usage.cost(self.config["model"]),
                }
            },
        }

    def checkpoint(self) -> None:
        write_json_atomic(self.partial_path, self.trace())

    def discuss(self) -> None:
        order = [(self.leader + offset) % 5 for offset in range(5)]
        template = self.config["prompts"]["discussion"]
        instruction = template.format(
            word_limit=self.config["game"]["discussion_word_limit"])
        word_limit = int(self.config["game"]["discussion_word_limit"])
        answers = self.ask_many([
            (player, "discussion", instruction,
             lambda text, limit=word_limit: parse_discussion(text, limit),
             lambda text, limit=word_limit: (
                 f"The public message has {len(text.split())} words; the maximum is {limit}."
             ))
            for player in order
        ])
        for player in order:
            answer = answers[player]
            public = f"Player {player} said: {answer.text}"
            self.record_agent(
                player, "discussion", answer, "public", public,
                f"Player {player} publicly said: \"{answer.text}\"",
            )

    def propose(self) -> tuple[int, list[int]]:
        team_size = self.config["game"]["quest_team_sizes"][self.quest - 1]
        answer = self.ask(
            self.leader, "proposal",
            self.config["prompts"]["proposal"].format(
                team_size=team_size, quest=self.quest),
            parser=lambda text: parse_team(text, team_size),
            repair_hint=lambda text: (
                f"Quest {self.quest} requires exactly {team_size} distinct player ids; "
                f"your last TEAM line contained {parsed_team_size(text)}."
            ),
        )
        leader = self.leader
        team = answer.action
        self.record_agent(
            leader, "proposal", answer, "public",
            f"Player {leader} proposed team {team}. Rationale: {answer.text}",
            f"Player {leader} publicly proposed team {team}. "
            f"Player {leader}'s public rationale was: \"{answer.text}\"",
        )
        self.leader = (self.leader + 1) % 5
        return leader, team

    def team_vote(self, team: list[int]) -> bool:
        votes: dict[int, str] = {}
        instruction = self.config["prompts"]["team_vote"].format(
            team=team, quest=self.quest)
        answers = self.ask_many([
            (player, "team-vote", instruction, parse_team_vote, None)
            for player in range(5)
        ])
        for player in range(5):
            answer = answers[player]
            votes[player] = answer.action
            self.record_agent(
                player, "team-vote", answer, "private-until-reveal", None,
                f"Player {player} privately decided to {answer.action} team {team} "
                f"for Quest {self.quest}. Player {player}'s private rationale was: "
                f"\"{answer.text}\"",
            )
        approved = sum(v == "APPROVE" for v in votes.values()) >= 3
        listing = ", ".join(f"Player {p} {vote}" for p, vote in votes.items())
        result = "approved" if approved else "rejected"
        self.record_environment(
            "team-vote-result",
            f"For Quest {self.quest}, team {team} received these public votes: "
            f"{listing}. The team was {result}.",
            {"team": team, "votes": votes, "approved": approved},
        )
        return approved

    def execute_mission(self, team: list[int]) -> bool:
        actions: dict[int, str] = {}
        instruction = self.config["prompts"]["mission_vote"].format(
            team=team, quest=self.quest)
        requests = []
        for player in team:
            good = self.roles[player] in {"Merlin", "Servant"}
            requests.append(
                (player, "mission-vote", instruction,
                 lambda text, is_good=good: parse_mission_vote(text, is_good), None)
            )
        answers = self.ask_many(requests)
        for player in team:
            answer = answers[player]
            actions[player] = answer.action
            self.record_agent(
                player, "mission-vote", answer, "secret", None,
                f"Player {player} secretly chose {answer.action} on Quest {self.quest}. "
                f"Player {player}'s private rationale was: \"{answer.text}\"",
            )
        fail_count = sum(action == "FAIL" for action in actions.values())
        success = fail_count == 0
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.quest_outcomes.append({
            "quest": self.quest,
            "team": team,
            "success": success,
            "fail_count": fail_count,
            "secret_actions": actions,
        })
        self.record_environment(
            "mission-result",
            f"Quest {self.quest} with team {team} "
            f"{'succeeded' if success else 'failed'} with {fail_count} FAIL card(s). "
            f"The score is Good {self.successes}, Evil {self.failures}.",
            {"team": team, "success": success, "fail_count": fail_count},
        )
        return success

    def assassinate(self) -> str:
        assassin = next(p for p, role in self.roles.items() if role == "Assassin")
        merlin = next(p for p, role in self.roles.items() if role == "Merlin")
        self.record_environment(
            "assassination-start",
            "Good completed three successful quests. The Assassin now chooses a target.",
        )
        answer = self.ask(
            assassin, "assassination", self.config["prompts"]["assassinate"],
            parser=lambda text: parse_target(text, assassin),
        )
        target = answer.action
        self.record_agent(
            assassin, "assassination", answer, "private-until-reveal", None,
            f"Player {assassin}, the Assassin, selected Player {target} as the "
            f"assassination target. The Assassin's rationale was: \"{answer.text}\"",
        )
        if target == merlin:
            outcome = "evil-assassinated-merlin"
            text = f"The Assassin targeted Player {target}, who was Merlin. Evil won."
        else:
            outcome = "good-survived-assassination"
            text = (f"The Assassin targeted Player {target}, who was not Merlin. "
                    f"Player {merlin} was Merlin, so Good won.")
        self.record_environment("assassination-result", text,
                                {"assassin": assassin, "target": target,
                                 "merlin": merlin, "hit": target == merlin})
        return outcome

    def run(self) -> dict[str, Any]:
        self.record_environment(
            "game-start",
            f"A five-player Avalon game began. Player {self.initial_leader} is the "
            "first quest leader.",
        )
        outcome: str | None = None
        while outcome is None:
            log(f"[{self.game_id}] Q{self.quest} proposal {self.proposal}: "
                f"leader P{self.leader}, discuss")
            self.discuss()
            _leader, team = self.propose()
            if self.proposal == int(self.config["game"]["max_proposals"]):
                self.record_environment(
                    "team-auto-approved",
                    f"Team {team} was the fifth proposal for Quest {self.quest} and "
                    "was automatically approved.",
                    {"team": team, "approved": True},
                )
                approved = True
            else:
                approved = self.team_vote(team)

            if not approved:
                self.proposal += 1
                log(f"[{self.game_id}] Q{self.quest}: team rejected; next proposal")
                continue

            self.execute_mission(team)
            log(f"[{self.game_id}] Q{self.quest} complete: "
                f"Good {self.successes} - Evil {self.failures}")
            if self.failures >= 3:
                outcome = "evil-three-failed-quests"
                self.record_environment("game-result", "Three quests failed. Evil won.")
            elif self.successes >= 3:
                outcome = self.assassinate()
            else:
                self.quest += 1
                self.proposal = 1

        role_list = ", ".join(f"Player {p} was {r}" for p, r in self.roles.items())
        self.record_environment("role-reveal", f"The game ended. Roles: {role_list}.")
        final = self.trace(complete=True, outcome=outcome)
        write_json_atomic(self.final_path, final)
        self.partial_path.unlink(missing_ok=True)
        (self.outdir / f"{self.game_id}.failed.json").unlink(missing_ok=True)
        return final


def run_one(config: dict[str, Any], game_index: int, outdir: Path) -> tuple[int, dict[str, Any]]:
    game = AvalonGame(config, game_index, outdir)
    try:
        trace = game.run()
    except Exception as exc:
        failure = game.trace(complete=False)
        failure["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ActionValidationError):
            failure["invalid_action_attempts"] = exc.attempts
        write_json_atomic(outdir / f"{game.game_id}.failed.json", failure)
        raise
    usage = trace["accounting"]["generation"]
    log(f"[{game.game_id}] DONE {trace['outcome']}; {len(trace['events'])} events; "
        f"{usage['calls']} calls; {usage['input_tokens']:,} in / "
        f"{usage['output_tokens']:,} out; {trace['elapsed_seconds']:.0f}s")
    return game_index, trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "experiments/datasets/avalon_5p.yaml")
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "experiments/avalon_5p_deepseek_v4_flash")
    ap.add_argument("--games", type=int, help="number of game ids starting at 1")
    ap.add_argument("--parallel-games", type=int)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    load_opencode_key()
    args.outdir.mkdir(parents=True, exist_ok=True)
    count = args.games or int(config.get("games", 10))
    parallel = args.parallel_games or int(config.get("parallel_games", 2))
    todo = [i for i in range(count)
            if not (args.outdir / f"avalon-5p-{i + 1:02d}.trace.json").exists()]
    log(f"plan: {count} five-player games; {len(todo)} pending; model "
        f"{config['model']}; temperature {config['temperature']}; parallel {parallel}")
    if not todo:
        log("all requested traces already exist")
        return 0

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run_one, config, i, args.outdir): i for i in todo}
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, trace = future.result()
                completed.append(trace)
            except Exception as exc:  # noqa: BLE001
                log(f"[avalon-5p-{i + 1:02d}] FAILED: {type(exc).__name__}: {exc}")

    traces = []
    for path in sorted(args.outdir.glob("avalon-5p-*.trace.json")):
        traces.append(json.loads(path.read_text()))
    usage = Usage()
    for trace in traces:
        row = trace["accounting"]["generation"]
        usage.input_tokens += row["input_tokens"]
        usage.output_tokens += row["output_tokens"]
        usage.calls += row["calls"]
    manifest = {
        "config": str(args.config),
        "expected_games": count,
        "complete_games": len(traces),
        "model": config["model"],
        "temperature": config["temperature"],
        "outcomes": {trace["execution_id"]: trace["outcome"] for trace in traces},
        "generation": {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost(config["model"]),
        },
    }
    write_json_atomic(args.outdir / "manifest.json", manifest)
    log(f"complete: {len(traces)}/{count}; {usage.report(config['model'])}")
    return 0 if len(traces) >= count else 1


if __name__ == "__main__":
    raise SystemExit(main())
