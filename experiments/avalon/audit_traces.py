#!/usr/bin/env python3
"""Offline structural and content sanity checks for generated Avalon traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def audit(path: Path) -> dict:
    trace = json.loads(path.read_text())
    events = trace["events"]
    agent_events = [event for event in events if event["kind"] == "agent"]
    discussions = [event for event in agent_events if event["phase"] == "discussion"]
    votes = [event for event in agent_events if event["phase"] == "team-vote"]
    missions = [event for event in agent_events if event["phase"] == "mission-vote"]
    normalized = [" ".join(event["output"].lower().split()) for event in discussions]
    duplicates = len(normalized) - len(set(normalized))
    roles = list(trace["roles"].values())
    legal_roles = Counter(roles) == Counter(
        ["Merlin", "Servant", "Servant", "Minion", "Assassin"])
    good_illegal = [
        event["event_id"] for event in missions
        if event["role"] in {"Merlin", "Servant"} and event["action"] != "SUCCESS"
    ]
    bad_visibility = [
        event["event_id"] for event in agent_events
        if ((event["phase"] in {"team-vote", "mission-vote", "assassination"}
             and len(event["recipients"]) != 1)
            or (event["phase"] in {"discussion", "proposal"}
                and len(event["recipients"]) != 5))
    ]
    invalid_attempts = sum(len(event.get("invalid_attempts", [])) for event in agent_events)
    words = [len(event["output"].split()) for event in discussions]
    mixed_votes = bool(votes) and len({event["action"] for event in votes}) > 1
    errors = []
    if not trace.get("complete"):
        errors.append("trace is incomplete")
    if not legal_roles:
        errors.append("illegal role multiset")
    if not 3 <= len(trace["quest_outcomes"]) <= 5:
        errors.append("game did not end after 3-5 quests")
    if any(not event.get("output", "").strip() for event in events):
        errors.append("empty event output")
    if good_illegal:
        errors.append(f"Good player chose FAIL at events {good_illegal}")
    if bad_visibility:
        errors.append(f"incorrect channel visibility at events {bad_visibility}")
    if duplicates > max(1, len(discussions) // 10):
        errors.append(f"too many exact duplicate discussions: {duplicates}")
    return {
        "file": path.name,
        "ok": not errors,
        "errors": errors,
        "outcome": trace.get("outcome"),
        "quests": len(trace["quest_outcomes"]),
        "events": len(events),
        "agent_actions": len(agent_events),
        "discussion_messages": len(discussions),
        "discussion_exact_duplicates": duplicates,
        "discussion_mean_words": round(statistics.mean(words), 1) if words else 0,
        "mixed_team_votes": mixed_votes,
        "invalid_action_attempts": invalid_attempts,
        "calls": trace["accounting"]["generation"]["calls"],
        "input_tokens": trace["accounting"]["generation"]["input_tokens"],
        "output_tokens": trace["accounting"]["generation"]["output_tokens"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    paths = sorted(args.outdir.glob("avalon-5p-*.trace.json"))
    if args.limit:
        paths = paths[:args.limit]
    rows = [audit(path) for path in paths]
    for row in rows:
        mark = "OK" if row["ok"] else "FAIL"
        print(f"{mark} {row['file']}: {row['outcome']}; {row['quests']} quests; "
              f"{row['calls']} calls; mean talk {row['discussion_mean_words']} words; "
              f"duplicates {row['discussion_exact_duplicates']}; "
              f"invalid attempts {row['invalid_action_attempts']}")
        for error in row["errors"]:
            print(f"    {error}")
    summary = {
        "games": len(rows),
        "passing": sum(row["ok"] for row in rows),
        "calls": sum(row["calls"] for row in rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "outcomes": Counter(row["outcome"] for row in rows),
    }
    print(json.dumps(summary, indent=2, default=dict))
    return 0 if rows and all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
