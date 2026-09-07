from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "avalon"))

from run_avalon import (  # noqa: E402
    parse_mission_vote,
    parse_discussion,
    parse_target,
    parse_team,
    parse_team_vote,
    private_observation,
    role_assignment,
)


ROLES = ["Merlin", "Servant", "Servant", "Minion", "Assassin"]


def test_action_parsers_accept_only_legal_moves():
    assert parse_team("Reason.\nTEAM: [0, 3]", 2) == [0, 3]
    assert parse_team("Reason.\n**TEAM:** [Player 0, P3]", 2) == [0, 3]
    assert parse_team("TEAM: [0, 0]", 2) is None
    assert parse_team("TEAM: [0, 3, 4]", 2) is None
    assert parse_team_vote("VOTE: yes") == "APPROVE"
    assert parse_team_vote("VOTE: REJECT") == "REJECT"
    assert parse_team_vote("**VOTE:** **APPROVE**") == "APPROVE"
    assert parse_team_vote("probably approve") is None
    assert parse_mission_vote("MISSION: FAIL", good_player=True) is None
    assert parse_mission_vote("MISSION: FAIL", good_player=False) == "FAIL"
    assert parse_mission_vote("MISSION: PASS", good_player=True) == "SUCCESS"
    assert parse_mission_vote("**MISSION:** **SUCCESS**", good_player=True) == "SUCCESS"
    assert parse_target("TARGET: Player 4", assassin=2) == 4
    assert parse_target("TARGET: 2", assassin=2) is None
    assert parse_discussion("short public message", 3) == "short public message"
    assert parse_discussion("one two three four", 3) is None


def test_ten_game_role_schedule_is_balanced_and_unique():
    assignments = [role_assignment(i, ROLES) for i in range(10)]
    assert len({tuple(row.values()) for row in assignments}) == 10
    for player in range(5):
        assert Counter(row[player] for row in assignments) == Counter({
            "Merlin": 2, "Servant": 4, "Minion": 2, "Assassin": 2,
        })


def test_private_observations_follow_avalon_information_sets():
    roles = {0: "Merlin", 1: "Servant", 2: "Assassin", 3: "Servant", 4: "Minion"}
    assert "Players 2 and 4 are Evil" in private_observation(0, roles)
    assert "Player 4 is your Evil partner" in private_observation(2, roles)
    servant = private_observation(1, roles)
    assert "no private knowledge" in servant
    assert "Player 2" not in servant and "Player 4" not in servant
