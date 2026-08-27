"""End-to-end check of the DDI join experiment without touching a network.

Quota is scarce, so the pipeline is proven against a stub before it is run for
real. The join detector is the part that matters: it is the whole claim of the
experiment, and a detector that silently returns False would make every
condition look identical.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "ddi"))

from fake_llm import FakeLLM  # noqa: E402

from cases import CASES, load_cases  # noqa: E402
from factflow.types import CanonicalFact, Channel, FactMention, FactStore, Provenance  # noqa: E402


def _store(source_facts, said):
    """Build a store: source facts, plus who expressed which at which round."""
    store = FactStore()
    ids = {}
    for i, text in enumerate(source_facts):
        m = FactMention(mention_id=f"s{i}", text=text,
                        provenance=Provenance(channel=Channel.SOURCE, round=0, doc_id="d"))
        store.add_mentions([m])
        f = CanonicalFact(fact_id=f"f{i}", canonical_text=text, mention_ids=[m.mention_id])
        store.assign(f)
        ids[text] = f
    k = 0
    for (agent, rnd), texts in said.items():
        for t in texts:
            m = FactMention(mention_id=f"o{k}", text=t,
                            provenance=Provenance(agent_id=agent, round=rnd, channel=Channel.OUTPUT))
            k += 1
            store.add_mentions([m])
            f = ids[t]
            f.mention_ids.append(m.mention_id)
            store.assign(f)
    return store


def test_cases_are_balanced_and_negatives_share_the_surface_pattern():
    pos, neg = load_cases("positive"), load_cases("negative")
    assert len(pos) == len(neg), "an unbalanced set lets a constant answer score well"
    for c in neg:
        assert c.enzyme == "none shared"
        # Both dossiers still name a CYP enzyme, so the pattern looks identical.
        assert "CYP" in c.drug_a.mechanism and "CYP" in c.drug_b.mechanism
    for c in pos:
        assert c.enzyme in c.drug_a.mechanism and c.enzyme in c.drug_b.mechanism, (
            f"{c.case_id}: the join enzyme must appear in both mechanism facts"
        )


def test_mechanism_fact_is_not_always_first_in_the_dossier():
    """If it always led, reading one bullet would be enough and the filtering
    load this task exists to create would vanish."""
    positions = set()
    for c in CASES:
        body = c.drug_a.dossier(shuffle_seed=7).splitlines()[1:]
        positions.add(next(i for i, ln in enumerate(body) if c.drug_a.mechanism in ln))
    assert len(positions) > 1, "mechanism fact sits at a fixed position in every dossier"


def test_join_detected_when_one_agent_holds_both_mechanism_facts():
    from analyze_ddi import find_join

    case = load_cases("positive")[0]
    fa, fb = case.critical_pair
    other = case.drug_a.clinical[0]
    store = _store([fa, fb, other], {
        ("A", 1): [fa],            # A only knows its own drug
        ("B", 1): [fb],            # B only knows its own
        ("B", 2): [fa, fb],        # B has now heard A's fact: the join
    })
    j = find_join(store, case.critical_pair)
    assert j["joined"] is True
    assert j["join_round"] == 2 and j["join_agent"] == "B"
    assert j["both_surfaced"] is True


def test_no_join_when_facts_stay_apart():
    from analyze_ddi import find_join

    case = load_cases("positive")[1]
    fa, fb = case.critical_pair
    store = _store([fa, fb], {("A", 1): [fa], ("B", 1): [fb], ("A", 2): [fa]})
    j = find_join(store, case.critical_pair)
    assert j["joined"] is False
    assert j["both_surfaced"] is True, "both surfaced somewhere, they just never met"


def test_both_surfaced_is_false_when_a_fact_never_reaches_output():
    from analyze_ddi import find_join

    case = load_cases("positive")[2]
    fa, fb = case.critical_pair
    store = _store([fa, fb], {("A", 1): [fa], ("A", 2): [fa]})
    j = find_join(store, case.critical_pair)
    assert j["joined"] is False and j["both_surfaced"] is False


def test_answer_parsing_handles_both_required_lines():
    from run_ddi import _parse

    ans, mech = _parse("reasoning here\nMECHANISM: CYP2C9 inhibition\nFINAL ANSWER: YES")
    assert ans == "YES" and "CYP2C9" in mech
    ans, _ = _parse("MECHANISM: insufficient information\nFINAL ANSWER: NO")
    assert ans == "NO"
    assert _parse("no verdict at all")[0] == ""


def test_trace_records_tag_the_critical_facts():
    from run_ddi import DDIRun, to_trace

    case = load_cases("positive")[0]
    run = DDIRun(case.case_id, "split", case.drug_a.name, case.drug_b.name, case.interacts,
                 case.enzyme, case.critical_pair, transcript={"A|1": "text", "B|1": "text"})
    recs = to_trace(run, case, "x")
    src = [r for r in recs if r["provenance"]["channel"] == "source"]
    crit = [r for r in src if r["provenance"]["extra"]["critical"]]
    assert len(crit) == 2, "exactly the two mechanism facts are critical"
    assert {r["text"] for r in crit} == set(case.critical_pair)
    assert len([r for r in recs if r["provenance"]["channel"] == "output"]) == 2


def test_empty_model_response_raises_instead_of_scoring_zero():
    """The failure that produced a spectacular but false result.

    deepseek-v4-pro spends its output budget on reasoning before emitting text,
    so max_tokens=700 returned empty strings. Empty scores as wrong, and the
    empty rate scales with prompt length -- so the split condition, which has the
    longest prompts, looked like catastrophic reasoning failure (6%) when it was
    truncation. A silent empty must be an error, not a data point.
    """
    import pytest

    from run_ddi import EmptyResponse, _require_text

    assert _require_text("FINAL ANSWER: YES", "A", 1) == "FINAL ANSWER: YES"
    for blank in ("", "   ", "\n\n"):
        with pytest.raises(EmptyResponse):
            _require_text(blank, "A", 1)


def test_default_token_budget_leaves_reasoning_headroom():
    import inspect

    from run_ddi import run_case

    assert inspect.signature(run_case).parameters["max_tokens"].default >= 4000
