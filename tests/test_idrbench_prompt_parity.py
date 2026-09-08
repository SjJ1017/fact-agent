"""The reconstructed idrbench cases must render the reference corpus's prompts.

`load_idrbench_generation` rebuilds its cases from the debates the full-topology
run produced, so that star and chain run on the same papers with the same
framing. That claim was once written into the loader's docstring without being
checked, and it was false: `public` was dropped, the panel lost the line saying
the paper pair had been selected as a valid integration candidate, and round-one
overlap under split disclosure moved from 17.2% to 2.3% on that one line. Eighty
debates had to be discarded.

The check costs a second, so it runs on every test invocation rather than
living in someone's memory.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

REFERENCE = ROOT / "experiments" / "idrbench_generation_10x5_r3"


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference corpus not present")
def test_round_one_context_matches_reference_byte_for_byte():
    from factflow.spec import load_spec, render_context
    from loaders import load_cases

    spec = load_spec(ROOT / "experiments/datasets/idrbench_idea_generation.yaml")
    cases = {c.id: c for c in load_cases(spec, n=10, seed=0)}
    assert cases, "loader produced no cases"

    checked = 0
    for path in sorted(glob.glob(str(REFERENCE / "*.debate.json"))):
        debate = json.loads(Path(path).read_text())
        case = cases.get(debate["case_id"])
        if case is None:
            continue
        held = (debate.get("disclosure") or {}).get("held") or {}
        for agent in sorted(debate["roles"]):
            recorded = debate["prompts"].get(f"{agent}|1")
            if recorded is None:
                continue
            items = tuple(i for i in case.items if i.id in held.get(agent, []))
            rendered = render_context(spec, case, items)
            assert recorded.startswith(rendered), (
                f"{Path(path).name} {agent}: rebuilt context is not a prefix of "
                f"the prompt this corpus was generated with.\n"
                f"rebuilt : {rendered[:200]!r}\n"
                f"recorded: {recorded[:200]!r}")
            checked += 1
    assert checked >= 100, f"only {checked} prompts compared; expected the full corpus"


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference corpus not present")
def test_public_framing_survives_the_rebuild():
    """The specific field that was lost, named so a regression says which one."""
    from factflow.spec import load_spec
    from loaders import load_cases

    spec = load_spec(ROOT / "experiments/datasets/idrbench_idea_generation.yaml")
    for case in load_cases(spec, n=10, seed=0):
        assert case.public.strip(), f"{case.id} rebuilt without its public framing"
        assert "integration candidate" in case.public
