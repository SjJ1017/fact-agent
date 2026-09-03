"""The generic runner must rebuild the frozen corpus's prompts exactly.

run_perspectrum.py generated every published number. If run_debate.py driven
by datasets/perspectrum.yaml produces a different prompt, the refactor has
silently changed the experiment the new datasets are compared against.
"""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from factflow.spec import load_spec
from factflow.tasks import Case, Item, deal
from run_debate import build_prompt


def _cases():
    spec = load_spec(ROOT / "experiments/datasets/perspectrum.yaml")
    for f in sorted(glob.glob(str(ROOT / "experiments/perspectrum_pilot_*/*.debate.json"))):
        d = json.loads(Path(f).read_text())
        case = Case(id=str(d["claim_id"]), question=d["claim"],
                    items=tuple(Item(e["id"], e["text"], side=e.get("stance"))
                                for e in d["evidence"]))
        yield spec, case, d


def test_prompts_match_corpus_byte_for_byte():
    n = 0
    for spec, case, d in _cases():
        dealt = deal(case, spec.agents, spec.disclosure, seed=0)
        agents = list(spec.agents)
        for key, recorded in d["prompts"].items():
            agent, rnd = key.split("|")
            rnd = int(rnd)
            # the frozen corpus is peer-only: no self history, peers are last round
            from frameworks import neighbours
            peers = [(p, d["transcript"][f"{p}|{rnd - 1}"])
                     for p in neighbours(d["topology"], agents, agent)
                     if f"{p}|{rnd - 1}" in d["transcript"]]
            built = build_prompt(spec, case, dealt[agent], peers, has_own_prior=False)
            assert built == recorded, (
                f"{d['execution_id']} {key}\n--- built ---\n{built[-300:]}\n"
                f"--- recorded ---\n{recorded[-300:]}")
            n += 1
    assert n > 1000, f"only checked {n} prompts"


def test_roles_match_corpus():
    for spec, _case, d in _cases():
        assert d["roles"] == spec.roles(d["panel"]), d["execution_id"]
