"""Reading a dataset's debate configuration.

One YAML file per dataset-task, so that adding PerspectruM's successor is a
config, not a second runner.  `experiments/run_qa.py` and
`experiments/run_perspectrum.py` already diverged into two panels, two prompt
builders and two output shapes for what is the same experiment; this is the
one description both of them should have read.

The persona ladder is deliberately fixed rather than free-form.  The study
manipulates persona and topology, so persona has to mean the same thing in
every dataset or the cross-dataset comparison measures wording instead:

    neutral   three identical agents -- the homogeneous control
    lenses    functionally differentiated, no position on the answer
    stance    committed opposing positions

A dataset may phrase all three however its task requires, but it must supply
all three, and they must stay in that order of increasing differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LADDER = ("neutral", "lenses", "stance")


@dataclass(frozen=True)
class TaskSpec:
    """Everything about a dataset-task that the runner would otherwise hard-code."""

    name: str
    loader: str
    loader_args: dict[str, Any]
    agents: tuple[str, ...]
    system: str
    answer_line: str
    max_words: int
    context_template: str
    personas: dict[str, tuple[str, ...]]
    disclosure: dict[str, Any]
    rounds: int
    notes: str = ""

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def roles(self, persona: str) -> dict[str, str]:
        return dict(zip(self.agents, self.personas[persona]))


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ValueError(f"{where}: missing required key {key!r}")
    return data[key]


def load_spec(path: Path) -> TaskSpec:
    data = yaml.safe_load(path.read_text())
    where = str(path)
    agents = tuple(data.get("agents") or ["A", "B", "C"])

    personas_raw = _require(data, "personas", where)
    missing = [p for p in LADDER if p not in personas_raw]
    if missing:
        raise ValueError(
            f"{where}: personas must define all of {LADDER}; missing {missing}. "
            "The persona ladder is the study's independent variable and has to "
            "be comparable across datasets.")
    context_template: str
    personas: dict[str, tuple[str, ...]] = {}
    for name, prompts in personas_raw.items():
        if len(prompts) != len(agents):
            raise ValueError(
                f"{where}: persona {name!r} has {len(prompts)} prompts for "
                f"{len(agents)} agents")
        personas[name] = tuple(prompts)

    task = _require(data, "task", where)
    disclosure = data.get("disclosure") or {"mode": "full"}
    if disclosure.get("mode") == "role_aligned":
        aff = disclosure.get("affinity")
        if not isinstance(aff, dict) or set(aff) - set(personas):
            raise ValueError(
                f"{where}: role_aligned disclosure needs an `affinity` map "
                f"keyed by persona name, so that the deal matches the roles "
                f"actually in play; got {aff!r}")

    return TaskSpec(
        name=data.get("name") or path.stem,
        loader=_require(data, "loader", where),
        loader_args=data.get("loader_args") or {},
        agents=agents,
        system=_require(task, "system", f"{where}:task"),
        answer_line=_require(task, "answer_line", f"{where}:task"),
        max_words=int(task.get("max_words", 180)),
        context_template=_require(task, "context", f"{where}:task"),
        personas=personas,
        disclosure=disclosure,
        rounds=int(data.get("rounds", 3)),
        notes=data.get("notes", ""),
    )


def affinity_for(spec: TaskSpec, persona: str) -> dict[str, Any]:
    """The disclosure spec as it applies to one persona.

    Role-aligned dealing only means anything relative to a particular set of
    roles: "the advocate gets the supporting evidence" is undefined under the
    neutral panel, where no agent is the advocate.  A persona with no entry
    falls back to symmetric disclosure rather than silently reusing another
    persona's affinity.
    """
    spec_d = dict(spec.disclosure)
    if spec_d.get("mode") != "role_aligned":
        return spec_d
    aff = spec_d.get("affinity", {}).get(persona)
    if aff is None:
        return {"mode": spec_d.get("fallback", "full")}
    return {"mode": "role_aligned", "affinity": aff}


def render_context(spec: TaskSpec, case: "Any", items: "Any") -> str:
    """The dossier text an agent is shown, built from the items it was dealt.

    Kept in the config rather than in code because the frame is part of the
    task: "Claim under review / Evidence dossier" is right for PerspectruM and
    wrong for a menu-design brief.  `{items}` is the dealt subset, not the
    whole case -- that substitution is the entire point of the dealing step.
    """
    return spec.context_template.format(
        question=case.question,
        public=case.public,
        items="\n\n".join(i.render() for i in items),
    ).strip()
