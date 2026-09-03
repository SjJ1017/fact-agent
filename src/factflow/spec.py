"""Reading a dataset's debate configuration.

One YAML file per dataset-task, so that adding PerspectruM's successor is a
config, not a second runner.  `experiments/run_qa.py` and
`experiments/run_perspectrum.py` already diverged into two panels, two prompt
builders and two output shapes for what is the same experiment; this is the
one description both of them should have read.

The original persona ladder is still the default for the PerspectruM corpus:

    neutral   three identical agents -- the homogeneous control
    lenses    functionally differentiated, no position on the answer
    stance    committed opposing positions

It is not a universal ontology.  A clinical case needs generic/specialist
roles and may cross those with full/partitioned disclosure; an asymmetric game
may get its roles and private observations directly from the data.  Specs can
therefore define named ``conditions`` that independently select a persona
profile and a disclosure policy.  Old specs without conditions retain the
three-condition ladder for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LADDER = ("neutral", "lenses", "stance")


@dataclass(frozen=True)
class ConditionSpec:
    persona: str
    disclosure: dict[str, Any] | None = None


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
    turn: dict[str, str]
    personas: dict[str, tuple[str, ...]]
    conditions: dict[str, ConditionSpec]
    disclosure: dict[str, Any]
    rounds: int
    notes: str = ""

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def condition_names(self) -> tuple[str, ...]:
        if self.conditions:
            return tuple(self.conditions)
        default = tuple(p for p in LADDER if p in self.personas)
        return default or tuple(self.personas)

    def persona_for(self, condition: str) -> str:
        if condition in self.conditions:
            return self.conditions[condition].persona
        if condition in self.personas:
            return condition
        raise KeyError(
            f"unknown condition {condition!r}; have {list(self.condition_names)}")

    def roles(self, condition: str) -> dict[str, str]:
        return dict(zip(self.agents, self.personas[self.persona_for(condition)]))

    def disclosure_for(self, condition: str) -> dict[str, Any]:
        """Resolve the information policy independently of the role profile."""
        if condition in self.conditions:
            override = self.conditions[condition].disclosure
            if override is not None:
                return dict(override)

        spec_d = dict(self.disclosure)
        if spec_d.get("mode") != "role_aligned":
            return spec_d
        persona = self.persona_for(condition)
        aff = spec_d.get("affinity", {}).get(persona)
        if aff is None:
            return {"mode": spec_d.get("fallback", "full")}
        return {"mode": "role_aligned", "affinity": aff}


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ValueError(f"{where}: missing required key {key!r}")
    return data[key]


def load_spec(path: Path) -> TaskSpec:
    data = yaml.safe_load(path.read_text())
    where = str(path)
    agents = tuple(data.get("agents") or ["A", "B", "C"])

    personas_raw = _require(data, "personas", where)
    if not personas_raw:
        raise ValueError(f"{where}: personas must define at least one role profile")
    context_template: str
    turn: dict[str, str]
    personas: dict[str, tuple[str, ...]] = {}
    for name, prompts in personas_raw.items():
        if len(prompts) != len(agents):
            raise ValueError(
                f"{where}: persona {name!r} has {len(prompts)} prompts for "
                f"{len(agents)} agents")
        personas[name] = tuple(prompts)

    conditions: dict[str, ConditionSpec] = {}
    for name, raw in (data.get("conditions") or {}).items():
        raw = raw or {}
        persona = raw.get("persona", name)
        if persona not in personas:
            raise ValueError(
                f"{where}: condition {name!r} names unknown persona {persona!r}; "
                f"have {sorted(personas)}")
        disclosure_override = raw.get("disclosure")
        if disclosure_override is not None and not isinstance(disclosure_override, dict):
            raise ValueError(
                f"{where}: condition {name!r} disclosure must be a mapping")
        conditions[name] = ConditionSpec(persona, disclosure_override)

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
        turn=task.get("turn") or {},
        personas=personas,
        conditions=conditions,
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
    return spec.disclosure_for(persona)


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
