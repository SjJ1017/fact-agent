"""Turning each dataset into `factflow.tasks.Case` objects.

One function per dataset-task, registered by the name a YAML config gives in
its `loader:` field.  A loader's whole job is to say what the question is,
what the shared frame is, and what the dealable pieces of information are --
never to decide who sees them, except where the dataset itself decides that
(the asymmetric-observation tasks), in which case it records the assignment
in `meta["held"]` and the config asks for `disclosure: {mode: dataset}`.
"""

from __future__ import annotations

import glob
import json
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factflow.tasks import Case, Item  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DELIB_SRC = ROOT / "data" / "delib_collab_src"

LOADERS: dict[str, Callable[..., list[Case]]] = {}


def loader(name: str):
    def wrap(fn):
        LOADERS[name] = fn
        return fn
    return wrap


def load_cases(spec, n: int, seed: int) -> list[Case]:
    if spec.loader not in LOADERS:
        raise ValueError(
            f"unknown loader {spec.loader!r}; have {sorted(LOADERS)}")
    return LOADERS[spec.loader](n=n, seed=seed, **spec.loader_args)


# --------------------------------------------------------------------------
# PerspectruM


@loader("perspectrum")
def load_perspectrum(n: int, seed: int, corpus: str = "experiments/perspectrum_pilot_*",
                     **_: Any) -> list[Case]:
    """Rebuild the published cases from the debates they produced.

    The PerspectruM source files are no longer on disk, and the debate records
    carry the claim and the full dossier verbatim, so this reproduces the
    exact cases rather than a fresh sample of them.  Each evidence entry keeps
    its support/undermine label, which is what role-aligned dealing matches on.
    """
    seen: dict[str, Case] = {}
    for f in sorted(glob.glob(str(ROOT / corpus / "*.debate.json"))):
        d = json.loads(Path(f).read_text())
        cid = str(d["claim_id"])
        if cid in seen:
            continue
        seen[cid] = Case(
            id=cid, question=d["claim"],
            items=tuple(Item(e["id"], e["text"], side=e.get("stance"))
                        for e in d["evidence"]),
            meta={"source": "perspectrum"})
    cases = list(seen.values())
    random.Random(seed).shuffle(cases)
    return cases[:n]


# --------------------------------------------------------------------------
# MMLU-Pro -- the symmetric-information baseline


@loader("mmlu_pro")
def load_mmlu_pro(n: int, seed: int, categories: str | list[str] = "",
                  split: str = "test", **_: Any) -> list[Case]:
    """Multiple choice, no dossier, no stance: the floor of the ladder.

    There is nothing to deal here -- every agent sees the same question, which
    is the point of including it.  It says what persona and topology do when
    no information is private, so the asymmetric datasets have a baseline.
    """
    from datasets import load_dataset

    want = ([c.strip() for c in categories.split(",") if c.strip()]
            if isinstance(categories, str) else list(categories))
    rows: list[dict] = []
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split, streaming=True)
    for row in ds:
        if want and row["category"] not in want:
            continue
        rows.append(row)
        if len(rows) >= n * 20:
            break
    random.Random(seed).shuffle(rows)

    cases = []
    for row in rows[:n]:
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(row["options"]))
        cases.append(Case(
            id=f"mmlu-{row['question_id']}", question=row["question"],
            public=f"Options:\n{opts}",
            items=(),
            meta={"gold": row["answer"], "category": row["category"],
                  "n_options": len(row["options"]), "source": "mmlu-pro"}))
    return cases


# --------------------------------------------------------------------------
# Deliberative collaboration (arXiv 2607.06157) -- asymmetric by construction


def _delib_modules() -> None:
    """Make the paper's pickles loadable without its generation dependencies.

    The pickles name modules that the released package no longer has under
    those names, and importing the game module drags in an LP solver that is
    only used to *build* games.  Statically disassembling one of these files
    shows it references TaskAllocationGame and numpy internals and nothing
    else, so aliasing is enough and no third-party code has to run.
    """
    if "pulp" not in sys.modules:
        stub = types.ModuleType("pulp")
        for sym in ("LpProblem", "LpVariable", "LpMaximize", "LpMinimize",
                    "lpSum", "LpStatus", "PULP_CBC_CMD", "value"):
            setattr(stub, sym, object)
        sys.modules["pulp"] = stub
    if str(DELIB_SRC) not in sys.path:
        sys.path.insert(0, str(DELIB_SRC))
    from delib_collab.data_generation.cooking import game as ck, persona as ck_p
    from delib_collab.data_generation.task_allocation import game as ta
    for alias, real in (("task_allocation_create_game_utils", ta),
                        ("create_game_utils", ck), ("persona", ck_p)):
        sys.modules.setdefault(alias, real)


def _games(subdir: str, n: int, seed: int) -> list[Any]:
    _delib_modules()
    files = sorted(glob.glob(str(DELIB_SRC / "data" / subdir / "**" / "*.pkl"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(
            f"no games under {DELIB_SRC / 'data' / subdir}; clone "
            "https://github.com/wcx21/deliberative-collaboration-agents "
            "into data/delib_collab_src")
    random.Random(seed).shuffle(files)
    out = []
    for f in files:
        try:
            out.append((Path(f).stem, pickle.load(open(f, "rb"))))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


@loader("delib_task_alloc")
def load_task_allocation(n: int, seed: int, **_: Any) -> list[Case]:
    """Three agents assign ten tasks; each sees only its own resources.

    The disclosure is the dataset's, not ours: `agent_i_obs` is what the paper
    gives agent i, including its slice of the public budget, whose fragments
    sum to the true total.  Splitting it any other way would be a different
    experiment from the one the benchmark defines.
    """
    cases = []
    for name, g in _games("task_allocation/games", n, seed):
        # Canonical order, workers before the leader, so that a persona written
        # for position C is always the leader.  The files happen to store them
        # that way already, but a config that silently binds the adjudicator
        # persona to a worker is not a failure anything downstream would show.
        agents = sorted(g.agents,
                        key=lambda a: (g.agents_config[a]["role"] == "Leader", a))
        labels = [chr(ord("A") + i) for i in range(len(agents))]
        items, held = [], {ell: [] for ell in labels}
        src_index = {a: i for i, a in enumerate(g.agents)}
        for agent, ell in zip(agents, labels):
            obs = getattr(g, f"agent_{src_index[agent]}_obs")
            cfg = g.agents_config[agent]
            for res, amt in obs.get("private_resources", {}).items():
                iid = f"{ell}-priv-{res}"
                items.append(Item(iid, f"{cfg['name']} privately holds {amt} unit(s) of {res}.",
                                  tags=("private", res)))
                held[ell].append(iid)
            for res, amt in obs.get("public_resources", {}).items():
                iid = f"{ell}-pub-{res}"
                items.append(Item(iid, f"{cfg['name']}'s records show {amt} unit(s) of the shared {res}.",
                                  tags=("public", res)))
                held[ell].append(iid)

        roster = "\n".join(
            f"- {g.agents_config[a]['name']} ({g.agents_config[a]['role']}), referred to as {ell}"
            for a, ell in zip(agents, labels))
        reqs = "\n".join(
            f"- {t}: " + ", ".join(
                f"{g.agents_config[a]['name']} would need "
                + ", ".join(f"{v} {r}" for r, v in g.task_requirements[t][a].items())
                for a in agents)
            for t in g.tasks)
        cases.append(Case(
            id=f"talloc-{name}", question=(
                "Assign every task to exactly one team member so that the team's "
                "total value is as high as possible without exceeding anyone's resources."),
            public=("Each member can see only their own private resources and a "
                    "fragment of the shared pool; the shared total is the sum of "
                    "those fragments.\n\n"
                    f"Team:\n{roster}\n\nTasks and what each member would need:\n{reqs}"),
            items=tuple(items),
            meta={"held": held, "source": "delib-task-alloc",
                  "gold": getattr(g, "best_allocation_dict", None),
                  "max_reward": float(getattr(g, "max_reward", 0.0)),
                  "roles_from_data": {ell: g.agents_config[a]["role"]
                                      for a, ell in zip(agents, labels)}}))
    return cases


@loader("delib_menu")
def load_menu_design(n: int, seed: int, semantic: bool = True, **_: Any) -> list[Case]:
    """Two agents design a menu from partial views of pantry and guests.

    Note the agent count: with two agents, full, star and chain are the same
    graph, so this task can move the persona axis but not the topology one.
    """
    cases = []
    for name, g in _games("cooking/games", n, seed):
        n_ag = len(getattr(g, "personas", ()) or (0, 0))
        labels = [chr(ord("A") + i) for i in range(2)]
        items, held = [], {ell: [] for ell in labels}
        for i, ell in enumerate(labels):
            for ing, amt in (getattr(g, f"agent_{i}_obs", {}) or {}).items():
                if not amt:
                    continue
                iid = f"{ell}-pantry-{ing.replace(' ', '_')}"
                items.append(Item(iid, f"You can personally account for {amt} unit(s) of {ing}.",
                                  tags=("pantry", ing)))
                held[ell].append(iid)
            if semantic:
                for j, prof in enumerate(getattr(g, f"agent_{i}_value_obs_l2", []) or []):
                    bits = ", ".join(f"{k}: {v}" for k, v in prof.items()
                                     if k in ("Name", "Nationality", "Age", "Occupation",
                                              "Recent Status", "Dietary Restriction"))
                    iid = f"{ell}-guest{j}"
                    items.append(Item(iid, f"What is known about guest {j + 1}: {bits}",
                                      tags=("guest", str(j))))
                    held[ell].append(iid)
            else:
                for dish, val in (getattr(g, f"nl_agent_{i}_values", {}) or {}).items():
                    iid = f"{ell}-val-{dish.replace(' ', '_')}"
                    items.append(Item(iid, f"Guest {i + 1} rates {dish} at {float(val):.2f}.",
                                      tags=("value", dish)))
                    held[ell].append(iid)

        dishes = sorted((getattr(g, "nl_recipes", {}) or {}))
        cases.append(Case(
            id=f"menu-{'sem' if semantic else 'num'}-{name}",
            question=("Agree on the menu that best satisfies the guests, given what "
                      "the pantry can actually supply."),
            public=("Each of you has counted a different part of the pantry, so the "
                    "amount actually available of any ingredient is the sum of what "
                    "the two of you can account for -- neither of you can read it off "
                    "alone.\n\nDishes you may choose from:\n"
                    + "\n".join(f"- {d}" for d in dishes)),
            items=tuple(items),
            meta={"held": held, "source": "delib-menu",
                  "variant": "semantic" if semantic else "numerical",
                  "gold": getattr(g, "best_menu", None),
                  "max_reward": float(getattr(g, "max_reward", 0.0)),
                  "n_agents_in_data": n_ag}))
    return cases
