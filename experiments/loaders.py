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
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factflow.tasks import Case, Item  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DELIB_SRC = ROOT / "data" / "delib_collab_src"
CLINICAL_SRC = ROOT / "data" / "clinicalbench_src"

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


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read an object, array, or JSONL file, tolerating the official demo's bad escapes.

    ClinicalLab's English example is nominally JSON but contains LaTeX-style
    ``\%`` and ``\#`` escapes, which JSON does not define.  The full licensed
    release may not have that defect, so strict parsing is attempted first and
    the narrow repair is only a fallback.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        repaired = re.sub(r"\\([%#])", r"\1", raw)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            data = [json.loads(re.sub(r"\\([%#])", r"\1", line))
                    for line in raw.splitlines() if line.strip()]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data
    raise ValueError(f"{path}: expected a JSON object, array of objects, or JSONL")


def _nonempty(value: Any) -> bool:
    """An unobserved field is present but empty, not absent."""
    if value is None or value == "" or value == {} or value == []:
        return False
    return True


def _flatten(value: Any) -> str:
    """Readable text for a nested profile field, so a prompt is not a dict repr."""
    if isinstance(value, dict):
        return "; ".join(f"{k.lower()} {_flatten(v)}" for k, v in value.items()
                         if _nonempty(v))
    if isinstance(value, (list, tuple)):
        return ", ".join(_flatten(v) for v in value if _nonempty(v))
    return str(value)


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
# ClinicalBench -- multi-department, text-only clinical records


CLINICAL_TASKS: dict[str, tuple[str, str]] = {
    "department_guide": (
        "Which clinical department should receive this patient first? Explain the evidence that determines the routing.",
        "clinical_department",
    ),
    "preliminary_diagnosis": (
        "Give the preliminary diagnoses supported by the presenting history and examination, before settling the final diagnosis.",
        "preliminary_diagnosis",
    ),
    "diagnostic_basis": (
        "Identify the case evidence that supports the leading diagnosis. Distinguish direct observations from inferences.",
        "diagnostic_basis",
    ),
    "differential_diagnosis": (
        "Construct the differential diagnosis and explain which observations support or weaken each alternative.",
        "differential_diagnosis",
    ),
    "final_diagnosis": (
        "Determine the principal diagnosis and justify it by integrating the history, examination, imaging, endoscopy, and laboratory evidence.",
        "principal_diagnosis",
    ),
    "treatment_principle": (
        "State the immediate treatment principles warranted by the case and tie each recommendation to a finding or risk.",
        "therapeutic_principle",
    ),
    "treatment_plan": (
        "Propose a concrete treatment plan, including urgent management, monitoring, and procedural steps, grounded in the case evidence.",
        "treatment_plan",
    ),
    "imaging_diagnosis": (
        "Provide an integrated imaging and endoscopic diagnosis from the textual reports, separating decisive findings from incidental findings.",
        "_imaging_gold",
    ),
}


def _clinical_items(row: dict[str, Any]) -> tuple[Item, ...]:
    """Keep source departments separate and exclude all gold answer fields."""
    items: list[Item] = []
    summary = str(row.get("clinical_case_summary") or "").strip()
    # The public example repeats every auxiliary report in the summary.  Keep
    # only the intake portion, then use the structured report fields below;
    # otherwise each observation enters the system twice under two source ids.
    intake = re.split(r"Auxiliary Examination", summary, maxsplit=1,
                      flags=re.IGNORECASE)[0].strip()
    if intake:
        items.append(Item("history-exam", intake, tags=("history", "examination")))

    for name, report in (row.get("imageological_examination") or {}).items():
        if not isinstance(report, dict):
            continue
        text = "\n".join(
            f"{label}: {report[key]}" for key, label in
            (("findings", "Findings"), ("impression", "Impression"))
            if report.get(key))
        if text:
            clean = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            items.append(Item(f"imaging-{clean}", f"{name.replace('_', ' ')}\n{text}",
                              tags=("imaging", "endoscopy" if "endosc" in name else "radiology")))

    for name, report in (row.get("laboratory_examination") or {}).items():
        if not isinstance(report, dict):
            continue
        # `abnormal` is a lossy duplicate of `result`; retaining both would
        # manufacture re-emphasis before any agent has spoken.
        text = str(report.get("result") or report.get("abnormal") or "").strip()
        if text:
            clean = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            items.append(Item(f"laboratory-{clean}",
                              f"{name.replace('_', ' ')}\n{text}",
                              tags=("laboratory",)))

    pathology = str(row.get("pathological_examination") or "").strip()
    if pathology and pathology.lower().rstrip(".") not in {"none", "not available", "n/a"}:
        items.append(Item("pathology", pathology, tags=("pathology", "laboratory")))
    return tuple(items)


def _imaging_gold(row: dict[str, Any]) -> str:
    parts = []
    for name, report in (row.get("imageological_examination") or {}).items():
        if isinstance(report, dict) and report.get("impression"):
            parts.append(f"{name.replace('_', ' ')}: {report['impression']}")
    return "\n".join(parts)


@loader("clinicalbench")
def load_clinicalbench(n: int, seed: int,
                       path: str = "data/clinicalbench_src/data_examples/data_example_en.json",
                       tasks: str | list[str] = "all", **_: Any) -> list[Case]:
    """Expand each ClinicalBench record into its eight published task types.

    ``n`` counts source records, not expanded task cases: ``-n 1`` with
    ``tasks: all`` returns eight cases from the one official public example.
    The full benchmark uses the same fields but requires a data-use
    application; pointing ``path`` at that JSON is the only change needed.
    """
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(
            f"ClinicalBench data not found at {source}. Clone the official "
            "ClinicalLab repository or set loader_args.path to an approved copy.")

    wanted = list(CLINICAL_TASKS) if tasks == "all" else (
        [x.strip() for x in tasks.split(",") if x.strip()]
        if isinstance(tasks, str) else list(tasks))
    unknown = sorted(set(wanted) - set(CLINICAL_TASKS))
    if unknown:
        raise ValueError(
            f"unknown ClinicalBench tasks {unknown}; have {sorted(CLINICAL_TASKS)}")

    rows = _read_json_records(source)
    random.Random(seed).shuffle(rows)
    cases: list[Case] = []
    for row in rows[:n]:
        uid = str(row.get("clinical_case_uid") or row.get("id") or len(cases))
        items = _clinical_items(row)
        if not items:
            raise ValueError(f"ClinicalBench record {uid} contains no observable case data")
        for task_name in wanted:
            question, gold_key = CLINICAL_TASKS[task_name]
            gold = _imaging_gold(row) if gold_key == "_imaging_gold" else row.get(gold_key)
            cases.append(Case(
                id=f"clinical-{uid[:12]}-{task_name}",
                question=question,
                public="De-identified ClinicalBench case conference. All imaging inputs are textual reports, not raw images.",
                items=items,
                meta={"gold": gold, "task_type": task_name,
                      "clinical_department": row.get("clinical_department"),
                      "question_is_source": False,
                      "source": "clinicalbench", "source_path": str(source)},
            ))
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
            # Each agent knows only its own row of the value matrix; the paper's
            # own agent gets partners' efficiencies solely through deliberation
            # (prompts/task_allocation: "partner_preferences ... Partners'
            # efficiency values").  Leaving these out left the panel asked to
            # maximise a value it was never shown.
            row = g.value_matrix[src_index[agent]]
            for task_idx, task in enumerate(g.tasks):
                eff = round(float(row[task_idx]), 3)
                if eff <= 0:
                    continue
                iid = f"{ell}-eff-{task_idx}"
                items.append(Item(iid, f"{cfg['name']} would complete \"{task}\" "
                                       f"at an efficiency of {eff}.",
                                  tags=("efficiency", task)))
                held[ell].append(iid)

        roster = "\n".join(
            f"- {g.agents_config[a]['name']} ({g.agents_config[a]['role']}), referred to as {ell}"
            for a, ell in zip(agents, labels))
        # The requirements are the same whoever does the task -- the reference
        # implementation reads agent_0's row and calls it the task's
        # requirement -- so listing them per member tripled the prompt with
        # three identical copies.  Assert rather than assume, and drop the
        # zero entries the way the reference formatter does.
        ref = g.agents[0]
        for t in g.tasks:
            rows = [g.task_requirements[t][a] for a in g.agents]
            if any(r != rows[0] for r in rows[1:]):
                raise ValueError(
                    f"{name}: task {t!r} has per-member requirements; the "
                    "shared-requirement rendering below would hide that")
        reqs = "\n".join(
            f"- {t}: " + ", ".join(
                f"{v:g} {r}" for r, v in g.task_requirements[t][ref].items() if v)
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
        seen_ids: set[str] = set()
        for i, ell in enumerate(labels):
            for ing, amt in (getattr(g, f"agent_{i}_obs", {}) or {}).items():
                if not amt:
                    continue
                iid = f"{ell}-pantry-{ing.replace(' ', '_')}"
                items.append(Item(iid, f"You can personally account for {amt} unit(s) of {ing}.",
                                  tags=("pantry", ing)))
                held[ell].append(iid)
            if semantic:
                # One item per (guest, field), and only fields this agent
                # actually observed.  The asymmetry in this variant is
                # field-level -- one agent has a guest's cultural preferences
                # and not their health goals, the other the reverse, with the
                # unobserved field present but empty.  Rendering a fixed
                # shortlist of fields dropped exactly the ones that differ and
                # left both agents with identical guest knowledge, which is
                # the property the task is built on.
                for j, prof in enumerate(getattr(g, f"agent_{i}_value_obs_l2", []) or []):
                    who = prof.get("Name") or f"guest {j + 1}"
                    for field, val in prof.items():
                        if field == "Name" or not _nonempty(val):
                            continue
                        iid = f"guest{j}-{field.replace(' ', '_')}"
                        if iid not in seen_ids:
                            seen_ids.add(iid)
                            items.append(Item(
                                iid, f"{who}'s {field.lower()}: {_flatten(val)}",
                                tags=("guest", str(j), field)))
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
