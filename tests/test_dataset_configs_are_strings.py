"""Every persona prompt must load as a string.

An unquoted YAML scalar containing ": " parses as a mapping, not a string, and
nothing complains: the run gets to the point of building a system prompt and
dies on `dict + str`, three datasets into a batch.  One prompt in
delib_task_alloc ("...actually feasible: whose resources...") did exactly that.
"""
import glob
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(glob.glob(str(ROOT / "experiments/datasets/*.yaml")))


@pytest.mark.parametrize("path", CONFIGS, ids=[Path(p).stem for p in CONFIGS])
def test_persona_prompts_are_strings(path):
    data = yaml.safe_load(Path(path).read_text())
    for name, prompts in (data.get("personas") or {}).items():
        assert isinstance(prompts, list), f"{name}: expected a list of prompts"
        for i, prompt in enumerate(prompts):
            assert isinstance(prompt, str), (
                f"{name}[{i}] parsed as {type(prompt).__name__}, not str -- "
                f"quote it if it contains a colon followed by a space: {prompt!r}")


@pytest.mark.parametrize("path", CONFIGS, ids=[Path(p).stem for p in CONFIGS])
def test_task_strings_are_strings(path):
    data = yaml.safe_load(Path(path).read_text())
    task = data.get("task") or {}
    for key in ("system", "answer_line", "context"):
        if key in task:
            assert isinstance(task[key], str), f"task.{key} is {type(task[key]).__name__}"
    for key, val in (task.get("turn") or {}).items():
        assert isinstance(val, str), f"task.turn.{key} is {type(val).__name__}"
