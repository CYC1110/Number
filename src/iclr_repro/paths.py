"""All inputs and outputs remain inside this checkout; no external mirrors."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]


def inside(*parts):
    path = ROOT.joinpath(*map(str, parts)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path escapes the project root (including symlink targets)")
    return path


def protocol():
    data=json.loads(inside("configs", "protocol.json").read_text())
    if set(data["state_cohorts"])!=set(data["states"]):raise ValueError("Every state needs an explicit cohort")
    for name,cohort in data["cohorts"].items():
        runs=cohort["runs"];seeds=cohort["seed_by_run"]
        if len(runs)!=cohort["n_runs"] or len(runs)!=len(set(runs)) or len(runs)<2:
            raise ValueError("Invalid configured cohort run count")
        if set(runs)!=set(seeds) or len(set(seeds.values()))!=len(runs):
            raise ValueError("Cohort needs a unique explicit seed for every run")
    if not set(data["state_cohorts"].values())<=set(data["cohorts"]):raise ValueError("Unknown cohort name")
    return data


def cohort(state):
    data=protocol()
    if state not in data["state_cohorts"]:raise ValueError("Unknown configured state")
    return data["cohorts"][data["state_cohorts"][state]]


def runs(state):
    return tuple(cohort(state)["runs"])


def run_seed(state,run):
    data=cohort(state)
    if run not in data["seed_by_run"]:raise ValueError("Run is not configured for this state")
    return int(data["seed_by_run"][run])


def state_counts():
    return {state:len(runs(state)) for state in protocol()["states"]}


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(value)):
        raise ValueError("Invalid state/run identifier")
    return str(value)


def dataset(name):
    names = {"np": "np", "numerosity": "numerosity", "imagenet": "imagenet",
             "noise": "synthetic/noise", "grating": "synthetic/grating",
             "statistical": "synthetic/statistical", "shaders": "synthetic/shaders"}
    return inside("data", names[name])


def checkpoint(state, run, epoch):
    return inside("checkpoints", identifier(state), identifier(run), f"epoch_{int(epoch):03d}.pth")


def result(state, run, epoch):
    return inside("results", identifier(state), identifier(run), f"epoch_{int(epoch):03d}")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, data):
    import numpy as np
    def clean(value):
        if isinstance(value, dict):return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):return [clean(v) for v in value]
        if isinstance(value, np.ndarray):return clean(value.tolist())
        if isinstance(value, np.generic):return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):return None
        return value
    path = inside(relative(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
