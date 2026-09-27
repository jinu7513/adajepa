import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

BASE_SHA = "66c5f18d00cc627ae35b9b183367f7e510972106"
REPO = Path(__file__).resolve().parents[1]


def absolute(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (REPO / p).resolve()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    digest = hashlib.sha256()
    code_paths = sorted((REPO / "tracka").glob("*.py")) + [REPO / "conf/tracka.yaml", REPO / "planning/image_corruption.py"]
    for path in code_paths:
        digest.update(str(path.relative_to(REPO)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    code_hash = digest.hexdigest()
    try:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
        return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "base_sha": BASE_SHA, "tracka_source_sha256": code_hash}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": "unknown", "git_dirty": None, "base_sha": BASE_SHA, "tracka_source_sha256": code_hash}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_checkpoint(path):
    # Only use trusted local experiment checkpoints, which include RNG tuples.
    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
