"""Smoke-test the B/C-only submitter with a fake Slurm command."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash")


def bash_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix()[2:]}" if os.name == "nt" else str(path)


@pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash unavailable")
def test_bc_submission_survives_quota_failure_without_duplicate_b(tmp_path):
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    (source / "hydra.yaml").write_text("env: pusht\n", encoding="utf-8")
    (source / "checkpoints/model_latest.pth").write_bytes(b"fixture")
    data = tmp_path / "data"
    (data / "pusht_noise").mkdir(parents=True)
    (data / "pusht_noise/transition_color_manifest.json").write_text("{}\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mocks = {
        "git": '#!/bin/bash\nif [[ "$*" == *rev-parse* ]]; then echo testcommit; fi\n',
        "sinfo": "#!/bin/bash\necho muma_2021\n",
        "flock": "#!/bin/bash\nexit 0\n",
        "sha256sum": "#!/bin/bash\nif [[ \"$1\" == -c ]]; then exit 0; fi\necho fixture\n",
        "sbatch": (
            '#!/bin/bash\n'
            'n=$(<"$FAKE_COUNTER_FILE")\n'
            'n=$((n + 1))\n'
            'printf "%s\\n" "$n" > "$FAKE_COUNTER_FILE"\n'
            'if [[ "$n" == 2 ]]; then\n'
            '  echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2\n'
            '  exit 1\n'
            'fi\n'
            'echo "$((1000 + n))"\n'
        ),
    }
    for name, source_text in mocks.items():
        path = bin_dir / name
        path.write_text(source_text, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    counter = tmp_path / "counter"
    counter.write_text("0\n", encoding="utf-8", newline="\n")
    run_root = tmp_path / "run"
    script = bash_path(ROOT / "scripts/submit_film_bc_train.bash")
    env = os.environ.copy()
    env.update(
        CIRCE_PARTITION="muma_2021",
        CIRCE_TIME="2-00:00:00",
        PYTHON_BIN="/bin/bash",
        DATASET_DIR=bash_path(data),
        SOURCE_MODEL_DIR=bash_path(source),
        FAKE_COUNTER_FILE=bash_path(counter),
    )
    command = [BASH, "-c", 'PATH="$1:$PATH" exec /bin/bash "$2" "$3"',
               "bash", bash_path(bin_dir), script, bash_path(run_root)]
    first = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert first.returncode != 0
    assert (run_root / "job_ids.txt").read_text(encoding="utf-8").splitlines() == ["train_B=1001"]
    second = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (run_root / "job_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "train_B=1001", "train_C=1003",
    ]
