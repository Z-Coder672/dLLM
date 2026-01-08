import json
import shutil
from pathlib import Path

import numpy as np

from src.checkpoint import prune_checkpoints


def _create_checkpoint(root: Path, step: int) -> Path:
    ckpt_dir = root / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Minimal files so list_checkpoints picks this up
    np.savez(ckpt_dir / "weights.npz", w=np.array([step], dtype=np.int32))
    with open(ckpt_dir / "state.json", "w") as f:
        json.dump({"step": step}, f)

    return ckpt_dir


def test_prune_keeps_recent_and_log_buckets(tmp_path: Path):
    save_interval = 50
    steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    for step in steps:
        _create_checkpoint(tmp_path, step)

    prune_checkpoints(
        output_dir=str(tmp_path),
        current_step=600,
        save_interval=save_interval,
        keep_last=5,
    )

    remaining = {
        p.name for p in tmp_path.iterdir() if (p / "weights.npz").exists()
    }

    # Keep last 5 unconditionally + one per log bucket for older checkpoints
    expected = {
        "step_600",
        "step_550",
        "step_500",
        "step_450",
        "step_400",
        "step_350",
        "step_200",
    }
    assert remaining == expected

    # Clean up in case tmp_path persists across tests
    for p in tmp_path.iterdir():
        shutil.rmtree(p)

