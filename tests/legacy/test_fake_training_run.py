import json
import math
import tempfile
from pathlib import Path

import numpy as np

from src.checkpoint import prune_checkpoints, save_checkpoint
from src.config import ModelConfig, TrainingConfig


def _compute_kept_steps(max_step: int, save_interval: int):
    """Pure-Python simulation of prune_checkpoints over integer steps."""
    steps = list(range(0, max_step + 1, save_interval))
    if not steps or save_interval <= 0:
        return []

    keep = set()
    bucket_best = {}

    for step in steps:
        age = max_step - step
        log_base = 2
        bucket = int(math.floor(math.log(age / save_interval, log_base))) if age > 0 else 0
        # Oldest per bucket wins (first seen in ascending order)
        if bucket not in bucket_best:
            bucket_best[bucket] = step

    keep.update(bucket_best.values())
    return sorted(keep)


class FakeModel:
    """Minimal stand-in that exposes no weights to save."""

    def named_modules(self):
        # Yield itself so the collector runs but finds no params.
        yield "", self


def test_fake_training_run_saves_empty_checkpoints(tmp_path: Path):
    """
    Simulate a short training loop that records fake metrics and writes empty
    checkpoints (no weights) to validate the checkpoint/plumbing without
    running any compute.
    """
    model = FakeModel()
    model_config = ModelConfig()
    training_config = TrainingConfig(output_dir=str(tmp_path), save_interval=2)

    fake_losses = {}
    saved_steps = []

    for step in range(0, 7):
        loss = round(42.0 + 0.1 * step, 3)
        fake_losses[step] = loss

        if step % training_config.save_interval == 0:
            saved_steps.append(step)
            ckpt_path = tmp_path / f"step_{step}"
            save_checkpoint(
                path=str(ckpt_path),
                model=model,
                optimizer=None,
                step=step,
                config=model_config,
                training_config=training_config,
                metrics={"loss": loss},
            )
            prune_checkpoints(
                output_dir=str(tmp_path),
                current_step=step,
                save_interval=training_config.save_interval,
            )

    remaining_steps = _compute_kept_steps(6, 2)
    remaining_names = sorted(f"step_{s}" for s in remaining_steps)
    
    actual = sorted(
        p.name for p in tmp_path.iterdir() if (p / "weights.npz").exists()
    )
    assert actual == remaining_names


if __name__ == "__main__":
    # Run a pure simulation for a large step count without touching disk.
    steps_to_run = 20000
    save_interval = 2

    kept_steps = _compute_kept_steps(
        max_step=steps_to_run,
        save_interval=save_interval,
    )

    print(
        f"Simulated kept checkpoints for {steps_to_run} steps "
        f"(save_interval={save_interval}):"
    )
    print(f"Total kept: {len(kept_steps)}")
    print(kept_steps)
