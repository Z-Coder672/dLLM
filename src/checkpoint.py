"""
Checkpoint utilities for saving and loading model state.

Saves:
- Model weights (INT8 + scales for ternary layers, BF16 for others)
- Optimizer state
- Training state (step, random state)
- Config
"""

import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np

from .config import ModelConfig, TrainingConfig
from .layers import TernaryLinear
from .model import TernaryTransformer
from .optimizer import AdamW


def save_checkpoint(
    path: str,
    model: TernaryTransformer,
    optimizer: Optional[AdamW],
    step: int,
    config: ModelConfig,
    training_config: Optional[TrainingConfig] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save a training checkpoint.
    
    Args:
        path: Directory to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        step: Current training step
        config: Model configuration
        training_config: Training configuration (optional)
        metrics: Current metrics (optional)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Collect and store model weights (compressed to save disk)
    weights = _collect_weights(model)
    _save_npz_compressed(path / "weights.npz", weights)
    
    # Save optimizer state if provided
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        # Convert arrays to saveable format
        opt_weights = {}
        opt_steps = {}
        for key, value in opt_state.get("state", {}).items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, mx.array):
                        opt_weights[f"{key}_{k}"] = v
                    elif k == "step":
                        opt_steps[key] = v
        
        if opt_weights:
            _save_npz_compressed(path / "optimizer.npz", opt_weights)
        
        # Save optimizer config
        opt_config = {k: v for k, v in opt_state.items() if k != "state"}
        if opt_steps:
            opt_config["state_steps"] = opt_steps
        with open(path / "optimizer_config.json", "w") as f:
            json.dump(opt_config, f, indent=2)
    
    # Save training state
    state = {
        "step": step,
        "metrics": metrics or {},
    }
    with open(path / "state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    # Save configs
    with open(path / "model_config.json", "w") as f:
        json.dump(asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__, f, indent=2)
    
    if training_config is not None:
        with open(path / "training_config.json", "w") as f:
            json.dump(asdict(training_config) if hasattr(training_config, '__dataclass_fields__') else training_config.__dict__, f, indent=2)
    
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: Optional[TernaryTransformer] = None,
    optimizer: Optional[AdamW] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Directory containing checkpoint
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Dictionary with loaded state including step, metrics, configs
    """
    path = Path(path)
    
    result = {}
    
    # Load model config
    config_path = path / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        result["config"] = ModelConfig(**config_dict)
    
    # Load training config
    train_config_path = path / "training_config.json"
    if train_config_path.exists():
        with open(train_config_path) as f:
            train_config_dict = json.load(f)
        result["training_config"] = TrainingConfig(**train_config_dict)
    
    # Load training state
    state_path = path / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        result["step"] = state.get("step", 0)
        result["metrics"] = state.get("metrics", {})
    
    # Load model weights
    weights_path = path / "weights.npz"
    if weights_path.exists():
        weights = {k: mx.array(v) for k, v in dict(np.load(weights_path)).items()}
        result["weights"] = weights
        
        if model is not None:
            _load_weights(model, weights)
            print(f"Loaded model weights from {weights_path}")
    
    # Load optimizer state
    opt_path = path / "optimizer.npz"
    opt_config_path = path / "optimizer_config.json"
    if opt_path.exists() and optimizer is not None:
        opt_weights = {k: mx.array(v) for k, v in dict(np.load(opt_path)).items()}
        
        if opt_config_path.exists():
            with open(opt_config_path) as f:
                opt_config = json.load(f)
            optimizer.load_state_dict({**opt_config, "state": {}})
            # Reconstruct per-parameter moments
            state_steps = opt_config.get("state_steps", {})
            optimizer.state = {}
            for full_key, arr in opt_weights.items():
                if "_" not in full_key:
                    continue
                param_name, stat = full_key.rsplit("_", 1)
                if param_name not in optimizer.state:
                    optimizer.state[param_name] = {
                        "m": None,
                        "v": None,
                        "step": state_steps.get(param_name, 0),
                    }
                optimizer.state[param_name][stat] = arr
            # Convert dicts to OptimizerState instances
            for name, s in optimizer.state.items():
                if s.get("m") is None or s.get("v") is None:
                    continue
                optimizer.state[name] = optimizer.init_state(s["m"])
                optimizer.state[name].m = s["m"]
                optimizer.state[name].v = s["v"]
                optimizer.state[name].step = s.get("step", 0)
            # Ensure step count restored
            optimizer._step_count = opt_config.get("step_count", optimizer._step_count)
        
        print(f"Loaded optimizer state from {opt_path}")
    
    return result


def _collect_weights(module: Any) -> Dict[str, mx.array]:
    """
    Collect weights from all submodules using named_modules so we don't
    miss children stored in the module registry.
    """
    weights: Dict[str, mx.array] = {}
    seen = set()

    for name, mod in module.named_modules():
        if id(mod) in seen:
            continue
        seen.add(id(mod))

        prefix = f"{name}." if name else ""

        if isinstance(mod, TernaryLinear):
            weights[f"{prefix}weight"] = mod._weight
            if mod._bias is not None:
                weights[f"{prefix}bias"] = mod._bias
            continue

        weight = getattr(mod, "weight", None)
        if isinstance(weight, mx.array):
            weights[f"{prefix}weight"] = weight

        bias = getattr(mod, "bias", None)
        if isinstance(bias, mx.array):
            weights[f"{prefix}bias"] = bias

    return weights


def _load_weights(module: Any, weights: Dict[str, mx.array]):
    """Load weights into all submodules using the same prefixes as save."""
    seen = set()

    for name, mod in module.named_modules():
        if id(mod) in seen:
            continue
        seen.add(id(mod))

        prefix = f"{name}." if name else ""

        if isinstance(mod, TernaryLinear):
            if f"{prefix}weight" in weights:
                mod._weight = weights[f"{prefix}weight"]
            if mod._bias is not None and f"{prefix}bias" in weights:
                mod._bias = weights[f"{prefix}bias"]
            continue

        if hasattr(mod, "weight") and f"{prefix}weight" in weights:
            mod.weight = weights[f"{prefix}weight"]

        if hasattr(mod, "bias") and f"{prefix}bias" in weights:
            mod.bias = weights[f"{prefix}bias"]


def _save_npz_compressed(path: Path, arrays: Dict[str, mx.array]):
    """Persist MLX arrays with ZIP compression to minimize disk usage."""
    if not arrays:
        with open(path, "wb") as f:
            np.savez_compressed(f)
        return

    def _to_numpy(arr: mx.array) -> np.ndarray:
        # Cast bfloat16 to float16 so NumPy can serialize and to cut size.
        if arr.dtype == mx.bfloat16:
            arr = arr.astype(mx.float16)
        # Avoid accidental float64; stick to float32 for config weights.
        if arr.dtype == mx.float64:
            arr = arr.astype(mx.float32)
        mx.eval(arr)
        return np.array(arr)

    np_arrays = {k: _to_numpy(v) for k, v in arrays.items()}
    np.savez_compressed(path, **np_arrays)


def list_checkpoints(output_dir: str) -> list:
    """List available checkpoints in output directory."""
    path = Path(output_dir)
    if not path.exists():
        return []
    
    checkpoints = []
    for item in path.iterdir():
        if item.is_dir() and (item / "weights.npz").exists():
            state_path = item / "state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                checkpoints.append({
                    "path": str(item),
                    "step": state.get("step", 0),
                    "metrics": state.get("metrics", {}),
                })
    
    return sorted(checkpoints, key=lambda x: x["step"])


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Get the path to the latest checkpoint."""
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]["path"]


def prune_checkpoints(
    output_dir: str,
    current_step: int,
    save_interval: int,
    keep_last: int = 5,
):
    """
    Log-spaced thinning with a protected recent window:
    - Always keep the latest `keep_last` checkpoints (including the current).
    - For older checkpoints, bucket by floor(log2(age / save_interval)) and
      keep the newest checkpoint in each bucket.
    """
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints or save_interval <= 0:
        return

    # Ensure deterministic ordering
    checkpoints = sorted(checkpoints, key=lambda x: x["step"])

    keep_paths = set()
    recent = checkpoints[-keep_last:] if keep_last > 0 else []

    # Always keep the most recent checkpoints
    for ckpt in recent:
        keep_paths.add(ckpt["path"])

    # Bucket older checkpoints to thin them out logarithmically
    bucket_best = {}
    older = checkpoints[:-keep_last] if keep_last > 0 else checkpoints

    for ckpt in older:
        step = ckpt["step"]
        path = ckpt["path"]
        age = max(0, current_step - step)

        # Skip any checkpoints already in the recent window
        if path in keep_paths:
            continue

        # Bucket by log2 spacing based on how many save intervals ago this is
        bucket = int(math.floor(math.log2(age / save_interval))) if age > 0 else 0
        best = bucket_best.get(bucket)

        # Prefer the most recent (highest step) checkpoint in each bucket
        if best is None or step > best[0]:
            bucket_best[bucket] = (step, path)

    # Add best-per-bucket to keep set
    for _, path in bucket_best.values():
        keep_paths.add(path)

    # Delete everything else
    for ckpt in checkpoints:
        path = ckpt["path"]
        if path in keep_paths:
            continue
        try:
            shutil.rmtree(path)
            print(f"Pruned checkpoint at step {ckpt['step']}: {path}")
        except Exception as e:
            print(f"Failed to prune checkpoint {path}: {e}")

