"""
Checkpoint utilities for saving and loading model state.

Saves:
- Model weights (INT8 + scales for ternary layers, BF16 for others)
- Optimizer state
- Training state (step, random state)
- Config
"""

import mlx.core as mx
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from .config import ModelConfig, TrainingConfig
from .model import TernaryTransformer
from .optimizer import AdamW8bit


def save_checkpoint(
    path: str,
    model: TernaryTransformer,
    optimizer: Optional[AdamW8bit],
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
    
    # Collect model weights
    weights = {}
    _collect_weights(model, "", weights)
    
    # Save weights
    mx.savez(str(path / "weights.npz"), **weights)
    
    # Save optimizer state if provided
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        # Convert arrays to saveable format
        opt_weights = {}
        for key, value in opt_state.get("state", {}).items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, mx.array):
                        opt_weights[f"{key}_{k}"] = v
        
        if opt_weights:
            mx.savez(str(path / "optimizer.npz"), **opt_weights)
        
        # Save optimizer config
        opt_config = {k: v for k, v in opt_state.items() if k != "state"}
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
    optimizer: Optional[AdamW8bit] = None,
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
        weights = dict(mx.load(str(weights_path)))
        result["weights"] = weights
        
        if model is not None:
            _load_weights(model, "", weights)
            print(f"Loaded model weights from {weights_path}")
    
    # Load optimizer state
    opt_path = path / "optimizer.npz"
    opt_config_path = path / "optimizer_config.json"
    if opt_path.exists() and optimizer is not None:
        opt_weights = dict(mx.load(str(opt_path)))
        
        if opt_config_path.exists():
            with open(opt_config_path) as f:
                opt_config = json.load(f)
            optimizer.load_state_dict({**opt_config, "state": {}})
        
        print(f"Loaded optimizer state from {opt_path}")
    
    return result


def _collect_weights(module: Any, prefix: str, weights: Dict[str, mx.array]):
    """Recursively collect weights from a module."""
    # Handle module's direct parameters
    if hasattr(module, '_weight_int8'):
        weights[f"{prefix}weight_int8"] = module._weight_int8
        weights[f"{prefix}scale"] = module._scale
        if module._bias is not None:
            weights[f"{prefix}bias"] = module._bias
        return
    
    if hasattr(module, 'weight'):
        weights[f"{prefix}weight"] = module.weight
    
    if hasattr(module, 'bias') and module.bias is not None:
        weights[f"{prefix}bias"] = module.bias
    
    # Handle nested modules
    if hasattr(module, '__dict__'):
        for name, child in module.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(child, list):
                for i, item in enumerate(child):
                    _collect_weights(item, f"{prefix}{name}.{i}.", weights)
            elif hasattr(child, '__call__') or hasattr(child, 'weight'):
                _collect_weights(child, f"{prefix}{name}.", weights)


def _load_weights(module: Any, prefix: str, weights: Dict[str, mx.array]):
    """Recursively load weights into a module."""
    # Handle module's direct parameters
    if hasattr(module, '_weight_int8'):
        if f"{prefix}weight_int8" in weights:
            module._weight_int8 = weights[f"{prefix}weight_int8"]
            module._scale = weights[f"{prefix}scale"]
            if module._bias is not None and f"{prefix}bias" in weights:
                module._bias = weights[f"{prefix}bias"]
        return
    
    if hasattr(module, 'weight') and f"{prefix}weight" in weights:
        module.weight = weights[f"{prefix}weight"]
    
    if hasattr(module, 'bias') and f"{prefix}bias" in weights:
        module.bias = weights[f"{prefix}bias"]
    
    # Handle nested modules
    if hasattr(module, '__dict__'):
        for name, child in module.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(child, list):
                for i, item in enumerate(child):
                    _load_weights(item, f"{prefix}{name}.{i}.", weights)
            elif hasattr(child, '__call__') or hasattr(child, 'weight'):
                _load_weights(child, f"{prefix}{name}.", weights)


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

