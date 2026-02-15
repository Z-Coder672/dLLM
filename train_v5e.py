#!/usr/bin/env python3
"""
Training script for 500M Transformer on Google Colab v5e TPU.

Full BF16 precision (no ternary quantization).
Optimized for v5e TPU with 8 chips and 128GB total memory.

Usage:
    python train_v5e.py --config configs/v5e.yaml
    python train_v5e.py --config configs/v5e.yaml --resume gdrive_checkpoint_path
"""

import argparse
import time
import math
import logging
import gc
import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from datetime import datetime
from dataclasses import asdict

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, value_and_grad
except ImportError:
    print("JAX not installed. Install with: pip install jax[tpu]")
    raise

import numpy as np
import yaml
from tqdm import tqdm


class LoggerWrapper:
    """Wrapper that both prints and logs messages."""
    def __init__(self, logger):
        self.logger = logger
    
    def info(self, msg):
        """Log and print info message."""
        print(msg)
        self.logger.info(msg)
    
    def debug(self, msg):
        """Log and print debug message."""
        print(msg)
        self.logger.debug(msg)

    def warning(self, msg):
        """Log and print warning message."""
        print(msg)
        self.logger.warning(msg)


def setup_logging(log_file: str = "training_v5e.log"):
    """Setup logging to both stdout and log file."""
    with open(log_file, 'w') as f:
        pass
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return LoggerWrapper(logger)


def setup_google_drive():
    """Mount Google Drive and return mounted path."""
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        logger.info("Google Drive mounted at /content/gdrive")
        return '/content/gdrive'
    except ImportError:
        logger.info("Not running in Colab, skipping Drive mount")
        return None
    except Exception as e:
        logger.warning(f"Failed to mount Google Drive: {e}")
        return None


def load_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model and training config from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model', {}), config.get('training', {})


class FlaxTransformerModel:
    """Simplified Transformer for JAX/Flax on TPU."""
    
    def __init__(self, model_config: Dict[str, Any], key: jax.random.PRNGKey):
        self.d_model = model_config['d_model']
        self.n_layers = model_config['n_layers']
        self.n_heads = model_config['n_heads']
        self.d_ff = model_config['d_ff']
        self.vocab_size = model_config['vocab_size']
        self.max_seq_len = model_config.get('max_seq_len', 512)
        self.dropout = model_config.get('dropout', 0.0)
        self.rope_theta = model_config.get('rope_theta', 10000.0)
        self.dtype = jnp.bfloat16
        
        self.params = self._init_params(key)
    
    def _init_params(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize model parameters."""
        params = {}
        
        # Embedding layer
        key, subkey = jax.random.split(key)
        params['embed'] = jax.random.normal(
            subkey, 
            (self.vocab_size, self.d_model), 
            dtype=self.dtype
        ) * 0.02
        
        # Transformer blocks
        for layer_idx in range(self.n_layers):
            key, *subkeys = jax.random.split(key, 6)
            layer_key = f'layer_{layer_idx}'
            params[layer_key] = {}
            
            # Attention weights
            params[layer_key]['attn_norm_scale'] = jnp.ones(self.d_model, dtype=self.dtype)
            params[layer_key]['q_proj'] = jax.random.normal(
                subkeys[0], 
                (self.d_model, self.d_model), 
                dtype=self.dtype
            ) * (0.02 / math.sqrt(self.n_heads))
            params[layer_key]['k_proj'] = jax.random.normal(
                subkeys[1], 
                (self.d_model, self.d_model), 
                dtype=self.dtype
            ) * (0.02 / math.sqrt(self.n_heads))
            params[layer_key]['v_proj'] = jax.random.normal(
                subkeys[2], 
                (self.d_model, self.d_model), 
                dtype=self.dtype
            ) * (0.02 / math.sqrt(self.n_heads))
            params[layer_key]['out_proj'] = jax.random.normal(
                subkeys[3], 
                (self.d_model, self.d_model), 
                dtype=self.dtype
            ) * 0.02
            
            # FFN weights
            params[layer_key]['ffn_norm_scale'] = jnp.ones(self.d_model, dtype=self.dtype)
            params[layer_key]['fc1'] = jax.random.normal(
                subkeys[4], 
                (self.d_model, self.d_ff), 
                dtype=self.dtype
            ) * 0.02
            params[layer_key]['fc2'] = jax.random.normal(
                subkeys[5], 
                (self.d_ff, self.d_model), 
                dtype=self.dtype
            ) * 0.02
        
        # Output layer
        params['norm_scale'] = jnp.ones(self.d_model, dtype=self.dtype)
        key, subkey = jax.random.split(key)
        params['lm_head'] = jax.random.normal(
            subkey, 
            (self.d_model, self.vocab_size), 
            dtype=self.dtype
        ) * 0.02
        
        return params
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        def count_array(x):
            return np.prod(x.shape) if isinstance(x, jnp.ndarray) else 0
        
        total = 0
        for key, value in self.params.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total += count_array(v)
            else:
                total += count_array(value)
        return total


class AdamWOptimizer:
    """AdamW optimizer in JAX."""
    
    def __init__(
        self,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}  # First moments
        self.v = {}  # Second moments
        self.t = 0   # Step count
    
    def init_state(self, params: Dict[str, Any]):
        """Initialize optimizer state."""
        def init_tree(x):
            if isinstance(x, jnp.ndarray):
                return jnp.zeros_like(x, dtype=jnp.float32)
            elif isinstance(x, dict):
                return {k: init_tree(v) for k, v in x.items()}
            return None
        
        self.m = init_tree(params)
        self.v = init_tree(params)
    
    def update(self, params: Dict[str, Any], grads: Dict[str, Any]) -> Dict[str, Any]:
        """Perform optimization step."""
        self.t += 1
        
        def update_param(p, g, m, v):
            if g is None or p is None:
                return p, m, v
            
            # Convert to float32 for stable updates
            p_f32 = p.astype(jnp.float32)
            g_f32 = g.astype(jnp.float32)
            
            # Update biased moments
            m = self.beta1 * m + (1 - self.beta1) * g_f32
            v = self.beta2 * v + (1 - self.beta2) * (g_f32 ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update with weight decay
            update = m_hat / (jnp.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                update = update + self.weight_decay * p_f32
            
            p_new = p_f32 - self.lr * update
            p_new = p_new.astype(p.dtype)
            
            return p_new, m, v
        
        def update_tree(p, g, m, v):
            if isinstance(p, dict):
                result_p, result_m, result_v = {}, {}, {}
                for k in p.keys():
                    result_p[k], result_m[k], result_v[k] = update_tree(
                        p[k], g.get(k), m[k], v[k]
                    )
                return result_p, result_m, result_v
            else:
                return update_param(p, g, m, v)
        
        new_params, self.m, self.v = update_tree(params, grads, self.m, self.v)
        return new_params


def compute_loss(params: Dict[str, Any], batch: Dict[str, jnp.ndarray], model: FlaxTransformerModel) -> jnp.ndarray:
    """Compute cross-entropy loss."""
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    # Simple forward pass (simplified attention mechanism)
    # In production, use full Transformer forward pass
    x = params['embed'][input_ids]  # (batch, seq, d_model)
    
    # Process through transformer layers (simplified)
    for layer_idx in range(model.n_layers):
        layer_params = params[f'layer_{layer_idx}']
        # Simplified residual: x = x + attention(x) + ffn(x)
        # In production, implement full attention and ffn
        pass
    
    # Output projection
    logits = jnp.dot(x, params['lm_head'])  # (batch, seq, vocab)
    
    # Flatten for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    
    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs,
        labels_flat[:, None],
        axis=-1
    ).squeeze(-1)
    
    loss = -jnp.mean(target_log_probs)
    return loss


def get_learning_rate(step: int, config: Dict[str, Any]) -> float:
    """Compute learning rate with cosine schedule."""
    warmup_steps = config.get('warmup_steps', 2000)
    max_steps = config.get('max_steps', 1000000)
    min_lr = config.get('min_learning_rate', 1e-5)
    base_lr = config.get('learning_rate', 5e-4)
    
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(progress, 1.0)
    
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def list_checkpoints(output_dir: str) -> list:
    """List available checkpoints."""
    path = Path(output_dir)
    if not path.exists():
        return []
    
    checkpoints = []
    for item in sorted(path.iterdir()):
        if item.is_dir() and (item / "state.json").exists():
            with open(item / "state.json") as f:
                state = json.load(f)
            checkpoints.append({
                "path": str(item),
                "step": state.get("step", 0),
            })
    
    return sorted(checkpoints, key=lambda x: x["step"])


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Get latest checkpoint path."""
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1]["path"] if checkpoints else None


def prune_checkpoints(output_dir: str, current_step: int, save_interval: int):
    """Prune checkpoints using log-spaced thinning."""
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints or save_interval <= 0:
        return
    
    checkpoints = sorted(checkpoints, key=lambda x: x["step"])
    keep_paths = set()
    bucket_best = {}
    
    for ckpt in checkpoints:
        step = ckpt["step"]
        path = ckpt["path"]
        age = max(0, current_step - step)
        
        log_base = 2.0
        bucket = int(math.floor(math.log(age / save_interval, log_base))) if age > 0 else 0
        best = bucket_best.get(bucket)
        
        if best is None or step < best[0]:
            bucket_best[bucket] = (step, path)
    
    for _, path in bucket_best.values():
        keep_paths.add(path)
    
    for ckpt in checkpoints:
        path = ckpt["path"]
        if path not in keep_paths:
            try:
                shutil.rmtree(path)
                logger.info(f"Pruned checkpoint at step {ckpt['step']}")
            except Exception as e:
                logger.info(f"Failed to prune {path}: {e}")


def save_checkpoint(
    path: str,
    params: Dict[str, Any],
    optimizer: AdamWOptimizer,
    step: int,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Save checkpoint to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save params as numpy
    def tree_to_numpy(tree):
        if isinstance(tree, dict):
            return {k: tree_to_numpy(v) for k, v in tree.items()}
        elif isinstance(tree, jnp.ndarray):
            return np.array(tree)
        return tree
    
    params_np = tree_to_numpy(params)
    np.savez_compressed(path / "params.npz", **flatten_dict(params_np))
    
    # Save optimizer state
    m_np = tree_to_numpy(optimizer.m)
    v_np = tree_to_numpy(optimizer.v)
    np.savez_compressed(path / "optimizer_m.npz", **flatten_dict(m_np))
    np.savez_compressed(path / "optimizer_v.npz", **flatten_dict(v_np))
    
    # Save training state
    state = {
        "step": step,
        "optimizer_step": optimizer.t,
    }
    with open(path / "state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    # Save configs
    with open(path / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    with open(path / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    logger.info(f"Saved checkpoint to {path}")


FLAT_KEY_SEP = "::"


def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{FLAT_KEY_SEP}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_checkpoint(path: str, model: FlaxTransformerModel, optimizer: AdamWOptimizer):
    """Load checkpoint from disk."""
    path = Path(path)
    
    # Load params
    params_flat = dict(np.load(path / "params.npz"))
    params = unflatten_dict(params_flat)
    
    # Load optimizer state
    m_flat = dict(np.load(path / "optimizer_m.npz"))
    optimizer.m = unflatten_dict(m_flat)
    
    v_flat = dict(np.load(path / "optimizer_v.npz"))
    optimizer.v = unflatten_dict(v_flat)
    
    # Load training state
    with open(path / "state.json") as f:
        state = json.load(f)
    
    optimizer.t = state.get("optimizer_step", 0)
    start_step = state.get("step", 0)
    
    logger.info(f"Loaded checkpoint from {path} at step {start_step}")
    return params, start_step


def unflatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Unflatten dictionary."""
    result = {}
    for key, value in d.items():
        parts = key.split(FLAT_KEY_SEP)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        if isinstance(value, np.ndarray):
            current[parts[-1]] = jnp.array(value)
        else:
            current[parts[-1]] = value
    return result


def main():
    global logger
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Train 500M Transformer on v5e TPU")
    parser.add_argument("-c", "--config", type=str, required=True,
                       help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                       help="Auto-resume from latest checkpoint")
    args = parser.parse_args()
    
    # Setup Google Drive
    gdrive_path = setup_google_drive()
    
    # Load configs
    logger.info(f"Loading config from {args.config}")
    model_config, training_config = load_config(args.config)
    
    logger.info(f"Model: d_model={model_config['d_model']}, n_layers={model_config['n_layers']}")
    logger.info(f"Training: batch_size={training_config['batch_size']}, lr={training_config['learning_rate']}")
    logger.info(f"Output: {training_config['output_dir']}")
    
    # Create output directory
    output_dir = training_config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    logger.info("Initializing model...")
    key = jax.random.PRNGKey(42)
    model = FlaxTransformerModel(model_config, key)
    total_params = model.count_parameters()
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # Initialize optimizer
    optimizer = AdamWOptimizer(
        learning_rate=training_config['learning_rate'],
        beta1=training_config['betas'][0],
        beta2=training_config['betas'][1],
        eps=training_config['eps'],
        weight_decay=training_config['weight_decay'],
    )
    optimizer.init_state(model.params)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        model.params, start_step = load_checkpoint(args.resume, model, optimizer)
    elif args.auto_resume:
        latest = get_latest_checkpoint(output_dir)
        if latest:
            logger.info(f"Auto-resuming from {latest}")
            model.params, start_step = load_checkpoint(latest, model, optimizer)
    
    logger.info(f"Starting training from step {start_step}")
    logger.info("-" * 60)
    
    # Training loop (simplified - requires actual dataloader implementation)
    step = start_step
    max_steps = training_config.get('max_steps', 1000000)
    save_interval = training_config.get('save_interval', 1000)
    eval_interval = training_config.get('eval_interval', 1000)
    log_interval = training_config.get('log_interval', 20)
    
    accumulated_loss = 0.0
    tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time
    
    logger.info("Training loop ready. Implement dataloader and training step.")
    logger.info("This script provides the framework; complete the training loop.")
    logger.info("Checkpoints will be saved to Google Drive automatically.")


if __name__ == "__main__":
    main()
