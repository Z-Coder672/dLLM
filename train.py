#!/usr/bin/env python3
"""
Training script for Ternary Transformer with INT8 shadow weights.

Usage:
    python train.py --config configs/500m.yaml
    python train.py --config configs/500m.yaml --resume checkpoints/500m/step_10000
"""

import argparse
import time
import math
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn

from src.config import ModelConfig, TrainingConfig
from src.model import TernaryTransformer, create_model
from src.layers import TernaryLinear
from src.optimizer import AdamW, get_cosine_schedule_with_warmup, clip_gradients
from src.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    prune_checkpoints,
)
from src.ste import compute_ternary_stats
from data.dataloader import create_dataloader, Batch, ValidationDataset


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


def setup_logging(log_file: str = "training.log"):
    """Setup logging to both stdout and log file."""
    # Clear the log file
    with open(log_file, 'w') as f:
        pass
    
    # Suppress noisy third-party library loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Only INFO and above from third-party libs
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler only (console output handled by wrapper's print())
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Return wrapper that prints to console and logs to file
    return LoggerWrapper(logger)


def compute_loss(
    model: TernaryTransformer,
    batch: Batch,
    use_checkpointing: bool = True,
    checkpoint_every: int = 6,
) -> mx.array:
    """
    Compute cross-entropy loss for a batch.
    
    Args:
        model: The transformer model
        batch: Training batch with input_ids and labels
        use_checkpointing: Whether to use gradient checkpointing
        checkpoint_every: Checkpoint every N layers
        
    Returns:
        Scalar loss value
    """
    if use_checkpointing:
        logits = model.forward_with_checkpointing(
            batch.input_ids,
            checkpoint_every=checkpoint_every,
            training=True,
        )
    else:
        logits, _ = model(batch.input_ids, training=True)
    
    # Reshape for cross-entropy: (batch * seq_len, vocab_size)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = batch.labels.reshape(-1)
    
    # Compute cross-entropy loss
    # Using log_softmax for numerical stability
    log_probs = nn.log_softmax(logits_flat.astype(mx.float32), axis=-1)
    
    # Gather log probs for target tokens
    # labels_flat: (batch * seq_len,)
    # log_probs: (batch * seq_len, vocab_size)
    target_log_probs = mx.take_along_axis(
        log_probs, 
        labels_flat[:, None], 
        axis=-1
    ).squeeze(-1)
    
    # Mean loss
    loss = -mx.mean(target_log_probs)
    
    return loss


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return math.exp(min(loss, 20))  # Cap to avoid overflow


def compute_ternary_strength(step: int, config: TrainingConfig) -> float:
    """
    Compute blend factor for ternary weights with fade-in schedule.
    
    Returns a value in [0,1]: 0 uses full-precision weights, 1 uses ternary.
    """
    if step < config.ternary_skip_steps:
        return 0.0
    
    if config.ternary_fadein_steps <= 0:
        return 1.0
    
    progress = (step - config.ternary_skip_steps) / config.ternary_fadein_steps
    return max(0.0, min(1.0, progress))


def get_trainable_params(model: TernaryTransformer) -> Dict[str, mx.array]:
    """
    Get all trainable parameters from the model.
    
    For TernaryLinear layers, we return the dequantized BF16 weights.
    """
    params: Dict[str, mx.array] = {}
    
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        
        # Handle ternary layers specially
        if isinstance(module, TernaryLinear):
            params[f"{prefix}weight"] = module.get_weight_bf16()
            if module.bias is not None:
                params[f"{prefix}bias"] = module.bias
            continue
        
        # Regular parameters
        weight = getattr(module, "weight", None)
        if isinstance(weight, mx.array):
            params[f"{prefix}weight"] = weight
        
        bias = getattr(module, "bias", None)
        if bias is not None and isinstance(bias, mx.array):
            params[f"{prefix}bias"] = bias
    
    return params


def set_trainable_params(model: TernaryTransformer, params: Dict[str, mx.array]):
    """
    Set updated parameters back into the model.
    
    For TernaryLinear layers, we re-quantize to INT8.
    """
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        
        if isinstance(module, TernaryLinear):
            w_key = f"{prefix}weight"
            if w_key in params:
                module.set_weight_bf16(params[w_key])
            
            b_key = f"{prefix}bias"
            if module.bias is not None and b_key in params:
                module._bias = params[b_key]
            continue
        
        w_key = f"{prefix}weight"
        if hasattr(module, "weight") and w_key in params:
            module.weight = params[w_key]
        
        b_key = f"{prefix}bias"
        if hasattr(module, "bias") and b_key in params:
            module.bias = params[b_key]


def apply_ternary_schedule(
    model: TernaryTransformer,
    step: int,
    config: TrainingConfig,
) -> float:
    """
    Apply ternary enable/strength schedule to the model.
    
    Returns:
        strength in [0,1] after scheduling.
    """
    strength = compute_ternary_strength(step, config)
    model.set_ternary_enabled(strength > 0.0)
    model.set_ternary_strength(strength)
    return strength


def get_lr(step: int, config: TrainingConfig) -> float:
    """Get learning rate based on schedule."""
    if config.lr_schedule.lower() == "cosine":
        return get_cosine_schedule_with_warmup(
            step,
            config.warmup_steps,
            config.max_steps,
            config.min_learning_rate,
            config.learning_rate,
        )
    elif config.lr_schedule.lower() == "constant":
        return config.learning_rate if step >= config.warmup_steps else config.learning_rate * step / config.warmup_steps
    else:
        # Default to cosine
        return get_cosine_schedule_with_warmup(
            step,
            config.warmup_steps,
            config.max_steps,
            config.min_learning_rate,
            config.learning_rate,
        )


def train_step(
    model: TernaryTransformer,
    optimizer: AdamW,
    batch: Batch,
    lr: float,
    gradient_clip: float,
    use_checkpointing: bool = True,
    checkpoint_every: int = 6,
) -> Tuple[float, float]:
    """
    Perform a single training step.
    
    Returns:
        loss: Training loss
        grad_norm: Gradient norm before clipping
    """
    # Get trainable parameters
    params = get_trainable_params(model)
    
    # Define loss function for value_and_grad
    def loss_fn(params_dict):
        # Temporarily set parameters
        set_trainable_params(model, params_dict)
        return compute_loss(model, batch, use_checkpointing, checkpoint_every)
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(params)
    
    # Clip gradients
    grads, grad_norm = clip_gradients(grads, gradient_clip)
    
    # Optimizer step
    updated_params = optimizer.step(params, grads, lr=lr)
    
    # Set updated parameters back
    set_trainable_params(model, updated_params)
    
    return loss.item(), grad_norm


def evaluate(
    model: TernaryTransformer,
    val_dataset: ValidationDataset,
    batch_size: int,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Returns:
        Dictionary with loss, perplexity
    """
    batches = val_dataset.get_batches(batch_size)
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch in batches:
        logits, _ = model(batch.input_ids, training=False)
        
        # Compute loss
        batch_size_actual, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = batch.labels.reshape(-1)
        
        log_probs = nn.log_softmax(logits_flat.astype(mx.float32), axis=-1)
        target_log_probs = mx.take_along_axis(
            log_probs,
            labels_flat[:, None],
            axis=-1
        ).squeeze(-1)
        
        batch_loss = -mx.sum(target_log_probs)
        batch_tokens = batch_size_actual * seq_len
        
        total_loss += batch_loss.item()
        total_tokens += batch_tokens
        
        mx.eval(batch_loss)
    
    avg_loss = total_loss / total_tokens
    perplexity = compute_perplexity(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


def log_ternary_stats(model: TernaryTransformer) -> Dict[str, float]:
    """Log statistics about ternary weight distribution."""
    total_pos = 0
    total_neg = 0
    total_zero = 0
    total_params = 0
    
    for _, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            w = module.get_weight_bf16()
            stats = compute_ternary_stats(
                w,
                module.threshold_factor,
                getattr(module, "ternary_temperature", 0.15),
            )
            total_pos += stats["n_positive"]
            total_neg += stats["n_negative"]
            total_zero += stats["n_zero"]
            total_params += stats["total"]
    
    if total_params > 0:
        return {
            "ternary_pct_positive": 100 * total_pos / total_params,
            "ternary_pct_negative": 100 * total_neg / total_params,
            "ternary_pct_zero": 100 * total_zero / total_params,
        }
    return {}


def main():
    # Setup logging
    logger = setup_logging("training.log")
    
    parser = argparse.ArgumentParser(description="Train Ternary Transformer")
    parser.add_argument("--config", type=str, default="configs/500m.yaml",
                        help="Path to config file")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Path to checkpoint to start from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-resume from latest checkpoint")
    args = parser.parse_args()
    
    # Load configs
    logger.info(f"Loading config from {args.config}")
    model_config = ModelConfig.from_yaml(args.config)
    training_config = TrainingConfig.from_yaml(args.config)
    
    logger.info(f"\nModel config:")
    logger.info(f"  d_model: {model_config.d_model}")
    logger.info(f"  n_layers: {model_config.n_layers}")
    logger.info(f"  n_heads: {model_config.n_heads}")
    logger.info(f"  d_ff: {model_config.d_ff}")
    logger.info(f"  vocab_size: {model_config.vocab_size}")
    
    logger.info(f"\nTraining config:")
    logger.info(f"  learning_rate: {training_config.learning_rate}")
    logger.info(f"  batch_size: {training_config.batch_size}")
    logger.info(f"  gradient_accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"  effective_batch_size: {training_config.effective_batch_size}")
    if training_config.datasets:
        logger.info(f"  datasets (mixed):")
        for d in training_config.datasets:
            cfg_str = f" ({d['config']})" if d.get("config") else ""
            logger.info(f"    - {d['name']}{cfg_str}: weight {d.get('weight', 1.0)}")
    else:
        logger.info(f"  dataset: {training_config.dataset_name}"
              f"{f'/{training_config.dataset_config}' if training_config.dataset_config else ''}")
    logger.info(f"  max_steps: {training_config.max_steps}")
    logger.info(f"  stop_steps: {training_config.stop_steps}")
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(model_config)
    
    # Count parameters
    param_counts = model.count_parameters()
    total_params = param_counts.get("total", 0)
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # Create optimizer
    if training_config.optimizer.lower() == "adamw":
        optimizer = AdamW(
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=tuple(training_config.betas),
            eps=training_config.eps,
        )
    elif training_config.optimizer.lower() == "sgd":
        from src.optimizer import SGDMomentum
        optimizer = SGDMomentum(
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config.optimizer}")
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.checkpoint:
        logger.info(f"\nLoading checkpoint from {args.checkpoint}")
        state = load_checkpoint(args.checkpoint, model, optimizer)
        start_step = state.get("step", 0)
        logger.info(f"Loaded checkpoint at step {start_step}")
    elif args.resume:
        logger.info(f"\nResuming from {args.resume}")
        state = load_checkpoint(args.resume, model, optimizer)
        start_step = state.get("step", 0)
        logger.info(f"Resumed at step {start_step}")
    elif args.auto_resume:
        latest = get_latest_checkpoint(training_config.output_dir)
        if latest:
            logger.info(f"\nAuto-resuming from {latest}")
            state = load_checkpoint(latest, model, optimizer)
            start_step = state.get("step", 0)
            logger.info(f"Resumed at step {start_step}")
    
    # Honor ternary warmup/fade-in (disable until skip, then fade)
    apply_ternary_schedule(model, start_step, training_config)
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader = create_dataloader(
        split="train",
        batch_size=training_config.batch_size,
        seq_len=training_config.sequence_length,
        dataset_name=training_config.dataset_name,
        dataset_config=training_config.dataset_config,
        datasets_config=training_config.datasets,
        streaming=training_config.streaming,
    )
    
    val_dataset_name = training_config.datasets[0]["name"] if training_config.datasets else training_config.dataset_name
    val_dataset_config = training_config.datasets[0].get("config") if training_config.datasets else training_config.dataset_config

    val_dataset = ValidationDataset(
        num_samples=500,
        seq_len=training_config.sequence_length,
        dataset_name=val_dataset_name,
        dataset_config=val_dataset_config,
    )
    
    # Training loop
    logger.info(f"\nStarting training from step {start_step}...")
    logger.info(f"Output directory: {training_config.output_dir}")
    logger.info("-" * 60)
    
    step = start_step
    accumulated_loss = 0.0
    accumulated_grad_norm = 0.0
    accumulation_count = 0
    
    tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time
    # Decide when to stop: honor stop_steps if > 0, otherwise use max_steps
    stop_at = training_config.stop_steps if training_config.stop_steps > 0 else training_config.max_steps
    
    for batch in train_loader:
        if stop_at is not None and step >= stop_at:
            break
        
        # Enable ternary weights with fade-in schedule
        _ = apply_ternary_schedule(model, step, training_config)
        
        # Get learning rate
        lr = get_lr(step, training_config)
        
        # Training step
        loss, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            lr=lr,
            gradient_clip=training_config.gradient_clip,
            use_checkpointing=True,
            checkpoint_every=training_config.checkpoint_layers,
        )
        
        # Accumulate
        accumulated_loss += loss
        accumulated_grad_norm += grad_norm
        accumulation_count += 1
        tokens_processed += batch.batch_size * batch.seq_len
        
        mx.eval(loss)
        mx.eval(accumulated_loss)
        for param in get_trainable_params(model).values():
            mx.eval(param)
        for state in optimizer.state.values():
            mx.eval(state.m)
            mx.eval(state.v)
        gc.collect()
        mx.clear_cache()
        
        # Actual step (after gradient accumulation)
        if accumulation_count >= training_config.gradient_accumulation_steps:
            step += 1
            
            avg_loss = accumulated_loss / accumulation_count
            avg_grad_norm = accumulated_grad_norm / accumulation_count
            
            # Logging
            if step % training_config.log_interval == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                
                perplexity = compute_perplexity(avg_loss)
                active = mx.get_active_memory() / 1e9
                peak = mx.get_peak_memory() / 1e9
                
                logger.info(f"Step {step:6d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"PPL: {perplexity:.2f} | "
                      f"LR: {lr:.2e} | "
                      f"Grad: {avg_grad_norm:.3f} | "
                      f"Tok/s: {tokens_per_sec:.0f}")
                logger.info(f"Mem: {active:.2f}GB active, {peak:.2f}GB peak")
                mx.reset_peak_memory()
                
                last_log_time = current_time
                tokens_processed = 0
            
            # Evaluation
            if step % training_config.eval_interval == 0:
                logger.info("\nRunning evaluation...")
                eval_metrics = evaluate(model, val_dataset, training_config.batch_size)
                ternary_stats = log_ternary_stats(model)
                
                logger.info(f"  Val Loss: {eval_metrics['loss']:.4f}")
                logger.info(f"  Val PPL: {eval_metrics['perplexity']:.2f}")
                if ternary_stats:
                    logger.info(f"  Ternary +1: {ternary_stats['ternary_pct_positive']:.1f}%")
                    logger.info(f"  Ternary  0: {ternary_stats['ternary_pct_zero']:.1f}%")
                    logger.info(f"  Ternary -1: {ternary_stats['ternary_pct_negative']:.1f}%")
                logger.info("")
            
            # Save checkpoint
            if step % training_config.save_interval == 0:
                ckpt_path = Path(training_config.output_dir) / f"step_{step}"
                save_checkpoint(
                    path=str(ckpt_path),
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=model_config,
                    training_config=training_config,
                    metrics={"loss": avg_loss},
                )
                prune_checkpoints(
                    output_dir=training_config.output_dir,
                    current_step=step,
                    save_interval=training_config.save_interval,
                )
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulated_grad_norm = 0.0
            accumulation_count = 0
            
            # Free buffers after each completed optimization step
            mx.eval(model.embed.weight)
            
            if stop_at is not None and step >= stop_at:
                break
    
    # Final save
    logger.info("\nTraining complete!")
    final_path = Path(training_config.output_dir) / "final"
    save_checkpoint(
        path=str(final_path),
        model=model,
        optimizer=optimizer,
        step=step,
        config=model_config,
        training_config=training_config,
    )
    
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
