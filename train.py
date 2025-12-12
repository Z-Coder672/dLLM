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
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import logging

import mlx.core as mx
import mlx.nn as nn

from src.config import ModelConfig, TrainingConfig
from src.model import TernaryTransformer, create_model
from src.layers import TernaryLinear
from src.optimizer import AdamW8bit, get_cosine_schedule_with_warmup, clip_gradients
from src.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from src.quantization import quantize_int8, dequantize_int8
from src.ste import compute_ternary_stats
from data.dataloader import create_dataloader, Batch, ValidationDataset
def setup_logger(log_path: Path) -> logging.Logger:
    """Configure a logger that logs to both stdout and a file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    return logger


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


def compute_loss_and_grads(
    model: TernaryTransformer,
    batch: Batch,
    use_checkpointing: bool = True,
    checkpoint_every: int = 6,
) -> Tuple[float, Dict[str, mx.array], Dict[str, mx.array]]:
    """
    Compute loss and gradients without applying an optimizer step.
    """
    params = get_trainable_params(model)
    
    def loss_fn(params_dict):
        set_trainable_params(model, params_dict)
        return compute_loss(model, batch, use_checkpointing, checkpoint_every)
    
    loss, grads = mx.value_and_grad(loss_fn)(params)
    
    # Materialize and detach grads to avoid holding computation graphs
    grads = {name: mx.stop_gradient(g) for name, g in grads.items()}
    mx.eval(*grads.values())
    
    return loss.item(), grads, params


def train_step(
    model: TernaryTransformer,
    optimizer: AdamW8bit,
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
            stats = compute_ternary_stats(w, module.threshold_factor)
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
    parser = argparse.ArgumentParser(description="Train Ternary Transformer")
    parser.add_argument("--config", type=str, default="configs/500m.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-resume from latest checkpoint")
    args = parser.parse_args()
    
    log_path = Path("training.log")
    logger = setup_logger(log_path)
    log = logger.info
    
    # Load configs
    log(f"Loading config from {args.config}")
    model_config = ModelConfig.from_yaml(args.config)
    training_config = TrainingConfig.from_yaml(args.config)
    
    log(f"\nModel config:")
    log(f"  d_model: {model_config.d_model}")
    log(f"  n_layers: {model_config.n_layers}")
    log(f"  n_heads: {model_config.n_heads}")
    log(f"  d_ff: {model_config.d_ff}")
    log(f"  vocab_size: {model_config.vocab_size}")
    
    log(f"\nTraining config:")
    log(f"  learning_rate: {training_config.learning_rate}")
    log(f"  batch_size: {training_config.batch_size}")
    log(f"  gradient_accumulation: {training_config.gradient_accumulation_steps}")
    log(f"  effective_batch_size: {training_config.effective_batch_size}")
    log(f"  dataset: {training_config.dataset_name}"
          f"{f'/{training_config.dataset_config}' if training_config.dataset_config else ''}")
    
    # Create model
    log("\nCreating model...")
    model = create_model(model_config)
    
    # Count parameters
    param_counts = model.count_parameters()
    total_params = param_counts.get("total", 0)
    log(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # Create optimizer
    optimizer = AdamW8bit(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        log(f"\nResuming from {args.resume}")
        state = load_checkpoint(args.resume, model, optimizer)
        start_step = state.get("step", 0)
        log(f"Resumed at step {start_step}")
    elif args.auto_resume:
        latest = get_latest_checkpoint(training_config.output_dir)
        if latest:
            log(f"\nAuto-resuming from {latest}")
            state = load_checkpoint(latest, model, optimizer)
            start_step = state.get("step", 0)
            log(f"Resumed at step {start_step}")
    
    # Create data loaders
    log("\nCreating data loaders...")
    train_loader = create_dataloader(
        split="train",
        batch_size=training_config.batch_size,
        seq_len=training_config.sequence_length,
        dataset_name=training_config.dataset_name,
        dataset_config=training_config.dataset_config,
    )
    
    val_dataset = ValidationDataset(
        num_samples=500,
        seq_len=training_config.sequence_length,
        dataset_name=training_config.dataset_name,
        dataset_config=training_config.dataset_config,
    )
    
    # Training loop
    log(f"\nStarting training from step {start_step}...")
    log(f"Output directory: {training_config.output_dir}")
    log("-" * 60)
    
    step = start_step
    accumulated_loss = 0.0
    accumulation_count = 0
    
    tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time
    grad_accum = None
    last_params = None
    
    for batch in train_loader:
        if step >= training_config.max_steps:
            break
        
        # Forward/backward; defer weight update until we accumulate enough microbatches
        loss, grads, params = compute_loss_and_grads(
            model=model,
            batch=batch,
            use_checkpointing=True,
            checkpoint_every=training_config.checkpoint_layers,
        )
        
        if grad_accum is None:
            grad_accum = grads
        else:
            grad_accum = {
                name: grad_accum[name] + grads[name]
                for name in grads
            }
        
        accumulated_loss += loss
        accumulation_count += 1
        tokens_processed += batch.batch_size * batch.seq_len
        last_params = params
        
        # Evaluate graph to free memory
        mx.eval(model.embed.weight)  # Force evaluation
        
        # Actual optimizer step (after gradient accumulation)
        if accumulation_count >= training_config.gradient_accumulation_steps:
            lr = get_cosine_schedule_with_warmup(
                step,
                training_config.warmup_steps,
                training_config.max_steps,
                training_config.min_learning_rate,
                training_config.learning_rate,
            )
            
            scale = 1.0 / accumulation_count
            mean_grads = {name: g * scale for name, g in grad_accum.items()}
            mean_grads, raw_grad_norm, clipped_grad_norm = clip_gradients(
                mean_grads,
                training_config.gradient_clip,
            )
            
            updated_params = optimizer.step(last_params, mean_grads, lr=lr)
            set_trainable_params(model, updated_params)
            
            step += 1
            
            avg_loss = accumulated_loss / accumulation_count
            
            # Logging
            if step % training_config.log_interval == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                
                perplexity = compute_perplexity(avg_loss)
                
                log(f"Step {step:6d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {perplexity:.2f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad: {clipped_grad_norm:.3f} (raw {raw_grad_norm:.3f}) | "
                    f"Tok/s: {tokens_per_sec:.0f}")
                
                last_log_time = current_time
                tokens_processed = 0
            
            # Evaluation
            if step % training_config.eval_interval == 0:
                log("\nRunning evaluation...")
                eval_metrics = evaluate(model, val_dataset, training_config.batch_size)
                ternary_stats = log_ternary_stats(model)
                
                log(f"  Val Loss: {eval_metrics['loss']:.4f}")
                log(f"  Val PPL: {eval_metrics['perplexity']:.2f}")
                if ternary_stats:
                    log(f"  Ternary +1: {ternary_stats['ternary_pct_positive']:.1f}%")
                    log(f"  Ternary  0: {ternary_stats['ternary_pct_zero']:.1f}%")
                    log(f"  Ternary -1: {ternary_stats['ternary_pct_negative']:.1f}%")
                log("")
            
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
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulation_count = 0
            grad_accum = None
    
    # Final save
    log("\nTraining complete!")
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
    log(f"Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()

