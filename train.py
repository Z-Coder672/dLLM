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

import mlx.core as mx
import mlx.nn as nn

from src.config import ModelConfig, TrainingConfig
from src.model import TernaryTransformer, create_model
from src.optimizer import AdamW8bit, get_cosine_schedule_with_warmup, clip_gradients
from src.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from src.quantization import quantize_int8, dequantize_int8
from src.ste import compute_ternary_stats
from data.dataloader import create_dataloader, Batch, ValidationDataset


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
    log_probs = mx.log_softmax(logits_flat.astype(mx.float32), axis=-1)
    
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
    params = {}
    _collect_params(model, "", params)
    return params


def _collect_params(module, prefix: str, params: Dict[str, mx.array]):
    """Recursively collect parameters."""
    # Handle TernaryLinear specially - get dequantized weights
    if hasattr(module, 'get_weight_bf16') and hasattr(module, '_weight_int8'):
        params[f"{prefix}weight"] = module.get_weight_bf16()
        if module._bias is not None:
            params[f"{prefix}bias"] = module._bias
        return
    
    # Regular parameters
    if hasattr(module, 'weight') and isinstance(module.weight, mx.array):
        params[f"{prefix}weight"] = module.weight
    
    if hasattr(module, 'bias') and module.bias is not None:
        params[f"{prefix}bias"] = module.bias
    
    # Recurse into children
    if hasattr(module, '__dict__'):
        for name, child in module.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(child, list):
                for i, item in enumerate(child):
                    _collect_params(item, f"{prefix}{name}.{i}.", params)
            elif hasattr(child, '__call__') or hasattr(child, 'weight'):
                _collect_params(child, f"{prefix}{name}.", params)


def set_trainable_params(model: TernaryTransformer, params: Dict[str, mx.array]):
    """
    Set updated parameters back into the model.
    
    For TernaryLinear layers, we re-quantize to INT8.
    """
    _set_params(model, "", params)


def _set_params(module, prefix: str, params: Dict[str, mx.array]):
    """Recursively set parameters."""
    # Handle TernaryLinear specially - requantize
    if hasattr(module, 'set_weight_bf16') and hasattr(module, '_weight_int8'):
        if f"{prefix}weight" in params:
            module.set_weight_bf16(params[f"{prefix}weight"])
        if module._bias is not None and f"{prefix}bias" in params:
            module._bias = params[f"{prefix}bias"]
        return
    
    # Regular parameters
    if hasattr(module, 'weight') and f"{prefix}weight" in params:
        module.weight = params[f"{prefix}weight"]
    
    if hasattr(module, 'bias') and f"{prefix}bias" in params:
        module.bias = params[f"{prefix}bias"]
    
    # Recurse into children
    if hasattr(module, '__dict__'):
        for name, child in module.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(child, list):
                for i, item in enumerate(child):
                    _set_params(item, f"{prefix}{name}.{i}.", params)
            elif hasattr(child, '__call__') or hasattr(child, 'weight'):
                _set_params(child, f"{prefix}{name}.", params)


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
        
        log_probs = mx.log_softmax(logits_flat.astype(mx.float32), axis=-1)
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
    
    def check_module(module):
        nonlocal total_pos, total_neg, total_zero, total_params
        if hasattr(module, 'get_weight_bf16') and hasattr(module, 'threshold_factor'):
            w = module.get_weight_bf16()
            stats = compute_ternary_stats(w, module.threshold_factor)
            total_pos += stats["n_positive"]
            total_neg += stats["n_negative"]
            total_zero += stats["n_zero"]
            total_params += stats["total"]
    
    def traverse(module):
        check_module(module)
        if hasattr(module, '__dict__'):
            for name, child in module.__dict__.items():
                if name.startswith('_'):
                    continue
                if isinstance(child, list):
                    for item in child:
                        traverse(item)
                elif hasattr(child, '__call__'):
                    traverse(child)
    
    traverse(model)
    
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
    
    # Load configs
    print(f"Loading config from {args.config}")
    model_config = ModelConfig.from_yaml(args.config)
    training_config = TrainingConfig.from_yaml(args.config)
    
    print(f"\nModel config:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  d_ff: {model_config.d_ff}")
    print(f"  vocab_size: {model_config.vocab_size}")
    
    print(f"\nTraining config:")
    print(f"  learning_rate: {training_config.learning_rate}")
    print(f"  batch_size: {training_config.batch_size}")
    print(f"  gradient_accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  effective_batch_size: {training_config.effective_batch_size}")
    print(f"  dataset: {training_config.dataset_name}"
          f"{f'/{training_config.dataset_config}' if training_config.dataset_config else ''}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(model_config)
    
    # Count parameters
    param_counts = model.count_parameters()
    total_params = param_counts.get("total", 0)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # Create optimizer
    optimizer = AdamW8bit(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        state = load_checkpoint(args.resume, model, optimizer)
        start_step = state.get("step", 0)
        print(f"Resumed at step {start_step}")
    elif args.auto_resume:
        latest = get_latest_checkpoint(training_config.output_dir)
        if latest:
            print(f"\nAuto-resuming from {latest}")
            state = load_checkpoint(latest, model, optimizer)
            start_step = state.get("step", 0)
            print(f"Resumed at step {start_step}")
    
    # Create data loaders
    print("\nCreating data loaders...")
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
    print(f"\nStarting training from step {start_step}...")
    print(f"Output directory: {training_config.output_dir}")
    print("-" * 60)
    
    step = start_step
    accumulated_loss = 0.0
    accumulated_grad_norm = 0.0
    accumulation_count = 0
    
    tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time
    
    for batch in train_loader:
        if step >= training_config.max_steps:
            break
        
        # Get learning rate
        lr = get_cosine_schedule_with_warmup(
            step,
            training_config.warmup_steps,
            training_config.max_steps,
            training_config.min_learning_rate,
            training_config.learning_rate,
        )
        
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
        
        # Evaluate graph to free memory
        mx.eval(model.embed.weight)  # Force evaluation
        
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
                
                print(f"Step {step:6d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"PPL: {perplexity:.2f} | "
                      f"LR: {lr:.2e} | "
                      f"Grad: {avg_grad_norm:.3f} | "
                      f"Tok/s: {tokens_per_sec:.0f}")
                
                last_log_time = current_time
                tokens_processed = 0
            
            # Evaluation
            if step % training_config.eval_interval == 0:
                print("\nRunning evaluation...")
                eval_metrics = evaluate(model, val_dataset, training_config.batch_size)
                ternary_stats = log_ternary_stats(model)
                
                print(f"  Val Loss: {eval_metrics['loss']:.4f}")
                print(f"  Val PPL: {eval_metrics['perplexity']:.2f}")
                if ternary_stats:
                    print(f"  Ternary +1: {ternary_stats['ternary_pct_positive']:.1f}%")
                    print(f"  Ternary  0: {ternary_stats['ternary_pct_zero']:.1f}%")
                    print(f"  Ternary -1: {ternary_stats['ternary_pct_negative']:.1f}%")
                print()
            
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
            accumulated_grad_norm = 0.0
            accumulation_count = 0
    
    # Final save
    print("\nTraining complete!")
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
    print(f"Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()

