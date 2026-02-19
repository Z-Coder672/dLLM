#!/usr/bin/env python3
"""
LoRA fine-tuning script for the Ternary Transformer.

Loads the TPU-trained step_120000 checkpoint, applies LoRA adapters to
attention projections, and fine-tunes on an interleaved mix of
90% UltraChat-200k + 10% WildChat using MLX.

Usage:
    python finetune_lora.py
    python finetune_lora.py --checkpoint checkpoints/500m/step_120000 \
        --lora-rank 16 --lr 2e-4 --max-steps 5000
"""

import argparse
import gc
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from src.config import ModelConfig
from src.layers import TernaryLinear
from src.model import TernaryTransformer
from data.dataloader import Batch, get_tokenizer

# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """LoRA wrapper around TernaryLinear.

    Freezes the base weight and adds low-rank A/B adapters so that the
    effective forward becomes:  x @ W^T + (x @ A^T) @ B^T  * scaling
    """

    def __init__(self, base: TernaryLinear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features
        # A: (rank, in_features)  — initialised with Kaiming uniform
        scale = 1.0 / math.sqrt(in_features)
        self.lora_A = mx.random.uniform(
            low=-scale, high=scale,
            shape=(rank, in_features),
            dtype=mx.float32,
        )
        # B: (out_features, rank) — zero-initialised so LoRA starts as identity
        self.lora_B = mx.zeros((out_features, rank), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        # LoRA path in float32 for stability
        x_f32 = x.astype(mx.float32)
        lora_out = (x_f32 @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out.astype(base_out.dtype)


def apply_lora(model: TernaryTransformer, rank: int = 16, alpha: float = 32.0):
    """Wrap every attention Q/K/V/O projection with a LoRA adapter."""
    for i in range(len(model.layers)):
        layer = model.layers[i]
        attn = layer.attention
        attn.q_proj = LoRALinear(attn.q_proj, rank, alpha)
        attn.k_proj = LoRALinear(attn.k_proj, rank, alpha)
        attn.v_proj = LoRALinear(attn.v_proj, rank, alpha)
        attn.o_proj = LoRALinear(attn.o_proj, rank, alpha)


def get_lora_params(model: TernaryTransformer) -> Dict[str, mx.array]:
    """Collect only LoRA A/B parameters (the only trainable params)."""
    params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"{name}." if name else ""
            params[f"{prefix}lora_A"] = module.lora_A
            params[f"{prefix}lora_B"] = module.lora_B
    return params


def set_lora_params(model: TernaryTransformer, params: Dict[str, mx.array]):
    """Write updated LoRA parameters back into the model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"{name}." if name else ""
            a_key = f"{prefix}lora_A"
            b_key = f"{prefix}lora_B"
            if a_key in params:
                module.lora_A = params[a_key]
            if b_key in params:
                module.lora_B = params[b_key]


def count_lora_params(model: TernaryTransformer) -> int:
    total = 0
    for p in get_lora_params(model).values():
        total += p.size
    return total


# ---------------------------------------------------------------------------
# Checkpoint loading (reuse chat.py legacy loader)
# ---------------------------------------------------------------------------

def load_base_model(checkpoint_path: str) -> Tuple[TernaryTransformer, ModelConfig]:
    """Load the TPU-trained checkpoint using the legacy loader from chat.py."""
    from chat import load_model
    model, config = load_model(checkpoint_path)
    # Freeze base weights — ternary disabled for SFT fine-tuning
    model.set_ternary_enabled(False)
    return model, config


# ---------------------------------------------------------------------------
# Chat-format data loading
# ---------------------------------------------------------------------------

EOT_TOKEN = 50256  # GPT-2 <|endoftext|>

# ChatML template — the standard multi-turn chat format used by
# OpenChat, Hermes, Nous, etc.  Encoded as plain text since the GPT-2
# tokenizer has no native ChatML special tokens.
CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"


def format_conversation(messages: list, tokenizer) -> Optional[List[int]]:
    """Format a conversation with the ChatML template and tokenize it.

    Each turn becomes:
        <|im_start|>role\ncontent<|im_end|>\n

    The full conversation is terminated with <|endoftext|>.
    Returns None if the conversation is empty or too short.
    """
    if not messages or len(messages) < 2:
        return None

    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or role not in ("user", "assistant", "system"):
            continue
        text_parts.append(f"{CHATML_START}{role}\n{content}{CHATML_END}\n")

    if len(text_parts) < 2:
        return None

    full_text = "".join(text_parts)
    tokens = tokenizer.encode(full_text)
    tokens.append(EOT_TOKEN)
    return tokens


class ChatSFTDataset:
    """Interleaved streaming SFT dataset: 90% UltraChat-200k + 10% WildChat."""

    def __init__(self, seq_len: int = 512, shuffle_buffer: int = 2000):
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer  # kept small for 16GB RAM
        self.tokenizer = get_tokenizer()

        self._ultrachat_iter = None
        self._wildchat_iter = None
        self._ultrachat_ds = None
        self._wildchat_ds = None
        self._buffers = [[], []]  # [ultrachat_buf, wildchat_buf]

    def _load_datasets(self):
        import os
        from datasets import DownloadConfig, load_dataset

        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")

        cache_dir = str(Path(__file__).resolve().parent / "data" / "cache")

        dl_cfg = DownloadConfig(cache_dir=cache_dir, max_retries=5)

        # UltraChat-200k — train_sft split
        self._ultrachat_ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True,
            cache_dir=cache_dir,
            download_config=dl_cfg,
        ).shuffle(buffer_size=self.shuffle_buffer, seed=42)
        self._ultrachat_iter = iter(self._ultrachat_ds)

        # WildChat-1M
        self._wildchat_ds = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
            download_config=dl_cfg,
        ).shuffle(buffer_size=self.shuffle_buffer, seed=43)
        self._wildchat_iter = iter(self._wildchat_ds)

    def _next_tokens(self, dataset_idx: int) -> Optional[List[int]]:
        """Pull the next conversation from dataset_idx and tokenize it."""
        if dataset_idx == 0:
            iterator = self._ultrachat_iter
            msg_key = "messages"
        else:
            iterator = self._wildchat_iter
            msg_key = "conversation"

        while True:
            try:
                example = next(iterator)
            except StopIteration:
                # Restart
                if dataset_idx == 0:
                    self._ultrachat_iter = iter(self._ultrachat_ds)
                    iterator = self._ultrachat_iter
                else:
                    self._wildchat_iter = iter(self._wildchat_ds)
                    iterator = self._wildchat_iter
                example = next(iterator)

            messages = example.get(msg_key, [])
            tokens = format_conversation(messages, self.tokenizer)
            if tokens and len(tokens) >= 4:
                return tokens

    def _get_sequence(self, dataset_idx: int) -> List[int]:
        """Pack tokens into a fixed-length sequence."""
        buf = self._buffers[dataset_idx]
        while len(buf) < self.seq_len + 1:
            new_tokens = self._next_tokens(dataset_idx)
            if new_tokens:
                buf.extend(new_tokens)
        seq = buf[: self.seq_len + 1]
        self._buffers[dataset_idx] = buf[self.seq_len :]
        return seq

    def __iter__(self) -> Iterator[Batch]:
        self._load_datasets()
        batch_sequences: List[List[int]] = []

        while True:
            # 90% UltraChat, 10% WildChat
            ds_idx = 0 if random.random() < 0.9 else 1
            seq = self._get_sequence(ds_idx)
            batch_sequences.append(seq)
            # Yield individual samples (batching handled outside)
            yield seq

    def get_batch_iterator(self, batch_size: int) -> Iterator[Batch]:
        batch_seqs: List[List[int]] = []
        for seq in self:
            batch_seqs.append(seq)
            if len(batch_seqs) >= batch_size:
                arr = np.array(batch_seqs, dtype=np.int32)
                input_ids = mx.array(arr[:, :-1])
                labels = mx.array(arr[:, 1:])
                yield Batch(input_ids=input_ids, labels=labels)
                batch_seqs = []


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_loss(model: TernaryTransformer, batch: Batch) -> mx.array:
    """Next-token cross-entropy with gradient checkpointing for M2 16GB."""
    logits = model.forward_with_checkpointing(
        batch.input_ids, checkpoint_every=4, training=True,
    )
    B, S, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = batch.labels.reshape(-1)
    log_probs = nn.log_softmax(logits_flat.astype(mx.float32), axis=-1)
    target_log_probs = mx.take_along_axis(
        log_probs, labels_flat[:, None], axis=-1
    ).squeeze(-1)
    return -mx.mean(target_log_probs)


class LoRAAdamW:
    """Lightweight AdamW that only tracks state for LoRA parameters.

    Keeps float32 moments — total overhead for rank-16 on 24-layer model
    is ~25 MB (negligible on 16 GB).
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[str, mx.array] = {}
        self.v: Dict[str, mx.array] = {}
        self.t = 0

    def step(self, params: Dict[str, mx.array], grads: Dict[str, mx.array],
             lr: float) -> Dict[str, mx.array]:
        self.t += 1
        updated = {}
        for name, param in params.items():
            g = grads[name].astype(mx.float32)
            p = param.astype(mx.float32)

            if name not in self.m:
                self.m[name] = mx.zeros_like(p)
                self.v[name] = mx.zeros_like(p)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g * g)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            update = m_hat / (mx.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                update = update + self.weight_decay * p

            updated[name] = (p - lr * update).astype(param.dtype)
        return updated


# Module-level optimizer instance (created in main)
_optimizer: Optional[LoRAAdamW] = None


def compute_grads(
    model: TernaryTransformer,
    batch: Batch,
) -> Tuple[float, Dict[str, mx.array]]:
    """Compute loss and LoRA gradients for one micro-batch (no optimizer step)."""
    params = get_lora_params(model)

    def loss_fn(lora_params):
        set_lora_params(model, lora_params)
        return compute_loss(model, batch)

    loss, grads = mx.value_and_grad(loss_fn)(params)
    return loss.item(), grads


def optimizer_step(
    model: TernaryTransformer,
    accumulated_grads: Dict[str, mx.array],
    num_accum: int,
    lr: float,
    gradient_clip: float,
) -> float:
    """Average accumulated gradients, clip, and run one AdamW update.

    Returns the pre-clip gradient norm.
    """
    # Average over accumulation steps
    avg_grads = {k: v / num_accum for k, v in accumulated_grads.items()}

    # Gradient clipping on the averaged gradients
    total_norm_sq = mx.array(0.0, dtype=mx.float32)
    for g in avg_grads.values():
        total_norm_sq = total_norm_sq + mx.sum(g.astype(mx.float32) ** 2)
    total_norm = mx.sqrt(total_norm_sq)
    clip_coef = mx.minimum(gradient_clip / (total_norm + 1e-6), mx.array(1.0))
    avg_grads = {k: v * clip_coef.astype(v.dtype) for k, v in avg_grads.items()}

    # AdamW update
    params = get_lora_params(model)
    updated = _optimizer.step(params, avg_grads, lr)
    set_lora_params(model, updated)

    return total_norm.item()


# ---------------------------------------------------------------------------
# Save / load LoRA adapters
# ---------------------------------------------------------------------------

def save_lora(model: TernaryTransformer, path: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    np_params = {}
    for k, v in params.items():
        arr = v
        if arr.dtype == mx.bfloat16:
            arr = arr.astype(mx.float16)
        mx.eval(arr)
        np_params[k] = np.array(arr)
    np.savez_compressed(str(path / "lora_adapters.npz"), **np_params)
    print(f"Saved LoRA adapters to {path}")


def load_lora(model: TernaryTransformer, path: str):
    lora_path = Path(path) / "lora_adapters.npz"
    raw = np.load(str(lora_path))
    params = {k: mx.array(v) for k, v in dict(raw).items()}
    set_lora_params(model, params)
    print(f"Loaded LoRA adapters from {lora_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Ternary Transformer")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/500m/step_120000",
                        help="Path to base checkpoint (TPU-trained)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA alpha scaling (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Micro batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Total optimizer steps (default: 5000)")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="LR warmup steps (default: 200)")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping (default: 1.0)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save every N steps (default: 500)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora_sft",
                        help="Output directory for LoRA adapters")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from saved LoRA adapters directory")
    args = parser.parse_args()

    print("=" * 60)
    print("LoRA Fine-Tuning — Ternary Transformer")
    print("=" * 60)

    # 1. Load base model
    print(f"\nLoading base model from {args.checkpoint}...")
    model, config = load_base_model(args.checkpoint)
    total_params = model.count_parameters().get("total", 0)
    print(f"Base model: {total_params / 1e6:.1f}M params, seq_len={config.max_seq_len}")

    # 2. Apply LoRA adapters
    apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    lora_param_count = count_lora_params(model)
    print(f"LoRA adapters: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Trainable LoRA params: {lora_param_count:,} ({lora_param_count / total_params * 100:.2f}%)")

    # Resume if requested
    if args.resume:
        load_lora(model, args.resume)

    # 3. Initialise LoRA-only AdamW optimizer
    global _optimizer
    _optimizer = LoRAAdamW(weight_decay=0.01)

    # 4. Create data loader
    print("\nCreating data loader (90% UltraChat-200k + 10% WildChat)...")
    dataset = ChatSFTDataset(seq_len=config.max_seq_len)
    data_iter = dataset.get_batch_iterator(args.batch_size)

    # 5. Training loop
    print(f"\nTraining config (optimised for M2 Mac Mini 16 GB):")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Seq len: {config.max_seq_len}")
    print(f"  Grad checkpointing: every 4 layers")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Warmup: {args.warmup_steps}")
    print(f"  Output: {args.output_dir}")
    print("-" * 60)

    step = 0
    accum_loss = 0.0
    accum_count = 0
    accum_grads: Optional[Dict[str, mx.array]] = None
    tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time

    for batch in data_iter:
        if step >= args.max_steps:
            break

        # --- micro-batch: compute grads only, no optimizer step ---
        loss, grads = compute_grads(model, batch)

        # Accumulate gradients
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = {k: accum_grads[k] + grads[k] for k in grads}

        accum_loss += loss
        accum_count += 1
        tokens_processed += batch.batch_size * batch.seq_len

        # Eagerly evaluate accumulated grads to avoid lazy-graph buildup
        mx.eval(mx.array(loss))
        for v in accum_grads.values():
            mx.eval(v)

        # --- optimizer step after grad_accum micro-batches ---
        if accum_count >= args.grad_accum:
            # Cosine schedule with warmup (computed per optimizer step)
            if step < args.warmup_steps:
                lr = args.lr * step / max(1, args.warmup_steps)
            else:
                progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

            grad_norm = optimizer_step(
                model, accum_grads, accum_count, lr, args.gradient_clip,
            )

            # Eagerly evaluate updated params + optimizer state
            for p in get_lora_params(model).values():
                mx.eval(p)
            for arr in _optimizer.m.values():
                mx.eval(arr)
            for arr in _optimizer.v.values():
                mx.eval(arr)
            gc.collect()
            mx.clear_cache()

            step += 1
            avg_loss = accum_loss / accum_count

            if step % args.log_interval == 0:
                now = time.time()
                elapsed = now - last_log_time
                tps = tokens_processed / elapsed if elapsed > 0 else 0
                ppl = math.exp(min(avg_loss, 20))
                mem_active = mx.get_active_memory() / 1e9
                mem_peak = mx.get_peak_memory() / 1e9
                print(
                    f"Step {step:5d} | Loss {avg_loss:.4f} | PPL {ppl:.2f} | "
                    f"LR {lr:.2e} | Grad {grad_norm:.3f} | Tok/s {tps:.0f} | "
                    f"Mem {mem_active:.1f}/{mem_peak:.1f} GB"
                )
                mx.reset_peak_memory()
                last_log_time = now
                tokens_processed = 0

            if step % args.save_interval == 0:
                ckpt = str(Path(args.output_dir) / f"step_{step}")
                save_lora(model, ckpt)

            accum_loss = 0.0
            accum_count = 0
            accum_grads = None

    # Final save
    final_path = str(Path(args.output_dir) / "final")
    save_lora(model, final_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete! {step} steps in {total_time / 3600:.2f} hours")
    print(f"Adapters saved to {args.output_dir}")


if __name__ == "__main__":
    main()
