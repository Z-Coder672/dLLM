#!/usr/bin/env python3
"""Debug generation - check raw model outputs"""
import mlx.core as mx
from src.model import create_model
from src.checkpoint import load_checkpoint
from data.dataloader import get_tokenizer

# Load model
state = load_checkpoint("checkpoints/500m/final")
model = create_model(state["config"])
load_checkpoint("checkpoints/500m/final", model=model)

tokenizer = get_tokenizer()

# Simple test
prompt = "The quick brown fox"
input_ids = mx.array([tokenizer.encode(prompt)])

# Get raw logits
logits, _ = model(input_ids, training=False)
next_token_logits = logits[0, -1, :]  # Last position

# Top 10 predictions
top_10_indices = mx.argsort(next_token_logits)[-10:].tolist()
top_10_tokens = [tokenizer.decode([i]) for i in top_10_indices]

print(f"Prompt: {prompt}")
print(f"Top 10 next tokens: {top_10_tokens}")

# Greedy decode (no sampling)
for _ in range(20):
    logits, _ = model(input_ids, training=False)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    input_ids = mx.concatenate([input_ids, next_token], axis=1)
    
greedy_output = tokenizer.decode(input_ids[0].tolist())
print(f"Greedy output: {greedy_output}")