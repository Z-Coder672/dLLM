#!/usr/bin/env python3
"""Autoregressive sampler for a v5e JAX checkpoint (base-LM text completion).

This is NOT chat.py. chat.py is the MLX stack for a different model and cannot
read the JAX .npz checkpoints train_v5e_complete.py writes. This loads such a
checkpoint, rebuilds the exact architecture from its saved model_config.json
(incl. scan_layers / tie_embeddings), and samples continuations with the tiktoken
GPT-2 tokenizer the trainer used.

It is a BASE model (pretraining only, no instruction tuning): it CONTINUES text,
it does not answer chat prompts. Feed it a prefix.

Runs anywhere JAX is installed; CPU is fine for short prompts (no KV cache — each
token re-runs a full forward over a fixed-length buffer, so it's simple, not fast).

Usage:
    python3 generate_v5e.py --checkpoint /path/to/step_70000 \
        --prompt "The history of the Roman Empire" \
        --max-new-tokens 80 --temperature 0.8 --top-k 40
    # greedy / most-likely continuation:
    python3 generate_v5e.py --checkpoint .../step_70000 --prompt "def quicksort(arr):" --temperature 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Reuse the exact model + loaders from the trainer (import is CPU-safe).
from train_v5e_complete import TransformerModel, unflatten_dict, _get_tokenizer


def load_model(ckpt_dir):
    ckpt = Path(ckpt_dir)
    with open(ckpt / "model_config.json") as f:
        mcfg = json.load(f)
    # The ctor reads scan_layers / tie_embeddings / dims from mcfg, so the rebuilt
    # tree matches the checkpoint; then overwrite the fresh params with the saved.
    model = TransformerModel(mcfg, jax.random.PRNGKey(0))
    model.remat_blocks = False  # inference: no backward pass to rematerialize
    params_flat = dict(np.load(ckpt / "params.npz"))
    model.params = unflatten_dict(params_flat, dtype=jnp.float32)
    return model, mcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint DIR (e.g. .../step_70000)")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/argmax")
    ap.add_argument("--top-k", type=int, default=40, help="0 = no top-k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, mcfg = load_model(args.checkpoint)
    max_seq = int(mcfg.get("max_seq_len", 512))
    tok = _get_tokenizer()
    eot = tok.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    ids = tok.encode_ordinary(args.prompt)
    if not ids:
        ids = [eot]
    # Fixed-length buffer so the forward jit compiles ONCE. Padding sits AFTER the
    # cursor; the causal mask means logits at the cursor never see padded positions.
    L = min(max_seq, len(ids) + args.max_new_tokens)
    ids = ids[-L:]

    @jax.jit
    def logits_at(buf, pos):
        lg = model.forward(buf, model.params, training=False)  # (1, L, vocab) f32
        return lg[0, pos]

    buf = np.zeros((1, L), dtype=np.int32)
    buf[0, : len(ids)] = ids
    pos = len(ids) - 1  # cursor = index of the last real token
    key = jax.random.PRNGKey(args.seed)

    print(args.prompt, end="", flush=True)
    while pos < L - 1 and (pos - (len(ids) - 1)) < args.max_new_tokens:
        logits = logits_at(jnp.asarray(buf), jnp.int32(pos))
        if args.temperature <= 0:
            nxt = int(jnp.argmax(logits))
        else:
            logits = logits / args.temperature
            if args.top_k > 0 and args.top_k < logits.shape[-1]:
                kth = jnp.sort(logits)[-args.top_k]
                logits = jnp.where(logits < kth, -1e9, logits)
            key, sub = jax.random.split(key)
            nxt = int(jax.random.categorical(sub, logits))
        if nxt == eot:
            break
        pos += 1
        buf[0, pos] = nxt
        print(tok.decode([nxt]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
