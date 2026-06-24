#!/usr/bin/env python3
"""Run a fixed set of completion prompts on a v5e checkpoint (loads the model once).

Base-LM sanity harness: same prompts + seeds every time so you can compare
checkpoints apples-to-apples (e.g. watch facts consolidate as the LR anneals).
bf16 weights (~0.7 GB). Not chat — these are completion prefixes.

Usage:
    python3 eval_prompts.py checkpoints/500m_scan/step_580000
"""
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp

# Importing from the repo works because this script lives in the repo dir.
import train_v5e_complete as T

CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/500m_scan/step_580000"
L = 96  # fixed buffer -> one forward compile; causal mask hides post-cursor padding

mcfg = json.load(open(f"{CK}/model_config.json"))
m = T.TransformerModel(mcfg, jax.random.PRNGKey(0))
m.remat_blocks = False
m.params = T.unflatten_dict(dict(np.load(f"{CK}/params.npz")), dtype=jnp.bfloat16)
tok = T._get_tokenizer()
eot = tok.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


@jax.jit
def logits_at(buf, pos):
    return m.forward(buf, m.params, training=False)[0, pos]


def gen(prompt, temp=0.8, topk=40, max_new=40, seed=0):
    ids = tok.encode_ordinary(prompt)[: L - max_new]
    buf = np.zeros((1, L), np.int32)
    buf[0, : len(ids)] = ids
    pos = len(ids) - 1
    start = pos
    key = jax.random.PRNGKey(seed)
    while pos < L - 1 and pos - start < max_new:
        lg = logits_at(jnp.asarray(buf), jnp.int32(pos))
        if temp <= 0:
            nxt = int(jnp.argmax(lg))
        else:
            lg = lg / temp
            kth = jnp.sort(lg)[-topk]
            lg = jnp.where(lg < kth, -1e9, lg)
            key, s = jax.random.split(key)
            nxt = int(jax.random.categorical(s, lg))
        if nxt == eot:
            break
        pos += 1
        buf[0, pos] = nxt
    return tok.decode(buf[0, start + 1 : pos + 1].tolist())


PROMPTS = [
    ("The Roman Empire was", 0.8),
    ("Photosynthesis is the process by which", 0.8),
    ("The capital of France is", 0.0),
    ("Water boils at a temperature of", 0.0),
    ("def fibonacci(n):", 0.7),
    ("Once upon a time, there was", 0.9),
]

print(f"CHECKPOINT: {CK}\n" + "=" * 70)
for i, (p, t) in enumerate(PROMPTS):
    print(f"\n[temp {t}] {p!r}\n  -> {p}{gen(p, temp=t, seed=i)}")
print("\n" + "=" * 70 + "\nDONE")
