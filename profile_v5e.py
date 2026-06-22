#!/usr/bin/env python3
"""Micro-profiler for the open questions in the final train_v5e audit.

TPU/Colab-ONLY (needs jax[tpu] + the same model code). Like
test_dataloader_resume.py it can't run on the dev box — importing
train_v5e_complete `sys.exit`s if JAX isn't installed. Run it on the v5e BEFORE
a long run to (a) validate the wall-clock estimate and batch headroom, and
(b) settle the three "is this actually free?" questions the audit flagged.

It profiles the REAL model code (imported from train_v5e_complete), reconstructing
only the `train_step` closure variants it needs to A/B (the real one lives inside
main()). Each reconstructed step mirrors main()'s logic exactly.

Usage:
    python3 profile_v5e.py                  # all tests, real dims, batch 16
    python3 profile_v5e.py --batch 8        # rerun for a clean per-batch peak HBM
    python3 profile_v5e.py --batch 32       # check headroom before bumping the config
    python3 profile_v5e.py --layers 4 --steps 20   # quick smoke (smaller graph)

Open questions tested (see AUDIT):
  Q1  Per-step non-finite `jnp.where` guard over the whole params/m/v tree —
      free (XLA fuses it into the AdamW update) or a real per-step tax?
  Q2  `accum_steps == 1` length-1 `lax.scan` — does the zero-grad alloc + scan
      cost anything vs a direct single-microbatch step?
  Q3  Naive materialized attention vs `jax.nn.dot_product_attention` (fused) —
      tok/s and peak-memory difference at this seq/batch.
  +   Real fused train_step tok/s + peak HBM at the chosen batch (validates the
      time estimate and shows how much room is left to grow the batch).
"""

import argparse
import math
import time

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, lax
from functools import partial

# Profile the real code, not a reimplementation.
from train_v5e_complete import (
    TransformerModel,
    AdamWOptimizer,
    build_decay_mask,
    clip_gradients,
    compute_loss,
    rope_tables,
    apply_rope,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _block(x):
    """Block until all arrays in a pytree are materialized on device."""
    jax.block_until_ready(x)
    return x


def time_fn(fn, n_iter, warmup=2):
    """Return mean seconds/call for `fn` (excludes `warmup` compile/warm runs)."""
    for _ in range(warmup):
        _block(fn())
    t0 = time.time()
    for _ in range(n_iter):
        _block(fn())
    return (time.time() - t0) / n_iter


def mem_stats():
    """(_bytes_in_use, peak_bytes_in_use) on device 0, in GB, or (None, None)."""
    try:
        s = jax.devices()[0].memory_stats()
        return (s.get("bytes_in_use", 0) / 1e9,
                s.get("peak_bytes_in_use", 0) / 1e9)
    except Exception:
        return (None, None)


def report_mem(label):
    cur, peak = mem_stats()
    if cur is None:
        print(f"  [{label}] memory_stats() unavailable on this backend")
    else:
        print(f"  [{label}] HBM in use: {cur:.2f} GB | peak: {peak:.2f} GB")


# ----------------------------------------------------------------------------
# model / state setup (mirrors main())
# ----------------------------------------------------------------------------
def build(args):
    model_config = dict(
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        d_ff=args.d_ff, vocab_size=args.vocab, max_seq_len=args.seq,
        rope_theta=10000.0, tie_embeddings=args.tie,
    )
    model = TransformerModel(model_config, jax.random.PRNGKey(0))
    model.ce_chunk_size = args.ce_chunk
    model.remat_blocks = True
    # f32 master, exactly like main().
    model.params = jax.tree_util.tree_map(lambda p: p.astype(jnp.float32), model.params)
    opt = AdamWOptimizer(beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.01)
    opt.init_state(model.params)
    decay_mask = build_decay_mask(model.params, decay_embeddings=True)
    n = sum(int(x.size) for x in jax.tree_util.tree_leaves(model.params))
    print(f"Params: {n/1e6:.1f}M | tie_embeddings={args.tie} | "
          f"layers={args.layers} d_model={args.d_model} | "
          f"batch={args.batch} seq={args.seq} accum={args.accum} "
          f"ce_chunk={args.ce_chunk}")
    return model, opt, decay_mask


def make_batch(args, accum):
    key = jax.random.PRNGKey(1)
    ids = jax.random.randint(key, (accum, args.batch, args.seq), 0, args.vocab)
    return _block({"input_ids": ids.astype(jnp.int32),
                   "labels": ids.astype(jnp.int32)})


def loss_fn_for(model):
    def _loss_fn(params, batch):
        params_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16), params)
        return compute_loss(params_bf16, batch, model)
    return _loss_fn


# ----------------------------------------------------------------------------
# train_step variants (mirror main()'s fused step)
# ----------------------------------------------------------------------------
def build_steps(model, opt, decay_mask, grad_clip=1.0):
    _loss_fn = loss_fn_for(model)
    b1, b2, eps, wd = opt.beta1, opt.beta2, opt.eps, opt.weight_decay

    def _scan_grads(params, batch):
        def micro(grad_acc, mb):
            loss_i, grads_i = value_and_grad(_loss_fn)(params, mb)
            grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad_acc, grads_i)
            return grad_acc, loss_i
        zero = jax.tree_util.tree_map(jnp.zeros_like, params)
        grad_acc, losses = lax.scan(micro, zero, batch)
        n = losses.shape[0]
        grads = jax.tree_util.tree_map(lambda g: g / n, grad_acc)
        return grads, jnp.mean(losses)

    def _apply(params, m, v, t, lr, grads):
        grads, gnorm = clip_gradients(grads, grad_clip)
        t_new = t + 1
        up, um, uv = AdamWOptimizer.apply(
            params, grads, m, v, t_new, lr, b1, b2, eps, wd, decay_mask)
        return up, um, uv, t_new, gnorm

    # --- REAL step: scan grad-accum + full non-finite where-guard (== main) ---
    @partial(jit, donate_argnums=(0, 1, 2))
    def step_guard(params, m, v, t, lr, batch):
        grads, loss = _scan_grads(params, batch)
        up, um, uv, t_new, gnorm = _apply(params, m, v, t, lr, grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        keep = lambda a, b: jax.tree_util.tree_map(
            lambda x, y: jnp.where(finite, x, y), a, b)
        return (keep(up, params), keep(um, m), keep(uv, v),
                jnp.where(finite, t_new, t), loss, gnorm)

    # --- Q1: SAME but no where-guard (commit unconditionally) ---
    @partial(jit, donate_argnums=(0, 1, 2))
    def step_noguard(params, m, v, t, lr, batch):
        grads, loss = _scan_grads(params, batch)
        up, um, uv, t_new, gnorm = _apply(params, m, v, t, lr, grads)
        return up, um, uv, t_new, loss, gnorm

    # --- Q2: guard step but NO scan (direct single micro-batch, accum==1) ---
    @partial(jit, donate_argnums=(0, 1, 2))
    def step_noscan(params, m, v, t, lr, mb):
        loss, grads = value_and_grad(_loss_fn)(params, mb)
        up, um, uv, t_new, gnorm = _apply(params, m, v, t, lr, grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        keep = lambda a, b: jax.tree_util.tree_map(
            lambda x, y: jnp.where(finite, x, y), a, b)
        return (keep(up, params), keep(um, m), keep(uv, v),
                jnp.where(finite, t_new, t), loss, gnorm)

    return step_guard, step_noguard, step_noscan


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def run_throughput_and_q1_q2(args):
    print("\n=== throughput + Q1 (where-guard) + Q2 (length-1 scan) ===")
    model, opt, decay_mask = build(args)
    report_mem("after init + optimizer state")
    step_guard, step_noguard, step_noscan = build_steps(model, opt, decay_mask)
    batch = make_batch(args, args.accum)
    lr = 7e-4
    tok_per_step = args.accum * args.batch * args.seq

    # train_step donates params/m/v, so every timed call deletes its input
    # buffers. Keep a pristine HOST copy and reload fresh device buffers before
    # each variant so we never feed a deleted buffer (and each variant starts
    # from identical state).
    host_p = jax.device_get(model.params)
    host_m = jax.device_get(opt.m)
    host_v = jax.device_get(opt.v)

    def fresh():
        return (jax.tree_util.tree_map(jnp.asarray, host_p),
                jax.tree_util.tree_map(jnp.asarray, host_m),
                jax.tree_util.tree_map(jnp.asarray, host_v), 0)

    def time_step(step_fn, b, label):
        # First call triggers the (heavy) XLA compile: value_and_grad over a
        # lax.scan of 24 rematerialized blocks + chunked CE. This can take
        # SEVERAL MINUTES on the full graph with no output — report it so a long
        # compile isn't mistaken for a hang.
        print(f"  compiling [{label}] (first fused-step build — can take "
              f"minutes on the full 24-layer graph)...", flush=True)
        p, m, v, t = fresh()
        tc = time.time()
        _block(step_fn(p, m, v, t, lr, b))
        print(f"  [{label}] compiled in {time.time()-tc:.1f}s; timing "
              f"{args.steps} steps...", flush=True)
        # One more warm run, then time on fresh state (donated buffers are spent).
        p, m, v, t = fresh()
        _block(step_fn(p, m, v, t, lr, b))

        state = {"s": fresh()}

        def once():
            p, m, v, t = state["s"]
            p, m, v, t, loss, gn = step_fn(p, m, v, t, lr, b)
            state["s"] = (p, m, v, t)
            return loss
        t0 = time.time()
        for _ in range(args.steps):
            _block(once())
        return (time.time() - t0) / args.steps

    # Real fused step (== main): tok/s + peak HBM.
    guard_sec = time_step(step_guard, batch, "scan+guard")
    report_mem(f"after {args.steps} guarded steps")
    print(f"  REAL step (scan+guard): {guard_sec*1e3:.1f} ms/step | "
          f"{tok_per_step/guard_sec:,.0f} tok/s")

    # Q1: same minus the where-guard.
    sec_ng = time_step(step_noguard, batch, "no-guard")
    print(f"  Q1 no-guard step:       {sec_ng*1e3:.1f} ms/step | "
          f"{tok_per_step/sec_ng:,.0f} tok/s")
    dq1 = (guard_sec - sec_ng) / guard_sec * 100
    print(f"  -> Q1 where-guard cost: {(guard_sec-sec_ng)*1e3:+.2f} ms/step "
          f"({dq1:+.1f}%). <~1% => XLA fuses it (free); larger => use lax.cond.")

    # Q2: only meaningful at accum==1 (scan length 1 vs no scan).
    if args.accum == 1:
        single = {"input_ids": batch["input_ids"][0], "labels": batch["labels"][0]}
        sec_ns = time_step(step_noscan, single, "no-scan")
        print(f"  Q2 no-scan step:        {sec_ns*1e3:.1f} ms/step | "
              f"{tok_per_step/sec_ns:,.0f} tok/s")
        dq2 = (guard_sec - sec_ns) / guard_sec * 100
        print(f"  -> Q2 length-1 scan cost: {(guard_sec-sec_ns)*1e3:+.2f} ms/step "
              f"({dq2:+.1f}%). <~1% => the scan is free; larger => add an "
              f"accum==1 fast path.")
    else:
        print("  Q2 skipped (run with --accum 1 to isolate the length-1 scan).")


def run_q3_attention(args):
    print("\n=== Q3 (naive materialized attention vs fused dot_product_attention) ===")
    B, H, S, D = args.batch, args.heads, args.seq, args.d_model // args.heads
    key = jax.random.PRNGKey(2)
    # Naive path uses the model's (B, H, S, D) layout.
    q = jax.random.normal(key, (B, H, S, D), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (B, H, S, D), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (B, H, S, D), dtype=jnp.bfloat16)
    cos, sin = rope_tables(S, D, 10000.0, jnp.bfloat16)
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))

    @jit
    def naive(q, k, v):
        qr, kr = apply_rope(q, k, cos, sin)
        scores = jnp.einsum("bhqd,bhkd->bhqk", qr, kr,
                            preferred_element_type=jnp.float32) / math.sqrt(D)
        scores = jnp.where(mask[None, None], scores, -1e9)
        w = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        return jnp.matmul(w, v)

    n_attn = B * H * S * S
    sec_naive = time_fn(lambda: naive(q, k, v), 50)
    print(f"  naive attention:  {sec_naive*1e3:.3f} ms  (materializes "
          f"{n_attn*4/1e6:.0f} MB f32 scores/layer)")

    dpa = getattr(jax.nn, "dot_product_attention", None)
    if dpa is None:
        print("  jax.nn.dot_product_attention not available in this JAX version "
              "— upgrade JAX to test the fused path.")
        return
    # dot_product_attention wants (B, S, H, D); RoPE applied first in NeoX layout.
    qr, kr = apply_rope(q, k, cos, sin)
    qd = qr.transpose(0, 2, 1, 3)
    kd = kr.transpose(0, 2, 1, 3)
    vd = v.transpose(0, 2, 1, 3)

    @jit
    def fused(qd, kd, vd):
        return dpa(qd, kd, vd, is_causal=True)

    try:
        sec_fused = time_fn(lambda: fused(qd, kd, vd), 50)
        spd = sec_naive / sec_fused
        print(f"  fused dot_product_attention: {sec_fused*1e3:.3f} ms "
              f"({spd:.2f}x vs naive)")
        print("  -> >1.3x and/or much lower peak => worth switching the block to "
              "the fused kernel before pushing the batch higher.")
    except Exception as e:
        print(f"  fused path errored ({e}); the naive path is the safe default at "
              f"seq={S}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--ce_chunk", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--tie", action="store_true", default=True)
    ap.add_argument("--no-tie", dest="tie", action="store_false")
    ap.add_argument("--skip-attn", action="store_true",
                    help="skip the Q3 attention micro-benchmark")
    args = ap.parse_args()

    print(f"JAX {jax.__version__} | backend={jax.default_backend()} | "
          f"devices={jax.device_count()}")
    print("Note: peak HBM is a per-process high-watermark — rerun with a single "
          "--batch for a clean peak at that batch size.\n")

    run_throughput_and_q1_q2(args)
    if not args.skip_attn:
        run_q3_attention(args)

    print("\nWall-clock check: multiply the REAL step tok/s above into your token "
          "budget.\n  e.g. 8.19B tokens (1M steps @ batch16) / tok_s = seconds; /3600 = hours.")


if __name__ == "__main__":
    main()
