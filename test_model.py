#!/usr/bin/env python3
"""
Test script to verify model components work correctly.

Run this before training to catch any issues early.

Usage:
    python test_model.py
"""

import mlx.core as mx
import numpy as np
import time

print("=" * 60)
print("Ternary Transformer - Component Tests")
print("=" * 60)

def test_ternarization():
    """Test ternarization with STE."""
    print("\n[2] Testing Ternarization with STE...")
    
    from src.ste import ternarize, compute_ternary_stats
    
    # Create test tensor
    w = mx.random.normal(shape=(64, 128), dtype=mx.bfloat16)
    
    # Ternarize
    w_ternary = ternarize(w, threshold_factor=0.7)
    
    # Check values are only {-1, 0, 1}
    unique_vals = set(w_ternary.reshape(-1).tolist())
    expected = {-1.0, 0.0, 1.0}
    assert unique_vals.issubset(expected), f"Unexpected values: {unique_vals - expected}"
    
    # Get stats
    stats = compute_ternary_stats(w, threshold_factor=0.7)
    print(f"  Distribution: +1={stats['pct_positive']:.1f}%, 0={stats['pct_zero']:.1f}%, -1={stats['pct_negative']:.1f}%")
    
    # Test gradient flow
    def loss_fn(x):
        return mx.sum(ternarize(x, 0.7))
    
    grad = mx.grad(loss_fn)(w)
    assert grad.shape == w.shape, "Gradient shape mismatch"
    print("  ✓ STE gradient flow working")
    print("  ✓ Ternarization test passed")


def test_ternary_linear():
    """Test TernaryLinear layer."""
    print("\n[3] Testing TernaryLinear Layer...")
    
    from src.layers import TernaryLinear
    
    layer = TernaryLinear(256, 128, threshold_factor=0.7)
    
    # Test forward pass
    x = mx.random.normal(shape=(2, 16, 256), dtype=mx.bfloat16)
    y = layer(x)
    
    assert y.shape == (2, 16, 128), f"Wrong output shape: {y.shape}"
    print(f"  Input: {x.shape} -> Output: {y.shape}")
    
    # Test gradient
    def loss_fn(x_in):
        return mx.sum(layer(x_in))
    
    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape, "Input gradient shape mismatch"
    print("  ✓ Forward and backward pass working")
    print("  ✓ TernaryLinear test passed")


def test_attention():
    """Test multi-head attention."""
    print("\n[4] Testing Multi-Head Attention...")
    
    from src.attention import MultiHeadAttention, create_causal_mask
    
    attn = MultiHeadAttention(
        d_model=256,
        n_heads=8,
        threshold_factor=0.7,
        max_seq_len=128,
    )
    
    # Test forward pass
    x = mx.random.normal(shape=(2, 32, 256), dtype=mx.bfloat16)
    mask = create_causal_mask(32, mx.bfloat16)
    
    out, cache = attn(x, mask=mask)
    
    assert out.shape == x.shape, f"Wrong output shape: {out.shape}"
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test with cache
    x_new = mx.random.normal(shape=(2, 1, 256), dtype=mx.bfloat16)
    out_new, new_cache = attn(x_new, cache=cache)
    
    assert out_new.shape == (2, 1, 256), f"Wrong cached output shape: {out_new.shape}"
    print("  ✓ KV cache working")
    print("  ✓ Attention test passed")


def test_transformer_block():
    """Test transformer block."""
    print("\n[5] Testing Transformer Block...")
    
    from src.config import ModelConfig
    from src.model import TransformerBlock
    
    config = ModelConfig(
        d_model=256,
        n_layers=2,
        n_heads=8,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=128,
    )
    
    block = TransformerBlock(config, layer_idx=0)
    
    x = mx.random.normal(shape=(2, 32, 256), dtype=mx.bfloat16)
    from src.attention import create_causal_mask
    mask = create_causal_mask(32, mx.bfloat16)
    
    out, cache = block(x, mask=mask)
    
    assert out.shape == x.shape, f"Wrong output shape: {out.shape}"
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print("  ✓ Transformer block test passed")


def test_full_model():
    """Test full transformer model."""
    print("\n[6] Testing Full Transformer Model...")
    
    from src.config import ModelConfig
    from src.model import create_model
    
    # Small test config
    config = ModelConfig(
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=128,
    )
    
    model = create_model(config)
    
    # Count parameters
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    
    # Test forward pass
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits, cache = model(input_ids)
    
    assert logits.shape == (1, 8, 1000), f"Wrong logits shape: {logits.shape}"
    print(f"  Input: {input_ids.shape} -> Logits: {logits.shape}")
    
    # Test generation
    generated = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    assert generated.shape[1] == 18, f"Wrong generated length: {generated.shape[1]}"
    print(f"  Generated {generated.shape[1] - 8} new tokens")
    
    # Test gradient
    def loss_fn(ids):
        logits, _ = model(ids)
        return mx.mean(logits)
    
    # This tests that gradients flow through the model
    grad = mx.grad(loss_fn)(input_ids.astype(mx.float32))
    print("  ✓ Gradient computation working")
    print("  ✓ Full model test passed")


def test_optimizer():
    """Test AdamW optimizer."""
    print("\n[7] Testing AdamW Optimizer...")
    
    from src.optimizer import AdamW, get_cosine_schedule_with_warmup
    
    optimizer = AdamW(learning_rate=1e-3, weight_decay=0.1)
    
    # Test parameter update
    params = {
        "layer1.weight": mx.random.normal(shape=(64, 128), dtype=mx.bfloat16),
        "layer1.bias": mx.random.normal(shape=(64,), dtype=mx.bfloat16),
    }
    
    grads = {
        "layer1.weight": mx.random.normal(shape=(64, 128), dtype=mx.bfloat16),
        "layer1.bias": mx.random.normal(shape=(64,), dtype=mx.bfloat16),
    }
    
    updated = optimizer.step(params, grads)
    
    assert "layer1.weight" in updated, "Weight not updated"
    assert updated["layer1.weight"].shape == params["layer1.weight"].shape
    
    # Test learning rate schedule
    lr_start = get_cosine_schedule_with_warmup(0, 100, 1000, 1e-5, 1e-3)
    lr_warmup = get_cosine_schedule_with_warmup(50, 100, 1000, 1e-5, 1e-3)
    lr_peak = get_cosine_schedule_with_warmup(100, 100, 1000, 1e-5, 1e-3)
    lr_end = get_cosine_schedule_with_warmup(1000, 100, 1000, 1e-5, 1e-3)
    
    print(f"  LR schedule: start={lr_start:.2e}, warmup={lr_warmup:.2e}, peak={lr_peak:.2e}, end={lr_end:.2e}")
    
    assert lr_start < lr_warmup < lr_peak, "Warmup not working"
    assert lr_end < lr_peak, "Decay not working"
    
    print("  ✓ Optimizer test passed")


def test_memory_usage():
    """Test memory usage estimation."""
    print("\n[8] Estimating Memory Usage...")
    
    from src.config import CONFIG_500M
    from src.model import create_model
    
    # Create 500M config
    model = create_model(CONFIG_500M)
    params = model.count_parameters()
    
    # Estimate memory
    int8_weights_mb = params['total'] / 1e6  # 1 byte per param
    bf16_weights_mb = params['total'] * 2 / 1e6
    
    print(f"  500M model parameter count: {params['total']:,}")
    print(f"  INT8 weight storage: {int8_weights_mb:.1f} MB")
    print(f"  BF16 weight storage: {bf16_weights_mb:.1f} MB")
    print(f"  Memory savings: {(1 - int8_weights_mb/bf16_weights_mb)*100:.0f}%")
    
    print("  ✓ Memory estimation complete")


def test_throughput():
    """Test forward pass throughput."""
    print("\n[9] Testing Throughput...")
    
    from src.config import ModelConfig
    from src.model import create_model
    
    # Medium test config
    config = ModelConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        vocab_size=1000,
        max_seq_len=512,
    )
    
    model = create_model(config)
    
    # Warmup
    input_ids = mx.array([[1] * 128])
    for _ in range(3):
        logits, _ = model(input_ids)
        mx.eval(logits)
    
    # Benchmark
    batch_size = 4
    seq_len = 256
    input_ids = mx.random.randint(0, 1000, shape=(batch_size, seq_len))
    
    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        logits, _ = model(input_ids)
        mx.eval(logits)
    elapsed = time.time() - start
    
    tokens_per_run = batch_size * seq_len
    total_tokens = tokens_per_run * n_runs
    tokens_per_sec = total_tokens / elapsed
    
    print(f"  Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"  Forward passes: {n_runs}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print("  ✓ Throughput test complete")


def main():
    tests = [
        ("Quantization", test_quantization),
        ("Ternarization", test_ternarization),
        ("TernaryLinear", test_ternary_linear),
        ("Attention", test_attention),
        ("Transformer Block", test_transformer_block),
        ("Full Model", test_full_model),
        ("Optimizer", test_optimizer),
        ("Memory Usage", test_memory_usage),
        ("Throughput", test_throughput),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()

