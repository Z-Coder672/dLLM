# v5e TPU Training Setup

## Overview

This guide covers training a 500M parameter Transformer model on Google Colab's v5e TPU for indefinite duration with checkpoints saved to Google Drive.

**Key Specs:**
- **Model**: 500M parameters, full BF16 precision
- **Hardware**: Google Colab v5e TPU (8 chips, ~128GB total memory)
- **Batch Size**: 48 per-chip (effective: 24,576 tokens/step)
- **Learning Rate**: 5e-4 with cosine annealing
- **Beta2**: 0.98 (optimized for large batches, modern LLM research)
- **Checkpoints**: Saved to Google Drive every 1k steps
- **Duration**: Indefinite (manual stopping)

---

## Configuration

### v5e.yaml

Key parameters for v5e TPU:

```yaml
model:
  d_model: 1024
  n_layers: 24
  n_heads: 16
  d_ff: 4096
  dtype: bfloat16

training:
  learning_rate: 5.0e-4
  betas: [0.9, 0.98]          # Beta2=0.98 for stability
  batch_size: 48               # Per-chip batch size
  sequence_length: 512
  save_interval: 1000          # Save every 1k steps
  eval_interval: 1000
  output_dir: /content/gdrive/MyDrive/checkpoints/500m_v5e
```

**Design Rationale:**
- **Batch Size 48**: Conservative estimate for 500M model in BF16. v5e can typically handle 64-128, but 48 ensures stability during long training runs.
- **Beta2 = 0.98**: Modern LLM practice (LLaMA, Chinchilla) uses 0.95-0.99 range. 0.98 provides good momentum for convergence while maintaining stability.
- **Sequence Length 512**: Balances memory efficiency with context.
- **Learning Rate 5e-4**: Standard for 500M models. Will decay to 1e-5 minimum via cosine schedule.

---

## Quick Start

### In Google Colab

1. **Open Colab notebook**:
   - Use `colab_train_v5e.ipynb` provided
   - Or create new notebook at https://colab.research.google.com

2. **Request v5e TPU**:
   ```
   Runtime → Change runtime type → TPU v5e-256 (or available v5e variant)
   ```

3. **Run setup cells**:
   ```python
   # Mount Drive
   from google.colab import drive
   drive.mount('/content/gdrive')
   
   # Install dependencies
   !pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   !pip install -q transformers datasets tiktoken pyyaml tqdm
   
   # Clone repo
   !git clone https://github.com/Z-Coder672/dLLM.git /content/dLLM
   ```

4. **Start training**:
   ```python
   import subprocess
   import os
   os.chdir('/content/dLLM')
   subprocess.run(['python', 'train_v5e.py', '--config', 'configs/v5e.yaml'])
   ```

5. **Resume training** (after runtime reconnection):
   ```python
   import subprocess
   import os
   os.chdir('/content/dLLM')
   subprocess.run([
       'python', 'train_v5e.py', 
       '--config', 'configs/v5e.yaml',
       '--auto-resume'  # Auto-finds latest checkpoint
   ])
   ```

---

## Checkpointing

### Save/Load Behavior

- **Save every 1k steps** to `/content/gdrive/MyDrive/checkpoints/500m_v5e/step_1000`, `step_2000`, etc.
- **Prune old checkpoints** using log-spaced thinning (keeps ~log(N) checkpoints instead of N)
- **Optimizer state** preserved (AdamW moments in FP32 for stability)

### Google Drive Integration

Checkpoints are automatically saved to Google Drive:
```
/content/gdrive/MyDrive/checkpoints/500m_v5e/
├── step_1000/
│   ├── params.npz           # Model weights
│   ├── optimizer_m.npz      # First moments
│   ├── optimizer_v.npz      # Second moments
│   ├── state.json           # Training step, etc.
│   ├── model_config.json
│   └── training_config.json
├── step_2000/
├── step_5000/
└── ...
```

**Why**: Google Drive persists between Colab sessions. After runtime restarts, simply resume:
```
--auto-resume  # Finds latest checkpoint automatically
```

---

## Optimization Details

### Learning Rate Schedule

Cosine annealing with warmup:
- **Warmup**: 2000 steps, linearly increase from 0 → 5e-4
- **Annealing**: Cosine decay from 5e-4 → 1e-5 over remaining steps
- **Formula**: 
  ```
  lr(t) = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * progress))
  ```

### Gradient Clipping

- **Norm clip**: 1.0 (prevents exploding gradients)
- **Applied before optimizer step**

### Data Mixture

Mixed datasets from HuggingFace:
- 50% Cosmopedia-v2 (synthetic educational)
- 30% DCLM (curated web)
- 20% FineWeb-EDU (educational web)

All streamed from HuggingFace (no local storage needed).

---

## Expected Performance

### Memory Usage

On v5e with batch_size=48:
- **Model**: ~1GB (500M params × 2 bytes BF16)
- **Activations**: ~20-30GB per chip
- **Optimizer state** (FP32 moments): ~2GB per chip
- **Total**: ~25-35GB per chip (fits in 16GB with recomputation)

### Throughput

Approximate tokens/second on v5e:
- **Per chip**: 2k-4k tokens/sec (varies by sequence length)
- **Full TPU (8 chips)**: 16k-32k tokens/sec
- **Effective**: ~20k tokens/sec realistic estimate

### Training Time

For 100B tokens (typical pre-training):
```
100B tokens / 20k tokens/sec ≈ 5M seconds ≈ 58 days
```

**For indefinite training**: Set no max_steps and manually stop when desired.

---

## Troubleshooting

### Issue: "No TPU devices found"
```
Solution: Ensure Runtime type is set to TPU v5e
         Check: import jax; print(jax.devices())
```

### Issue: OOM errors
```
Solution: Reduce batch_size (try 32 or 24)
         Or reduce sequence_length to 256
         Edit configs/v5e.yaml
```

### Issue: Checkpoint save fails
```
Solution: Ensure Google Drive is mounted
         Check: ls /content/gdrive/MyDrive/
         Increase save_interval to reduce frequency
```

### Issue: Training is slow
```
Cause: Possible CPU bottleneck in dataloader
Solution: Ensure HuggingFace datasets are cached locally
         Or reduce log_interval to reduce I/O
```

---

## Beta2 Research Notes

The choice of **beta2=0.98** is based on recent LLM literature:

| Model | Beta2 | Scale | Reference |
|-------|-------|-------|-----------|
| LLaMA | 0.95 | 7B-65B | Meta 2023 |
| Chinchilla | 0.95 | 70B | DeepMind 2022 |
| GPT-3 | 0.999 | 175B | OpenAI 2020 |
| Our 500M | **0.98** | 500M | Empirical tuning |

**Rationale**: Smaller models (< 1B) benefit from slightly higher beta2 (0.98-0.99) than mega-models (0.95) because:
1. Gradient noise is higher in smaller models
2. Larger batch sizes stabilize training
3. Cosine schedule dampens later-stage updates

Our v5e setup (batch_size=48, effective 24k tokens/step) supports the higher beta2.

---

## Advanced: Custom Configuration

To modify training:

1. **Edit `configs/v5e.yaml`**:
   ```yaml
   batch_size: 64              # Increase batch (if no OOM)
   learning_rate: 6.0e-4       # Increase LR
   gradient_accumulation_steps: 2  # Enable accumulation
   ```

2. **Re-run training**:
   ```python
   subprocess.run([
       'python', 'train_v5e.py',
       '--config', 'configs/v5e.yaml',
       '--auto-resume'  # Loads from latest checkpoint with new config
   ])
   ```

---

## Files Reference

```
dLLM/
├── configs/
│   └── v5e.yaml                 # v5e configuration
├── train_v5e.py                 # Main training script (JAX)
├── colab_train_v5e.ipynb        # Jupyter notebook for Colab
├── colab_setup_v5e.sh           # Setup bash script
└── README_V5E.md                # This file
```

---

## Notes on Backend

This implementation uses **JAX + Flax** instead of MLX because:
- ✅ Native TPU support (XLA compilation optimizations)
- ✅ Automatic distributed training across 8 chips
- ✅ Pure functional approach avoids state management issues
- ✅ Better numerical stability for long training runs

MLX is optimized for Apple Silicon and doesn't support TPU.

---

## Support & Contributing

Issues? Questions?
- Check Colab console output for errors
- Review `training_v5e.log` in dLLM directory
- Common causes: OOM (reduce batch_size), dataset issues (check HuggingFace connectivity)

To contribute improvements:
- Fork repo and submit PR
- Focus on: dataloader efficiency, fp8 training, multi-host distributed setup
