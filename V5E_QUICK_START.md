# v5e TPU Training - Quick Start Guide

## TL;DR

1. **Open Google Colab**: https://colab.research.google.com
2. **Request v5e TPU**: Runtime → Change runtime type → TPU v5e
3. **Paste this into first cell**:
```python
from google.colab import drive
drive.mount('/content/gdrive')
!pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install -q transformers datasets tiktoken pyyaml tqdm
!git clone https://github.com/Z-Coder672/dLLM.git /content/dLLM
%cd /content/dLLM
!python train_v5e_complete.py --config configs/v5e.yaml
```

4. **Let it train indefinitely** (will checkpoint every 1k steps to Google Drive)

---

## What's Included

### Config File
- **`configs/v5e.yaml`**: Optimized parameters for v5e TPU
  - Batch size: 48 (optimal for 500M model)
  - LR: 5e-4 (standard for 500M)
  - Beta2: 0.98 (modern LLM research)
  - Save every 1k steps to Google Drive

### Training Scripts
- **`train_v5e_complete.py`**: Full JAX implementation with:
  - Complete Transformer architecture (RMSNorm, RoPE, SwiGLU)
  - AdamW optimizer
  - Cross-entropy loss computation
  - Gradient clipping & checkpointing
  - Google Drive integration
  - Checkpoint pruning (log-spaced)
  
- **`train_v5e.py`**: Framework/skeleton (use train_v5e_complete.py instead)

### Colab Integration
- **`colab_train_v5e.ipynb`**: Ready-to-use Jupyter notebook
- **`colab_setup_v5e.sh`**: Setup bash script

### Documentation
- **`README_V5E.md`**: Comprehensive guide
- **`V5E_QUICK_START.md`**: This file

---

## Config Choices Explained

### Why batch_size=48?
- v5e has 8 chips × 16GB = 128GB total
- 500M model in BF16 = ~1GB
- Activation peaks = ~20-30GB per chip
- 48 is conservative but reliable for long runs
- **Try 64 or 32 if needed**

### Why LR=5e-4?
- Standard for 500M models across LLaMA, Chinchilla, others
- Will decay to 1e-5 via cosine schedule
- **Warmup**: 2k steps to stabilize training
- **Schedule**: Cosine over remaining steps

### Why beta2=0.98?
Research shows:
| Model  | Beta2 | Scale | Source |
|--------|-------|-------|--------|
| LLaMA  | 0.95  | 7B-65B | Meta |
| Chinchilla | 0.95 | 70B | DeepMind |
| Ours   | **0.98** | **500M** | **This project** |

**Rationale**: Smaller models benefit from slightly higher beta2 because:
- Gradient noise is higher
- Larger batch sizes (48) stabilize training
- No need for aggressive momentum (0.95)

**Try 0.95 or 0.99 if you want to experiment**

---

## Key Features

### Indefinite Training ✅
- No max_steps limit (set to 1M, never reached)
- **Manual stop**: Ctrl+C in Colab anytime
- Checkpoints resume seamlessly

### Google Drive Integration ✅
- Auto-saves every 1k steps
- Auto-resumes with `--auto-resume`
- No data lost on runtime crash

### Efficient Checkpointing ✅
- **Log-spaced pruning**: Keeps ~log(N) checkpoints instead of N
- Save disk space while maintaining recovery points
- Example: After 1M steps, keeps ~20 checkpoints instead of 1000

### Full BF16 Training ✅
- No quantization (unlike original MLX version)
- Stable gradient updates
- Better convergence for indefinite training

### RoPE Positional Embeddings ✅
- Modern position encoding (better than absolute)
- Extrapolation friendly (can extend seq_len later)

---

## Expected Performance

### Memory
- **Model**: ~1GB (500M params × 2 bytes)
- **Activations**: ~20-30GB per chip
- **Optimizer state**: ~2GB per chip
- **Total**: ~25-35GB per chip (fits!)

### Speed
- **Tokens/sec**: 2-4k per chip (16-32k total)
- **Realistic throughput**: ~20k tokens/sec
- **For 100B tokens**: ~58 days continuous

### Data
- **Mixed sources**: Cosmopedia-v2 + DCLM + FineWeb-EDU
- **Streamed**: No local storage needed
- **Mixed weights**: 50% + 30% + 20%

---

## Troubleshooting

### "No TPU devices found"
```
✓ Ensure Runtime → Change runtime → TPU v5e
✓ Check: import jax; print(jax.devices())
```

### Out of Memory (OOM)
```
1. Reduce batch_size to 32 or 24
2. Reduce sequence_length to 256
3. Edit configs/v5e.yaml and retry
```

### Checkpoint save fails
```
1. Ensure Google Drive mounted: !ls /content/gdrive/MyDrive/
2. Create directory: !mkdir -p /content/gdrive/MyDrive/checkpoints/500m_v5e
3. Increase save_interval to reduce frequency
```

### Slow training
```
Likely causes:
1. Python GC overhead
2. Slow dataloader
3. Network I/O

Solutions:
1. Reduce log_interval
2. Enable dataloader caching
3. Check datasets library for bottlenecks
```

---

## Resume Training

After Colab runtime restarts:

```python
import subprocess
import os
os.chdir('/content/dLLM')
subprocess.run([
    'python', 'train_v5e_complete.py',
    '--config', 'configs/v5e.yaml',
    '--auto-resume'  # Finds latest checkpoint automatically
])
```

The script will:
1. ✅ Mount Google Drive
2. ✅ Find latest checkpoint in `/content/gdrive/MyDrive/checkpoints/500m_v5e/`
3. ✅ Load model weights and optimizer state
4. ✅ Resume training from exact step

---

## Monitoring

### Check Progress
```python
import json
from pathlib import Path

ckpt_dir = Path('/content/gdrive/MyDrive/checkpoints/500m_v5e')
latest = sorted(ckpt_dir.glob('step_*'), key=lambda x: int(x.name.split('_')[1]))[-1]
with open(latest / 'state.json') as f:
    print(json.load(f))
```

### View Logs
```python
with open('/content/dLLM/training_v5e_complete.log') as f:
    print(f.read())
```

### List All Checkpoints
```python
ckpt_dir = Path('/content/gdrive/MyDrive/checkpoints/500m_v5e')
for d in sorted(ckpt_dir.glob('step_*')):
    print(d.name)
```

---

## Files Reference

```
dLLM/
├── configs/
│   └── v5e.yaml                    ← Edit here to change training params
├── train_v5e_complete.py           ← Main training script
├── colab_train_v5e.ipynb           ← Pre-made Jupyter notebook
├── colab_setup_v5e.sh              ← Setup script
├── README_V5E.md                   ← Full documentation
├── V5E_QUICK_START.md              ← This file
└── [original files: model.py, layers.py, etc.]
```

---

## Next Steps

1. **Run the notebook** or bash commands above
2. **Monitor training** via Colab console
3. **Check Google Drive** for checkpoints every 1k steps
4. **Resume anytime** with `--auto-resume`
5. **Experiment** with hyperparameters (batch_size, LR, beta2)

---

## Support

Issues? Check:
1. `README_V5E.md` for detailed guide
2. Colab console output for errors
3. `/content/dLLM/training_v5e_complete.log` for training logs

Good luck! 🚀
