#!/bin/bash
# Setup script for single v5e TPU training in Google Colab
# Run this in Colab cells before training

REPO="https://raw.githubusercontent.com/Z-Coder672/dLLM/v5e-tpu-training"
DEST="/content/dLLM"

echo "=== v5e TPU Training Setup ==="
echo ""

# Clone the repo (or pull if it already exists)
if [ -d "$DEST/.git" ]; then
  echo "Repo exists, pulling latest..."
  cd "$DEST" && git fetch origin && git reset --hard origin/v5e-tpu-training
else
  echo "Cloning repo..."
  rm -rf "$DEST"
  git clone --branch v5e-tpu-training --single-branch https://github.com/Z-Coder672/dLLM.git "$DEST"
fi
echo "Got latest code from v5e-tpu-training branch"

# Install JAX for TPU
echo ""
echo "Installing JAX for TPU..."
pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install dependencies
echo "Installing dependencies..."
pip install -q transformers datasets tiktoken pyyaml tqdm numpy

# Check TPU availability
echo ""
echo "Checking TPU..."
python3 << 'EOF'
import jax
devices = jax.devices()
print(f"Available devices: {devices}")
print(f"Device type: {devices[0].device_kind if devices else 'None'}")

# Check memory
try:
    import jax.experimental
    print(f"JAX version: {jax.__version__}")
except:
    pass
EOF

echo ""
echo "Setup complete! Ready for training."
echo ""
echo "Next steps (run in separate Colab cells):"
echo "  1. from google.colab import drive; drive.mount('/content/gdrive')"
echo "  2. !python /content/dLLM/train_v5e_complete.py --config /content/dLLM/configs/v5e.yaml"
echo ""
