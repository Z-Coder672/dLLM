#!/bin/bash
# Setup script for single v5e TPU training in Google Colab
# Run this in Colab cells before training

REPO="https://raw.githubusercontent.com/Z-Coder672/dLLM/main"
DEST="/content/dLLM"

echo "=== v5e TPU Training Setup ==="
echo ""

# Download only the files needed for v5e training
echo "Downloading training files from GitHub..."
mkdir -p "$DEST/configs"
curl -fsSL "$REPO/train_v5e_complete.py" -o "$DEST/train_v5e_complete.py"
curl -fsSL "$REPO/configs/v5e.yaml"       -o "$DEST/configs/v5e.yaml"
echo "Downloaded train_v5e_complete.py and configs/v5e.yaml"

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

# Mount Google Drive for checkpoints
echo ""
echo "Mounting Google Drive..."
python3 -c "from google.colab import drive; drive.mount('/content/gdrive')"

echo ""
echo "Setup complete! Ready for training."
echo ""
echo "Next step:"
echo "  python /content/dLLM/train_v5e_complete.py --config /content/dLLM/configs/v5e.yaml"
echo ""
