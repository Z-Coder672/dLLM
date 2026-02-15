#!/bin/bash
# Setup script for v5e TPU training in Google Colab
# Run this in Colab cells before training

echo "=== v5e TPU Training Setup ==="
echo ""

# Install JAX for TPU
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
echo "Next steps:"
echo "1. Mount Google Drive: from google.colab import drive; drive.mount('/content/gdrive')"
echo "2. Clone/upload your dLLM repo to /content"
echo "3. Run: python /content/dLLM/train_v5e.py --config /content/dLLM/configs/v5e.yaml"
echo ""
