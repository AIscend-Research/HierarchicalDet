#!/usr/bin/env bash
# HierarchicalDet reproduction environment setup
# Run this on a GPU machine with CUDA 12.1.
# This script REPLACES the roadmap's setup commands, which contain errors.
#
# ERRORS IN ORIGINAL ROADMAP:
#   1. "pip install -r requirements.txt" — no requirements.txt exists
#   2. "pip install detectron2" — DO NOT do this; detectron2 is bundled in the repo
#
# Tested on: Linux x86_64, CUDA 12.1, conda 23+

set -euo pipefail

# ── 0. Clone official repo ────────────────────────────────────────────────────
REPO_DIR="$(pwd)/HierarchicalDet_official"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/ibrahimethemhamamci/HierarchicalDet "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 1. Create conda environment ───────────────────────────────────────────────
conda create -n hierarchicaldet python=3.9 -y
# All subsequent commands run inside the env via `conda run`
CONDA_RUN="conda run -n hierarchicaldet --no-capture-output"

# ── 2. Install PyTorch (CUDA 12.1) ───────────────────────────────────────────
$CONDA_RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. Install core dependencies ─────────────────────────────────────────────
# These are inferred from the import statements in the codebase.
# requirements.txt does not exist in the official repo.
$CONDA_RUN pip install \
    opencv-python \
    fvcore \
    iopath \
    omegaconf \
    hydra-core \
    timm \
    scipy \
    pycocotools \
    tqdm \
    termcolor \
    yacs \
    tabulate \
    cloudpickle \
    matplotlib \
    Pillow \
    scikit-image \
    huggingface_hub

# ── 4. Install bundled detectron2 ────────────────────────────────────────────
# The repo ships its own detectron2/. Build and install it from source.
# Do NOT run: pip install detectron2
$CONDA_RUN pip install -e .

# Verify detectron2 installation
$CONDA_RUN python -c "import detectron2; print('detectron2 version:', detectron2.__version__)"

# ── 5. Download Swin-B backbone weights ──────────────────────────────────────
# The nonpretrain config uses swin_base_patch4_window7_224_22k, the standard
# ImageNet-22k pretrained Swin-B. Convert it to detectron2 pickle format.
mkdir -p models
$CONDA_RUN python - <<'EOF'
import torch
import pickle
import urllib.request

# Official Swin-B 22k weights from Microsoft
url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
print(f"Downloading Swin-B 22k backbone from: {url}")
urllib.request.urlretrieve(url, "models/swin_base_patch4_window7_224_22k.pth")

# Convert to detectron2 pickle format
ckpt = torch.load("models/swin_base_patch4_window7_224_22k.pth", map_location="cpu")
state_dict = ckpt.get("model", ckpt)

# Wrap in detectron2 expected format
d2_ckpt = {"model": state_dict, "__author__": "third_party", "matching_heuristics": True}
with open("models/swin_base_patch4_window7_224_22k.pkl", "wb") as f:
    pickle.dump(d2_ckpt, f)

print("Backbone saved to models/swin_base_patch4_window7_224_22k.pkl")
EOF

# ── 6. Verify GPU is visible ──────────────────────────────────────────────────
$CONDA_RUN python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete. Activate with: conda activate hierarchicaldet"
echo "  Then run dataset download: bash ../download_dataset.sh"
echo "══════════════════════════════════════════════════════════════"
