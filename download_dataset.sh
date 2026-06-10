#!/usr/bin/env bash
# Download DENTEX dataset from Hugging Face and organize for HierarchicalDet training.
# Dataset: https://huggingface.co/datasets/ibrahimhamamci/DENTEX
# Size: ~11.8 GB
# License: CC BY-SA 4.0
#
# Expected output directory structure (mirrors what train_net.py expects):
#   sorted/
#     challenge/
#       train_merged_disease_coco3class_onlyd_fixed.json
#       test_merged_disease_coco3class.json
#       for_coco_disease_train/   (training images)
#       for_coco_disease_test/    (test images)
#
# Run from the parent directory of HierarchicalDet_official/, so that
# "../sorted/" resolves correctly from inside the cloned repo.

set -euo pipefail

# Destination: parent dir of wherever this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/sorted"

echo "Downloading DENTEX dataset to: $DEST"
mkdir -p "$DEST"

# ── Option A: huggingface_hub CLI (recommended) ───────────────────────────────
python3 - <<EOF
from huggingface_hub import snapshot_download
import os, shutil, pathlib

print("Downloading DENTEX dataset (~11.8 GB) — this may take a while...")
local_dir = snapshot_download(
    repo_id="ibrahimhamamci/DENTEX",
    repo_type="dataset",
    local_dir="$DEST/dentex_raw",
    ignore_patterns=["*.git*"],
)
print(f"Downloaded to: {local_dir}")
EOF

# ── Inspect what was downloaded ────────────────────────────────────────────────
echo ""
echo "Downloaded files:"
find "$DEST/dentex_raw" -maxdepth 3 -type f | sort | head -60
echo ""
echo "Directory structure:"
find "$DEST/dentex_raw" -maxdepth 3 -type d | sort

# ── Note on data organization ─────────────────────────────────────────────────
# train_net.py registers datasets with these hardcoded paths:
#   "../sorted/challenge/train_merged_disease_coco3class_onlyd_fixed.json"
#   "../sorted/challenge/for_coco_disease_train"
#   "../sorted/challenge/test_merged_disease_coco3class.json"
#   "../sorted/challenge/for_coco_disease_test"
#
# After download, inspect $DEST/dentex_raw/ to find the actual JSON and image dirs,
# then either:
#   (a) Symlink/move them to match the expected paths above, OR
#   (b) Edit train_net.py to use configurable DATA_ROOT env variable (see train_net_patched.py)
#
# Option (b) is cleaner for reproducibility. See the patched train_net script.

echo ""
echo "Next step: inspect $DEST/dentex_raw/ and run organize_dataset.py"
echo "Or edit train_net.py DATA paths to point to the downloaded files."
