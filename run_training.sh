#!/usr/bin/env bash
# ============================================================
# HierarchicalDet — Full Hierarchical Training Pipeline
# ============================================================
# Trains all three tiers sequentially:
#   Tier 1 (quadrant)    → Tier 2 (enumeration) → Tier 3 (diagnosis)
#
# Between tiers, runs inference on train+val to generate the noisy-box
# JSON files that seed the next tier's dataset mapper.
#
# Usage:
#   export DATA_ROOT=/path/to/sorted/challenge
#   export NUM_GPUS=1           # or 2, 4, etc.
#   export BACKBONE_WEIGHTS=models/swin_base_patch4_window7_224_22k.pkl
#   bash run_training.sh 2>&1 | tee logs/training_full.log
#
# Outputs:
#   output/tier1/model_final.pth
#   output/tier2/model_final.pth
#   output/tier3/model_final.pth
#   logs/                        — tier-by-tier logs
#   noisy_boxes/                 — intermediate inference JSONs

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-../sorted/challenge}"
NUM_GPUS="${NUM_GPUS:-1}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-models/swin_base_patch4_window7_224_22k.pkl}"
CONFIG_FILE="configs/diffdet.custom.swinbase.nonpretrain.yaml"
SEED=40244023

mkdir -p logs output/tier1 output/tier2 output/tier3 noisy_boxes

# Verify backbone exists
if [ ! -f "$BACKBONE_WEIGHTS" ]; then
    echo "ERROR: Backbone weights not found: $BACKBONE_WEIGHTS"
    echo "Run setup.sh first to download the Swin-B backbone."
    exit 1
fi

# Log hardware and package versions for reproducibility
python3 - <<'EOF' | tee logs/environment.txt
import torch, sys, platform
print("=" * 50)
print("ENVIRONMENT LOG")
print("=" * 50)
print(f"Python:    {sys.version}")
print(f"PyTorch:   {torch.__version__}")
print(f"CUDA:      {torch.version.cuda}")
print(f"GPU:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Platform:  {platform.platform()}")
try:
    import detectron2
    print(f"detectron2: {detectron2.__version__}")
except: pass
try:
    import fvcore
    print(f"fvcore:    {fvcore.__version__}")
except: pass
EOF

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1: Quadrant Detection
# No noisy boxes — uses GT boxes only (standard DiffusionDet behaviour)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  TIER 1: Quadrant Detection (k=0)"
echo "  Config: $CONFIG_FILE"
echo "  Iterations: 40000"
echo "════════════════════════════════════════════════"
echo ""

export DATA_ROOT
export DENTEX_TIER=tier1
export USE_NOISY_BOXES=0

python3 train_net_patched.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus "$NUM_GPUS" \
    SEED "$SEED" \
    MODEL.WEIGHTS "$BACKBONE_WEIGHTS" \
    OUTPUT_DIR output/tier1 \
    MODEL.ROI_HEADS.NUM_CLASSES 4 \
    MODEL.DiffusionDet.NUM_CLASSES "[4,8,4]" \
    SOLVER.MAX_ITER 40000 \
    2>&1 | tee logs/tier1_train.log

echo "Tier 1 training complete. Checkpoint: output/tier1/model_final.pth"

# ─────────────────────────────────────────────────────────────────────────────
# INTERMEDIATE: Run tier-1 inference on train+val to generate noisy boxes
# These become the noisy box proposals for tier-2 training.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  GENERATING NOISY BOXES: Tier-1 inference on train"
echo "═══════════════════════════════════════════════════"

for SPLIT in train val; do
    if [ "$SPLIT" = "train" ]; then
        DATASET="custom_train_class"
    else
        DATASET="custom_validation_class"
    fi

    python3 phase2_generate_noisy_boxes.py \
        --config-file "$CONFIG_FILE" \
        --weights output/tier1/model_final.pth \
        --dataset "$DATASET" \
        --tier 0 \
        --out "noisy_boxes/tier1_${SPLIT}_boxes.json" \
        --score-thresh 0.5 \
        2>&1 | tee "logs/tier1_inference_${SPLIT}.log"

    echo "  Saved: noisy_boxes/tier1_${SPLIT}_boxes.json"
done

# ─────────────────────────────────────────────────────────────────────────────
# TIER 2: Quadrant-Enumeration Detection
# Uses tier-1 inference boxes as noisy proposals
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  TIER 2: Quadrant-Enumeration Detection (k=1)"
echo "  Noisy boxes: noisy_boxes/tier1_*.json"
echo "════════════════════════════════════════════════"
echo ""

export DENTEX_TIER=tier2
export USE_NOISY_BOXES=1
export NOISY_BOX_TRAIN="noisy_boxes/tier1_train_boxes.json"
export NOISY_BOX_VAL="noisy_boxes/tier1_val_boxes.json"
export NOISY_BOX_THRESH=0.5

python3 train_net_patched.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus "$NUM_GPUS" \
    SEED "$SEED" \
    MODEL.WEIGHTS output/tier1/model_final.pth \
    OUTPUT_DIR output/tier2 \
    MODEL.ROI_HEADS.NUM_CLASSES 4 \
    MODEL.DiffusionDet.NUM_CLASSES "[4,8,4]" \
    SOLVER.MAX_ITER 40000 \
    2>&1 | tee logs/tier2_train.log

echo "Tier 2 training complete. Checkpoint: output/tier2/model_final.pth"

# Tier-2 inference for tier-3 noisy boxes
echo ""
echo "══════════════════════════════════════════════════════"
echo "  GENERATING NOISY BOXES: Tier-2 inference on train+val"
echo "══════════════════════════════════════════════════════"

for SPLIT in train val; do
    if [ "$SPLIT" = "train" ]; then
        DATASET="custom_train_class"
    else
        DATASET="custom_validation_class"
    fi

    python3 phase2_generate_noisy_boxes.py \
        --config-file "$CONFIG_FILE" \
        --weights output/tier2/model_final.pth \
        --dataset "$DATASET" \
        --tier 1 \
        --out "noisy_boxes/tier2_${SPLIT}_boxes.json" \
        --score-thresh 0.5 \
        2>&1 | tee "logs/tier2_inference_${SPLIT}.log"

    echo "  Saved: noisy_boxes/tier2_${SPLIT}_boxes.json"
done

# ─────────────────────────────────────────────────────────────────────────────
# TIER 3: Full Diagnosis Detection
# Uses tier-2 inference boxes as noisy proposals
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  TIER 3: Full Diagnosis Detection (k=2)"
echo "  Noisy boxes: noisy_boxes/tier2_*.json"
echo "════════════════════════════════════════════════"
echo ""

export DENTEX_TIER=tier3
export USE_NOISY_BOXES=1
export NOISY_BOX_TRAIN="noisy_boxes/tier2_train_boxes.json"
export NOISY_BOX_VAL="noisy_boxes/tier2_val_boxes.json"

python3 train_net_patched.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus "$NUM_GPUS" \
    SEED "$SEED" \
    MODEL.WEIGHTS output/tier2/model_final.pth \
    OUTPUT_DIR output/tier3 \
    MODEL.ROI_HEADS.NUM_CLASSES 4 \
    MODEL.DiffusionDet.NUM_CLASSES "[4,8,4]" \
    SOLVER.MAX_ITER 40000 \
    2>&1 | tee logs/tier3_train.log

echo ""
echo "════════════════════════════════════════════════"
echo "  ALL TIERS COMPLETE"
echo "  Tier 1: output/tier1/model_final.pth"
echo "  Tier 2: output/tier2/model_final.pth"
echo "  Tier 3: output/tier3/model_final.pth"
echo "════════════════════════════════════════════════"
echo ""
echo "Next: bash run_evaluation.sh"
