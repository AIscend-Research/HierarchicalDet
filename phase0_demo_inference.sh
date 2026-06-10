#!/usr/bin/env bash
# phase0_demo_inference.sh — Phase 0 verification: official demo inference
#
# Runs the official demo.py command on a single test image (or a handful of
# images from the validation split) and saves proof artifacts:
#   1. Terminal output (stdout + stderr) → logs/demo_inference.log
#   2. Detection visualization images   → demo_output/
#   3. Raw score file (JSON)            → demo_output/raw_scores.json
#
# This verifies the full inference path end-to-end before committing to
# a full training or evaluation run.
#
# Usage:
#   export DATA_ROOT=/path/to/sorted/challenge
#   bash phase0_demo_inference.sh
#   # Then check: demo_output/ and logs/demo_inference.log

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../sorted/challenge}"
WEIGHTS="${WEIGHTS:-output/tier3/model_final.pth}"
CONFIG="${CONFIG:-configs/diffdet.custom.swinbase.nonpretrain.yaml}"
TIER="${TIER:-2}"           # 0=quadrant, 1=enumeration, 2=diagnosis
THRESHOLD="${THRESHOLD:-0.4}"

OUTDIR="demo_output"
LOGDIR="logs"

mkdir -p "$OUTDIR" "$LOGDIR"

# ── Pick test images ───────────────────────────────────────────────────────────
# Try to find a few images from the validation or test set.
IMG_DIR="$DATA_ROOT/validation_images"
if [[ ! -d "$IMG_DIR" ]]; then
    # Fall back to first match in the data root
    IMG_DIR=$(find "$DATA_ROOT" -type d -name "*valid*" -o -name "*val*" | head -1)
fi

# Collect up to 5 images
mapfile -t IMAGES < <(find "$IMG_DIR" -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) | head -5)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "ERROR: No images found in $IMG_DIR"
    echo "       Set DATA_ROOT to the DENTEX dataset root."
    exit 1
fi

echo "Found ${#IMAGES[@]} images for demo inference."
echo "Config:  $CONFIG"
echo "Weights: $WEIGHTS"
echo "Tier:    $TIER"
echo "Images:  ${IMAGES[*]}"
echo ""

# ── Check that weights exist ───────────────────────────────────────────────────
if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Weights not found: $WEIGHTS"
    echo "       Complete Phase 2 training (run_training.sh) first,"
    echo "       or set WEIGHTS= to an existing checkpoint."
    exit 1
fi

# ── Run official demo.py ───────────────────────────────────────────────────────
echo "Running demo.py (official inference command from README)..."
python demo.py \
    --config-file "$CONFIG" \
    --input "${IMAGES[@]}" \
    --output "$OUTDIR" \
    --confidence-threshold "$THRESHOLD" \
    --nclass "$TIER" \
    MODEL.WEIGHTS "$WEIGHTS" \
    2>&1 | tee "$LOGDIR/demo_inference.log"

# ── Capture raw scores ─────────────────────────────────────────────────────────
# Run a minimal Python snippet to extract raw prediction scores and boxes
# from the same images and save as JSON proof.
echo ""
echo "Extracting raw scores to $OUTDIR/raw_scores.json..."

python3 - << 'PYEOF'
import json, os, sys, time
sys.path.insert(0, '.')

import torch
import numpy as np
import cv2

CONFIG   = os.environ.get("CONFIG",  "configs/diffdet.custom.swinbase.nonpretrain.yaml")
WEIGHTS  = os.environ.get("WEIGHTS", "output/tier3/model_final.pth")
TIER     = int(os.environ.get("TIER", "2"))
OUTDIR   = os.environ.get("OUTDIR",  "demo_output")
IMG_DIR  = os.environ.get("IMG_DIR", "")

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import transforms as T
from hierarchialdet import add_diffusiondet_config
from hierarchialdet.util.model_ema import (
    add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer
)

cfg = get_cfg()
add_diffusiondet_config(cfg)
add_model_ema_configs(cfg)
cfg.merge_from_file(CONFIG)
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.freeze()

model = build_model(cfg)
kwargs = may_get_ema_checkpointer(cfg, model)
if cfg.MODEL_EMA.ENABLED:
    EMADetectionCheckpointer(model, **kwargs).resume_or_load(WEIGHTS, resume=False)
else:
    DetectionCheckpointer(model).resume_or_load(WEIGHTS, resume=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

import glob
images = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_DIR, "*.png"))
images = sorted(images)[:5]
if not images:
    print("  No images found — raw_scores.json will be empty.")
    json.dump([], open(os.path.join(OUTDIR, "raw_scores.json"), "w"))
    sys.exit(0)

resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
all_results = []

with torch.no_grad():
    for img_path in images:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1]
        img_r, _ = T.apply_transform_gens([resize], img_rgb)
        img_t = torch.as_tensor(
            np.ascontiguousarray(img_r.transpose(2, 0, 1))
        ).float().to(device)

        t0 = time.perf_counter()
        out = model([{"image": img_t, "height": h, "width": w,
                      "image_id": 0, "file_name": img_path}], k=TIER)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        inst = out[0]["instances"].to("cpu")
        scores = inst.scores.numpy().tolist()
        boxes  = inst.pred_boxes.tensor.numpy().tolist()
        labels = inst.pred_classes.numpy().tolist() if inst.has("pred_classes") else []

        all_results.append({
            "file_name": os.path.basename(img_path),
            "tier": TIER,
            "inference_ms": round(elapsed_ms, 1),
            "n_detections": len(scores),
            "predictions": [
                {"score": round(s, 4), "box_xyxy": [round(v, 1) for v in b],
                 "class_id": c if labels else None}
                for s, b, c in zip(scores, boxes, labels or [None]*len(scores))
            ],
        })
        print(f"  {os.path.basename(img_path)}: {len(scores)} detections, {elapsed_ms:.0f} ms")

out_path = os.path.join(OUTDIR, "raw_scores.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nRaw scores saved to {out_path}")
PYEOF

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Phase 0 Demo Inference — DONE"
echo "══════════════════════════════════════════════════"
echo "  Terminal log:      $LOGDIR/demo_inference.log"
echo "  Visualizations:    $OUTDIR/  (PNG files)"
echo "  Raw scores (JSON): $OUTDIR/raw_scores.json"
echo ""
echo "  Include these three artifacts as proof-of-inference in your"
echo "  MLRC reproducibility report."
