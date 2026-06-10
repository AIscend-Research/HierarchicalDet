#!/usr/bin/env bash
# ============================================================
# HierarchicalDet — Baseline Model Training and Evaluation
# ============================================================
# Trains and evaluates the baseline models from the paper:
#   - RetinaNet (Swin-B backbone to match HierarchicalDet)
#   - Faster R-CNN (Swin-B backbone)
#   - DiffusionDet standard (without hierarchical noisy-box manipulation)
#
# Note on DETR: DETR requires a separate installation (detr package)
# and is not included in the bundled detectron2. If DETR is available,
# add it manually following the official detr/d2 setup.
#
# Approach: Each baseline is trained on the FULL DENTEX training set
# (all three annotation tiers merged), treating the three-label detection
# as a single multi-label classification problem. This is the fairest
# comparison — each baseline gets the same training data as HierarchicalDet.
#
# Since the baselines use standard COCO loaders, we need to convert the
# triple-category annotations to single-category COCO format (collapse
# to a single 16-class problem: 4 quadrants × 1 + 8 enum × 1 + 4 diag × 1).
#
# Usage:
#   export DATA_ROOT=/path/to/sorted/challenge
#   bash run_baselines.sh 2>&1 | tee logs/baselines.log

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../sorted/challenge}"
NUM_GPUS="${NUM_GPUS:-1}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-models/swin_base_patch4_window7_224_22k.pkl}"
SEED=40244023

mkdir -p logs results/retinanet_eval results/fasterrcnn_eval results/diffusiondet_std_eval

# ── Step 0: Convert annotations to standard COCO format for baselines ─────────
echo "Converting DENTEX triple-label annotations to single-label COCO..."
python3 phase2_convert_to_single_label_coco.py \
    --train-json "$DATA_ROOT/train_merged_disease_coco3class_onlyd_fixed.json" \
    --val-json   "$DATA_ROOT/test_merged_disease_coco3class.json" \
    --out-dir    "$DATA_ROOT/single_label/" \
    --mode       diagnosis_only
# This creates:
#   $DATA_ROOT/single_label/train.json  (single category_id = diagnosis class)
#   $DATA_ROOT/single_label/val.json

export DENTEX_TIER=tier3_single
export DATA_ROOT_SINGLE="$DATA_ROOT/single_label"

# ── Baseline 1: RetinaNet ──────────────────────────────────────────────────────
echo ""
echo "═══ BASELINE: RetinaNet ═══"
# RetinaNet with Swin-B backbone and FPN — closest architecture to paper's comparison
python3 train_net_patched.py \
    --config-file configs/baseline_retinanet.yaml \
    --num-gpus "$NUM_GPUS" \
    SEED "$SEED" \
    MODEL.WEIGHTS "$BACKBONE_WEIGHTS" \
    OUTPUT_DIR output/retinanet \
    SOLVER.MAX_ITER 40000 \
    2>&1 | tee logs/retinanet_train.log

python3 train_net_patched.py \
    --config-file configs/baseline_retinanet.yaml \
    --eval-only \
    MODEL.WEIGHTS output/retinanet/model_final.pth \
    OUTPUT_DIR results/retinanet_eval \
    2>&1 | tee logs/retinanet_eval.log

# ── Baseline 2: Faster R-CNN ──────────────────────────────────────────────────
echo ""
echo "═══ BASELINE: Faster R-CNN ═══"
python3 train_net_patched.py \
    --config-file configs/baseline_faster_rcnn.yaml \
    --num-gpus "$NUM_GPUS" \
    SEED "$SEED" \
    MODEL.WEIGHTS "$BACKBONE_WEIGHTS" \
    OUTPUT_DIR output/faster_rcnn \
    SOLVER.MAX_ITER 40000 \
    2>&1 | tee logs/fasterrcnn_train.log

python3 train_net_patched.py \
    --config-file configs/baseline_faster_rcnn.yaml \
    --eval-only \
    MODEL.WEIGHTS output/faster_rcnn/model_final.pth \
    OUTPUT_DIR results/fasterrcnn_eval \
    2>&1 | tee logs/fasterrcnn_eval.log

# ── Baseline 3: DiffusionDet (standard, no hierarchical noisy boxes) ──────────
echo ""
echo "═══ BASELINE: DiffusionDet (non-hierarchical) ═══"
# Same architecture as HierarchicalDet but trained end-to-end on the diagnosis
# tier only, WITHOUT the hierarchical noisy-box manipulation.
# This isolates the contribution of the hierarchical training scheme.
export USE_NOISY_BOXES=0
export DENTEX_TIER=tier3

python3 train_net_patched.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --num-gpus "$NUM_GPUS" \
    SEED "$SEED" \
    MODEL.WEIGHTS "$BACKBONE_WEIGHTS" \
    OUTPUT_DIR output/diffusiondet_standard \
    SOLVER.MAX_ITER 40000 \
    2>&1 | tee logs/diffusiondet_std_train.log

python3 train_net_patched.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --eval-only \
    MODEL.WEIGHTS output/diffusiondet_standard/model_final.pth \
    OUTPUT_DIR results/diffusiondet_std_eval \
    2>&1 | tee logs/diffusiondet_std_eval.log

# ── Baseline 4: DETR ──────────────────────────────────────────────────────────
# DETR is listed as a paper baseline but is NOT included in the bundled
# detectron2 and has no official config in this repo.
#
# To add DETR you must:
#   1. Install the facebook DETR detectron2 wrapper:
#        git clone https://github.com/facebookresearch/detr /tmp/detr
#        pip install -e /tmp/detr/d2
#   2. Use a DETR config with a ResNet-50 or Swin backbone, e.g.:
#        configs/COCO-Detection/detr_256_6_6_torchvision.yaml
#   3. Convert annotations with --mode diagnosis_only (already done above)
#   4. Run training analogous to the Faster R-CNN block above
#
# If DETR is set up, uncomment the block below.
#
# echo ""
# echo "═══ BASELINE: DETR ═══"
# python3 train_net_patched.py \
#     --config-file /tmp/detr/d2/configs/detr_256_6_6_torchvision.yaml \
#     --num-gpus "$NUM_GPUS" \
#     SEED "$SEED" \
#     OUTPUT_DIR output/detr \
#     SOLVER.MAX_ITER 40000 \
#     2>&1 | tee logs/detr_train.log
#
# python3 train_net_patched.py \
#     --config-file /tmp/detr/d2/configs/detr_256_6_6_torchvision.yaml \
#     --eval-only \
#     MODEL.WEIGHTS output/detr/model_final.pth \
#     OUTPUT_DIR results/detr_eval \
#     2>&1 | tee logs/detr_eval.log
#
# DEVIATION TO DISCLOSE: DETR baseline requires a separate installation not
# bundled in the official HierarchicalDet repository. If unable to run, note
# this as a reproduction gap and compare only against RetinaNet, Faster R-CNN,
# and non-hierarchical DiffusionDet.

# ── Collect baseline results ──────────────────────────────────────────────────
echo ""
echo "═══ COLLECTING BASELINE RESULTS ═══"
python3 phase2_collect_results.py \
    --eval-dirs results/tier1_eval results/tier2_eval results/tier3_eval \
    --tier-names "Quadrant" "Enumeration" "Diagnosis" \
    --log-files logs/eval_tier1.log logs/eval_tier2.log logs/eval_tier3.log \
    --baseline-dirs results/retinanet_eval results/fasterrcnn_eval results/diffusiondet_std_eval \
    --baseline-names RetinaNet "Faster R-CNN" "DiffusionDet (no hierarchy)" \
    --out results/full_comparison.json

echo ""
echo "All baselines complete. See results/full_comparison.json"
