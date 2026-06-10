#!/usr/bin/env bash
# ============================================================
# HierarchicalDet — Evaluation Pipeline
# ============================================================
# Runs mAP evaluation at all three detection tiers on the
# official test split, then records runtime per image.
#
# Usage:
#   export DATA_ROOT=/path/to/sorted/challenge
#   bash run_evaluation.sh 2>&1 | tee logs/evaluation.log
#
# Outputs:
#   results/tier1_eval/    — mAP for quadrant tier
#   results/tier2_eval/    — mAP for enumeration tier
#   results/tier3_eval/    — mAP for diagnosis tier
#   results/eval_summary.json — aggregated metrics table

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../sorted/challenge}"
CONFIG_FILE="configs/diffdet.custom.swinbase.nonpretrain.yaml"

mkdir -p results/tier1_eval results/tier2_eval results/tier3_eval logs

# ── Tier 1: Quadrant mAP ──────────────────────────────────────────────────────
echo "═══ EVAL TIER 1: Quadrant Detection (k=0) ═══"
export DENTEX_TIER=tier1
export USE_NOISY_BOXES=0

python3 train_net_patched.py \
    --config-file "$CONFIG_FILE" \
    --eval-only \
    MODEL.WEIGHTS output/tier1/model_final.pth \
    OUTPUT_DIR results/tier1_eval \
    2>&1 | tee logs/eval_tier1.log

# ── Tier 2: Enumeration mAP ──────────────────────────────────────────────────
echo ""
echo "═══ EVAL TIER 2: Quadrant-Enumeration Detection (k=0,1) ═══"
export DENTEX_TIER=tier2
export USE_NOISY_BOXES=1
export NOISY_BOX_VAL="noisy_boxes/tier1_val_boxes.json"

python3 train_net_patched.py \
    --config-file "$CONFIG_FILE" \
    --eval-only \
    MODEL.WEIGHTS output/tier2/model_final.pth \
    OUTPUT_DIR results/tier2_eval \
    2>&1 | tee logs/eval_tier2.log

# ── Tier 3: Diagnosis mAP ────────────────────────────────────────────────────
echo ""
echo "═══ EVAL TIER 3: Full Diagnosis Detection (k=0,1,2) ═══"
export DENTEX_TIER=tier3
export USE_NOISY_BOXES=1
export NOISY_BOX_VAL="noisy_boxes/tier2_val_boxes.json"

python3 train_net_patched.py \
    --config-file "$CONFIG_FILE" \
    --eval-only \
    MODEL.WEIGHTS output/tier3/model_final.pth \
    OUTPUT_DIR results/tier3_eval \
    2>&1 | tee logs/eval_tier3.log

# ── Collect all results ───────────────────────────────────────────────────────
echo ""
echo "═══ COLLECTING RESULTS ═══"
python3 phase2_collect_results.py \
    --eval-dirs results/tier1_eval results/tier2_eval results/tier3_eval \
    --tier-names "Quadrant" "Enumeration" "Diagnosis" \
    --log-files logs/eval_tier1.log logs/eval_tier2.log logs/eval_tier3.log \
    --out results/eval_summary.json

# ── Runtime benchmark ────────────────────────────────────────────────────────
echo ""
echo "═══ RUNTIME BENCHMARK ═══"
python3 phase2_runtime_benchmark.py \
    --config-file "$CONFIG_FILE" \
    --weights output/tier3/model_final.pth \
    --tier 2 \
    --n-images 50 \
    --out results/runtime_benchmark.json \
    2>&1 | tee logs/runtime_benchmark.log

echo ""
echo "All evaluation complete. See results/eval_summary.json"
