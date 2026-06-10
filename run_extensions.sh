#!/usr/bin/env bash
# run_extensions.sh — Phase 3 Extension Experiments
#
# Runs all Phase 3 analyses in sequence after Phase 2 training + evaluation.
# Prerequisite: run_training.sh and run_evaluation.sh must have completed.
#
# Usage:
#   export DATA_ROOT=/path/to/sorted/challenge
#   bash run_extensions.sh
#
# All results are written to ./results/

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

DATA_ROOT="${DATA_ROOT:-/data/DENTEX/sorted/challenge}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$REPO_ROOT/results"
LOGS="$REPO_ROOT/logs"
OUTPUT="$REPO_ROOT/output"

# Dataset paths
T3_VAL_JSON="$DATA_ROOT/validation_triple.json"
T3_IMG_DIR="$DATA_ROOT/validation_images"
T3_SUBSET_CLEAN="$REPO_ROOT/results/subsets/clean_subset.json"
T3_SUBSET_STRESS="$REPO_ROOT/results/subsets/stress_subset.json"

# Noisy-box files (written by run_training.sh → phase2_generate_noisy_boxes.py)
NB_T1_VAL="$REPO_ROOT/noisy_boxes/tier1_val_boxes.json"
NB_T2_VAL="$REPO_ROOT/noisy_boxes/tier2_val_boxes.json"

# Checkpoints
WEIGHTS_T1="$OUTPUT/tier1/model_final.pth"
WEIGHTS_T2="$OUTPUT/tier2/model_final.pth"
WEIGHTS_T3="$OUTPUT/tier3/model_final.pth"

CONFIG="$REPO_ROOT/configs/diffdet.custom.swinbase.nonpretrain.yaml"

# Inference image limit for extension experiments (reduce to 50 for speed)
N_IMAGES="${N_IMAGES:-50}"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1\n  Run Phase 2 training first (run_training.sh)."
}

mkdir -p "$RESULTS" "$LOGS"

# ── Preflight checks ─────────────────────────────────────────────────────────

log "Preflight checks"
require_file "$WEIGHTS_T3"
require_file "$T3_VAL_JSON"
require_file "$CONFIG"

if [[ ! -f "$T3_SUBSET_CLEAN" || ! -f "$T3_SUBSET_STRESS" ]]; then
    log "Subsets not found — running phase1_select_subsets.py first"
    python "$REPO_ROOT/phase1_select_subsets.py" \
        --data-root "$DATA_ROOT" \
        --out-dir "$RESULTS/subsets" \
        2>&1 | tee "$LOGS/p3_subsets.log"
fi

require_file "$NB_T1_VAL" || true   # noisy-box files optional — not all experiments need them
require_file "$NB_T2_VAL" || true

echo "  Weights T3:  $WEIGHTS_T3"
echo "  Val JSON:    $T3_VAL_JSON"
echo "  N images:    $N_IMAGES"
echo ""

# ── Experiment 1: Diffusion Step Sensitivity ─────────────────────────────────

log "EXPERIMENT 1: Diffusion step sensitivity (mAP vs SAMPLE_STEP)"
python "$REPO_ROOT/phase3_diffusion_step_map.py" \
    --config-file "$CONFIG" \
    --weights "$WEIGHTS_T3" \
    --tier 2 \
    --val-json "$T3_VAL_JSON" \
    --img-dir "$T3_IMG_DIR" \
    --out-dir "$RESULTS/diffusion_steps" \
    --steps 1 2 5 10 \
    --n-images "$N_IMAGES" \
    2>&1 | tee "$LOGS/p3_diffusion_steps.log"

echo "  Output: $RESULTS/diffusion_steps/"

# ── Experiment 2: Image Degradation Robustness ───────────────────────────────

log "EXPERIMENT 2: Image degradation robustness"
python "$REPO_ROOT/phase3_image_degradation.py" \
    --config-file "$CONFIG" \
    --weights "$WEIGHTS_T3" \
    --tier 2 \
    --img-dir "$T3_IMG_DIR" \
    --val-json "$T3_VAL_JSON" \
    --n-images "$N_IMAGES" \
    --out-dir "$RESULTS/degradation" \
    --save-samples \
    2>&1 | tee "$LOGS/p3_degradation.log"

echo "  Output: $RESULTS/degradation/"

# ── Experiment 3: Noisy-Box Perturbation (Tier 2) ────────────────────────────

if [[ -f "$NB_T1_VAL" ]]; then
    log "EXPERIMENT 3a: Noisy-box perturbation — tier 1 boxes → tier 2 model"
    python "$REPO_ROOT/phase3_noise_injection.py" \
        --noisy-box-json "$NB_T1_VAL" \
        --config-file "$CONFIG" \
        --weights "$WEIGHTS_T2" \
        --tier 1 \
        --val-json "$T3_VAL_JSON" \
        --img-dir "$T3_IMG_DIR" \
        --n-images "$N_IMAGES" \
        --out "$RESULTS/noise_injection/tier2_results.json" \
        2>&1 | tee "$LOGS/p3_noise_t2.log"
else
    echo "  Skipping experiment 3a — noisy-box file not found: $NB_T1_VAL"
fi

if [[ -f "$NB_T2_VAL" ]]; then
    log "EXPERIMENT 3b: Noisy-box perturbation — tier 2 boxes → tier 3 model"
    python "$REPO_ROOT/phase3_noise_injection.py" \
        --noisy-box-json "$NB_T2_VAL" \
        --config-file "$CONFIG" \
        --weights "$WEIGHTS_T3" \
        --tier 2 \
        --val-json "$T3_VAL_JSON" \
        --img-dir "$T3_IMG_DIR" \
        --n-images "$N_IMAGES" \
        --out "$RESULTS/noise_injection/tier3_results.json" \
        2>&1 | tee "$LOGS/p3_noise_t3.log"
else
    echo "  Skipping experiment 3b — noisy-box file not found: $NB_T2_VAL"
fi

# ── Experiment 4: Failure Analysis ───────────────────────────────────────────

log "EXPERIMENT 4: Failure case analysis (clean vs stress-test)"
python "$REPO_ROOT/phase3_failure_analysis.py" \
    --config-file "$CONFIG" \
    --weights "$WEIGHTS_T3" \
    --tier 2 \
    --val-json "$T3_VAL_JSON" \
    --img-dir "$T3_IMG_DIR" \
    --clean-subset "$T3_SUBSET_CLEAN" \
    --stress-subset "$T3_SUBSET_STRESS" \
    --out-dir "$RESULTS/failure_analysis" \
    2>&1 | tee "$LOGS/p3_failure.log"

echo "  Output: $RESULTS/failure_analysis/"

# ── Experiment 5: Prediction Visualization ───────────────────────────────────

log "EXPERIMENT 5: Prediction visualization (all 3 tiers)"
if [[ -f "$WEIGHTS_T1" && -f "$WEIGHTS_T2" ]]; then
    python "$REPO_ROOT/phase3_visualize_predictions.py" \
        --config-file "$CONFIG" \
        --weights-t1 "$WEIGHTS_T1" \
        --weights-t2 "$WEIGHTS_T2" \
        --weights-t3 "$WEIGHTS_T3" \
        --val-json "$T3_VAL_JSON" \
        --img-dir "$T3_IMG_DIR" \
        --out-dir "$RESULTS/visualizations" \
        --n-images "$N_IMAGES" \
        2>&1 | tee "$LOGS/p3_visualize.log"
    echo "  Output: $RESULTS/visualizations/"
else
    echo "  Skipping experiment 5 — tier-1 or tier-2 checkpoint not found"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

log "Phase 3 complete — summary of outputs"
echo ""
printf "%-40s %s\n" "Experiment" "Output directory"
printf "%-40s %s\n" "-----------" "----------------"
printf "%-40s %s\n" "1. Diffusion step sensitivity"  "$RESULTS/diffusion_steps/"
printf "%-40s %s\n" "2. Image degradation"           "$RESULTS/degradation/"
printf "%-40s %s\n" "3. Noisy-box perturbation"      "$RESULTS/noise_injection/"
printf "%-40s %s\n" "4. Failure analysis"            "$RESULTS/failure_analysis/"
printf "%-40s %s\n" "5. Prediction visualization"    "$RESULTS/visualizations/"
echo ""
echo "Logs: $LOGS/p3_*.log"
echo ""
echo "Next: fill in Phase 3 results in experiment_log.md and"
echo "      update reproducibility_checklist.md section 5."
