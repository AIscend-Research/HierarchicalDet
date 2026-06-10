# HierarchicalDet — Reproduction Study

Reproducing **"Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays"** (MICCAI 2023).

- Paper: https://arxiv.org/abs/2303.06500
- Official repo: https://github.com/ibrahimethemhamamci/HierarchicalDet
- Dataset (DENTEX): https://huggingface.co/datasets/ibrahimhamamci/DENTEX

---

## Phase Status

| Phase | Target | Status |
|-------|--------|--------|
| 0 — Setup & Audit | June 3 | Scripts ready (run on GPU machine) |
| 1 — Data Preparation | June 3 | Scripts ready (run on GPU machine) |
| 2 — Reproduction | June 4 | Scripts ready (run on GPU machine) |
| 3 — Extensions | June 5 | Scripts ready (run on GPU machine) |
| 4 — Writing | June 6 | Pending |

---

## Phase 0 — Key Findings (READ BEFORE RUNNING ANYTHING)

### 1. No pretrained HierarchicalDet weights are released
You must train from scratch. See compute estimate below.

### 2. No `requirements.txt` exists
The roadmap's `pip install -r requirements.txt` will fail. Use `setup.sh` instead.

### 3. detectron2 is bundled in the official repo
Do **not** run `pip install detectron2`. The repo ships its own copy under `detectron2/`. The `setup.sh` script builds it from source.

### 4. The main config requires a custom pretrained backbone (not public)
`diffdet.custom.swinbase.enumeration.yaml` references `../Swin-Transformer/output_dental_pretrain2/train_smim/ckpt_epoch_99.pth` — a custom dental X-ray pretrained Swin-B that is not publicly available.

**We use `diffdet.custom.swinbase.nonpretrain.yaml` instead**, which uses the standard ImageNet-22k Swin-B. Results will differ slightly from the paper; this is disclosed in the report.

### 5. Dataset paths are hardcoded in `train_net.py`
The original `train_net.py` has hardcoded relative paths (`../sorted/challenge/...`). Use `train_net_patched.py` instead, which reads from `DATA_ROOT` and `DENTEX_TIER` environment variables.

### 6. Compute requirements
Training from scratch with `diffdet.custom.swinbase.nonpretrain.yaml`:
- Swin-B backbone, 40,000 iterations, batch size 2
- Requires ~8 GB VRAM minimum, 16 GB recommended
- Estimated: 12–20 hours on a single A100 or V100

---

## Correct Setup (GPU Machine Only)

This machine (Mac) has no GPU. Run the following on a Linux GPU server.

```bash
# 1. Clone the official repo
git clone https://github.com/ibrahimethemhamamci/HierarchicalDet

# 2. Copy our reproduction scripts into the cloned repo
cp setup.sh train_net_patched.py download_dataset.sh HierarchicalDet/

# 3. Run setup (creates conda env, installs deps, downloads Swin-B backbone)
cd HierarchicalDet
bash setup.sh

# 4. Download DENTEX dataset (~11.8 GB)
conda activate hierarchicaldet
bash download_dataset.sh

# 5. Organize dataset and set env vars
export DATA_ROOT=/path/to/sorted/challenge
export DENTEX_TIER=tier3

# 6. Train from scratch
python train_net_patched.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS models/swin_base_patch4_window7_224_22k.pkl \
    OUTPUT_DIR output/tier3

# 7. Evaluate
python train_net_patched.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --eval-only \
    MODEL.WEIGHTS output/tier3/model_final.pth \
    OUTPUT_DIR output/tier3_eval

# 8. Demo inference on a single image
python demo.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --input path/to/xray.jpg \
    --output output/demo/ \
    --confidence-threshold 0.5 \
    --nclass 3 \
    MODEL.WEIGHTS output/tier3/model_final.pth
```

---

## Repository Contents

### Setup
| File | Purpose |
|------|---------|
| `setup.sh` | Corrected environment setup (replaces roadmap's broken commands) |
| `download_dataset.sh` | Downloads DENTEX dataset from Hugging Face |
| `train_net_patched.py` | Fixed training script with DATA_ROOT/DENTEX_TIER env vars |
| `dataset_mapper_patched.py` | Fixed mapper with NOISY_BOX_TRAIN/VAL env vars (tier-1 GT-only fallback) |
| `phase0_findings.md` | Full audit: paper analysis, blockers, annotation format, compute estimate |
| `phase0_demo_inference.sh` | Runs official `demo.py` on test images; saves terminal log + visualizations + raw scores JSON as proof |

### Phase 1 — Data
| File | Purpose |
|------|---------|
| `phase1_audit_dataset.py` | Image counts, resolution, annotation completeness table |
| `phase1_verify_dataloader.py` | Dataloader smoke test on 10 real samples |
| `phase1_select_subsets.py` | Produces clean (100 imgs) and stress-test (50 imgs) eval subsets |
| `phase1_visualize_annotations.py` | 5-image visual verification with bbox + label overlay |

### Phase 2 — Reproduction
| File | Purpose |
|------|---------|
| `run_training.sh` | Full 3-tier training pipeline (tier1→2→3 with noisy box generation) |
| `phase2_generate_noisy_boxes.py` | Runs inference to produce intermediate noisy-box JSONs |
| `run_evaluation.sh` | Evaluates all 3 tiers, runs runtime benchmark, collects results |
| `run_baselines.sh` | Trains/evaluates RetinaNet, Faster R-CNN, non-hierarchical DiffusionDet |
| `phase2_convert_to_single_label_coco.py` | Converts triple-label annotations to standard COCO for baselines |
| `phase2_collect_results.py` | Parses logs, formats comparison table vs paper values |
| `phase2_runtime_benchmark.py` | Per-image timing (GPU+CPU), failure cases, diffusion step sweep |
| `experiment_log.md` | Fill-in template for all experiment metadata (required for paper) |

### Phase 3 — Extensions
| File | Purpose |
|------|---------|
| `run_extensions.sh` | Runs all Phase 3 experiments in sequence |
| `phase3_diffusion_step_map.py` | mAP vs SAMPLE_STEP (1/2/5/10) sweep with runtime comparison |
| `phase3_image_degradation.py` | Detection robustness under blur, JPEG, resolution reduction, noise |
| `phase3_noise_injection.py` | Hierarchical robustness: perturbs noisy-box proposals between tiers |
| `phase3_failure_analysis.py` | Categorizes failure modes; compares clean vs stress-test subsets |
| `phase3_visualize_predictions.py` | 3-panel GT vs prediction overlay figures for all tiers |
| `phase3_colab_notebook.ipynb` | Colab T4 feasibility test (inference-only; training not feasible) |
| `reproducibility_checklist.md` | Standalone checklist covering all reproduction steps and deviations |

---

## DENTEX Dataset Summary

| Tier | Images | Annotation |
|------|--------|------------|
| Quadrant | 693 | 4 quadrant classes |
| Quadrant + Enumeration | 634 | Quadrant + FDI tooth number |
| Quadrant + Enumeration + Diagnosis | 1005 (705/50/250 train/val/test) | Full hierarchy: Q + N + D |

Total size: ~11.8 GB. License: CC-BY-NC-SA 4.0 (non-commercial).

---

## Model Configuration

Config used: `diffdet.custom.swinbase.nonpretrain.yaml`

| Parameter | Value |
|-----------|-------|
| Backbone | Swin-B (ImageNet-22k) |
| Proposals | 1000 |
| Classes | `[4, 8, 4]` (quadrant, enumeration, diagnosis) |
| Loss | Federated loss (handles partial annotations) |
| Diffusion steps | 1 (inference default) |
| Max iterations | 40,000 |
| Optimizer | AdamW |
