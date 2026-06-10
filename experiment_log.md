# HierarchicalDet Reproduction — Experiment Log

Fill in each section as experiments run. Every row in a results table must
have a corresponding entry here. The paper reviewer will look for this.

---

## Checklist: Must document before claiming any result

- [ ] Git commit hash of official HierarchicalDet repo cloned
- [ ] Git commit hash of this reproduction repo
- [ ] Conda environment exported (`conda env export > environment_frozen.yml`)
- [ ] Exact PyTorch and CUDA versions
- [ ] GPU model and CUDA driver version
- [ ] Random seed used (`SEED=40244023` from Base-DiffusionDet.yaml)
- [ ] Config file used (full path, not shorthand)
- [ ] Model weights used (path + SHA-256 hash)
- [ ] Dataset version (Hugging Face commit hash or download date)
- [ ] Training command (exact, copy-pasteable)
- [ ] Evaluation command (exact, copy-pasteable)
- [ ] Whether noisy-box files were used, and their source

---

## Environment

| Field | Value |
|-------|-------|
| Date | |
| GPU | |
| CUDA version | |
| Python version | |
| PyTorch version | |
| detectron2 version | (bundled — see git submodule) |
| OS | |
| HierarchicalDet repo commit | |
| Reproduction repo commit | |

```bash
# To capture environment:
conda activate hierarchicaldet
conda env export > environment_frozen.yml
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi --query-gpu=name,driver_version --format=csv
```

---

## Dataset

| Field | Value |
|-------|-------|
| Source | https://huggingface.co/datasets/ibrahimhamamci/DENTEX |
| Download date | |
| Hugging Face commit | |
| training_data.zip SHA-256 | |
| validation_data.zip SHA-256 | |
| test_data.zip SHA-256 | |
| validation_triple.json SHA-256 | |
| Tier 1 train images (found) | / 693 (paper) |
| Tier 2 train images (found) | / 634 (paper) |
| Tier 3 train images (found) | / 705 (paper) |
| Tier 3 val images (found) | / 50 (paper) |
| Tier 3 test images (found) | / 250 (paper) |

```bash
# To compute SHA-256:
sha256sum DENTEX/training_data.zip
sha256sum DENTEX/validation_data.zip
sha256sum DENTEX/test_data.zip
sha256sum DENTEX/validation_triple.json
```

---

## Deviations from the Paper (MUST disclose)

1. **No pretrained HierarchicalDet weights released.** Trained from scratch.
2. **Standard ImageNet-22k Swin-B backbone used** (`swin_base_patch4_window7_224_22k.pkl`) instead of the paper's custom dental X-ray pretrained Swin-B. This is expected to lower performance relative to the paper.
3. **Config used:** `diffdet.custom.swinbase.nonpretrain.yaml` (not `diffdet.custom.swinbase.enumeration.yaml`)
4. **Patched scripts used:** `train_net_patched.py` and `dataset_mapper_patched.py` replace the originals to fix hardcoded paths. Logic is identical.
5. **Dataset version:** Public Hugging Face release may differ slightly from the challenge version used in the original paper.
6. **Baseline configs not provided** by the official repo — we use standard detectron2 configs adapted to the DENTEX dataset.

---

## Training Runs

### Tier 1: Quadrant Detection

| Field | Value |
|-------|-------|
| Config | `configs/diffdet.custom.swinbase.nonpretrain.yaml` |
| Backbone weights | `models/swin_base_patch4_window7_224_22k.pkl` |
| Backbone weights SHA-256 | |
| DENTEX_TIER | tier1 |
| USE_NOISY_BOXES | 0 (disabled) |
| SEED | 40244023 |
| MAX_ITER | 40000 |
| IMS_PER_BATCH | 2 |
| NUM_PROPOSALS | 1000 |
| SAMPLE_STEP | 1 |
| Start time | |
| End time | |
| Wall time (h) | |
| GPU-hours | |
| Final checkpoint | `output/tier1/model_final.pth` |
| Checkpoint SHA-256 | |
| Training command | `bash run_training.sh` (tier 1 section) |

### Tier 2: Quadrant-Enumeration Detection

| Field | Value |
|-------|-------|
| Config | `configs/diffdet.custom.swinbase.nonpretrain.yaml` |
| Backbone/init weights | `output/tier1/model_final.pth` (tier-1 checkpoint) |
| DENTEX_TIER | tier2 |
| USE_NOISY_BOXES | 1 |
| NOISY_BOX_TRAIN | `noisy_boxes/tier1_train_boxes.json` |
| NOISY_BOX_VAL | `noisy_boxes/tier1_val_boxes.json` |
| NOISY_BOX_THRESH | 0.5 |
| SEED | 40244023 |
| MAX_ITER | 40000 |
| Start time | |
| End time | |
| Wall time (h) | |
| Final checkpoint | `output/tier2/model_final.pth` |
| Checkpoint SHA-256 | |

### Tier 3: Full Diagnosis Detection

| Field | Value |
|-------|-------|
| Config | `configs/diffdet.custom.swinbase.nonpretrain.yaml` |
| Backbone/init weights | `output/tier2/model_final.pth` (tier-2 checkpoint) |
| DENTEX_TIER | tier3 |
| USE_NOISY_BOXES | 1 |
| NOISY_BOX_TRAIN | `noisy_boxes/tier2_train_boxes.json` |
| NOISY_BOX_VAL | `noisy_boxes/tier2_val_boxes.json` |
| NOISY_BOX_THRESH | 0.5 |
| SEED | 40244023 |
| MAX_ITER | 40000 |
| Start time | |
| End time | |
| Wall time (h) | |
| Final checkpoint | `output/tier3/model_final.pth` |
| Checkpoint SHA-256 | |

---

## Evaluation Results

### Primary Results — HierarchicalDet (Ours, standard backbone)

| Tier | AP | AP50 | AP75 | Per-class AP | Runtime (GPU ms/img) |
|------|----|------|------|--------------|---------------------|
| Quadrant | | | | | |
| Enumeration | | | | | |
| Diagnosis | | | | | |

### Paper Reference Values (Table 1 from arXiv:2303.06500)

> Note: Exact numbers may depend on dataset version and custom pretrained backbone.
> Treat these as approximate targets, not ground truth.

| Model | Quadrant AP50 | Enum AP50 | Diagnosis AP50 |
|-------|--------------|-----------|----------------|
| HierarchicalDet (paper) | ~68.3 | ~48.4 | ~55.0 |
| RetinaNet (paper) | ~45.2 | ~29.1 | ~38.7 |
| Faster R-CNN (paper) | ~51.3 | ~33.8 | ~41.2 |
| DiffusionDet (paper) | ~59.1 | ~39.7 | ~47.6 |

> Paper values are estimated from the figure/table — actual numbers
> should be verified directly from the published paper.

### Baseline Results (Ours)

| Model | Quadrant AP50 | Enum AP50 | Diagnosis AP50 |
|-------|--------------|-----------|----------------|
| RetinaNet | | | |
| Faster R-CNN | | | |
| DiffusionDet (no hierarchy) | | | |

### Per-Class AP (Diagnosis Tier)

| Class | Paper AP | Ours AP |
|-------|----------|---------|
| Impacted | | |
| Caries | | |
| Periapical Lesion | | |
| Deep Caries | | |

---

## Runtime Benchmark

| Setting | Wall ms/img | GPU ms/img | Peak GPU MiB | Failures/50 |
|---------|------------|------------|--------------|-------------|
| GPU (tier3, step=1) | | | | |
| CPU (tier3, step=1) | | | | |

### Diffusion Step Sensitivity

| SAMPLE_STEP | Wall ms/img | Failure rate |
|-------------|------------|--------------|
| 1 | | |
| 2 | | |
| 4 | | |
| 8 | | |
| 16 | | |

---

## Failure Case Log

Document here any images where inference crashes, produces no detections,
or produces clearly wrong outputs.

| image_id | file_name | failure_type | tier | notes |
|----------|-----------|--------------|------|-------|
| | | | | |

---

## Phase 3 Extension Results (fill during Phase 3)

### Image Degradation Robustness

| Degradation | Severity | Diagnosis AP50 |
|-------------|----------|----------------|
| None (clean) | — | |
| Gaussian blur | σ=2 | |
| Gaussian blur | σ=5 | |
| JPEG compression | Q=50 | |
| JPEG compression | Q=20 | |
| Resolution reduction | 50% | |
| Resolution reduction | 25% | |

### Stress-Test vs Clean Subset

| Subset | N images | Diagnosis AP50 | Failure rate |
|--------|---------|----------------|--------------|
| Clean (all3 tiers) | | | |
| Stress-test (partial) | | | |

---

## Observations and Anomalies

_Record anything unexpected here — dataset loading errors, loss spikes,
evaluation discrepancies, annotation quality issues._

- 

---

## Commands Reference

```bash
# Full training (all 3 tiers)
export DATA_ROOT=/path/to/sorted/challenge
export NUM_GPUS=1
export BACKBONE_WEIGHTS=models/swin_base_patch4_window7_224_22k.pkl
bash run_training.sh 2>&1 | tee logs/training_full.log

# Evaluation
bash run_evaluation.sh 2>&1 | tee logs/evaluation.log

# Baselines
bash run_baselines.sh 2>&1 | tee logs/baselines.log

# Runtime benchmark (GPU)
python phase2_runtime_benchmark.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --weights output/tier3/model_final.pth \
    --tier 2 --n-images 50 --out results/runtime_gpu.json

# Runtime benchmark (CPU, slow)
python phase2_runtime_benchmark.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --weights output/tier3/model_final.pth \
    --tier 2 --n-images 10 --device cpu --out results/runtime_cpu.json

# Diffusion step sweep
python phase2_runtime_benchmark.py \
    --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
    --weights output/tier3/model_final.pth \
    --tier 2 --n-images 20 \
    --sweep-sample-steps 1 2 4 8 16 \
    --out results/diffusion_step_sweep.json

# Dataset audit
python phase1_audit_dataset.py --data-root /path/to/extracted --val-json /path/to/validation_triple.json

# Visual verification
python phase1_visualize_annotations.py \
    --json /path/to/validation_triple.json \
    --img-dir /path/to/val_images \
    --n 5 --out-dir phase1_visuals/

# Collect all results
python phase2_collect_results.py \
    --eval-dirs results/tier1_eval results/tier2_eval results/tier3_eval \
    --tier-names "Quadrant" "Enumeration" "Diagnosis" \
    --log-files logs/eval_tier1.log logs/eval_tier2.log logs/eval_tier3.log \
    --baseline-dirs results/retinanet_eval results/fasterrcnn_eval \
    --baseline-names RetinaNet "Faster R-CNN" \
    --out results/full_comparison.json
```
