# Phase 0 Findings: HierarchicalDet Reproduction Audit

Date: 2026-06-07  
Auditor: Christopher Huang  

---

## 1. Paper Analysis — Feasible Computational Claims

### What is reproducible
- **Inference pipeline**: The full code is public (GitHub). `demo.py` and `train_net.py` are the entry points.
- **Training from scratch**: The `diffdet.custom.swinbase.nonpretrain.yaml` config uses a publicly available Swin-B ImageNet-22k backbone (`swin_base_patch4_window7_224_22k`), making full training reproducible without the authors' custom pretrained backbone.
- **Dataset**: The DENTEX dataset (all three annotation tiers) is publicly available on Hugging Face (`ibrahimhamamci/DENTEX`, 11.8 GB).
- **Evaluation metrics**: COCO-style mAP evaluation is implemented in `hierarchialdet/util/coco_3class_eval.py`.
- **Architecture**: Swin-B backbone + FPN + DiffusionDet head with hierarchical multi-label output `NUM_CLASSES: [4, 8, 4]` (4 quadrants, 8 tooth types per quadrant, 4 diagnoses).

### What is NOT reproducible as described in the paper
- **Pretrained HierarchicalDet weights**: Not released. No checkpoint files are available in the repo or on Hugging Face.
- **Custom pretrained Swin-B backbone**: The `diffdet.custom.swinbase.enumeration.yaml` config (used for the tooth enumeration tier) references `../Swin-Transformer/output_dental_pretrain2/train_smim/ckpt_epoch_99.pth` — a custom dental X-ray pretrained backbone that is NOT publicly available. This means the exact weights from the paper cannot be reproduced without rerunning that pretraining step.
- **Exact dataset version**: The paper may use a slightly different annotation version than what is currently on Hugging Face (the DENTEX challenge data was updated for MICCAI 2023).
- **Baseline model results**: RetinaNet, Faster R-CNN, DETR, and DiffusionDet baseline configs are not provided in the repo.

### Decision: Training from scratch with standard backbone
Use `diffdet.custom.swinbase.nonpretrain.yaml` with the publicly available `swin_base_patch4_window7_224_22k.pkl` backbone. This is the honest path. Results will likely differ slightly from the paper (which used a domain-pretrained backbone) — this deviation must be disclosed in the report.

---

## 1b. Additional Blockers Found During Phase 1 Audit (2026-06-07)

### BLOCKER 2: dataset_mapper.py crashes on init — hardcoded noisy-box paths
`hierarchialdet/dataset_mapper.py.__init__()` immediately tries to `open()` two pre-computed inference JSON files that are NOT in the public release:
```
ibrahim/Diseasedataset_base_enumeration_m_t_inference_train/inference/coco_instances_results.json
ibrahim/Diseasedataset_base_enumeration_m_t_inference_val/inference/coco_instances_results.json
```
These are COCO results from running the enumeration-tier model on train/val, used as noisy boxes for the diagnosis-tier training. Without them, the mapper crashes before loading any data.

**Fix**: Use `dataset_mapper_patched.py`, which reads paths from `NOISY_BOX_TRAIN` / `NOISY_BOX_VAL` env vars and falls back to GT-only mode when not set (correct for tier-1 training).

### BLOCKER 3: Non-standard COCO annotation format
DENTEX annotations use a non-standard triple-category format:
- `categories_1` / `category_id_1`: quadrant (1–4)
- `categories_2` / `category_id_2`: tooth enumeration (1–8, 0 = not annotated)  
- `categories_3` / `category_id_3`: diagnosis (1–4, 0 = not annotated; 1=Impacted, 2=Caries, 3=PeriapicalLesion, 4=DeepCaries)

Standard COCO tools and detectron2 expect a single `categories` key. The bundled detectron2 has been modified to handle this, but any external evaluation tools (e.g., pycocotools directly) will fail unless they use the bundled version.

### Finding: Validation split HAS ground truth labels
`validation_triple.json` (50 images, 182 annotations) contains full three-tier labels, contradicting the original README which implies validation labels are withheld for the challenge. Use this for primary mAP evaluation.

### Finding: License is CC-BY-NC-SA 4.0 (not CC BY-SA 4.0)
The dataset license is **CC-BY-NC-SA 4.0** (non-commercial, share-alike). This must be cited correctly in the paper and constrains deployment.

### Finding: Federated loss uses hardcoded class frequencies
`loss.py` embeds hardcoded frequency weights for the fed loss:
```python
class_freq[0] = torch.tensor([247, 195, 3051, 865])   # diagnosis
class_freq[1] = torch.tensor([628, 623, 618, 588, 565, 565, 578, 341])  # enumeration
class_freq[2] = torch.tensor([693, 693, 693, 693])    # quadrant (uniform)
```
These were computed from the training data but are not derived at runtime. If the dataset distribution differs from these counts, the fed loss reweighting will be suboptimal.

### DENTEX dataset file structure (confirmed)
Files available for download on Hugging Face:
```
DENTEX/training_data.zip    10.9 GB
DENTEX/validation_data.zip   150 MB
DENTEX/test_data.zip         765 MB
DENTEX/validation_triple.json  64.4 kB  ← ground-truth labels for 50 val images
```
Zip contents unknown until extracted — annotation JSONs and image directories are inside.

---

## 2. Repository Audit

**Official repo**: https://github.com/ibrahimethemhamamci/HierarchicalDet  

### File structure
```
HierarchicalDet/
├── configs/
│   ├── Base-DiffusionDet.yaml            # Base config (ResNet-50 backbone, 270k iters)
│   ├── diffdet.custom.swinbase.enumeration.yaml   # Enumeration tier (needs custom backbone)
│   └── diffdet.custom.swinbase.nonpretrain.yaml   # All-tier (standard Swin-B, 40k iters) ← USE THIS
├── demo.py                               # Single-image inference entry point
├── train_net.py                          # Training + eval entry point
├── evaluator.py                          # Dataset evaluator
├── hierarchialdet/                       # Core model package
│   ├── config.py                         # DiffusionDet config extensions
│   ├── detector.py                       # DiffusionDet model
│   ├── dataset_mapper.py                 # Data loading + noisy box manipulation
│   ├── head.py                           # Detection head
│   ├── swintransformer.py                # Swin-B backbone
│   ├── predictor.py                      # Inference visualizer
│   └── util/
│       ├── coco_3class_eval.py           # Custom COCO evaluator for 3 label tiers
│       └── model_ema.py                  # EMA weight averaging
├── detectron2/                           # BUNDLED detectron2 (not pip-installed)
└── pycocotools/                          # BUNDLED pycocotools
```

### Critical observation: detectron2 is bundled
The repo ships its own copy of `detectron2/` and `pycocotools/`. Do NOT `pip install detectron2` separately — use what's in the repo.

### No requirements.txt
There is no `requirements.txt`. Dependencies must be installed manually (see setup script).

---

## 3. Setup Instructions — CORRECTED

The roadmap's setup commands contain two errors:
1. `pip install -r requirements.txt` — there is no requirements.txt
2. The standard `pip install detectron2` would conflict with the bundled version

**Correct setup for a CUDA 12.1 GPU machine:**
See `setup.sh` in this repo.

**This machine (Mac, no GPU)**: Environment setup must run on a GPU server or cloud instance. CPU-only inference is possible but very slow.

---

## 4. Dataset Audit

**Source**: https://huggingface.co/datasets/ibrahimhamamci/DENTEX  
**Size**: 11.8 GB total  
**License**: CC BY-SA 4.0  

### Three annotation tiers (confirmed present)
| Tier | Images | Labels |
|------|--------|--------|
| Quadrant only | 693 | 4 quadrant classes |
| Quadrant + enumeration | 634 | Quadrant + tooth number (FDI system) |
| Quadrant + enumeration + diagnosis | 1005 | Full hierarchy: Q + N + D (caries, deep caries, periapical lesions, impacted teeth) |

### Evaluation split (fully annotated tier)
| Split | Images |
|-------|--------|
| Train | 705 |
| Val | 50 |
| Test | 250 |

Additional: 1,571 unlabeled X-rays available for pretraining (not needed for reproduction).

### Known issue
Hugging Face dataset viewer reports a format inconsistency: "Couldn't infer the same data file format for all splits." Train/test use imagefolder format; val uses JSON. Use the download script (`download_dataset.sh`) instead of the Hugging Face streaming API.

---

## 5. Pretrained Weights — CONFIRMED NOT RELEASED

No pretrained HierarchicalDet weights are available publicly.

**Implication**: The team must train from scratch. Estimated compute:
- Config: `diffdet.custom.swinbase.nonpretrain.yaml`
- Iterations: 40,000
- Batch size: 2 images/iter
- Backbone: Swin-B (requires ~8 GB VRAM minimum; 16 GB recommended)
- Estimated time: ~12–20 hours on a single A100 or V100

**Backbone weights that ARE available:**
- `swin_base_patch4_window7_224_22k.pkl` — standard ImageNet-22k pretrained Swin-B. Download from the Swin Transformer official release and convert to detectron2 pickle format using the script in `train_net.py` comments.

---

## 6. Model Configuration Summary

From `diffdet.custom.swinbase.nonpretrain.yaml`:

| Parameter | Value |
|-----------|-------|
| Backbone | Swin-B (ImageNet-22k pretrained) |
| FPN inputs | swin0–swin3 |
| NUM_CLASSES | `[4, 8, 4]` (quadrant, enumeration, diagnosis) |
| NUM_PROPOSALS | 1000 |
| USE_FED_LOSS | True (federated loss for partial annotations) |
| SAMPLE_STEP | 1 (diffusion steps at inference) |
| MAX_ITER | 40,000 |
| EVAL_PERIOD | 1,000 iterations |
| Optimizer | AdamW |

**Key insight on `NUM_CLASSES: [4, 8, 4]`**: The model predicts three label groups simultaneously per detected box — 4 quadrant classes, 8 enumeration classes (tooth types within a quadrant), and 4 diagnosis classes. This is the hierarchical multi-label structure.

---

## 7. Inference Entry Point (demo.py)

Example command structure (from code inspection):
```bash
python demo.py \
  --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
  --input path/to/xray.jpg \
  --output output/ \
  --confidence-threshold 0.5 \
  --nclass 3 \
  MODEL.WEIGHTS path/to/checkpoint.pth
```

`--nclass` controls which tier is being run (1=quadrant, 2=enumeration, 3=diagnosis). The `k=(args.nclass-1)` in the code selects the label tier.

---

## 8. Hardcoded Data Paths (Reproducibility Risk)

`train_net.py` registers datasets with hardcoded paths:
```python
register_coco_instances('custom_train_class', {}, 
    "../sorted/challenge/train_merged_disease_coco3class_onlyd_fixed.json", 
    "../sorted/challenge/for_coco_disease_train")
register_coco_instances('custom_validation_class', {},
    "../sorted/challenge/test_merged_disease_coco3class.json",
    "../sorted/challenge/for_coco_disease_test")
```

The repo must be cloned such that `../sorted/challenge/` exists relative to it, or these paths must be modified. **We will modify `train_net.py`** to use configurable paths (environment variables or command-line arguments) rather than hardcoded relative paths.

---

## 9. Phase 0 Checklist

- [x] Read the paper and identified feasible computational claims
- [x] Read the GitHub README
- [x] Inspected demo.py, train_net.py, all config files
- [x] Identified that detectron2 is bundled (not pip-installed)
- [x] Confirmed no requirements.txt exists
- [x] Confirmed no pretrained weights are released
- [x] Confirmed DENTEX dataset is publicly available (11.8 GB, Hugging Face)
- [x] Confirmed all three annotation tiers are present in the dataset
- [x] Identified hardcoded data paths as a reproducibility risk
- [x] Confirmed `diffdet.custom.swinbase.nonpretrain.yaml` is the correct config to use
- [x] Documented estimated training compute (40k iters, ~12–20h on A100)
- [ ] Clone official repo and verify it runs on GPU machine
- [ ] Download DENTEX dataset and verify all three tiers load correctly
- [ ] Download standard Swin-B backbone weights
- [ ] Run example inference on a single test image (blocked until training completes)
- [ ] Save terminal output, detection visualizations, and raw score files

## Phase 1 Scripts (ready to run on GPU machine)
- `phase1_audit_dataset.py` — image counts, resolution stats, annotation completeness table
- `phase1_verify_dataloader.py` — dataloader smoke test using patched mapper
- `phase1_select_subsets.py` — select clean (100 imgs) and stress-test (50 imgs) eval subsets
- `phase1_visualize_annotations.py` — 5-image visual verification with bbox overlay

## Phase 1 Usage
```bash
# Step 1: Audit extracted dataset
python phase1_audit_dataset.py \
    --data-root /path/to/extracted/dentex \
    --val-json /path/to/DENTEX/validation_triple.json

# Step 2: Verify dataloader
export DATA_ROOT=/path/to/sorted/challenge
export DENTEX_TIER=tier3
python phase1_verify_dataloader.py --n-samples 10

# Step 3: Select evaluation subsets
python phase1_select_subsets.py \
    --json /path/to/train_merged_disease_coco3class_onlyd_fixed.json \
    --json /path/to/validation_triple.json \
    --out-dir phase1_subsets/

# Step 4: Visual verification (5 images)
python phase1_visualize_annotations.py \
    --json /path/to/validation_triple.json \
    --img-dir /path/to/validation_images \
    --n 5 \
    --out-dir phase1_visuals/
```
