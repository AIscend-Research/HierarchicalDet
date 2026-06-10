# HierarchicalDet Reproducibility Checklist

MLRC 2023 — "Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays"  
Paper: MICCAI 2023 (Hamamci et al.)  
Reproduced by: [your name] · [affiliation]

---

## 0  Environment

| # | Item | Status | Notes |
|---|------|--------|-------|
| 0.1 | Python version matches original (3.9.x) | ☐ | |
| 0.2 | CUDA and GPU model recorded | ☐ | |
| 0.3 | PyTorch version recorded (torch.__version__) | ☐ | |
| 0.4 | detectron2 is the **bundled** copy (`pip install -e .`), NOT `pip install detectron2` | ☐ | Critical — different API |
| 0.5 | Swin-B backbone file present and SHA-256 recorded | ☐ | |
| 0.6 | Swin-B checkpoint converted to detectron2 pickle format | ☐ | Via setup.sh |
| 0.7 | All dependencies installed (fvcore, iopath, timm, scipy, omegaconf, pycocotools) | ☐ | No requirements.txt in original repo |
| 0.8 | Environment snapshot saved (`pip freeze > logs/pip_freeze.txt`) | ☐ | |

**Deviations to disclose**:
- [ ] `requirements.txt` does not exist; dependencies inferred from imports
- [ ] `pip install detectron2` would install a **different** detectron2 version; bundled copy has modified `transform_instance_annotations` signature required for noisy-box injection
- [ ] Swin-B backbone (.pth) downloaded from official SwinTransformer release; custom dental-pretrained backbone referenced in paper is not publicly released

---

## 1  Dataset

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1.1 | DENTEX dataset downloaded from HuggingFace (`ibrahimhamamci/DENTEX`) | ☐ | 11.8 GB |
| 1.2 | License acknowledged: CC-BY-NC-SA 4.0 (non-commercial only) | ☐ | |
| 1.3 | Annotation file counts match paper: 693+634+1005 images | ☐ | phase1_audit_report.txt |
| 1.4 | Tier-1 split (quadrant only): 693/50 train/val | ☐ | |
| 1.5 | Tier-2 split (quad+enum): 634/50 train/val | ☐ | |
| 1.6 | Tier-3 split (full diagnosis): 705/50/250 train/val/test | ☐ | |
| 1.7 | Non-standard COCO format verified: `category_id_1/2/3` fields per annotation | ☐ | |
| 1.8 | `category_id_3=0` means "no diagnosis" (partial annotation) documented | ☐ | |
| 1.9 | Hierarchy consistency checks pass (Q→N→D ordering, valid IDs) | ☐ | phase1_visualize_annotations.py |
| 1.10 | Image resolution distribution recorded | ☐ | phase1_audit_report.txt |
| 1.11 | Clean and stress-test subsets selected and saved | ☐ | phase1_select_subsets.py |

**Deviations to disclose**:
- [ ] Validation set has full ground-truth labels available (used as primary eval set); test set may not have public GT
- [ ] "256×256 data structure" in repo README refers to DENTEX challenge structure, not image size

---

## 2  Training

| # | Item | Status | Notes |
|---|------|--------|-------|
| 2.1 | Config file used recorded: `diffdet.custom.swinbase.nonpretrain.yaml` | ☐ | See deviation note |
| 2.2 | Tier-1 trained for 40k iterations (paper default) | ☐ | |
| 2.3 | Tier-2 trained for 40k iterations | ☐ | |
| 2.4 | Tier-3 trained for 40k iterations | ☐ | |
| 2.5 | Training seed recorded (default: SOLVER.SEED in config) | ☐ | |
| 2.6 | Tier-1 model checkpoint SHA-256 recorded | ☐ | |
| 2.7 | Tier-2 model checkpoint SHA-256 recorded | ☐ | |
| 2.8 | Tier-3 model checkpoint SHA-256 recorded | ☐ | |
| 2.9 | Tier-1 noisy boxes generated before tier-2 training | ☐ | phase2_generate_noisy_boxes.py |
| 2.10 | Tier-2 noisy boxes generated before tier-3 training | ☐ | phase2_generate_noisy_boxes.py |
| 2.11 | Noisy-box score threshold (default 0.5) recorded | ☐ | NOISY_BOX_THRESH env var |
| 2.12 | Federated loss enabled (USE_FED_LOSS=True) for partial annotations | ☐ | |
| 2.13 | Training time per tier recorded (hours) | ☐ | logs/ timestamps |
| 2.14 | GPU utilization and batch size recorded | ☐ | |
| 2.15 | Training loss curves saved | ☐ | detectron2 tensorboard logs |
| 2.16 | No mid-training restarts due to OOM or disconnection | ☐ | |

**Deviations to disclose**:
- [ ] Paper uses custom dental-pretrained Swin-B backbone (not publicly available); this reproduction uses standard ImageNet-22k Swin-B (`nonpretrain.yaml`)
- [ ] Hardcoded paths in original `train_net.py` replaced by `train_net_patched.py` with env vars
- [ ] Hardcoded noisy-box files in original `dataset_mapper.py` replaced by `dataset_mapper_patched.py` with env vars and graceful fallback

---

## 3  Evaluation

| # | Item | Status | Notes |
|---|------|--------|-------|
| 3.1 | Tier-1 AP50 (quadrant) within 5% of paper: ~68.3 | ☐ | |
| 3.2 | Tier-2 AP50 (enumeration) within 5% of paper: ~48.4 | ☐ | |
| 3.3 | Tier-3 AP50 (diagnosis) within 5% of paper: ~55.0 | ☐ | |
| 3.4 | Per-class AP recorded for all three tiers | ☐ | phase2_collect_results.py |
| 3.5 | Evaluation uses SAMPLE_STEP=1 (paper default) | ☐ | |
| 3.6 | Evaluation uses EMA weights if MODEL_EMA.ENABLED=True | ☐ | |
| 3.7 | Inference is on validation set (50 images with GT) | ☐ | |
| 3.8 | COCO evaluator used: detectron2 COCOEvaluator | ☐ | |

---

## 4  Baselines

| # | Item | Status | Notes |
|---|------|--------|-------|
| 4.1 | RetinaNet baseline trained on diagnosis tier (single-label) | ☐ | run_baselines.sh |
| 4.2 | Faster R-CNN baseline trained on diagnosis tier | ☐ | run_baselines.sh |
| 4.3 | Non-hierarchical DiffusionDet baseline trained (USE_NOISY_BOXES=0) | ☐ | run_baselines.sh |
| 4.4 | Baseline annotations converted to single category_id format | ☐ | phase2_convert_to_single_label_coco.py |
| 4.5 | Baseline AP50 values compared to paper Table 1 | ☐ | phase2_collect_results.py |
| 4.6 | Hierarchical ablation: confirms noisy-box injection improves mAP | ☐ | |
| 4.7 | DETR baseline attempted or gap documented | ☐ | Requires separate install — see run_baselines.sh comments |

---

## 5  Extensions (Phase 3)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 5.1 | Diffusion step sensitivity: mAP vs SAMPLE_STEP (1, 2, 5, 10) measured | ☐ | phase3_diffusion_step_map.py |
| 5.2 | Diffusion step vs runtime tradeoff plot saved | ☐ | results/diffusion_steps/plot.png |
| 5.3 | Image degradation: clean, blur, JPEG, resize, noise tested | ☐ | phase3_image_degradation.py |
| 5.4 | Image degradation: zero-detection images counted per condition | ☐ | |
| 5.5 | Noisy-box perturbation: jitter, scale, dropout, injection tested | ☐ | phase3_noise_injection.py |
| 5.6 | Noisy-box perturbation: relative detection count and score shift recorded | ☐ | |
| 5.7 | Failure case categories analyzed: missed detections, FP, wrong label | ☐ | phase3_failure_analysis.py |
| 5.8 | Failure cases compared: clean vs stress-test subset | ☐ | |
| 5.9 | Prediction visualization: 3-panel T1/T2/T3 figures generated | ☐ | phase3_visualize_predictions.py |
| 5.10 | Colab T4 feasibility tested: inference-only confirmed feasible | ☐ | phase3_colab_notebook.ipynb |
| 5.11 | Colab T4 inference time and peak GPU memory recorded | ☐ | |

---

## 6  Documentation and Reporting

| # | Item | Status | Notes |
|---|------|--------|-------|
| 6.1 | All deviations from the paper documented | ☐ | experiment_log.md |
| 6.2 | Reproduction code publicly available (GitHub URL) | ☐ | |
| 6.3 | Training commands and environment fully specified | ☐ | run_training.sh |
| 6.4 | All model checkpoints archived (Google Drive / Zenodo) | ☐ | |
| 6.5 | phase0_findings.md documents all setup blockers and resolutions | ☐ | |
| 6.6 | experiment_log.md filled in with actual measured values | ☐ | |
| 6.7 | Deviation severity labeled: Minor / Significant / Major | ☐ | See below |

---

## Deviation Severity Reference

| Deviation | Severity | Impact |
|-----------|----------|--------|
| No `requirements.txt` | Minor | Solved by inspecting imports |
| Bundled detectron2 must be built from source | Significant | Breaks if `pip install detectron2` is used instead |
| Custom dental-pretrained Swin-B not released | **Major** | Results will differ; ImageNet-22k backbone is a weaker starting point |
| Hardcoded paths in `train_net.py` / `dataset_mapper.py` | Significant | Original code crashes without private paths; patched files required |
| DETR baseline not runnable out-of-box | Significant | Requires `facebookresearch/detr/d2` package separate from bundled detectron2; instructions provided in run_baselines.sh but not automated |
| License CC-BY-NC-SA 4.0 (non-commercial) | Note | Use in commercial settings requires permission |

---

## Reproducibility Score (fill in after completion)

| Dimension | Score (1–5) | Justification |
|-----------|-------------|---------------|
| Code availability | | |
| Dataset availability | | |
| Hyperparameter documentation | | |
| Checkpoint availability | | |
| Result match (AP50 within 10%) | | |
| **Overall** | | |

Score guide: 5 = fully reproducible, 4 = minor effort needed, 3 = significant effort, 2 = partially reproducible, 1 = not reproducible

---

## Auditor Sign-off

- Auditor: ___________________________
- Date: ___________________________
- Compute: ___________________________  (GPU model, hours)
- Repository commit: ___________________________
