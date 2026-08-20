# Scope, claims and what a reviewer can re-verify

Run mode: **micro**. Measured GPU-hours (training records + evaluation results): **7.27**.

## Claim coverage

| # | Claim in the original paper | Status | Evidence |
|---|---|---|---|
| 1 | The hierarchical multi-label model beats base DiffusionDet at every tier. | cited, untested | not trained in RUN_MODE=micro (missing: DiffusionDet_base_tier2) |
| 2 | Noisy-box manipulation is the component that contributes the accuracy gain. | undertrained, not conclusive | notebook 03 (matched budgets) + notebook 05; Ours_full vs Ours_wo_Manipulation; every arm is far below the reported scale (Ours_full AP 4.97 vs 37.6 reported; Ours_wo_Manipulation AP 2.78 vs 36.3 reported), so the ordering cannot be separated from undertraining |
| 3 | Weight transfer alone does not improve accuracy. | cited, untested | not trained in RUN_MODE=micro (missing: Ours_wo_Manip_Transfer) |
| 4 | The full model outperforms RetinaNet, Faster R-CNN and DETR at every tier. | cited, untested | not trained in RUN_MODE=micro (missing: RetinaNet, FasterRCNN, DETR) |
| 5 | SimMIM pretraining on 1,571 unlabelled X-rays contributes to the result. | out of scope | the authors' SimMIM checkpoint is not published; notebook opt-11 exists for anyone with spare quota |
| 6 | The reported numbers are obtained on the DENTEX test split (250 images). | tested | notebook 01 converts and verifies the test ground truth; every number in main_results is computed on that split |
| 7 | The approach is robust enough for panoramic X-ray analysis in practice. | partially supported | notebook 06: degradation grid, clean/stress subsets and hierarchy fault injection - none of which the original paper tests; NOT supported by: figure:fault_injection (recorded as not run) |
| 8 | Diffusion sampling steps trade accuracy against latency. | extended beyond the paper | notebook 06 step sweep with both axes measured |

## Compute actually spent

| source | GPU-hours | provenance |
|---|---|---|
| training runs (retained run records) | 5.31 | measured; `run_record.json` per stage |
| evaluation runs in `results_raw/` | 1.96 | measured; every run records its own `wall_seconds` |
| total measured | 7.27 | sum of the two rows above |

**Provenance: the run records below document the retraining written at 2026-08-20T06:50:02+0000; every evaluation number in this build was produced earlier (written at 2026-08-16T10:41:49+0000) from checkpoints whose training records were not retained. Records and numbers describe DIFFERENT checkpoints until notebook 03 is re-run against the retained runs.**

| run | wall (s) | GPUs |
|---|---|---|
| quadrant_stage | 3101.4 | 2 |
| enumeration_stage | 3253.8 | 2 |
| diagnosis_full | 4151.6 | 2 |
| diagnosis_wo_manipulation | 4306.2 | 2 |
| diagnosis_wo_transfer | 4295.5 | 2 |

## Re-verifying without retraining

Attach `dentex-repro-ckpts` and `dentex-repro-data`, open `notebooks/03_evaluate_and_build_assets.ipynb` with Accelerator = GPU T4, set `RUN_MODE = "micro"`, and Run All.

That reproduces every number in `tables/main_results.csv` from the released checkpoints, with no training. Re-running the same notebook on a CPU session rebuilds every table and figure for free.

## Not a clinical claim

Every artifact here is a research artifact produced under a constrained compute budget. None of it is validated for, or intended for, clinical use.

