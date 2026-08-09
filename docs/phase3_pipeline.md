# Phase 3/4 pipeline — what exists, how to run it, what it does not cover

Written 2026-08-09, closing the gap between the Phase 2 state (data loadable,
nothing trained) and a runnable reproduction. Everything below is code that
exists in this repo now; **no training run has been executed yet**, so this
document describes a pipeline, not results.

## Two bugs in the released code that had to be fixed first

Both are in `hierarchialdet/detector.py`, both were found by reading the
inference path rather than by running it, and both block roadmap items
outright.

1. **Multi-step DDIM sampling crashes.** In `ddim_sample`'s box-renewal branch
   the released code calls `self.num_proposal` (the attribute is
   `num_proposals`) and `torch.stack(bbox_pre)` on a list that is
   unconditionally empty. That branch is only reached when
   `MODEL.DiffusionDet.SAMPLE_STEP > 1`, which is why it survives the default
   config — and why the roadmap's diffusion-step sensitivity experiment could
   not have been run at all. Now replenishes `num_proposals - num_remain`
   boxes, matching upstream DiffusionDet.

2. **Inference-time noisy box injection does not exist.** The paper's noisy box
   manipulation is applied at training time through the dataset mapper, but the
   equivalent inference-time path is present only as commented-out lines
   pointing at `ibrahim/...` paths that were never released. The default here
   still reproduces the released behaviour exactly (pure random proposals);
   setting `NOISY_BOX_INFER` opts into injection, and
   `NOISY_BOX_INFER_JITTER` / `NOISY_BOX_INFER_DROP` deliberately corrupt the
   injected boxes, which is what the "is the hierarchy robust to imperfect
   early tiers?" experiment needs.

A third, smaller one in `train_net_patched.py`: the evaluation tier was
hardcoded (`EvalHook(..., 0)` during training, `k=2` in `ema_test`), so a
tier-2 stage would have been scored with only its quadrant head, or a tier-0
stage with heads that have no supervision yet. It now comes from `TIER`.

## The tools

| Script | What it does |
|---|---|
| `tools/normalize_dentex_tiers.py` | (Phase 2) makes the quadrant / enumeration tiers loadable |
| `tools/labelme_to_coco.py` | converts the 250-image test split from LabelMe + Turkish strings to 3-tier COCO |
| `tools/run_curriculum.py` | the 3-stage curriculum: train → dump noisy boxes → train → dump → train |
| `tools/dump_predictions.py` | inference over a split: COCO-results predictions, per-image runtime, failure records |
| `tools/evaluate_tiers.py` | per-tier AP / AP50 / AP75 / per-class AP via the authors' own evaluator |
| `tools/coco_eval_standalone.py` | the same predictions scored by stock `pycocotools`, as a cross-check and for the baselines |
| `tools/degrade_images.py` | blur / JPEG / downscale copies of an eval split |
| `tools/run_experiments.py` | the Phase 4 sweeps, collected into one comparable table |
| `tools/benchmark_resources.py` | parameters, checkpoint size, peak GPU memory, throughput, CPU vs GPU |
| `tools/error_analysis.py` | failure taxonomy per tier + how errors accumulate down the hierarchy |
| `tools/visualize_predictions.py` | GT-vs-prediction overlays with quadrant/tooth/diagnosis labels |
| `tools/baselines/train_baseline.py` | RetinaNet / Faster R-CNN on a single flat tier |
| `tools/baselines/train_detr.py` | DETR via Hugging Face `transformers` |
| `tools/make_results_table.py` | the comparison table |
| `tools/make_repro_manifest.py` | the reproducibility checklist |

`kaggle/kaggle_setup.ipynb` cells 5–10 drive all of it in order.

## Running the curriculum across Kaggle sessions

One stage is ~10.3h at `SOLVER.MAX_ITER=40000` on a T4 (Phase 0 measurement),
so ~31h for three — against a 12h session limit and a 30h weekly quota. The
driver is built for that: a stage whose `model_final.pth` exists is skipped
entirely, an interrupted stage resumes from its last checkpoint, and the
noisy-box dumps are cached. So the procedure is to re-run the same cell each
session with `--output-root` on persistent storage, and check
`curriculum_manifest.json` for where it got to.

Run once with `--dry-run` first: it prints the exact command sequence,
including which checkpoint feeds which stage and which images each noisy-box
dump covers.

`--ims-per-batch 1` is the default here because batch 2, the config's own
setting, hit a real CUDA OOM on a T4 during the Phase 0 benchmark.

## Deliberate deviations, to be stated in the report

- **No pretrained HierarchicalDet weights are published**, so every number will
  come from training done here, under this reproduction's compute budget, not
  from the authors' checkpoints. Any gap against the paper is confounded by
  that and must not be reported as a failure to reproduce on its own.
- **DETR is the Hugging Face implementation**, since the authors released no
  DETR config. It reproduces the comparison, not the authors' DETR run.
- **The evaluation reported twice**: the repo's forked evaluator and stock
  `pycocotools`. If they disagree, the disagreement is a finding about the
  fork, and both numbers go in the report.
- **The test split's labels were parsed, not received.** The Turkish diagnosis
  strings are mapped by an explicit alias table; unrecognized strings abort the
  conversion instead of being dropped. The parse report from `--report-only`
  should be kept as evidence.
- The cross-tier image overlap documented in `docs/phase1_dataset_audit.md`
  (38% of test images appear in a training tier) is inherited from the dataset
  release and affects every number produced here.

## Not covered

- Nothing has been trained or evaluated yet — the smoke values in the notebook
  (`MAX_ITER=200`, 1 DETR epoch) exist to prove the chain runs, not to produce
  numbers.
- SimMIM self-supervised pretraining on the 1,571 unlabelled images (the paper
  does this before the hierarchical training; the released config uses plain
  ImageNet-22k Swin-B weights instead, so this reproduction follows the config,
  not the paper's description).
- Test-time augmentation (`TEST.AUG`) is available in the codebase but is not
  part of any pipeline here.
