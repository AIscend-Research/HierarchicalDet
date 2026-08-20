# Reproducibility checklist

Generated from executed-run records only.

## Environment

- repo commit: `0ae20744c59f3ffa39f31c638bcdb5fd9c83dfc3`
- python: 3.12.13
- platform: Linux-6.12.90+-x86_64-with-glibc2.35
- GPU: [{"index": 0, "name": "Tesla T4", "total_memory_gb": 14.56, "capability": "7.5"}, {"index": 1, "name": "Tesla T4", "total_memory_gb": 14.56, "capability": "7.5"}]
- CUDA / cuDNN: 12.8 / 91002
- run mode: **micro**
- training seed: 40244023 (the repo's own SEED)
- inference seeds: [0, 1, 2]
- multi-GPU: {"requested": 2, "works": true, "error": null}

## Converted dataset

- `flat_tier0_test.json`: `ff60d750379c5a688a9153a27c310ba1164608a97e1310b43d72cf7b962ea3b0`
- `flat_tier0_train.json`: `4bc04823ffc278a9fe432001d252fe0cf652173cd7b0cc78873b9c9afbea1f5e`
- `flat_tier1_test.json`: `2b5a77aef30713a41a99644ca59d2e73445612a2f83d9b3a6a7ee0b607c39fbf`
- `flat_tier1_train.json`: `acdabc303b4c5eb3448be2f47c9defd7b6c61e0c82971871ff5d6b965ebd33c4`
- `flat_tier2_test.json`: `781d26f5c5b4dee9262d61f0085cb40338931246b62b9d0a7503f5359d2a2aa5`
- `flat_tier2_train.json`: `897a61d4a0e4532ed5bdf4dc36ee02324ea571ff6330d2472db67071dfa3fdb8`
- `test_diagnosis.json`: `9e0af81132f46c2517bd6f8e8379c27cec02053cc7ebbdbb202d625d62fb6a68`
- `train_diagnosis.json`: `6e1f702cfd6c83bc63660d9a0fd79fe5b26d5a54b63d93b954a02cc254314483`
- `train_enumeration.json`: `ac924d8bd52ccc5a82d8bf8e6b9447dddcdb8ab4fa8f7245f300a87441a331d7`
- `train_quadrant.json`: `4bc04823ffc278a9fe432001d252fe0cf652173cd7b0cc78873b9c9afbea1f5e`
- `val_diagnosis.json`: `d058afd35d2849923c7c045e61fd3e05d231dcf74d55009993fff88bd9b6f5a2`

## Runs

**Provenance: the run records below document the retraining written at 2026-08-20T06:50:02+0000; every evaluation number in this build was produced earlier (written at 2026-08-16T10:41:49+0000) from checkpoints whose training records were not retained. Records and numbers describe DIFFERENT checkpoints until notebook 03 is re-run against the retained runs.**

| run | config hash | iterations | seed | batch | wall (s) | GPUs | stopped on budget |
|---|---|---|---|---|---|---|---|
| quadrant_stage | `7c74c8f06efa` | 600 | 40244023 | 2 | 3101.4 | 2 | False |
| enumeration_stage | `89ed50e90e33` | 600 | 40244023 | 2 | 3253.8 | 2 | False |
| diagnosis_full | `87a8386638c9` | 900 | 40244023 | 2 | 4151.6 | 2 | False |
| diagnosis_wo_manipulation | `f4418137b3d6` | 900 | 40244023 | 2 | 4306.2 | 2 | False |
| diagnosis_wo_transfer | `375d5feecaac` | 900 | 40244023 | 2 | 4295.5 | 2 | False |

## Exact commands

```
/usr/bin/python3 /kaggle/working/repo/dentex-repro/src/train_entry.py --config-file /kaggle/working/repo/dentex-repro/configs_repro/diffdet.dentex.quadrant.yaml --num-gpus 2 --resume --budget-seconds 5960.1 --disk-floor-gb 3.0 --heartbeat /kaggle/working/runs/micro/quadrant_stage/heartbeat.json OUTPUT_DIR /kaggle/working/runs/micro/quadrant_stage MODEL.WEIGHTS /kaggle/working/repo/models/swin_base_patch4_window7_224_22k.pth SOLVER.MAX_ITER 600 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 150 SOLVER.AMP.ENABLED True MODEL_EMA.ENABLED False TEST.EVAL_PERIOD 0 SEED 40244023
```
```
/usr/bin/python3 /kaggle/working/repo/dentex-repro/src/train_entry.py --config-file /kaggle/working/repo/dentex-repro/configs_repro/diffdet.dentex.enumeration.yaml --num-gpus 2 --resume --budget-seconds 6300.0 --disk-floor-gb 3.0 --heartbeat /kaggle/working/runs/micro/enumeration_stage/heartbeat.json OUTPUT_DIR /kaggle/working/runs/micro/enumeration_stage MODEL.WEIGHTS /kaggle/working/runs/micro/quadrant_stage/model_final.pth SOLVER.MAX_ITER 600 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 150 SOLVER.AMP.ENABLED True MODEL_EMA.ENABLED False TEST.EVAL_PERIOD 0 SEED 40244023
```
```
/usr/bin/python3 /kaggle/working/repo/dentex-repro/src/train_entry.py --config-file /kaggle/working/repo/dentex-repro/configs_repro/diffdet.dentex.diagnosis.yaml --num-gpus 2 --resume --budget-seconds 9900.0 --trajectory {"300": "traj_033", "600": "traj_067"} --disk-floor-gb 3.0 --heartbeat /kaggle/working/runs/micro/diagnosis_full/heartbeat.json OUTPUT_DIR /kaggle/working/runs/micro/diagnosis_full MODEL.WEIGHTS /kaggle/working/runs/micro/enumeration_stage/model_final.pth SOLVER.MAX_ITER 900 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 225 SOLVER.AMP.ENABLED True MODEL_EMA.ENABLED False TEST.EVAL_PERIOD 0 SEED 40244023
```
```
/usr/bin/python3 /kaggle/working/repo/dentex-repro/src/train_entry.py --config-file /kaggle/working/repo/dentex-repro/configs_repro/diffdet.dentex.diagnosis.yaml --num-gpus 2 --resume --budget-seconds 9900.0 --trajectory {"300": "traj_033", "600": "traj_067"} --disk-floor-gb 3.0 --heartbeat /kaggle/working/runs/micro/diagnosis_wo_manipulation/heartbeat.json OUTPUT_DIR /kaggle/working/runs/micro/diagnosis_wo_manipulation MODEL.WEIGHTS /kaggle/working/runs/micro/enumeration_stage/model_final.pth SOLVER.MAX_ITER 900 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 225 SOLVER.AMP.ENABLED True MODEL_EMA.ENABLED False TEST.EVAL_PERIOD 0 SEED 40244023
```
```
/usr/bin/python3 /kaggle/working/repo/dentex-repro/src/train_entry.py --config-file /kaggle/working/repo/dentex-repro/configs_repro/diffdet.dentex.diagnosis.yaml --num-gpus 2 --resume --budget-seconds 9900.0 --trajectory {"300": "traj_033", "600": "traj_067"} --disk-floor-gb 3.0 --heartbeat /kaggle/working/runs/micro/diagnosis_wo_transfer/heartbeat.json OUTPUT_DIR /kaggle/working/runs/micro/diagnosis_wo_transfer MODEL.WEIGHTS /kaggle/working/repo/models/swin_base_patch4_window7_224_22k.pth SOLVER.MAX_ITER 900 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 225 SOLVER.AMP.ENABLED True MODEL_EMA.ENABLED False TEST.EVAL_PERIOD 0 SEED 40244023
```

## Remaining nondeterminism

- mixed-precision (AMP) reduction order during training
- atomics in the torchvision ROIAlign / NMS CUDA kernels
- DataLoader worker interleaving (NUM_WORKERS=2)
- inference starts from random noisy boxes, which is why every reported number is a mean over 3 inference seeds

