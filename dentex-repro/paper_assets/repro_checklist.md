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
- multi-GPU: {}

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

| run | config hash | iterations | seed | batch | wall (s) | GPUs | stopped on budget |
|---|---|---|---|---|---|---|---|

## Exact commands


## Remaining nondeterminism

- mixed-precision (AMP) reduction order during training
- atomics in the torchvision ROIAlign / NMS CUDA kernels
- DataLoader worker interleaving (NUM_WORKERS=2)
- inference starts from random noisy boxes, which is why every reported number is a mean over 3 inference seeds

