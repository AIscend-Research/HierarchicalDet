# Deviations from the original paper and released code

Auto-accumulated by `src.setup_env.log_deviation`. Every entry was
written by a notebook that actually executed; nothing here is planned
or hypothetical.
- **01_setup_and_data** — test-split ground truth missing entirely for ['Deep Caries']
  - Why: The released DENTEX test annotations use a 9-code clinical labelling scheme and contain NO 'Deep Caries' label: every carious tooth is plain 'çürük' (code 1). The public test release therefore cannot distinguish Caries from Deep Caries, so test-split evaluation covers 3 of the paper's 4 diagnosis classes and the Deep Caries column has no ground truth. This is a property of the data release, not of this conversion; the authors' own test file (test_merged_disease_coco3class.json) was never published.
  - Impact on results: test-split evaluation covers 3 of 4 diagnosis classes; the missing class contributes no ground truth, so its per-class AP is undefined and the diagnosis-tier mean AP averages over the classes that have ground truth
- **01_setup_and_data** — DENTEX license discrepancy (CC BY-SA on GitHub vs CC BY-NC-SA on HuggingFace)
  - Why: the two published statements disagree; the stricter non-commercial reading is assumed
- **03_evaluate_and_build_assets** — diffusion step sweep runs without cross-timestep ensembling
  - Why: the released ensemble path is unrunnable in the multi-label fork: inference() returns single-label variable names that do not exist here, and ddim_sample then treats the aggregate as both a tensor and a per-tier list. N-step denoising itself is unaffected
  - Impact on results: AP-vs-steps measures denoising depth alone, without the multi-timestep detection ensemble the upstream single-label model would apply
- **03_evaluate_and_build_assets** — inference-time prior-tier box injection used for the fault-injection experiment
  - Why: the released code injects prior-tier boxes only during training; the inference-time path exists in detector.py but is entirely commented out and referenced unpublished paths. It is off for every other experiment, so no main-table number depends on it
