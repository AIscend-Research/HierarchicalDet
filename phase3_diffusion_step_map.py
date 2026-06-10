"""
Phase 3 — Diffusion Step Sensitivity: mAP vs Compute Tradeoff

Evaluates detection accuracy (mAP) AND inference speed at multiple
SAMPLE_STEP values. Phase 2's runtime benchmark only measures timing;
this script adds full COCO mAP evaluation per step count.

The paper uses SAMPLE_STEP=1 (one denoising step). Standard DiffusionDet
uses more steps (4–16). This experiment tests:
  - Does accuracy improve significantly with more steps?
  - What is the speed-accuracy Pareto frontier?
  - Is SAMPLE_STEP=1 a reasonable default for clinical deployment?

Usage:
    export DATA_ROOT=/path/to/sorted/challenge
    python phase3_diffusion_step_map.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier3/model_final.pth \
        --tier 2 \
        --steps 1 2 4 8 \
        --dataset custom_validation_class \
        --out results/diffusion_step_map.json

Outputs:
    results/diffusion_step_map.json  — AP/AP50/AP75 + timing per step count
    results/diffusion_step_map.png   — plot of mAP vs steps vs runtime
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file",  required=True)
    p.add_argument("--weights",      required=True)
    p.add_argument("--tier",         type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--steps",        type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--dataset",      default="custom_validation_class")
    p.add_argument("--n-warmup",     type=int, default=3)
    p.add_argument("--out",          default="results/diffusion_step_map.json")
    p.add_argument("--no-plot",      action="store_true")
    return p.parse_args()


def build_cfg_for_step(config_file, weights, sample_step):
    from detectron2.config import get_cfg
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = sample_step
    cfg.freeze()
    return cfg


def evaluate_at_step(config_file, weights, sample_step, tier, dataset_name, n_warmup):
    """
    Run full COCO evaluation at a given SAMPLE_STEP.
    Returns dict with mAP metrics and per-image timing.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.data import build_detection_test_loader
    from hierarchialdet.util.model_ema import may_get_ema_checkpointer, \
        EMADetectionCheckpointer
    from hierarchialdet.util.coco_3class_eval import COCOEvaluator
    from evaluator import inference_on_dataset
    from dataset_mapper_patched import DiffusionDetDatasetMapper

    cfg = build_cfg_for_step(config_file, weights, sample_step)

    # Model
    model = build_model(cfg)
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, **kwargs).resume_or_load(weights, resume=False)
    else:
        DetectionCheckpointer(model).resume_or_load(weights, resume=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Data
    mapper = DiffusionDetDatasetMapper(cfg, is_train=False)
    loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    output_dir = f"/tmp/step{sample_step}_eval"
    os.makedirs(output_dir, exist_ok=True)
    evaluator = COCOEvaluator(dataset_name, cfg, True, output_dir)

    # Warmup
    warmup_iter = iter(loader)
    with torch.no_grad():
        for _ in range(min(n_warmup, len(loader))):
            batch = next(warmup_iter)
            model(batch, k=tier)

    # Timed evaluation
    t0 = time.perf_counter()
    results = inference_on_dataset(model, loader, tier, evaluator)
    elapsed = time.perf_counter() - t0
    n_images = len(loader.dataset)

    # Parse results — inference_on_dataset returns list of dicts (one per tier up to k)
    tier_result = results[tier] if isinstance(results, list) and len(results) > tier else results

    metrics = {}
    if isinstance(tier_result, dict) and "bbox" in tier_result:
        metrics = tier_result["bbox"]
    elif isinstance(tier_result, dict):
        metrics = tier_result

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sample_step": sample_step,
        "metrics": metrics,
        "total_eval_time_s": elapsed,
        "n_images": n_images,
        "ms_per_image": (elapsed / n_images * 1000) if n_images > 0 else None,
    }


def plot_results(step_results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    steps      = [r["sample_step"]         for r in step_results]
    ap50_vals  = [r["metrics"].get("AP50", 0) for r in step_results]
    ap_vals    = [r["metrics"].get("AP",   0) for r in step_results]
    ms_vals    = [r["ms_per_image"] or 0     for r in step_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # mAP vs steps
    ax1.plot(steps, ap50_vals, "o-", color="steelblue", label="AP50", linewidth=2)
    ax1.plot(steps, ap_vals,   "s--", color="coral",    label="AP",   linewidth=2)
    ax1.set_xlabel("Diffusion SAMPLE_STEP")
    ax1.set_ylabel("mAP (%)")
    ax1.set_title("Detection Accuracy vs Diffusion Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(steps)

    # Runtime vs steps
    ax2.bar(steps, ms_vals, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Diffusion SAMPLE_STEP")
    ax2.set_ylabel("Inference time (ms/image)")
    ax2.set_title("Runtime vs Diffusion Steps")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, (s, ms) in enumerate(zip(steps, ms_vals)):
        ax2.text(s, ms + 1, f"{ms:.0f}ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))

    print(f"\n{'='*55}")
    print(f"  Diffusion Step Sensitivity — mAP sweep")
    print(f"  Tier: {args.tier}  Steps: {args.steps}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*55}\n")

    # Register datasets
    from train_net_patched import _register_datasets
    _register_datasets()

    all_results = []
    for step in args.steps:
        print(f"\n─── SAMPLE_STEP = {step} ─────────────────────────────")
        result = evaluate_at_step(
            args.config_file, args.weights, step,
            args.tier, args.dataset, args.n_warmup,
        )
        all_results.append(result)

        ap50 = result["metrics"].get("AP50", "?")
        ap   = result["metrics"].get("AP",   "?")
        ms   = result.get("ms_per_image", "?")
        print(f"  SAMPLE_STEP={step}: AP50={ap50}, AP={ap}, {ms:.1f}ms/img")

    # Summary table
    print(f"\n{'='*55}")
    print(f"  SUMMARY — Step Count vs Accuracy/Speed")
    print(f"{'='*55}")
    print(f"{'Steps':>6}  {'AP50':>6}  {'AP':>6}  {'ms/img':>8}  {'rel speed':>10}")
    baseline_ms = all_results[0].get("ms_per_image", 1) or 1
    for r in all_results:
        step = r["sample_step"]
        ap50 = r["metrics"].get("AP50", float("nan"))
        ap   = r["metrics"].get("AP",   float("nan"))
        ms   = r.get("ms_per_image") or float("nan")
        rel  = ms / baseline_ms
        print(f"{step:>6}  {ap50:>6.1f}  {ap:>6.1f}  {ms:>8.1f}  {rel:>10.2f}x")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config":  args.config_file,
            "weights": args.weights,
            "tier":    args.tier,
            "dataset": args.dataset,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if not args.no_plot:
        plot_path = out_path.with_suffix(".png")
        plot_results(all_results, str(plot_path))


if __name__ == "__main__":
    main()
