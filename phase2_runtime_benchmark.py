"""
Phase 2 — Inference Runtime and Failure Case Benchmarking

Measures:
  - Per-image inference time on GPU (with CUDA synchronization)
  - Per-image inference time on CPU
  - Peak GPU memory usage
  - Model size (parameter count, checkpoint size)
  - Failure cases: crashes, zero-detection images, invalid box outputs

Also benchmarks the effect of varying SAMPLE_STEP (diffusion steps)
on both runtime and detection quality (box score distribution).

Usage:
    # GPU benchmark
    python phase2_runtime_benchmark.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier3/model_final.pth \
        --tier 2 \
        --n-images 50 \
        --out results/runtime_benchmark.json

    # CPU benchmark (slow — use small n-images)
    python phase2_runtime_benchmark.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier3/model_final.pth \
        --tier 2 \
        --n-images 10 \
        --device cpu \
        --out results/runtime_cpu.json

    # Diffusion step sensitivity
    python phase2_runtime_benchmark.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier3/model_final.pth \
        --tier 2 \
        --n-images 20 \
        --sweep-sample-steps 1 2 4 8 16 \
        --out results/diffusion_step_sensitivity.json
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
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--tier", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--sweep-sample-steps", type=int, nargs="+", default=None,
                   help="If set, benchmark multiple SAMPLE_STEP values (Phase 3 extension)")
    p.add_argument("--n-warmup", type=int, default=5,
                   help="Number of warmup images before timing starts")
    p.add_argument("--out", default="results/runtime_benchmark.json")
    return p.parse_args()


def load_model_and_cfg(config_file, weights, device, sample_step=None):
    sys.path.insert(0, str(Path(__file__).parent))
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, \
        EMADetectionCheckpointer

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    if device == "cpu":
        cfg.MODEL.DEVICE = "cpu"
    if sample_step is not None:
        cfg.MODEL.DiffusionDet.SAMPLE_STEP = sample_step
    cfg.freeze()

    model = build_model(cfg)
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, **kwargs).resume_or_load(weights, resume=False)
    else:
        DetectionCheckpointer(model).resume_or_load(weights, resume=False)

    if device == "cpu":
        model = model.cpu()
    elif torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, cfg


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable,
            "total_params_M": round(total / 1e6, 2)}


def run_benchmark(model, cfg, tier, n_images, n_warmup, device):
    from detectron2.data import build_detection_test_loader
    from dataset_mapper_patched import DiffusionDetDatasetMapper

    mapper = DiffusionDetDatasetMapper(cfg, is_train=False)
    loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    times_gpu = []
    times_wall = []
    n_detections = []
    failure_cases = []
    peak_memory_mb = 0

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats()

    img_count = 0
    with torch.no_grad():
        for batch in loader:
            if img_count >= n_images + n_warmup:
                break

            is_warmup = img_count < n_warmup

            # Wall time
            wall_start = time.perf_counter()

            # GPU time (separate from CPU→GPU transfer)
            if torch.cuda.is_available() and device != "cpu":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()

            try:
                outputs = model(batch, k=tier)
                if torch.cuda.is_available() and device != "cpu":
                    end_event.record()
                    torch.cuda.synchronize()
                    gpu_ms = start_event.elapsed_time(end_event)
                else:
                    gpu_ms = None

                wall_ms = (time.perf_counter() - wall_start) * 1000

                if not is_warmup:
                    for inp, out in zip(batch, outputs):
                        n_det = len(out["instances"]) if "instances" in out else 0
                        scores = out["instances"].scores.cpu().numpy() if n_det > 0 else np.array([])

                        failure = None
                        if n_det == 0:
                            failure = "no_detections"
                        elif any(np.isnan(scores)) or any(np.isinf(scores)):
                            failure = "invalid_scores"
                        elif hasattr(out["instances"], "pred_boxes"):
                            boxes = out["instances"].pred_boxes.tensor.cpu().numpy()
                            if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
                                failure = "invalid_boxes"

                        n_detections.append(n_det)
                        if failure:
                            failure_cases.append({
                                "image_id": inp.get("image_id"),
                                "file_name": inp.get("file_name"),
                                "failure_type": failure,
                                "n_detections": n_det,
                            })

                    times_wall.append(wall_ms)
                    if gpu_ms is not None:
                        times_gpu.append(gpu_ms)

                    if torch.cuda.is_available() and device != "cpu":
                        mem_mb = torch.cuda.max_memory_allocated() / 1e6
                        peak_memory_mb = max(peak_memory_mb, mem_mb)

            except Exception as e:
                if not is_warmup:
                    failure_cases.append({
                        "image_id": batch[0].get("image_id") if batch else None,
                        "failure_type": "inference_crash",
                        "error": str(e),
                    })

            img_count += 1
            if img_count % 10 == 0:
                print(f"  Processed {img_count}/{n_images + n_warmup} images...")

    return {
        "n_measured": len(times_wall),
        "n_warmup": n_warmup,
        "wall_ms_mean": float(np.mean(times_wall)) if times_wall else None,
        "wall_ms_std":  float(np.std(times_wall))  if times_wall else None,
        "wall_ms_median": float(np.median(times_wall)) if times_wall else None,
        "wall_ms_p95":  float(np.percentile(times_wall, 95)) if times_wall else None,
        "gpu_ms_mean":  float(np.mean(times_gpu))  if times_gpu  else None,
        "gpu_ms_std":   float(np.std(times_gpu))   if times_gpu  else None,
        "n_detections_mean": float(np.mean(n_detections)) if n_detections else None,
        "n_detections_std":  float(np.std(n_detections))  if n_detections else None,
        "n_zero_detection_images": sum(1 for n in n_detections if n == 0),
        "failure_cases": failure_cases,
        "n_failures": len(failure_cases),
        "peak_gpu_memory_mb": peak_memory_mb if peak_memory_mb else None,
    }


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"  Runtime Benchmark — Tier {args.tier}")
    print(f"  Device: {device}")
    print(f"  Images: {args.n_images} (+ {args.n_warmup} warmup)")
    print(f"{'='*55}\n")

    results = {
        "config": args.config_file,
        "weights": args.weights,
        "tier": args.tier,
        "device": device,
        "n_images": args.n_images,
    }

    # ── Single run benchmark ───────────────────────────────────────────────────
    if args.sweep_sample_steps is None:
        print("Loading model...")
        model, cfg = load_model_and_cfg(args.config_file, args.weights, device)
        param_info = count_parameters(model)
        results["model_info"] = param_info
        results["checkpoint_size_mb"] = round(
            os.path.getsize(args.weights) / 1e6, 2
        ) if os.path.exists(args.weights) else None
        results["sample_step"] = cfg.MODEL.DiffusionDet.SAMPLE_STEP

        print(f"Model parameters: {param_info['total_params_M']}M")
        print(f"Checkpoint size: {results.get('checkpoint_size_mb')} MB")
        print(f"SAMPLE_STEP: {results['sample_step']}")
        print(f"\nRunning benchmark ({args.n_images} images, device={device})...")

        bench = run_benchmark(model, cfg, args.tier, args.n_images, args.n_warmup, device)
        results["benchmark"] = bench

        print(f"\nRESULTS:")
        print(f"  Wall time: {bench['wall_ms_mean']:.1f} ± {bench['wall_ms_std']:.1f} ms/image")
        if bench["gpu_ms_mean"]:
            print(f"  GPU time:  {bench['gpu_ms_mean']:.1f} ± {bench['gpu_ms_std']:.1f} ms/image")
        print(f"  Peak GPU memory: {bench['peak_gpu_memory_mb']:.0f} MB")
        print(f"  Avg detections/image: {bench['n_detections_mean']:.1f} ± {bench['n_detections_std']:.1f}")
        print(f"  Zero-detection images: {bench['n_zero_detection_images']}/{args.n_images}")
        print(f"  Failure cases: {bench['n_failures']}")
        if bench["failure_cases"]:
            for fc in bench["failure_cases"][:5]:
                print(f"    img {fc.get('image_id')}: {fc['failure_type']}")

    # ── Diffusion step sweep ───────────────────────────────────────────────────
    else:
        print(f"Sweeping SAMPLE_STEP values: {args.sweep_sample_steps}")
        sweep_results = {}

        for step in args.sweep_sample_steps:
            print(f"\n─── SAMPLE_STEP = {step} ───")
            model, cfg = load_model_and_cfg(args.config_file, args.weights, device,
                                            sample_step=step)
            bench = run_benchmark(model, cfg, args.tier, args.n_images, args.n_warmup, device)
            sweep_results[step] = bench
            print(f"  Wall: {bench['wall_ms_mean']:.1f}ms  "
                  f"GPU: {bench.get('gpu_ms_mean') or '?'}ms  "
                  f"Failures: {bench['n_failures']}")
            del model  # free GPU memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["sample_step_sweep"] = sweep_results

        print("\nSWEEP SUMMARY")
        print(f"{'Steps':>6}  {'Wall ms':>9}  {'GPU ms':>9}  {'Failures':>9}")
        for step, bench in sweep_results.items():
            gpu_ms = f"{bench['gpu_ms_mean']:.1f}" if bench.get("gpu_ms_mean") else "N/A"
            print(f"{step:>6}  {bench['wall_ms_mean']:>9.1f}  {gpu_ms:>9}  {bench['n_failures']:>9}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
