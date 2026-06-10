"""
Phase 3 — Image Degradation Robustness

Evaluates HierarchicalDet's detection stability under realistic clinical
image quality degradation:
  - Gaussian blur (simulates motion artifact or poor focus)
  - JPEG compression artifacts (simulates low-quality digital storage)
  - Resolution reduction (simulates older or lower-cost X-ray equipment)

For each degradation type and severity, runs inference on the validation
set and reports:
  - Detection score distribution shift (mean/std of predicted scores)
  - Number of zero-detection images (complete failures)
  - mAP degradation relative to clean images

Usage:
    export DATA_ROOT=/path/to/sorted/challenge
    python phase3_image_degradation.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier3/model_final.pth \
        --tier 2 \
        --img-dir /path/to/val_images \
        --val-json /path/to/validation_triple.json \
        --out-dir results/degradation/ \
        --n-images 50

Outputs:
    results/degradation/results.json      — per-degradation metrics
    results/degradation/results.png       — summary plot
    results/degradation/samples/          — example degraded images
"""

import argparse
import io
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights",     required=True)
    p.add_argument("--tier",        type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--img-dir",     required=True, help="Directory containing validation images")
    p.add_argument("--val-json",    required=True, help="Path to validation_triple.json")
    p.add_argument("--n-images",    type=int, default=50)
    p.add_argument("--out-dir",     default="results/degradation")
    p.add_argument("--save-samples", action="store_true",
                   help="Save example degraded images for visual inspection")
    p.add_argument("--n-sample-images", type=int, default=3,
                   help="Number of example images to save per degradation")
    return p.parse_args()


# ── Degradation functions ──────────────────────────────────────────────────────

DEGRADATIONS = [
    ("clean",          lambda img: img.copy(),                        "No degradation (baseline)"),
    ("blur_sigma2",    lambda img: cv2.GaussianBlur(img, (0,0), 2),  "Gaussian blur σ=2"),
    ("blur_sigma5",    lambda img: cv2.GaussianBlur(img, (0,0), 5),  "Gaussian blur σ=5"),
    ("blur_sigma10",   lambda img: cv2.GaussianBlur(img, (0,0), 10), "Gaussian blur σ=10"),
    ("jpeg_q50",       lambda img: jpeg_compress(img, 50),            "JPEG compression Q=50"),
    ("jpeg_q20",       lambda img: jpeg_compress(img, 20),            "JPEG compression Q=20"),
    ("jpeg_q10",       lambda img: jpeg_compress(img, 10),            "JPEG compression Q=10"),
    ("resize_50pct",   lambda img: resize_and_back(img, 0.5),         "Resolution ↓ 50%"),
    ("resize_25pct",   lambda img: resize_and_back(img, 0.25),        "Resolution ↓ 25%"),
    ("noise_gauss",    lambda img: add_gaussian_noise(img, std=25),   "Additive Gaussian noise σ=25"),
]


def jpeg_compress(img, quality):
    """Compress to JPEG and decompress — simulates JPEG artifact degradation."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def resize_and_back(img, scale):
    """Downsample then upsample — simulates lower-resolution acquisition."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(img, std=25):
    """Add zero-mean Gaussian noise."""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


# ── Inference with degraded images ────────────────────────────────────────────

def load_model(config_file, weights):
    import torch
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
    cfg.freeze()

    model = build_model(cfg)
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, **kwargs).resume_or_load(weights, resume=False)
    else:
        DetectionCheckpointer(model).resume_or_load(weights, resume=False)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, cfg


def run_degraded_inference(model, cfg, tier, image_paths, degradation_fn):
    """
    Run inference on a list of image paths, applying degradation_fn to each.
    Returns per-image stats: n_detections, top score, mean score, timing.
    """
    import torch
    from detectron2.data.detection_utils import read_image
    from detectron2.data import transforms as T

    resize = T.ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice"
    )

    per_image = []
    with torch.no_grad():
        for img_path in image_paths:
            t0 = time.perf_counter()

            # Load and degrade
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                per_image.append({"error": f"cannot load {img_path}", "n_detections": 0})
                continue

            img_degraded = degradation_fn(img_bgr)
            img_rgb = img_degraded[:, :, ::-1]  # BGR → RGB

            # Preprocess (matching DiffusionDetDatasetMapper)
            img_resized, _ = T.apply_transform_gens([resize], img_rgb)
            img_tensor = torch.as_tensor(
                np.ascontiguousarray(img_resized.transpose(2, 0, 1))
            ).float()
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            h_orig, w_orig = img_rgb.shape[:2]
            inp = [{
                "image": img_tensor,
                "height": h_orig,
                "width": w_orig,
                "image_id": 0,
                "file_name": str(img_path),
            }]

            try:
                out = model(inp, k=tier)
                instances = out[0]["instances"].to("cpu")
                scores = instances.scores.numpy()
                n_det = len(scores)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                per_image.append({
                    "n_detections": n_det,
                    "top_score": float(np.max(scores)) if n_det > 0 else 0.0,
                    "mean_score": float(np.mean(scores)) if n_det > 0 else 0.0,
                    "all_scores": scores.tolist(),
                    "ms": elapsed_ms,
                })
            except Exception as e:
                per_image.append({"error": str(e), "n_detections": 0, "ms": 0})

    return per_image


def aggregate_stats(per_image_results):
    valid = [r for r in per_image_results if "error" not in r]
    n_det_list = [r["n_detections"] for r in valid]
    ms_list = [r["ms"] for r in valid]
    score_list = [s for r in valid for s in r.get("all_scores", [])]

    return {
        "n_images": len(per_image_results),
        "n_valid": len(valid),
        "n_errors": len(per_image_results) - len(valid),
        "n_zero_detection": sum(1 for n in n_det_list if n == 0),
        "mean_detections": float(np.mean(n_det_list)) if n_det_list else 0,
        "std_detections":  float(np.std(n_det_list))  if n_det_list else 0,
        "mean_score": float(np.mean(score_list)) if score_list else 0,
        "std_score":  float(np.std(score_list))  if score_list else 0,
        "mean_ms":    float(np.mean(ms_list))    if ms_list else 0,
    }


def save_sample_degradations(img_paths, degradations_to_sample, out_dir, n_samples=3):
    """Save side-by-side comparisons of clean vs degraded for visual inspection."""
    out_dir = Path(out_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths[:n_samples]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stem = Path(img_path).stem

        panels = []
        labels = []
        for name, fn, label in degradations_to_sample:
            deg = fn(img)
            # Scale down for display (panoramics are huge)
            scale = min(1.0, 800 / img.shape[1])
            panels.append(cv2.resize(deg, (int(img.shape[1]*scale), int(img.shape[0]*scale))))
            labels.append(f"{label}")

        if not panels:
            continue

        # Stack horizontally with labels
        h = panels[0].shape[0]
        labeled = []
        for panel, label in zip(panels, labels):
            p = panel.copy()
            cv2.putText(p, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            labeled.append(p)

        # Arrange in a 2-row grid
        n = len(labeled)
        ncols = min(5, n)
        nrows = (n + ncols - 1) // ncols
        rows = []
        for r in range(nrows):
            row_panels = labeled[r*ncols:(r+1)*ncols]
            while len(row_panels) < ncols:
                row_panels.append(np.zeros_like(row_panels[0]))
            rows.append(np.hstack(row_panels))
        grid = np.vstack(rows)
        cv2.imwrite(str(out_dir / f"{stem}_degradations.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"  Sample degradation images saved to {out_dir}/")


def plot_summary(all_stats, out_path):
    if not MATPLOTLIB:
        return
    names  = [s["name"] for s in all_stats]
    n_zero = [s["stats"]["n_zero_detection"] for s in all_stats]
    n_det  = [s["stats"]["mean_detections"]   for s in all_stats]
    scores = [s["stats"]["mean_score"]         for s in all_stats]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(names))

    axes[0].bar(x, n_zero, color="crimson", alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("# Zero-detection images"); axes[0].set_title("Detection Failures")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, n_det, color="steelblue", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean detections/image"); axes[1].set_title("Detection Count")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].axhline(n_det[0], color="red", linestyle="--", label="Clean baseline")
    axes[1].legend()

    axes[2].bar(x, scores, color="darkorange", alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean detection score"); axes[2].set_title("Prediction Confidence")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].axhline(scores[0], color="red", linestyle="--", label="Clean baseline")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out_path}")
    plt.close()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotation to get image filenames
    with open(args.val_json) as f:
        val_data = json.load(f)

    from pathlib import Path as _Path
    img_dir = _Path(args.img_dir)
    image_paths = []
    for img_meta in val_data["images"][:args.n_images]:
        fname = img_meta.get("file_name", "")
        candidates = [
            img_dir / fname,
            img_dir / _Path(fname).name,
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            image_paths.append(found)
        else:
            print(f"  WARNING: image not found: {fname}")

    if not image_paths:
        print("ERROR: No images found. Check --img-dir and --val-json.")
        return

    print(f"\n{'='*60}")
    print(f"  Image Degradation Robustness — Tier {args.tier}")
    print(f"  Images: {len(image_paths)}  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*60}\n")

    # Load model once
    print("Loading model...")
    model, cfg = load_model(args.config_file, args.weights)

    # Save sample degradation images
    if args.save_samples:
        save_sample_degradations(image_paths, DEGRADATIONS, out_dir, args.n_sample_images)

    # Run all degradations
    all_stats = []
    for name, fn, label in DEGRADATIONS:
        print(f"\n  Degradation: {label}")
        per_image = run_degraded_inference(model, cfg, args.tier, image_paths, fn)
        stats = aggregate_stats(per_image)
        all_stats.append({"name": name, "label": label, "stats": stats})

        print(f"    Zero-detection: {stats['n_zero_detection']}/{stats['n_valid']}")
        print(f"    Mean detections: {stats['mean_detections']:.1f} ± {stats['std_detections']:.1f}")
        print(f"    Mean score: {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")

    # Summary relative to clean baseline
    clean_stats = all_stats[0]["stats"]
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Relative to clean baseline")
    print(f"{'='*60}")
    print(f"{'Degradation':<22} {'ZeroDet':>8} {'Δ Dets':>8} {'Δ Score':>9}")
    print("-" * 60)
    for entry in all_stats:
        s = entry["stats"]
        delta_det   = s["mean_detections"] - clean_stats["mean_detections"]
        delta_score = s["mean_score"]      - clean_stats["mean_score"]
        print(f"  {entry['label'][:20]:<22} {s['n_zero_detection']:>8} "
              f"{delta_det:>+8.1f} {delta_score:>+9.3f}")

    # Save
    result = {
        "config": args.config_file, "weights": args.weights,
        "tier": args.tier, "n_images": len(image_paths),
        "degradations": all_stats,
    }
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_json}")

    plot_summary(all_stats, str(out_dir / "results.png"))


if __name__ == "__main__":
    main()
