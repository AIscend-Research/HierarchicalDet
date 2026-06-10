"""
Phase 3 — Hierarchical Inference Robustness: Noisy-Box Perturbation

Tests whether the hierarchical inference order is robust when early-tier
predictions are imperfect. Injects controlled perturbations into the
noisy-box JSONs (tier-k inference outputs used to seed tier-(k+1)), then
re-runs tier-(k+1) inference and measures performance change.

Perturbation types applied to predicted boxes:
  1. Box coordinate jitter — add Gaussian noise to (x,y,w,h)
  2. Box scale perturbation — randomly scale box size up/down
  3. Score perturbation — lower high-confidence scores (simulate overconfident wrong boxes)
  4. Random box injection — replace fraction of boxes with random-position boxes
  5. Box dropout — randomly remove a fraction of boxes

This isolates a key question: does the hierarchical training make tier-(k+1)
robust to imperfect tier-k proposals, or does it depend on high-quality proposals?

Usage:
    python phase3_noise_injection.py \
        --noisy-box-json noisy_boxes/tier1_val_boxes.json \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier2/model_final.pth \
        --tier 1 \
        --val-json /path/to/validation_triple.json \
        --img-dir /path/to/val_images \
        --out results/noise_injection.json

Outputs:
    results/noise_injection.json  — per-perturbation detection stats
    results/noise_injection.png   — comparison plot
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy-box-json", required=True,
                   help="Tier-k inference boxes JSON (used as noisy proposals for tier-(k+1))")
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights",     required=True)
    p.add_argument("--tier",        type=int, required=True, choices=[1, 2],
                   help="Tier to evaluate after perturbation (1=enumeration, 2=diagnosis)")
    p.add_argument("--val-json",    required=True)
    p.add_argument("--img-dir",     required=True)
    p.add_argument("--n-images",    type=int, default=50)
    p.add_argument("--out",         default="results/noise_injection.json")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ── Perturbation functions ─────────────────────────────────────────────────────

def perturb_jitter(boxes, sigma_frac=0.1, rng=None):
    """Add Gaussian noise scaled by box size to each coordinate."""
    rng = rng or np.random.default_rng(42)
    perturbed = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        dx = rng.normal(0, sigma_frac * w)
        dy = rng.normal(0, sigma_frac * h)
        dw = rng.normal(0, sigma_frac * w)
        dh = rng.normal(0, sigma_frac * h)
        perturbed.append({**b, "bbox": [
            max(0, x + dx), max(0, y + dy),
            max(1, w + dw), max(1, h + dh)
        ]})
    return perturbed


def perturb_scale(boxes, scale_range=(0.5, 2.0), rng=None):
    """Randomly scale each box size, keeping center fixed."""
    rng = rng or np.random.default_rng(42)
    perturbed = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        cx, cy = x + w / 2, y + h / 2
        s = rng.uniform(*scale_range)
        nw, nh = w * s, h * s
        perturbed.append({**b, "bbox": [cx - nw/2, cy - nh/2, nw, nh]})
    return perturbed


def perturb_score_reduction(boxes, max_score=0.6, rng=None):
    """Cap high scores — simulates low-confidence tier-k proposals."""
    rng = rng or np.random.default_rng(42)
    return [{**b, "score": min(b.get("score", 1.0), rng.uniform(0.3, max_score))}
            for b in boxes]


def perturb_random_injection(boxes, inject_frac=0.3, img_w=2900, img_h=1316, rng=None):
    """Replace inject_frac of boxes with random-position boxes."""
    rng = rng or np.random.default_rng(42)
    n_inject = max(1, int(len(boxes) * inject_frac))
    kept = list(rng.choice(boxes, size=max(0, len(boxes) - n_inject), replace=False))
    random_boxes = []
    for _ in range(n_inject):
        w = rng.integers(50, 300)
        h = rng.integers(50, 250)
        x = rng.integers(0, max(1, img_w - w))
        y = rng.integers(0, max(1, img_h - h))
        random_boxes.append({
            "image_id": boxes[0]["image_id"] if boxes else 0,
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": float(rng.uniform(0.5, 0.9)),
        })
    return kept + random_boxes


def perturb_dropout(boxes, drop_frac=0.5, rng=None):
    """Randomly remove drop_frac of boxes."""
    rng = rng or np.random.default_rng(42)
    if not boxes:
        return boxes
    n_keep = max(0, int(len(boxes) * (1 - drop_frac)))
    return list(rng.choice(boxes, size=n_keep, replace=False))


PERTURBATIONS = [
    ("clean",           lambda boxes, rng: boxes,                         "No perturbation (baseline)"),
    ("jitter_10pct",    lambda b, rng: perturb_jitter(b, 0.10, rng),      "Box jitter σ=10% of size"),
    ("jitter_30pct",    lambda b, rng: perturb_jitter(b, 0.30, rng),      "Box jitter σ=30% of size"),
    ("scale_random",    lambda b, rng: perturb_scale(b, (0.5, 2.0), rng), "Random scale 0.5×–2.0×"),
    ("score_cap_0.6",   lambda b, rng: perturb_score_reduction(b, 0.6, rng), "Score capped at 0.6"),
    ("dropout_30pct",   lambda b, rng: perturb_dropout(b, 0.30, rng),     "Box dropout 30%"),
    ("dropout_70pct",   lambda b, rng: perturb_dropout(b, 0.70, rng),     "Box dropout 70%"),
    ("random_inj_30pct",lambda b, rng: perturb_random_injection(b, 0.30, rng=rng), "30% random box injection"),
]


# ── Inference helpers ──────────────────────────────────────────────────────────

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


def run_with_perturbed_boxes(model, cfg, tier, image_paths, perturbed_boxes_by_img):
    """
    Run inference using perturbed_boxes_by_img as noisy proposals.
    Returns per-image detection stats.
    """
    import torch
    from detectron2.data import transforms as T

    resize = T.ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice"
    )
    per_image = []

    with torch.no_grad():
        for img_id, img_path in image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                per_image.append({"image_id": img_id, "error": "load_fail", "n_detections": 0})
                continue

            img_rgb = img_bgr[:, :, ::-1]
            h, w = img_rgb.shape[:2]
            img_resized, _ = T.apply_transform_gens([resize], img_rgb)
            import torch as _torch
            img_tensor = _torch.as_tensor(
                np.ascontiguousarray(img_resized.transpose(2, 0, 1))
            ).float()
            if _torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # Build noisy_boxes annotation for this image
            bboxes_pre = [b["bbox"] for b in perturbed_boxes_by_img.get(img_id, [])]

            inp = [{
                "image": img_tensor,
                "height": h,
                "width": w,
                "image_id": img_id,
                "file_name": str(img_path),
                "bbox_pre": bboxes_pre,  # injected as precomputed proposals
            }]

            t0 = time.perf_counter()
            try:
                out = model(inp, k=tier)
                instances = out[0]["instances"].to("cpu")
                scores = instances.scores.numpy()
                per_image.append({
                    "image_id": img_id,
                    "n_detections": len(scores),
                    "n_noisy_boxes": len(bboxes_pre),
                    "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
                    "ms": (time.perf_counter() - t0) * 1000,
                })
            except Exception as e:
                per_image.append({"image_id": img_id, "error": str(e), "n_detections": 0})

    return per_image


def aggregate(per_image):
    valid = [r for r in per_image if "error" not in r]
    dets  = [r["n_detections"] for r in valid]
    scores = [r["mean_score"]  for r in valid]
    return {
        "n_images": len(per_image), "n_valid": len(valid),
        "n_errors": len(per_image) - len(valid),
        "n_zero_detection": sum(1 for n in dets if n == 0),
        "mean_detections": float(np.mean(dets))    if dets   else 0,
        "std_detections":  float(np.std(dets))     if dets   else 0,
        "mean_score":      float(np.mean(scores))  if scores else 0,
    }


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))

    rng = np.random.default_rng(args.seed)

    # Load base noisy boxes
    print(f"Loading noisy boxes: {args.noisy_box_json}")
    with open(args.noisy_box_json) as f:
        base_boxes = json.load(f)

    boxes_by_img = {}
    for b in base_boxes:
        boxes_by_img.setdefault(b["image_id"], []).append(b)
    print(f"  Loaded boxes for {len(boxes_by_img)} images")

    # Load val image list
    with open(args.val_json) as f:
        val_data = json.load(f)

    img_dir = Path(args.img_dir)
    image_paths = []
    for img_meta in val_data["images"][:args.n_images]:
        iid   = img_meta["id"]
        fname = img_meta.get("file_name", "")
        path  = img_dir / fname
        if not path.exists():
            path = img_dir / Path(fname).name
        if path.exists():
            image_paths.append((iid, path))

    print(f"Found {len(image_paths)} images for inference")

    print("Loading model...")
    model, cfg = load_model(args.config_file, args.weights)

    results_all = []
    for name, perturb_fn, label in PERTURBATIONS:
        print(f"\n  Perturbation: {label}")

        # Apply perturbation to all boxes
        perturbed_by_img = {}
        for img_id, boxes in boxes_by_img.items():
            perturbed_by_img[img_id] = perturb_fn(list(boxes), rng)

        n_before = sum(len(v) for v in boxes_by_img.values())
        n_after  = sum(len(v) for v in perturbed_by_img.values())
        print(f"    Boxes: {n_before} → {n_after}")

        per_image = run_with_perturbed_boxes(model, cfg, args.tier, image_paths, perturbed_by_img)
        stats = aggregate(per_image)
        results_all.append({"name": name, "label": label, "stats": stats,
                             "boxes_before": n_before, "boxes_after": n_after})

        print(f"    Zero-det: {stats['n_zero_detection']}/{stats['n_valid']}")
        print(f"    Mean dets: {stats['mean_detections']:.1f} ± {stats['std_detections']:.1f}")
        print(f"    Mean score: {stats['mean_score']:.3f}")

    # Summary table
    clean = results_all[0]["stats"]
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Relative to unperturbed baseline")
    print(f"{'='*60}")
    print(f"{'Perturbation':<30} {'ZeroDet':>8} {'ΔDets':>7} {'ΔScore':>8}")
    print("-" * 60)
    for r in results_all:
        s = r["stats"]
        print(f"  {r['label'][:28]:<30} {s['n_zero_detection']:>8} "
              f"{s['mean_detections'] - clean['mean_detections']:>+7.1f} "
              f"{s['mean_score'] - clean['mean_score']:>+8.3f}")

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "config": args.config_file, "weights": args.weights,
            "tier": args.tier, "n_images": len(image_paths),
            "noisy_box_json": args.noisy_box_json,
            "perturbations": results_all,
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
