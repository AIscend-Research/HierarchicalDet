"""
Phase 2 Helper — Generate Noisy Boxes from a Trained Tier Model

Runs inference with a trained tier-k model on a dataset and saves the
detections as a COCO results JSON. This JSON is then used as the
NOISY_BOX_TRAIN / NOISY_BOX_VAL input for the next tier's training.

This is the core of the hierarchical noisy-box manipulation described
in Section 3.2 of the HierarchicalDet paper.

Usage:
    python phase2_generate_noisy_boxes.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier1/model_final.pth \
        --dataset custom_train_class \
        --tier 0 \
        --out noisy_boxes/tier1_train_boxes.json \
        --score-thresh 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True, help="Path to trained model checkpoint")
    p.add_argument("--dataset", required=True, help="Registered dataset name to run inference on")
    p.add_argument("--tier", type=int, required=True, choices=[0, 1, 2],
                   help="Tier index: 0=quadrant, 1=enumeration, 2=diagnosis")
    p.add_argument("--out", required=True, help="Output COCO results JSON path")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Score threshold for including a detection as a noisy box")
    p.add_argument("--num-gpus", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, \
        EMADetectionCheckpointer, apply_model_ema_and_restore
    from dataset_mapper_patched import DiffusionDetDatasetMapper

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, **kwargs).resume_or_load(args.weights, resume=False)
    else:
        DetectionCheckpointer(model).resume_or_load(args.weights, resume=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # ── Dataloader ────────────────────────────────────────────────────────────
    mapper = DiffusionDetDatasetMapper(cfg, is_train=False)
    loader = build_detection_test_loader(cfg, args.dataset, mapper=mapper)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"Running tier-{args.tier} inference on '{args.dataset}'...")
    print(f"  Checkpoint: {args.weights}")
    print(f"  Score threshold: {args.score_thresh}")

    results = []
    total_images = 0
    total_detections = 0
    images_with_no_det = 0

    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch, k=args.tier)
            for inp, out in zip(batch, outputs):
                image_id = inp["image_id"]
                instances = out["instances"].to("cpu")
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()

                n_kept = 0
                for box, score in zip(boxes, scores):
                    if score >= args.score_thresh:
                        x1, y1, x2, y2 = box.tolist()
                        results.append({
                            "image_id": image_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: x,y,w,h
                            "score": float(score),
                        })
                        n_kept += 1

                if n_kept == 0:
                    images_with_no_det += 1
                total_detections += n_kept
                total_images += 1

    elapsed = time.perf_counter() - start

    print(f"\nInference complete:")
    print(f"  Images processed: {total_images}")
    print(f"  Detections (score >= {args.score_thresh}): {total_detections}")
    print(f"  Images with no detections: {images_with_no_det}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/max(total_images,1):.3f}s/image)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)

    meta = {
        "tier": args.tier,
        "dataset": args.dataset,
        "weights": args.weights,
        "score_thresh": args.score_thresh,
        "n_images": total_images,
        "n_detections": len(results),
        "n_images_no_detection": images_with_no_det,
        "inference_time_s": elapsed,
        "s_per_image": elapsed / max(total_images, 1),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(results)} detections → {out_path}")
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
