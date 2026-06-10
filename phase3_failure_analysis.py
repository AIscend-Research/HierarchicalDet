"""
Phase 3 — Failure Case Analysis and Per-Tier Error Clustering

Categorizes every detected failure on the validation set:
  1. Missed detections (GT box has no matching prediction)
  2. False positives (predicted box has no matching GT)
  3. Wrong label — correct box, wrong quadrant / tooth number / diagnosis
  4. Multi-tooth box — predicted box spans multiple GT annotations (IoU with several)
  5. Degenerate box — zero area or extreme aspect ratio
  6. Score collapse — all predictions below 0.1 score threshold

Additionally tests the clean vs stress-test subsets created in Phase 1
and reports whether errors cluster by annotation tier (quadrant vs diagnosis).

Usage:
    export DATA_ROOT=/path/to/sorted/challenge
    python phase3_failure_analysis.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights output/tier3/model_final.pth \
        --tier 2 \
        --val-json /path/to/validation_triple.json \
        --img-dir /path/to/val_images \
        --clean-subset phase1_subsets/clean_subset.json \
        --stress-subset phase1_subsets/stress_test_subset.json \
        --out-dir results/failure_analysis/

Outputs:
    results/failure_analysis/failure_report.json  — full per-image failure records
    results/failure_analysis/failure_summary.txt  — human-readable summary
    results/failure_analysis/failure_images/      — annotated images for worst cases
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np


IOU_MATCH_THRESH = 0.5   # IoU threshold to count a detection as a match
SCORE_THRESH     = 0.3   # Score threshold for counting predictions as active
MULTITOOTH_THRESH = 2    # N GT boxes with IoU > 0.3 → "multi-tooth box" flag

TIER_NAMES  = {0: "quadrant", 1: "enumeration", 2: "diagnosis"}
DIAG_NAMES  = {0: "None", 1: "Impacted", 2: "Caries", 3: "PeriapicalLesion", 4: "DeepCaries"}
QUAD_NAMES  = {0: "?", 1: "UpperRight", 2: "UpperLeft", 3: "LowerLeft", 4: "LowerRight"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file",    required=True)
    p.add_argument("--weights",        required=True)
    p.add_argument("--tier",           type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--val-json",       required=True)
    p.add_argument("--img-dir",        required=True)
    p.add_argument("--clean-subset",   default=None)
    p.add_argument("--stress-subset",  default=None)
    p.add_argument("--n-images",       type=int, default=50)
    p.add_argument("--n-worst-cases",  type=int, default=10,
                   help="Number of worst-case images to save for inspection")
    p.add_argument("--out-dir",        default="results/failure_analysis")
    return p.parse_args()


# ── IoU ───────────────────────────────────────────────────────────────────────

def box_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(a_xyxy, b_xyxy):
    ax1, ay1, ax2, ay2 = a_xyxy
    bx1, by1, bx2, by2 = b_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


# ── Load and run inference ────────────────────────────────────────────────────

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


def infer_image(model, cfg, tier, img_path):
    """Run inference on a single image. Returns (pred_boxes_xyxy, pred_scores, pred_labels)."""
    import torch
    from detectron2.data import transforms as T

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, None

    img_rgb = img_bgr[:, :, ::-1]
    h, w = img_rgb.shape[:2]

    resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
    img_resized, _ = T.apply_transform_gens([resize], img_rgb)
    img_tensor = torch.as_tensor(np.ascontiguousarray(img_resized.transpose(2, 0, 1))).float()
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    inp = [{"image": img_tensor, "height": h, "width": w, "image_id": 0, "file_name": str(img_path)}]
    with torch.no_grad():
        out = model(inp, k=tier)

    instances = out[0]["instances"].to("cpu")
    boxes  = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    labels = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(len(scores))
    return boxes, scores, labels


# ── Failure categorization ────────────────────────────────────────────────────

def categorize_failures(gt_annotations, pred_boxes, pred_scores, pred_labels, tier, score_thresh):
    """
    Returns a dict of failure categories for this image.
    gt_annotations: list of annotation dicts (category_id_1/2/3, bbox)
    pred_boxes: N×4 array (xyxy)
    """
    failures = defaultdict(list)

    gt_boxes = [box_xywh_to_xyxy(a["bbox"]) for a in gt_annotations]
    gt_cats  = [(a.get("category_id_1", 0), a.get("category_id_2", 0), a.get("category_id_3", 0))
                for a in gt_annotations]

    active_preds = [(pred_boxes[i], pred_scores[i], pred_labels[i])
                    for i in range(len(pred_scores))
                    if pred_scores[i] >= score_thresh] if pred_boxes is not None else []

    # Score collapse
    if pred_boxes is None or len(pred_boxes) == 0:
        failures["crash"].append({"note": "inference returned no instances"})
        return dict(failures)

    if len(active_preds) == 0:
        failures["score_collapse"].append({"max_score": float(np.max(pred_scores)) if len(pred_scores) else 0})

    # GT matching
    gt_matched = [False] * len(gt_boxes)
    for pred_box, pred_score, pred_label in active_preds:
        best_iou, best_gt_idx = 0.0, -1
        for gi, gt_box in enumerate(gt_boxes):
            iou_val = iou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou, best_gt_idx = iou_val, gi

        if best_iou >= IOU_MATCH_THRESH and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            # Check label correctness for the active tier
            gt_cat = gt_cats[best_gt_idx]
            if tier == 0 and pred_label != gt_cat[0] - 1:
                failures["wrong_quadrant_label"].append({
                    "pred": int(pred_label), "gt": gt_cat[0], "score": float(pred_score)
                })
            elif tier == 1 and pred_label != gt_cat[1] - 1:
                failures["wrong_enumeration_label"].append({
                    "pred": int(pred_label), "gt": gt_cat[1], "score": float(pred_score)
                })
            elif tier == 2 and pred_label != gt_cat[2] - 1:
                failures["wrong_diagnosis_label"].append({
                    "pred": int(pred_label), "gt": gt_cat[2], "score": float(pred_score)
                })
        else:
            # False positive
            failures["false_positive"].append({"best_iou": round(best_iou, 3), "score": float(pred_score)})

        # Multi-tooth box check
        n_overlapping_gt = sum(1 for gt_box in gt_boxes if iou(pred_box, gt_box) > 0.3)
        if n_overlapping_gt >= MULTITOOTH_THRESH:
            failures["multi_tooth_box"].append({"n_gt_overlapping": n_overlapping_gt})

        # Degenerate box
        x1, y1, x2, y2 = pred_box
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            failures["degenerate_box"].append({"box": list(pred_box)})
        aspect = (x2 - x1) / max(1, y2 - y1)
        if aspect > 10 or aspect < 0.1:
            failures["extreme_aspect_ratio"].append({"aspect": round(aspect, 2)})

    # Missed detections (GT boxes with no matching prediction)
    for gi, matched in enumerate(gt_matched):
        if not matched:
            gt_cat = gt_cats[gi]
            failures["missed_detection"].append({
                "gt_quadrant": gt_cat[0],
                "gt_enumeration": gt_cat[1],
                "gt_diagnosis": gt_cat[2],
                "diagnosis_name": DIAG_NAMES.get(gt_cat[2], str(gt_cat[2])),
            })

    return {k: list(v) for k, v in failures.items()}


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_failure_overlay(img_bgr, gt_anns, pred_boxes, pred_scores, score_thresh, out_path):
    out = img_bgr.copy()
    scale = min(1.0, 1200 / img_bgr.shape[1])
    out = cv2.resize(out, (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale)))

    # GT boxes — green
    for ann in gt_anns:
        x, y, w, h = [int(v * scale) for v in ann["bbox"]]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 200, 0), 2)
        diag = ann.get("category_id_3", 0)
        cv2.putText(out, f"GT Q{ann.get('category_id_1',0)} D{diag}",
                    (x, max(0, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    # Predictions — red (low confidence) or blue (high confidence)
    if pred_boxes is not None:
        for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            color = (255, 80, 0) if score >= 0.5 else (0, 80, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{score:.2f}", (x1, max(0, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Legend
    cv2.putText(out, "GT=green  Pred(high)=orange  Pred(low)=blue",
                (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ── Main ───────────────────────────────────────────────────────────────────────

def run_analysis(model, cfg, tier, images_with_gt, out_dir, score_thresh=SCORE_THRESH):
    """Run failure analysis on a list of (img_id, img_path, gt_anns) tuples."""
    all_records = []
    failure_counter = Counter()

    for img_id, img_path, gt_anns in images_with_gt:
        pred_boxes, pred_scores, pred_labels = infer_image(model, cfg, tier, img_path)
        failures = categorize_failures(gt_anns, pred_boxes, pred_scores, pred_labels,
                                       tier, score_thresh)
        for ftype, flist in failures.items():
            failure_counter[ftype] += len(flist)

        n_failures = sum(len(v) for v in failures.values())
        all_records.append({
            "image_id": img_id,
            "file_name": str(img_path),
            "n_gt": len(gt_anns),
            "n_pred_active": sum(1 for s in (pred_scores or []) if s >= score_thresh),
            "failures": failures,
            "n_failures": n_failures,
        })

    return all_records, failure_counter


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    # Load val annotations
    with open(args.val_json) as f:
        val_data = json.load(f)

    ann_by_img = defaultdict(list)
    for ann in val_data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    img_map = {img["id"]: img for img in val_data["images"]}

    def find_img(fname):
        for p in [img_dir / fname, img_dir / Path(fname).name]:
            if p.exists():
                return p
        return None

    # Build image list for full val set
    full_set = []
    for img_meta in list(img_map.values())[:args.n_images]:
        iid = img_meta["id"]
        path = find_img(img_meta.get("file_name", ""))
        if path:
            full_set.append((iid, path, ann_by_img[iid]))

    # Load optional subsets
    def load_subset(json_path, set_name):
        if not json_path or not Path(json_path).exists():
            return None
        with open(json_path) as f:
            data = json.load(f)
        sub_ann_by_img = defaultdict(list)
        for ann in data["annotations"]:
            sub_ann_by_img[ann["image_id"]].append(ann)
        result = []
        for img in data["images"]:
            iid = img["id"]
            path = find_img(img.get("file_name", ""))
            if path:
                result.append((iid, path, sub_ann_by_img[iid]))
        print(f"  {set_name}: {len(result)} images")
        return result

    clean_set  = load_subset(args.clean_subset,  "Clean subset")
    stress_set = load_subset(args.stress_subset, "Stress-test subset")

    print("Loading model...")
    model, cfg = load_model(args.config_file, args.weights)

    # ── Run analysis ──────────────────────────────────────────────────────────
    print(f"\nAnalyzing {len(full_set)} images (tier={args.tier}, score_thresh={SCORE_THRESH})...")
    records, counter = run_analysis(model, cfg, args.tier, full_set, out_dir)

    # Save worst-case visualizations
    worst = sorted(records, key=lambda r: r["n_failures"], reverse=True)[:args.n_worst_cases]
    print(f"\nSaving {len(worst)} worst-case visualizations...")
    fail_img_dir = out_dir / "failure_images"
    for r in worst:
        img_bgr = cv2.imread(r["file_name"])
        if img_bgr is None:
            continue
        gt_anns = ann_by_img[r["image_id"]]
        pred_boxes, pred_scores, _ = infer_image(model, cfg, args.tier, Path(r["file_name"]))
        draw_failure_overlay(
            img_bgr, gt_anns, pred_boxes, pred_scores, SCORE_THRESH,
            fail_img_dir / f"img{r['image_id']}_failures{r['n_failures']}.jpg",
        )

    # ── Subset comparison ─────────────────────────────────────────────────────
    subset_results = {}
    for subset_name, subset_data in [("clean", clean_set), ("stress_test", stress_set)]:
        if subset_data is None:
            continue
        print(f"\nAnalyzing {subset_name} subset ({len(subset_data)} images)...")
        sub_records, sub_counter = run_analysis(model, cfg, args.tier, subset_data, out_dir)
        n = len(sub_records)
        subset_results[subset_name] = {
            "n_images": n,
            "failure_counts": dict(sub_counter),
            "failure_rate_per_image": {k: v/n for k, v in sub_counter.items()},
            "images_with_any_failure": sum(1 for r in sub_records if r["n_failures"] > 0),
        }

    # ── Print summary ─────────────────────────────────────────────────────────
    n = len(records)
    print(f"\n{'='*60}")
    print(f"  FAILURE ANALYSIS SUMMARY — Tier {TIER_NAMES.get(args.tier)}")
    print(f"  Total images: {n}  Score threshold: {SCORE_THRESH}")
    print(f"{'='*60}")
    print(f"\n  Failure type breakdown:")
    for ftype, count in counter.most_common():
        rate = count / n
        print(f"    {ftype:<35} {count:>5}  ({rate:.1%} per image)")

    print(f"\n  Images with ≥1 failure: {sum(1 for r in records if r['n_failures'] > 0)}/{n}")

    if subset_results:
        print(f"\n  SUBSET COMPARISON:")
        print(f"  {'Metric':<35} {'Clean':>10} {'Stress':>10}")
        print("  " + "-" * 58)
        clean_r  = subset_results.get("clean", {})
        stress_r = subset_results.get("stress_test", {})
        for ftype in sorted(set(list(clean_r.get("failure_rate_per_image", {}).keys()) +
                                list(stress_r.get("failure_rate_per_image", {}).keys()))):
            c = clean_r.get("failure_rate_per_image", {}).get(ftype, 0)
            s = stress_r.get("failure_rate_per_image", {}).get(ftype, 0)
            print(f"  {ftype:<35} {c:>10.2%} {s:>10.2%}")

    # Save full report
    report = {
        "config": args.config_file, "weights": args.weights,
        "tier": args.tier, "tier_name": TIER_NAMES.get(args.tier),
        "n_images": n, "score_thresh": SCORE_THRESH,
        "failure_counts": dict(counter),
        "failure_rate_per_image": {k: v/n for k, v in counter.items()},
        "images_with_any_failure": sum(1 for r in records if r["n_failures"] > 0),
        "per_image_records": records,
        "subset_comparison": subset_results,
    }
    out_json = out_dir / "failure_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    lines = [
        "=" * 60,
        f"Failure Analysis — Tier: {TIER_NAMES.get(args.tier)}",
        "=" * 60, "",
        f"Images analyzed: {n}",
        f"Score threshold: {SCORE_THRESH}",
        "",
        "FAILURE COUNTS:",
    ]
    for ftype, count in counter.most_common():
        lines.append(f"  {ftype:<35} {count:>5}  ({count/n:.1%}/img)")
    with open(out_dir / "failure_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
