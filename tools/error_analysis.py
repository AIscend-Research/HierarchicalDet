"""
Match predictions to ground truth and classify what went wrong, per tier.

The per-tier mAP numbers say how much is wrong but not what kind of wrong,
which is what matters clinically: a missed impacted tooth, a box drawn across
two teeth, and a correct box with the wrong diagnosis are three very different
failures that all cost the same AP.

Categories reported per tier:

  matched_correct        prediction matched a GT box (IoU >= --iou) with the
                         right class at this tier
  matched_wrong_class    box was found, class at this tier was wrong
                         (broken out as a confusion matrix)
  false_positive         confident prediction with no GT box above the IoU
                         threshold
  missed_gt              GT box with no matching prediction
  spans_multiple_teeth   prediction containing >= 50% of the area of two or more
                         GT boxes (one box drawn around a group of teeth)
  degenerate             near-zero-area boxes, or boxes covering most of the film

Also reports how errors cluster across tiers -- specifically, of the boxes that
are localized correctly and get the right quadrant, how many then get the tooth
number right, and of those how many get the diagnosis right. That is the direct
measure of whether the hierarchy degrades tier by tier.
"""
import argparse
import json
import os
from collections import Counter, defaultdict


def to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = to_xyxy(box_a)
    bx0, by0, bx1, by1 = to_xyxy(box_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def contained_fraction(inner, outer):
    """How much of `inner`'s area falls inside `outer`."""
    ax0, ay0, ax1, ay1 = to_xyxy(inner)
    bx0, by0, bx1, by1 = to_xyxy(outer)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    return (iw * ih) / area if area > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-json", required=True, help="3-tier DENTEX COCO annotations")
    p.add_argument("--predictions", required=True, help="from tools/dump_predictions.py")
    p.add_argument("--output", required=True)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2),
                   help="highest tier available in the predictions")
    p.add_argument("--iou", type=float, default=0.5, help="match threshold")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="predictions below this score are ignored (a detector emits "
                        "hundreds of low-confidence boxes that no clinician would see)")
    p.add_argument("--max-examples", type=int, default=15,
                   help="how many concrete example images to list per failure kind")
    args = p.parse_args()

    with open(args.gt_json) as f:
        gt_json = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)

    images = {im["id"]: im for im in gt_json["images"]}
    class_names = {
        tier: {c["id"]: str(c["name"]) for c in gt_json["categories_{}".format(tier + 1)]}
        for tier in range(3) if "categories_{}".format(tier + 1) in gt_json
    }

    gt_by_image = defaultdict(list)
    for ann in gt_json["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    pred_by_image = defaultdict(list)
    for record in predictions:
        if record.get("score", 1.0) >= args.score_thresh:
            pred_by_image[record["image_id"]].append(record)

    report = {
        "gt_json": os.path.abspath(args.gt_json),
        "predictions": os.path.abspath(args.predictions),
        "iou_threshold": args.iou,
        "score_threshold": args.score_thresh,
        "images": len(images),
        "gt_boxes": len(gt_json["annotations"]),
        "predicted_boxes_above_threshold": sum(len(v) for v in pred_by_image.values()),
        "per_tier": {},
        "hierarchy_cascade": {},
        "geometry": {},
        "examples": {},
    }

    # --- geometry-level problems (tier-independent) ---
    geometry = Counter()
    examples = defaultdict(list)
    for image_id, records in pred_by_image.items():
        image = images[image_id]
        gts = gt_by_image[image_id]
        for record in records:
            x, y, w, h = record["bbox"]
            if w <= 1 or h <= 1:
                geometry["degenerate"] += 1
                examples["degenerate"].append(image["file_name"])
            elif w * h > 0.9 * image["width"] * image["height"]:
                geometry["covers_whole_image"] += 1
                examples["covers_whole_image"].append(image["file_name"])
            # Containment, not IoU: a box drawn around two teeth has a low IoU
            # with each of them precisely because it is too big, so IoU would
            # miss exactly the case this is meant to catch.
            overlapping = sum(1 for g in gts if contained_fraction(g["bbox"], record["bbox"]) >= 0.5)
            if overlapping >= 2:
                geometry["spans_multiple_teeth"] += 1
                examples["spans_multiple_teeth"].append(image["file_name"])
    for image_id in images:
        if not pred_by_image.get(image_id):
            geometry["images_with_no_detections"] += 1
            examples["images_with_no_detections"].append(images[image_id]["file_name"])
    report["geometry"] = dict(geometry)

    # --- matching, per tier ---
    cascade = Counter()
    for tier in range(args.tier + 1):
        key = "category_id_{}".format(tier + 1)
        counts = Counter()
        confusion = Counter()
        tier_examples = defaultdict(list)

        for image_id, image in images.items():
            gts = [g for g in gt_by_image[image_id] if g.get(key) is not None]
            records = sorted(pred_by_image.get(image_id, []),
                             key=lambda r: r.get("score", 1.0), reverse=True)
            claimed = set()
            for record in records:
                best_iou, best_index = 0.0, None
                for index, gt in enumerate(gts):
                    if index in claimed:
                        continue
                    value = iou(record["bbox"], gt["bbox"])
                    if value > best_iou:
                        best_iou, best_index = value, index
                if best_index is None or best_iou < args.iou:
                    counts["false_positive"] += 1
                    tier_examples["false_positive"].append(image["file_name"])
                    continue
                claimed.add(best_index)
                gt = gts[best_index]
                predicted_class = record.get(key)
                if predicted_class is not None and int(predicted_class) == int(gt[key]):
                    counts["matched_correct"] += 1
                    if tier == 0:
                        cascade["quadrant_correct"] += 1
                    elif tier == 1:
                        cascade["enumeration_correct"] += 1
                    elif tier == 2:
                        cascade["diagnosis_correct"] += 1
                else:
                    counts["matched_wrong_class"] += 1
                    confusion["{} -> {}".format(
                        class_names[tier].get(int(gt[key]), gt[key]),
                        class_names[tier].get(int(predicted_class), predicted_class)
                        if predicted_class is not None else "none")] += 1
                    tier_examples["matched_wrong_class"].append(image["file_name"])
                if tier == 0:
                    cascade["localized"] += 1
            counts["missed_gt"] += len(gts) - len(claimed)
            if len(gts) - len(claimed) > 0:
                tier_examples["missed_gt"].append(image["file_name"])

        total_gt = sum(1 for a in gt_json["annotations"] if a.get(key) is not None)
        recall_denominator = max(total_gt, 1)
        report["per_tier"][str(tier)] = {
            "gt_boxes_at_this_tier": total_gt,
            "counts": dict(counts),
            "recall_localized_and_classified": round(
                100.0 * counts["matched_correct"] / recall_denominator, 2),
            "class_confusion": dict(confusion.most_common(20)),
        }
        for kind, names in tier_examples.items():
            report["examples"]["tier{}_{}".format(tier, kind)] = sorted(set(names))[:args.max_examples]

    # Of the boxes found at all, how many survive each additional tier?
    localized = cascade.get("localized", 0)
    report["hierarchy_cascade"] = {
        "localized_boxes": localized,
        "quadrant_correct": cascade.get("quadrant_correct", 0),
        "enumeration_correct": cascade.get("enumeration_correct", 0),
        "diagnosis_correct": cascade.get("diagnosis_correct", 0),
        "note": "counts are over matched predictions at each tier; a monotone drop "
                "from quadrant to enumeration to diagnosis is the expected pattern "
                "if errors accumulate down the hierarchy",
    }
    for kind, names in examples.items():
        report["examples"][kind] = sorted(set(names))[:args.max_examples]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    print("\nwrote {}".format(args.output))


if __name__ == "__main__":
    main()
