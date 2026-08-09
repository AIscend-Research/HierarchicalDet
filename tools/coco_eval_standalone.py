"""
Score a predictions dump against a DENTEX split with a stock COCO evaluator,
independently of the repo's forked evaluation code.

Two uses:

1. **Cross-check.** The repo ships a modified `pycocotools` and a modified
   `COCOEvaluator` to handle its three-tier annotations. Numbers produced by a
   fork are worth confirming against the standard implementation before they go
   into a reproduction report, so this script flattens one tier of the 3-tier
   GT into ordinary COCO and evaluates with the pip-installed pycocotools.

2. **Baselines.** RetinaNet / Faster R-CNN / DETR predict a single flat class
   set, so their outputs cannot go through the tier-aware evaluator at all.
   Scoring them here, and the HierarchicalDet dumps here as well, makes the
   comparison an apples-to-apples one computed by the same code.

Predictions are read in COCO-results format as written by
tools/dump_predictions.py (`bbox` in absolute XYWH, `score`, and either
`category_id` or the per-tier `category_id_N`).
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np

from repro_common import TIERS, load_pip_pycocotools  # noqa: E402


def flatten_gt(coco_json, tier):
    """
    Turn the 3-tier DENTEX GT into standard COCO for one tier.

    Annotations whose `category_id_{tier+1}` is null carry no label at that tier
    (that is how the partially-annotated tiers are encoded) and are dropped --
    keeping them would count unlabelled teeth as missed detections.
    """
    key = "category_id_{}".format(tier + 1)
    categories = coco_json["categories_{}".format(tier + 1)]
    annotations = []
    for ann in coco_json["annotations"]:
        if ann.get(key) is None:
            continue
        flat = {k: v for k, v in ann.items() if not k.startswith("category_id_")}
        flat["category_id"] = ann[key]
        flat.setdefault("iscrowd", 0)
        flat.setdefault("area", ann["bbox"][2] * ann["bbox"][3])
        annotations.append(flat)
    return {
        "images": coco_json["images"],
        "annotations": annotations,
        "categories": [{"id": c["id"], "name": str(c["name"])} for c in categories],
    }


def flatten_predictions(predictions, tier):
    key = "category_id_{}".format(tier + 1)
    out = []
    for record in predictions:
        category_id = record.get(key, record.get("category_id"))
        if category_id is None:
            continue
        out.append({
            "image_id": record["image_id"],
            "category_id": int(category_id),
            "bbox": [float(v) for v in record["bbox"]],
            "score": float(record.get("score", 1.0)),
        })
    return out


def evaluate(gt_flat, predictions_flat, quiet=False):
    COCO, COCOeval = load_pip_pycocotools()

    import tempfile

    # pycocotools' COCO() only reads GT from a file path.
    fd, gt_path = tempfile.mkstemp(suffix=".json", prefix="dentex_gt_flat_")
    with os.fdopen(fd, "w") as f:
        json.dump(gt_flat, f)

    # COCOeval prints its tables to stdout; capture them so callers can choose
    # whether to show them, and so they never interleave with progress output.
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            coco_gt = COCO(gt_path)
            if not predictions_flat:
                coco_eval = None
            else:
                coco_dt = coco_gt.loadRes(predictions_flat)
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
    finally:
        if gt_path:
            os.unlink(gt_path)
    if not quiet:
        print(sink.getvalue())
    if coco_eval is None:
        return {"AP": -1.0, "AP50": -1.0, "AP75": -1.0, "per_class_AP": {}, "num_predictions": 0,
                "summary_text": sink.getvalue()}

    stats = coco_eval.stats
    # precisions: [iou_thresholds, recall, class, area_range, max_dets]
    precisions = coco_eval.eval["precision"]
    per_class = {}
    for index, category_id in enumerate(coco_eval.params.catIds):
        precision = precisions[:, :, index, 0, -1]
        precision = precision[precision > -1]
        name = coco_gt.loadCats([category_id])[0]["name"]
        per_class[str(name)] = float(np.mean(precision) * 100) if precision.size else float("nan")

    return {
        "AP": float(stats[0] * 100),
        "AP50": float(stats[1] * 100),
        "AP75": float(stats[2] * 100),
        "APs": float(stats[3] * 100),
        "APm": float(stats[4] * 100),
        "APl": float(stats[5] * 100),
        "AR100": float(stats[8] * 100),
        "per_class_AP": per_class,
        "num_predictions": len(predictions_flat),
        "summary_text": sink.getvalue(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-json", required=True, help="DENTEX 3-tier COCO annotations (or already-flat COCO)")
    p.add_argument("--predictions", required=True, help="predictions JSON from tools/dump_predictions.py")
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--all-tiers", action="store_true",
                   help="evaluate tiers 0..--tier instead of only --tier")
    p.add_argument("--output", default=None, help="write results JSON here")
    p.add_argument("--quiet", action="store_true", help="suppress the COCOeval summary tables")
    args = p.parse_args()

    with open(args.gt_json) as f:
        gt = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)

    tiers = range(args.tier + 1) if args.all_tiers else [args.tier]
    results = {}
    for tier in tiers:
        if "categories_{}".format(tier + 1) not in gt:
            print("skipping tier {}: GT has no categories_{}".format(tier, tier + 1))
            continue
        print("\n=== tier {} ({}) ===".format(tier, TIERS[tier][1]))
        metrics = evaluate(flatten_gt(gt, tier), flatten_predictions(predictions, tier), quiet=args.quiet)
        metrics.pop("summary_text", None)
        results[str(tier)] = metrics
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_AP"}, indent=2))
        print("per-class AP: {}".format(
            json.dumps({k: round(v, 2) for k, v in metrics["per_class_AP"].items()})))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "gt_json": os.path.abspath(args.gt_json),
                "predictions": os.path.abspath(args.predictions),
                "evaluator": "standard pycocotools COCOeval (not the repo's fork)",
                "results_by_tier": results,
            }, f, indent=2)
        print("\nwrote {}".format(args.output))


if __name__ == "__main__":
    main()
