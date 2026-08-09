"""
Overlay ground-truth and predicted boxes on panoramic X-rays for qualitative
comparison, with the full quadrant / tooth / diagnosis label on each box.

Ground truth is drawn in green, predictions in red, on the same image, so
missed teeth and false positives are visible at a glance. Labels are rendered
at whichever tiers are present, e.g. `Q2 N7 Impacted 0.83`.

Reads the same prediction dumps as everything else
(tools/dump_predictions.py), so any run -- clean, degraded, different diffusion
step counts -- can be visualized without re-running inference.
"""
import argparse
import json
import os
from collections import defaultdict

import cv2

GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 0, 235)


def draw_box(image, bbox, color, text, thickness=2):
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    if not text:
        return
    font, scale, text_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, text_thickness)
    top = max(y - text_height - baseline, 0)
    cv2.rectangle(image, (x, top), (x + text_width + 4, top + text_height + baseline), color, -1)
    cv2.putText(image, text, (x + 2, top + text_height), font, scale, (255, 255, 255), text_thickness,
                cv2.LINE_AA)


def label_for(record, class_names, tier, score=None):
    parts = []
    for t in range(tier + 1):
        key = "category_id_{}".format(t + 1)
        value = record.get(key)
        if value is None:
            continue
        name = class_names.get(t, {}).get(int(value), value)
        parts.append({0: "Q{}", 1: "N{}", 2: "{}"}[t].format(name))
    if score is not None:
        parts.append("{:.2f}".format(score))
    return " ".join(str(p) for p in parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--predictions", default=None,
                   help="prediction dump; omit to render ground truth only")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=20, help="number of images to render")
    p.add_argument("--images", nargs="*", default=None,
                   help="specific file names to render (overrides --limit ordering)")
    args = p.parse_args()

    with open(args.gt_json) as f:
        gt_json = json.load(f)
    class_names = {
        tier: {c["id"]: str(c["name"]) for c in gt_json["categories_{}".format(tier + 1)]}
        for tier in range(3) if "categories_{}".format(tier + 1) in gt_json
    }
    gt_by_image = defaultdict(list)
    for ann in gt_json["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    pred_by_image = defaultdict(list)
    if args.predictions:
        with open(args.predictions) as f:
            for record in json.load(f):
                if record.get("score", 1.0) >= args.score_thresh:
                    pred_by_image[record["image_id"]].append(record)

    images = gt_json["images"]
    if args.images:
        wanted = {os.path.basename(n) for n in args.images}
        images = [im for im in images if im["file_name"] in wanted]
    else:
        images = images[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    for image_info in images:
        path = os.path.join(args.image_dir, image_info["file_name"])
        image = cv2.imread(path)
        if image is None:
            print("could not read {}".format(path))
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for ann in gt_by_image[image_info["id"]]:
            draw_box(image, ann["bbox"], GT_COLOR, label_for(ann, class_names, args.tier))
        for record in pred_by_image.get(image_info["id"], []):
            draw_box(image, record["bbox"], PRED_COLOR,
                     label_for(record, class_names, args.tier, record.get("score")))

        cv2.putText(image, "green = ground truth   red = prediction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(
            args.output_dir, os.path.splitext(image_info["file_name"])[0] + "_overlay.jpg")
        cv2.imwrite(out_path, image)
        print("wrote {} (gt: {}, pred: {})".format(
            out_path, len(gt_by_image[image_info["id"]]), len(pred_by_image.get(image_info["id"], []))))


if __name__ == "__main__":
    main()
