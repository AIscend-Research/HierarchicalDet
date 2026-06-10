"""
Phase 2 — Convert DENTEX Triple-Label Annotations to Standard COCO Format

Baseline models (RetinaNet, Faster R-CNN) expect a single category_id per
annotation. This script converts the DENTEX triple-label format to standard
COCO by selecting ONE category dimension.

Modes:
  diagnosis_only   → category_id = category_id_3 (0 excluded = no detection)
  quadrant_only    → category_id = category_id_1
  enumeration_only → category_id = category_id_2
  fdi              → FDI tooth number = (category_id_1 - 1) * 8 + category_id_2
                     (combines quadrant + position into a single 32-class problem)

Usage:
    python phase2_convert_to_single_label_coco.py \
        --train-json /path/to/train_merged_disease_coco3class_onlyd_fixed.json \
        --val-json   /path/to/test_merged_disease_coco3class.json \
        --out-dir    /path/to/single_label/ \
        --mode       diagnosis_only
"""

import argparse
import json
from pathlib import Path


DIAGNOSIS_CATEGORIES = [
    {"id": 1, "name": "Impacted",         "supercategory": "diagnosis"},
    {"id": 2, "name": "Caries",           "supercategory": "diagnosis"},
    {"id": 3, "name": "PeriapicalLesion", "supercategory": "diagnosis"},
    {"id": 4, "name": "DeepCaries",       "supercategory": "diagnosis"},
]

QUADRANT_CATEGORIES = [
    {"id": 1, "name": "Quadrant1", "supercategory": "quadrant"},
    {"id": 2, "name": "Quadrant2", "supercategory": "quadrant"},
    {"id": 3, "name": "Quadrant3", "supercategory": "quadrant"},
    {"id": 4, "name": "Quadrant4", "supercategory": "quadrant"},
]

ENUMERATION_CATEGORIES = [
    {"id": i, "name": f"Tooth{i}", "supercategory": "enumeration"}
    for i in range(1, 9)
]

FDI_CATEGORIES = [
    {"id": (q - 1) * 8 + n, "name": f"Q{q}N{n}", "supercategory": "fdi"}
    for q in range(1, 5) for n in range(1, 9)
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json",   required=True)
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--mode", choices=["diagnosis_only", "quadrant_only",
                                       "enumeration_only", "fdi"],
                   default="diagnosis_only")
    return p.parse_args()


def convert(data: dict, mode: str) -> dict:
    """Convert a triple-label COCO dict to single-label."""

    if mode == "diagnosis_only":
        categories = DIAGNOSIS_CATEGORIES
        def get_cat_id(ann):
            return ann.get("category_id_3", 0)

    elif mode == "quadrant_only":
        categories = QUADRANT_CATEGORIES
        def get_cat_id(ann):
            return ann.get("category_id_1", 0)

    elif mode == "enumeration_only":
        categories = ENUMERATION_CATEGORIES
        def get_cat_id(ann):
            return ann.get("category_id_2", 0)

    elif mode == "fdi":
        categories = FDI_CATEGORIES
        def get_cat_id(ann):
            q = ann.get("category_id_1", 0)
            n = ann.get("category_id_2", 0)
            if q == 0 or n == 0:
                return 0
            return (q - 1) * 8 + n

    # Filter annotations where the chosen category is 0 (not annotated)
    valid_anns = []
    skipped = 0
    for ann in data["annotations"]:
        cat_id = get_cat_id(ann)
        if cat_id == 0:
            skipped += 1
            continue
        new_ann = {
            "id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": cat_id,
            "bbox": ann["bbox"],
            "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
            "iscrowd": ann.get("iscrowd", 0),
        }
        valid_anns.append(new_ann)

    print(f"  Annotations: {len(data['annotations'])} total, "
          f"{len(valid_anns)} kept, {skipped} skipped (category=0)")

    # Keep only images that have at least one annotation
    img_ids_with_ann = {ann["image_id"] for ann in valid_anns}
    valid_images = [img for img in data["images"] if img["id"] in img_ids_with_ann]
    print(f"  Images: {len(data['images'])} total, {len(valid_images)} with annotations")

    return {
        "images": valid_images,
        "annotations": valid_anns,
        "categories": categories,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, json_path in [("train", args.train_json), ("val", args.val_json)]:
        print(f"\nConverting {split_name}: {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        if "categories_1" not in data:
            print(f"  WARNING: {json_path} does not appear to be a triple-label JSON. Skipping.")
            continue

        converted = convert(data, args.mode)
        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(converted, f)
        print(f"  Saved → {out_path}")

    # Write mode metadata
    meta = {"mode": args.mode, "train": args.train_json, "val": args.val_json}
    with open(out_dir / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Mode: {args.mode}")
    print(f"Output: {out_dir}/{{train,val}}.json")


if __name__ == "__main__":
    main()
