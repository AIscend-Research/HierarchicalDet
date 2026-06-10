"""
Phase 1 — Evaluation Subset Selection
Creates two evaluation subsets from the DENTEX annotation JSONs:

  1. CLEAN subset — images with complete annotations across all 3 tiers
     (every annotation has non-zero category_id_1, category_id_2, category_id_3)
     Used for: primary mAP evaluation matching paper conditions.

  2. STRESS-TEST subset — images with partial annotations, high annotation
     density, or edge-case tooth configurations.
     Used for: robustness analysis, failure case study.

Inputs:
    One or more COCO-triple JSON files (use --json flags).
    Typically: training_data annotation JSON + validation_triple.json

Usage:
    python phase1_select_subsets.py \
        --json /path/to/train_merged_disease_coco3class_onlyd_fixed.json \
        --json /path/to/validation_triple.json \
        --out-dir phase1_subsets/

Outputs:
    phase1_subsets/clean_subset.json         — COCO-triple format
    phase1_subsets/stress_test_subset.json   — COCO-triple format
    phase1_subsets/subset_summary.txt        — human-readable report
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="append", dest="json_files", required=True,
                   help="Path to COCO-triple JSON file. Repeat for multiple files.")
    p.add_argument("--out-dir", default="phase1_subsets")
    p.add_argument("--clean-size", type=int, default=100,
                   help="Target size of clean subset (may be smaller if data is limited)")
    p.add_argument("--stress-size", type=int, default=50,
                   help="Target size of stress-test subset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_and_merge(json_files):
    """Load and merge multiple COCO-triple JSONs into a single in-memory dataset."""
    merged = {
        "images": [],
        "annotations": [],
        "categories_1": None,
        "categories_2": None,
        "categories_3": None,
    }
    img_id_offset = 0
    ann_id_offset = 0
    img_id_remap = {}

    for path in json_files:
        print(f"  Loading {path}")
        with open(path) as f:
            data = json.load(f)

        # Sanity check
        assert "categories_1" in data, f"{path} is not a COCO-triple JSON"

        # Category consistency check
        if merged["categories_1"] is None:
            merged["categories_1"] = data["categories_1"]
            merged["categories_2"] = data["categories_2"]
            merged["categories_3"] = data["categories_3"]
        else:
            for k in ("categories_1", "categories_2", "categories_3"):
                assert len(data[k]) == len(merged[k]), (
                    f"Category count mismatch in {path} for {k}: "
                    f"{len(data[k])} vs {len(merged[k])}"
                )

        # Remap IDs to avoid collisions
        local_id_remap = {}
        for img in data["images"]:
            new_id = img["id"] + img_id_offset
            local_id_remap[img["id"]] = new_id
            new_img = dict(img)
            new_img["id"] = new_id
            new_img["source_file"] = str(path)
            merged["images"].append(new_img)

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = ann["id"] + ann_id_offset
            new_ann["image_id"] = local_id_remap[ann["image_id"]]
            merged["annotations"].append(new_ann)

        img_id_offset += max(img["id"] for img in data["images"]) + 1
        ann_id_offset += max(ann["id"] for ann in data["annotations"]) + 1

    print(f"  Merged: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    return merged


def classify_images(data):
    """
    For each image, compute per-tier completeness and edge-case flags.
    Returns dict: image_id -> classification dict.
    """
    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    classifications = {}
    for img in data["images"]:
        iid = img["id"]
        anns = ann_by_img[iid]

        if not anns:
            classifications[iid] = {
                "n_annotations": 0,
                "has_all3": False,
                "has_quadrant": False,
                "has_enumeration": False,
                "has_diagnosis": False,
                "partial": True,
                "stress_flags": ["no_annotations"],
            }
            continue

        has_q = any(a.get("category_id_1", 0) != 0 for a in anns)
        has_e = any(a.get("category_id_2", 0) != 0 for a in anns)
        has_d = any(a.get("category_id_3", 0) != 0 for a in anns)
        has_all3 = has_q and has_e and has_d

        # Diagnosis distribution (for stress-test: unusual pathology mix)
        diag_counts = defaultdict(int)
        for a in anns:
            d = a.get("category_id_3", 0)
            if d != 0:
                diag_counts[d] += 1

        # Edge-case flags
        stress_flags = []
        n = len(anns)

        if not has_all3:
            stress_flags.append("partial_annotation")
        if n >= 20:
            stress_flags.append("high_density")  # many teeth, complex panoramic
        if n <= 2 and has_d:
            stress_flags.append("sparse_with_diagnosis")  # very few annotations
        if len(diag_counts) >= 3:
            stress_flags.append("multi_diagnosis_type")  # multiple pathology types
        if diag_counts.get(1, 0) > 0:  # Impacted (rarest class)
            stress_flags.append("has_impacted")
        if any(a.get("category_id_2", 0) == 0 and a.get("category_id_3", 0) != 0 for a in anns):
            stress_flags.append("diagnosis_without_enumeration")  # annotation inconsistency

        # Quadrant distribution (for imbalance stress)
        quad_counts = Counter([a.get("category_id_1", 0) for a in anns if a.get("category_id_1", 0) != 0])
        if len(quad_counts) == 1:
            stress_flags.append("single_quadrant")  # only one quadrant annotated

        classifications[iid] = {
            "n_annotations": n,
            "has_all3": has_all3,
            "has_quadrant": has_q,
            "has_enumeration": has_e,
            "has_diagnosis": has_d,
            "partial": not has_all3,
            "diag_types": list(diag_counts.keys()),
            "stress_flags": stress_flags,
            "is_stress": len(stress_flags) > 0,
        }

    return classifications


try:
    from collections import Counter
except ImportError:
    from collections import Counter


def select_subsets(data, classifications, clean_size, stress_size, seed):
    """
    Select clean and stress-test subsets.
    Returns (clean_ids, stress_ids) as lists of image IDs.
    """
    rng = random.Random(seed)

    all_ids = [img["id"] for img in data["images"]]
    clean_candidates = [iid for iid in all_ids
                        if classifications[iid]["has_all3"]
                        and not classifications[iid]["is_stress"]]
    stress_candidates = [iid for iid in all_ids
                         if classifications[iid]["is_stress"]]
    # Also include partial-annotation images in stress
    partial_only = [iid for iid in all_ids
                    if classifications[iid]["partial"]
                    and iid not in stress_candidates]
    stress_candidates = list(set(stress_candidates + partial_only))

    # Sample
    rng.shuffle(clean_candidates)
    rng.shuffle(stress_candidates)
    clean_ids  = clean_candidates[:clean_size]
    stress_ids = stress_candidates[:stress_size]

    print(f"  Clean candidates: {len(clean_candidates)} → selected {len(clean_ids)}")
    print(f"  Stress candidates: {len(stress_candidates)} → selected {len(stress_ids)}")

    return clean_ids, stress_ids


def filter_to_subset(data, image_ids):
    """Return a new COCO-triple dict containing only the specified image_ids."""
    id_set = set(image_ids)
    images = [img for img in data["images"] if img["id"] in id_set]
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in id_set]
    return {
        "images": images,
        "annotations": annotations,
        "categories_1": data["categories_1"],
        "categories_2": data["categories_2"],
        "categories_3": data["categories_3"],
    }


def write_subset_summary(clean_ids, stress_ids, classifications, data, out_path):
    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    lines = [
        "=" * 65,
        "Phase 1 — Evaluation Subset Summary",
        "=" * 65,
        "",
        f"CLEAN SUBSET ({len(clean_ids)} images)",
        "  Criterion: all 3 annotation tiers present, no edge-case flags",
        "-" * 65,
        f"  {'image_id':>10}  {'n_anns':>6}  {'diag_types'}",
    ]
    for iid in sorted(clean_ids)[:20]:  # show first 20
        c = classifications[iid]
        lines.append(f"  {iid:>10}  {c['n_annotations']:>6}  {c.get('diag_types', [])}")
    if len(clean_ids) > 20:
        lines.append(f"  ... ({len(clean_ids)-20} more)")

    lines += [
        "",
        f"STRESS-TEST SUBSET ({len(stress_ids)} images)",
        "  Criterion: partial annotations, edge cases, high density, or unusual pathology",
        "-" * 65,
        f"  {'image_id':>10}  {'n_anns':>6}  {'flags'}",
    ]
    for iid in sorted(stress_ids)[:20]:
        c = classifications[iid]
        lines.append(f"  {iid:>10}  {c['n_annotations']:>6}  {c.get('stress_flags', [])}")
    if len(stress_ids) > 20:
        lines.append(f"  ... ({len(stress_ids)-20} more)")

    lines += [
        "",
        "STRESS FLAG BREAKDOWN",
        "-" * 65,
    ]
    flag_counts = Counter()
    for iid in stress_ids:
        for flag in classifications[iid].get("stress_flags", []):
            flag_counts[flag] += 1
    for flag, count in flag_counts.most_common():
        lines.append(f"  {flag:<35} {count:>4}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and merging annotation files...")
    data = load_and_merge(args.json_files)

    print("\nClassifying images...")
    classifications = classify_images(data)

    n_all3  = sum(1 for c in classifications.values() if c["has_all3"])
    n_part  = sum(1 for c in classifications.values() if c["partial"])
    n_stress = sum(1 for c in classifications.values() if c["is_stress"])
    print(f"  Images with all 3 tiers: {n_all3}")
    print(f"  Images with partial annotations: {n_part}")
    print(f"  Images with stress flags: {n_stress}")

    print("\nSelecting subsets...")
    clean_ids, stress_ids = select_subsets(
        data, classifications, args.clean_size, args.stress_size, args.seed
    )

    # Save subset JSONs
    clean_data  = filter_to_subset(data, clean_ids)
    stress_data = filter_to_subset(data, stress_ids)

    clean_path  = out_dir / "clean_subset.json"
    stress_path = out_dir / "stress_test_subset.json"

    with open(clean_path, "w") as f:
        json.dump(clean_data, f, indent=2)
    with open(stress_path, "w") as f:
        json.dump(stress_data, f, indent=2)

    print(f"\nSaved clean subset ({len(clean_ids)} images) → {clean_path}")
    print(f"Saved stress-test subset ({len(stress_ids)} images) → {stress_path}")

    # Summary
    write_subset_summary(clean_ids, stress_ids, classifications, data,
                         out_dir / "subset_summary.txt")

    # Machine-readable metadata
    meta = {
        "clean_image_ids": clean_ids,
        "stress_image_ids": stress_ids,
        "classifications": {str(k): v for k, v in classifications.items()},
        "params": vars(args),
    }
    with open(out_dir / "subset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {out_dir / 'subset_metadata.json'}")


if __name__ == "__main__":
    main()
