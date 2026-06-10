"""
Phase 1 — Dataset Audit Script
Run this AFTER extracting the DENTEX zip files.

Usage:
    python phase1_audit_dataset.py --data-root /path/to/extracted/dentex

Expected directory structure after extraction (inspect and adjust --data-root as needed):
    <data-root>/
        training_data/
            quadrant/           or similar per-tier folders
            quadrant_enumeration/
            quadrant_enumeration_diagnosis/
        validation_data/
        test_data/
        validation_triple.json  (from Hugging Face, already downloaded separately)

Outputs:
    phase1_audit_report.txt   — human-readable summary table
    phase1_audit_data.json    — machine-readable full audit results
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow not installed. Image resolution checks will be skipped.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Root directory of extracted DENTEX dataset")
    p.add_argument("--val-json", default=None, help="Path to validation_triple.json (auto-detected if omitted)")
    p.add_argument("--out", default="phase1_audit_report.txt")
    return p.parse_args()


# ── JSON discovery ─────────────────────────────────────────────────────────────

def find_json_files(root: Path):
    """Recursively find all .json files under root."""
    return sorted(root.rglob("*.json"))


def classify_json(path: Path):
    """
    Try to load a JSON and classify it:
      - coco_gt: standard COCO ground-truth format (has 'images', 'annotations', 'categories')
      - coco_triple: non-standard triple-category format (has 'categories_1', 'categories_2', 'categories_3')
      - coco_results: COCO results format (list of dicts with 'image_id', 'bbox', 'score')
      - unknown
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and "image_id" in data[0] and "bbox" in data[0]:
                return "coco_results", data
            return "unknown_list", data
        if isinstance(data, dict):
            if "categories_1" in data and "categories_2" in data:
                return "coco_triple", data
            if "annotations" in data and "categories" in data:
                return "coco_gt", data
            if "images" in data:
                return "coco_gt_partial", data
        return "unknown_dict", data
    except Exception as e:
        return "error", str(e)


# ── Image auditing ─────────────────────────────────────────────────────────────

def audit_images(img_dir: Path, sample_limit=None):
    """
    Return dict with:
      count, resolutions (list of (w,h)), missing (list), errors (list)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in img_dir.rglob("*") if f.suffix.lower() in exts]
    if sample_limit:
        files = files[:sample_limit]

    resolutions = []
    errors = []
    for f in files:
        try:
            if PIL_AVAILABLE:
                with Image.open(f) as img:
                    resolutions.append(img.size)  # (W, H)
            elif CV2_AVAILABLE:
                img = cv2.imread(str(f))
                if img is None:
                    errors.append(str(f))
                else:
                    h, w = img.shape[:2]
                    resolutions.append((w, h))
        except Exception as e:
            errors.append(f"{f}: {e}")

    return {
        "count": len(files),
        "files": [str(f) for f in files],
        "resolutions": resolutions,
        "errors": errors,
    }


# ── Annotation auditing ────────────────────────────────────────────────────────

def audit_triple_json(data: dict):
    """
    Audit a coco_triple format JSON (the HierarchicalDet annotation format).
    Returns a structured summary.
    """
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    cats1 = data.get("categories_1", [])
    cats2 = data.get("categories_2", [])
    cats3 = data.get("categories_3", [])

    img_ids = {img["id"] for img in images}
    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann["image_id"]].append(ann)

    # Per-annotation stats
    cat1_counter = Counter()
    cat2_counter = Counter()
    cat3_counter = Counter()
    no_diag_count = 0    # category_id_3 == 0 → partial annotation (no diagnosis)
    no_enum_count = 0    # category_id_2 == 0 → partial annotation (no enumeration)
    bbox_areas = []
    degenerate_boxes = []

    for ann in annotations:
        c1 = ann.get("category_id_1", ann.get("category_id", None))
        c2 = ann.get("category_id_2", 0)
        c3 = ann.get("category_id_3", 0)
        cat1_counter[c1] += 1
        cat2_counter[c2] += 1
        cat3_counter[c3] += 1
        if c3 == 0:
            no_diag_count += 1
        if c2 == 0:
            no_enum_count += 1
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4:
            w, h = bbox[2], bbox[3]
            area = w * h
            bbox_areas.append(area)
            if w <= 0 or h <= 0:
                degenerate_boxes.append(ann["id"])

    # Per-image completeness
    complete_all3 = 0
    partial_q_only = 0
    partial_qe = 0
    imgs_no_ann = 0

    for img_id in img_ids:
        anns = ann_by_img[img_id]
        if not anns:
            imgs_no_ann += 1
            continue
        has_diag = any(a.get("category_id_3", 0) != 0 for a in anns)
        has_enum = any(a.get("category_id_2", 0) != 0 for a in anns)
        if has_diag and has_enum:
            complete_all3 += 1
        elif has_enum:
            partial_qe += 1
        else:
            partial_q_only += 1

    return {
        "n_images": len(images),
        "n_annotations": len(annotations),
        "n_categories_1": len(cats1),
        "n_categories_2": len(cats2),
        "n_categories_3": len(cats3),
        "categories_1": cats1,
        "categories_2": cats2,
        "categories_3": cats3,
        "images_with_all3_tiers": complete_all3,
        "images_quadrant_enumeration_only": partial_qe,
        "images_quadrant_only": partial_q_only,
        "images_no_annotation": imgs_no_ann,
        "annotations_no_diagnosis": no_diag_count,
        "annotations_no_enumeration": no_enum_count,
        "degenerate_boxes": degenerate_boxes,
        "bbox_area_stats": {
            "min": float(np.min(bbox_areas)) if bbox_areas else None,
            "max": float(np.max(bbox_areas)) if bbox_areas else None,
            "mean": float(np.mean(bbox_areas)) if bbox_areas else None,
            "median": float(np.median(bbox_areas)) if bbox_areas else None,
        },
        "category_1_distribution": dict(cat1_counter),
        "category_2_distribution": dict(cat2_counter),
        "category_3_distribution": dict(cat3_counter),
    }


def audit_coco_gt(data: dict):
    """Audit a standard COCO GT JSON."""
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    img_ids_with_ann = {a["image_id"] for a in annotations}
    return {
        "n_images": len(images),
        "n_annotations": len(annotations),
        "n_categories": len(categories),
        "categories": categories,
        "images_with_annotations": len(img_ids_with_ann),
        "images_without_annotations": len(images) - len(img_ids_with_ann),
    }


# ── Resolution analysis ────────────────────────────────────────────────────────

def resolution_stats(resolutions):
    if not resolutions:
        return {}
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    unique = Counter(resolutions)
    return {
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": float(np.mean(widths)),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": float(np.mean(heights)),
        "n_unique_resolutions": len(unique),
        "top5_resolutions": unique.most_common(5),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root = Path(args.data_root)
    assert root.exists(), f"Data root not found: {root}"

    report = {}

    print(f"\n{'='*60}")
    print(f"  DENTEX Dataset Audit")
    print(f"  Root: {root}")
    print(f"{'='*60}\n")

    # ── 1. Discover all JSON annotation files ─────────────────────────────────
    print("[1/5] Discovering JSON annotation files...")
    json_files = find_json_files(root)
    print(f"  Found {len(json_files)} JSON files")

    json_audit = {}
    for jf in json_files:
        kind, data = classify_json(jf)
        key = str(jf.relative_to(root))
        print(f"  [{kind}] {key}")
        if kind == "coco_triple":
            json_audit[key] = {"type": kind, "audit": audit_triple_json(data)}
        elif kind in ("coco_gt", "coco_gt_partial"):
            json_audit[key] = {"type": kind, "audit": audit_coco_gt(data)}
        elif kind == "coco_results":
            json_audit[key] = {"type": kind, "n_predictions": len(data)}
        else:
            json_audit[key] = {"type": kind}

    report["json_files"] = json_audit

    # ── 2. Validate_triple.json (Hugging Face separate download) ─────────────
    val_json_path = args.val_json or root / "validation_triple.json"
    if val_json_path and Path(val_json_path).exists():
        print(f"\n[2/5] Auditing validation_triple.json...")
        kind, data = classify_json(Path(val_json_path))
        if kind == "coco_triple":
            vt = audit_triple_json(data)
            report["validation_triple"] = vt
            print(f"  Images: {vt['n_images']}")
            print(f"  Annotations: {vt['n_annotations']}")
            print(f"  Images with all 3 tiers: {vt['images_with_all3_tiers']}")
            print(f"  Categories_3 (diagnoses): {vt['categories_3']}")
    else:
        print(f"\n[2/5] validation_triple.json not found at {val_json_path} — skip")

    # ── 3. Audit image directories ────────────────────────────────────────────
    print(f"\n[3/5] Auditing image directories (this may take a while)...")
    img_audit = {}
    # Common expected dirs — adjust if the zip extracts differently
    candidate_dirs = [
        root / "training_data",
        root / "validation_data",
        root / "test_data",
        root / "train",
        root / "val",
        root / "test",
    ]
    # Also check subdirs of root
    for d in root.iterdir():
        if d.is_dir() and d not in candidate_dirs:
            candidate_dirs.append(d)

    for d in candidate_dirs:
        if not d.exists():
            continue
        print(f"  Scanning {d.name}/ ...")
        result = audit_images(d, sample_limit=500)
        res_stats = resolution_stats(result["resolutions"])
        img_audit[d.name] = {
            "n_images": result["count"],
            "resolution_stats": res_stats,
            "load_errors": result["errors"],
            "n_errors": len(result["errors"]),
        }
        print(f"    {result['count']} images, {len(result['errors'])} load errors")
        if res_stats:
            print(f"    Resolution: W={res_stats['width_min']}–{res_stats['width_max']}, "
                  f"H={res_stats['height_min']}–{res_stats['height_max']}")

    report["image_directories"] = img_audit

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print(f"\n[4/5] Building summary table...")

    summary_rows = []
    for json_key, json_info in json_audit.items():
        if json_info.get("type") in ("coco_triple", "coco_gt", "coco_gt_partial"):
            a = json_info.get("audit", {})
            row = {
                "json": json_key,
                "type": json_info["type"],
                "n_images": a.get("n_images", "?"),
                "n_annotations": a.get("n_annotations", "?"),
                "all3_tiers": a.get("images_with_all3_tiers", "?"),
                "partial_qe": a.get("images_quadrant_enumeration_only", "?"),
                "partial_q": a.get("images_quadrant_only", "?"),
                "no_ann": a.get("images_no_annotation", "?"),
                "degenerate_boxes": len(a.get("degenerate_boxes", [])),
            }
            summary_rows.append(row)

    if "validation_triple" in report:
        vt = report["validation_triple"]
        summary_rows.append({
            "json": "validation_triple.json (HF)",
            "type": "coco_triple",
            "n_images": vt["n_images"],
            "n_annotations": vt["n_annotations"],
            "all3_tiers": vt["images_with_all3_tiers"],
            "partial_qe": vt["images_quadrant_enumeration_only"],
            "partial_q": vt["images_quadrant_only"],
            "no_ann": vt["images_no_annotation"],
            "degenerate_boxes": len(vt["degenerate_boxes"]),
        })

    report["summary_table"] = summary_rows

    # ── 5. Cross-check: paper vs public release ───────────────────────────────
    print(f"\n[5/5] Cross-checking against paper description...")

    paper_claims = {
        "quadrant_only": 693,
        "quadrant_enumeration": 634,
        "full_diagnosis_total": 1005,
        "full_diagnosis_train": 705,
        "full_diagnosis_val": 50,
        "full_diagnosis_test": 250,
    }

    total_images = sum(v["n_images"] for v in img_audit.values())
    print(f"  Total images found: {total_images}")
    print(f"  Paper claims total labeled: {693+634+1005} + 1571 unlabeled")

    report["paper_cross_check"] = {
        "paper_claims": paper_claims,
        "found_image_counts": {k: v["n_images"] for k, v in img_audit.items()},
        "found_json_image_counts": {k: v.get("audit", {}).get("n_images", None) for k, v in json_audit.items()},
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    with open("phase1_audit_data.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull audit data saved to phase1_audit_data.json")

    # Human-readable report
    lines = [
        "=" * 70,
        "DENTEX Dataset Audit Report — Phase 1",
        f"Root: {root}",
        "=" * 70,
        "",
        "SUMMARY TABLE",
        "-" * 70,
        f"{'JSON File':<45} {'Imgs':>5} {'Anns':>6} {'All3':>5} {'QE':>4} {'Q':>4} {'NoAnn':>6} {'BadBox':>7}",
        "-" * 70,
    ]
    for row in summary_rows:
        lines.append(
            f"{row['json'][:44]:<45} {str(row['n_images']):>5} {str(row['n_annotations']):>6} "
            f"{str(row['all3_tiers']):>5} {str(row['partial_qe']):>4} {str(row['partial_q']):>4} "
            f"{str(row['no_ann']):>6} {str(row['degenerate_boxes']):>7}"
        )
    lines += [
        "",
        "IMAGE DIRECTORY SUMMARY",
        "-" * 70,
    ]
    for dir_name, di in img_audit.items():
        rs = di.get("resolution_stats", {})
        lines.append(f"  {dir_name}/: {di['n_images']} images, {di['n_errors']} load errors")
        if rs:
            lines.append(f"    Width: {rs.get('width_min')}–{rs.get('width_max')} px  "
                         f"Height: {rs.get('height_min')}–{rs.get('height_max')} px")
    lines += [
        "",
        "PAPER vs PUBLIC RELEASE",
        "-" * 70,
        f"  Paper claims: 693 quadrant-only, 634 quadrant-enum, 1005 full-diag",
        f"  Paper full-diag split: 705 train / 50 val / 250 test",
        "",
        "ANNOTATION FORMAT",
        "-" * 70,
        "  Non-standard COCO with 3 parallel category fields per annotation:",
        "  category_id_1: quadrant (1–4)",
        "  category_id_2: tooth enumeration (1–8, 0 = not annotated)",
        "  category_id_3: diagnosis (1–4, 0 = not annotated)",
        "  Diagnoses: 1=Impacted, 2=Caries, 3=Periapical Lesion, 4=Deep Caries",
        "",
        "DATALOADER COMPATIBILITY",
        "-" * 70,
        "  BLOCKER: DiffusionDetDatasetMapper.__init__() tries to open TWO",
        "  hardcoded pre-computed noisy-box JSON files that are NOT in the",
        "  public release. Use dataset_mapper_patched.py (see repo).",
        "  Paths expected by original code:",
        "    ibrahim/Diseasedataset_base_enumeration_m_t_inference_train/",
        "      inference/coco_instances_results.json",
        "    ibrahim/Diseasedataset_base_enumeration_m_t_inference_val/",
        "      inference/coco_instances_results.json",
    ]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Human-readable report saved to {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
