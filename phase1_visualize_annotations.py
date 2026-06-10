"""
Phase 1 — 5-Image Visual Annotation Verification

Renders bounding boxes and hierarchical labels on panoramic X-rays to confirm:
  1. Bounding boxes align with visible teeth
  2. Quadrant labels match the anatomical quadrant
  3. Enumeration labels are consistent within each quadrant
  4. Diagnosis labels are plausible (diagnosed teeth look abnormal)
  5. Annotation tier labels are consistent across the hierarchy

Usage:
    python phase1_visualize_annotations.py \
        --json /path/to/validation_triple.json \
        --img-dir /path/to/validation_data/images \
        --n 5 \
        --out-dir phase1_visuals/

Output:
    phase1_visuals/<image_name>_annotated.png  — one image per X-ray
    phase1_visuals/visual_check_report.txt     — pass/fail table for each of the 5 images
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import numpy as np


# ── FDI system reference ───────────────────────────────────────────────────────
# Quadrant mapping: 1=Upper Right, 2=Upper Left, 3=Lower Left, 4=Lower Right
QUADRANT_NAMES = {1: "UpperRight", 2: "UpperLeft", 3: "LowerLeft", 4: "LowerRight"}

# Diagnosis labels (from validation_triple.json categories_3)
DIAGNOSIS_NAMES = {0: "None", 1: "Impacted", 2: "Caries", 3: "PeriapicalLesion", 4: "DeepCaries"}

# Visually distinct colors per quadrant (BGR for OpenCV)
QUADRANT_COLORS = {
    1: (0, 200, 0),    # green
    2: (200, 120, 0),  # orange
    3: (0, 0, 220),    # red
    4: (180, 0, 180),  # purple
    0: (128, 128, 128),  # gray (unknown)
}

# Diagnosis marker colors
DIAGNOSIS_COLORS = {
    0: None,
    1: (0, 255, 255),    # yellow (impacted)
    2: (0, 165, 255),    # orange (caries)
    3: (0, 0, 255),      # red (periapical)
    4: (255, 0, 255),    # magenta (deep caries)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to COCO-triple annotation JSON")
    p.add_argument("--img-dir", required=True, help="Directory containing the X-ray images")
    p.add_argument("--n", type=int, default=5, help="Number of images to visualize")
    p.add_argument("--image-ids", nargs="+", type=int, default=None,
                   help="Specific image IDs to visualize (overrides --n random selection)")
    p.add_argument("--out-dir", default="phase1_visuals")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=float, default=0.4,
                   help="Scale factor for output image (panoramics are ~3000px wide)")
    return p.parse_args()


def find_image_file(img_dir, filename):
    """Try to find an image by filename in img_dir, trying common extensions."""
    base = Path(filename).stem
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]
    for ext in exts:
        p = Path(img_dir) / (base + ext)
        if p.exists():
            return p
        p = Path(img_dir) / filename
        if p.exists():
            return p
    # Recursive search fallback
    for f in Path(img_dir).rglob(base + ".*"):
        return f
    return None


def draw_annotations_cv2(img, annotations, scale=1.0):
    """Draw bounding boxes and labels on image using OpenCV."""
    h, w = img.shape[:2]
    out = img.copy()

    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        x, y, bw, bh = int(x*scale), int(y*scale), int(bw*scale), int(bh*scale)

        c1 = ann.get("category_id_1", 0)
        c2 = ann.get("category_id_2", 0)
        c3 = ann.get("category_id_3", 0)

        color = QUADRANT_COLORS.get(c1, (128, 128, 128))
        thickness = 2 if c3 == 0 else 3

        # Draw bounding box
        cv2.rectangle(out, (x, y), (x+bw, y+bh), color, thickness)

        # Label: Q<c1> N<c2> [D<c3>]
        label_parts = [f"Q{c1}"]
        if c2 != 0:
            label_parts.append(f"N{c2}")
        if c3 != 0:
            diag = DIAGNOSIS_NAMES.get(c3, str(c3))
            label_parts.append(diag[:4])  # truncate for readability

        label = " ".join(label_parts)
        font_scale = 0.45 * scale
        font_thickness = max(1, int(thickness * scale))

        # Background rectangle for label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(out, (x, y-th-4), (x+tw+4, y), color, -1)
        cv2.putText(out, label, (x+2, y-3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Diagnosis marker: colored dot inside box
        if c3 != 0:
            d_color = DIAGNOSIS_COLORS.get(c3, (255, 255, 255))
            cx, cy = x + bw//2, y + bh//2
            cv2.circle(out, (cx, cy), max(4, int(8*scale)), d_color, -1)

    return out


def consistency_checks(img_id, annotations):
    """
    Perform automated consistency checks for a single image.
    Returns list of (check_name, pass/fail, message).
    """
    checks = []

    if not annotations:
        checks.append(("has_annotations", False, "No annotations for this image"))
        return checks

    checks.append(("has_annotations", True, f"{len(annotations)} annotations"))

    # 1. Every annotation has a valid quadrant
    bad_q = [a["id"] for a in annotations if a.get("category_id_1", 0) not in (1, 2, 3, 4)]
    checks.append((
        "valid_quadrant_ids",
        len(bad_q) == 0,
        f"Annotations with invalid quadrant: {bad_q}" if bad_q else "All quadrant IDs valid (1-4)"
    ))

    # 2. Enumeration IDs in 0-8 range
    bad_e = [a["id"] for a in annotations if a.get("category_id_2", 0) not in range(9)]
    checks.append((
        "valid_enumeration_ids",
        len(bad_e) == 0,
        f"Annotations with invalid enumeration: {bad_e}" if bad_e else "All enumeration IDs valid (0-8)"
    ))

    # 3. Diagnosis IDs in 0-4 range
    bad_d = [a["id"] for a in annotations if a.get("category_id_3", 0) not in range(5)]
    checks.append((
        "valid_diagnosis_ids",
        len(bad_d) == 0,
        f"Annotations with invalid diagnosis: {bad_d}" if bad_d else "All diagnosis IDs valid (0-4)"
    ))

    # 4. No degenerate bounding boxes
    bad_bbox = [a["id"] for a in annotations
                if a.get("bbox", [0,0,0,0])[2] <= 0 or a.get("bbox", [0,0,0,0])[3] <= 0]
    checks.append((
        "valid_bboxes",
        len(bad_bbox) == 0,
        f"Degenerate boxes (w or h <= 0): {bad_bbox}" if bad_bbox else "All bounding boxes valid"
    ))

    # 5. Quadrant/enumeration hierarchy consistency
    # FDI: Q1=teeth 11-18, Q2=21-28, Q3=31-38, Q4=41-48
    # Our encoding: Q in {1,2,3,4}, N in {1..8} within quadrant
    # Check that if N is annotated, Q is also annotated
    broken_hierarchy = [
        a["id"] for a in annotations
        if a.get("category_id_2", 0) != 0 and a.get("category_id_1", 0) == 0
    ]
    checks.append((
        "hierarchy_q_before_n",
        len(broken_hierarchy) == 0,
        f"Annotations with N but no Q: {broken_hierarchy}" if broken_hierarchy
        else "Hierarchy consistent (Q present whenever N present)"
    ))

    # 6. If diagnosis present, enumeration should be present
    diag_without_enum = [
        a["id"] for a in annotations
        if a.get("category_id_3", 0) != 0 and a.get("category_id_2", 0) == 0
    ]
    checks.append((
        "hierarchy_n_before_d",
        len(diag_without_enum) == 0,
        f"Annotations with D but no N: {diag_without_enum}" if diag_without_enum
        else "Hierarchy consistent (N present whenever D present)"
    ))

    # 7. Annotation density sanity
    n = len(annotations)
    checks.append((
        "annotation_count_sane",
        1 <= n <= 32,  # max 32 teeth in FDI system
        f"Unusual annotation count: {n}" if not (1 <= n <= 32) else f"Annotation count OK: {n}"
    ))

    return checks


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CV2_AVAILABLE:
        print("ERROR: OpenCV (cv2) required. Run: pip install opencv-python")
        return

    # Load annotations
    print(f"Loading {args.json}...")
    with open(args.json) as f:
        data = json.load(f)

    assert "categories_1" in data, "Expected COCO-triple format JSON"

    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    img_map = {img["id"]: img for img in data["images"]}

    # Select images
    if args.image_ids:
        selected_ids = [iid for iid in args.image_ids if iid in img_map]
    else:
        rng = random.Random(args.seed)
        # Prefer images with all 3 tiers for the visual check
        all3_ids = [
            iid for iid, anns in ann_by_img.items()
            if all(a.get(f"category_id_{k}", 0) != 0 for a in anns for k in [1, 2, 3])
            and iid in img_map
        ]
        partial_ids = [iid for iid in img_map if iid not in all3_ids]

        # Take 3 complete + 2 partial for a representative 5-image check
        selected_ids = (
            rng.sample(all3_ids, min(3, len(all3_ids))) +
            rng.sample(partial_ids, min(2, len(partial_ids)))
        )[:args.n]

    print(f"Selected {len(selected_ids)} images for visual verification")

    # Process each image
    all_check_results = []
    for iid in selected_ids:
        img_meta = img_map[iid]
        filename = img_meta.get("file_name", f"{iid}.png")
        anns = ann_by_img[iid]

        print(f"\nImage {iid}: {filename} ({len(anns)} annotations)")

        # Find image file
        img_path = find_image_file(args.img_dir, filename)
        if img_path is None:
            print(f"  ✗ Image file not found in {args.img_dir}")
            all_check_results.append({
                "image_id": iid,
                "filename": filename,
                "found": False,
                "checks": [],
            })
            continue

        print(f"  Found: {img_path}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ✗ Failed to load image")
            all_check_results.append({
                "image_id": iid, "filename": filename,
                "found": True, "loaded": False, "checks": [],
            })
            continue

        h, w = img.shape[:2]
        print(f"  Resolution: {w}×{h} px")

        # Run consistency checks
        checks = consistency_checks(iid, anns)
        for check_name, passed, msg in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {msg}")

        # Draw annotations
        scale = args.scale
        display = cv2.resize(img, (int(w * scale), int(h * scale)))
        display = draw_annotations_cv2(display, anns, scale=scale)

        # Add legend
        legend_y = 20
        for q, qname in QUADRANT_NAMES.items():
            color = QUADRANT_COLORS[q]
            cv2.rectangle(display, (5, legend_y-12), (20, legend_y+2), color, -1)
            cv2.putText(display, f"Q{q}: {qname}", (25, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            legend_y += 18

        for d, dname in DIAGNOSIS_NAMES.items():
            if d == 0:
                continue
            d_color = DIAGNOSIS_COLORS[d]
            cv2.circle(display, (12, legend_y-5), 6, d_color, -1)
            cv2.putText(display, dname, (25, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, d_color, 1, cv2.LINE_AA)
            legend_y += 16

        # Annotation tier info
        has_q = any(a.get("category_id_1", 0) != 0 for a in anns)
        has_e = any(a.get("category_id_2", 0) != 0 for a in anns)
        has_d = any(a.get("category_id_3", 0) != 0 for a in anns)
        tier_str = f"Tiers: Q={'Y' if has_q else 'N'} E={'Y' if has_e else 'N'} D={'Y' if has_d else 'N'}"
        cv2.putText(display, tier_str, (5, h*scale - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        out_path = out_dir / (Path(filename).stem + "_annotated.png")
        cv2.imwrite(str(out_path), display)
        print(f"  Saved: {out_path}")

        all_check_results.append({
            "image_id": iid,
            "filename": filename,
            "resolution": f"{w}×{h}",
            "n_annotations": len(anns),
            "has_q": has_q, "has_e": has_e, "has_d": has_d,
            "all_checks_pass": all(c[1] for c in checks),
            "checks": [(c[0], c[1], c[2]) for c in checks],
            "output_file": str(out_path),
        })

    # Write report
    report_lines = [
        "=" * 65,
        "Phase 1 — Visual Annotation Verification Report",
        "=" * 65,
        "",
        f"{'Image ID':>10}  {'Filename':<30}  {'Res':>12}  {'Anns':>5}  {'Q':>2}  {'E':>2}  {'D':>2}  {'Pass':>5}",
        "-" * 65,
    ]
    for r in all_check_results:
        report_lines.append(
            f"{r['image_id']:>10}  {r.get('filename','?')[:29]:<30}  "
            f"{r.get('resolution','?'):>12}  {r.get('n_annotations',0):>5}  "
            f"{'Y' if r.get('has_q') else 'N':>2}  {'Y' if r.get('has_e') else 'N':>2}  "
            f"{'Y' if r.get('has_d') else 'N':>2}  {'✓' if r.get('all_checks_pass') else '✗':>5}"
        )
    report_lines += ["", "DETAILED CHECK RESULTS", "-" * 65]
    for r in all_check_results:
        report_lines.append(f"\n  Image {r['image_id']} ({r.get('filename','?')}):")
        for check_name, passed, msg in r.get("checks", []):
            report_lines.append(f"    {'✓' if passed else '✗'} {check_name}: {msg}")

    report_path = out_dir / "visual_check_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n" + "\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
