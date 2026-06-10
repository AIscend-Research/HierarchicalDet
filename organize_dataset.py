"""
organize_dataset.py — Map extracted DENTEX files into the directory structure
expected by train_net_patched.py.

Run this after extracting all three zips from sorted/dentex_raw/DENTEX/.

What this does:
  1. Inspects the JSON formats in training_data/ and validates they match
     the expected triple-label COCO format (category_id_1/2/3)
  2. Creates sorted/challenge/ with symlinks to the actual files so
     train_net_patched.py can find them via DATA_ROOT=sorted/challenge
  3. Prints the export commands to set DATA_ROOT correctly

Expected input layout (after unzip):
  sorted/
    training_data/
      quadrant/train_quadrant.json + xrays/
      quadrant_enumeration/train_quadrant_enumeration.json + xrays/
      quadrant-enumeration-disease/train_quadrant_enumeration_disease.json + xrays/
    validation_data/
      quadrant_enumeration_disease/xrays/
    disease/input/   (test images)
    dentex_raw/DENTEX/validation_triple.json

Output layout (symlinks in sorted/challenge/):
  sorted/challenge/
    train_merged_disease_coco3class_onlyd_fixed.json  → tier3 train JSON
    train_enumeration_coco.json                        → tier2 train JSON
    train_quadrant_coco.json                           → tier1 train JSON
    test_merged_disease_coco3class.json                → validation JSON (all tiers)
    test_enumeration_coco.json                         → validation JSON
    test_quadrant_coco.json                            → validation JSON
    for_coco_disease_train/                            → tier3 train images
    for_coco_enumeration_train/                        → tier2 train images
    for_coco_quadrant_train/                           → tier1 train images
    for_coco_disease_test/                             → validation images
    for_coco_enumeration_test/                         → validation images (symlink)
    for_coco_quadrant_test/                            → validation images (symlink)
"""

import json
import os
import sys
from pathlib import Path


def check_json(path, label):
    if not Path(path).exists():
        print(f"  MISSING: {label} → {path}")
        return False
    with open(path) as f:
        d = json.load(f)
    keys = list(d.keys())
    n_img = len(d.get("images", []))
    n_ann = len(d.get("annotations", []))
    ann0 = d["annotations"][0] if d.get("annotations") else {}
    has_triple = all(f"category_id_{i}" in ann0 for i in [1, 2, 3])
    print(f"  ✓ {label}: {n_img} images, {n_ann} anns, triple={has_triple}")
    return True


def symlink(src, dst):
    src = Path(src).resolve()
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink() if dst.is_symlink() else None
        if dst.is_dir():
            print(f"  EXISTS (dir): {dst.name}")
            return
    if not src.exists():
        print(f"  MISSING SOURCE: {src}")
        return
    dst.symlink_to(src)
    print(f"  → {dst.name}")


def _convert_tier1_json(src, dst):
    """Convert tier1 quadrant JSON from standard COCO to triple-label format."""
    with open(src) as f:
        d = json.load(f)
    # Remap: tier1 category ordering differs from tier3/val
    # tier1: {0:Q2, 1:Q1, 2:Q3, 3:Q4} → target: {0:Q1, 1:Q2, 2:Q3, 3:Q4}
    tier1_cats = {c["id"]: int(c["name"]) for c in d["categories"]}
    name_to_t3id = {1: 0, 2: 1, 3: 2, 4: 3}
    remap = {old_id: name_to_t3id[name] for old_id, name in tier1_cats.items()}
    for ann in d["annotations"]:
        ann["category_id_1"] = remap[ann.get("category_id", 0)]
        ann["category_id_2"] = 0
        ann["category_id_3"] = 0
    d["categories_1"] = [{"id": i, "name": str(i+1), "supercategory": str(i+1)} for i in range(4)]
    d["categories_2"] = []
    d["categories_3"] = []
    d.pop("categories", None)
    with open(dst, "w") as f:
        json.dump(d, f)
    print(f"  Converted tier1 JSON: {dst.name}")


def main():
    repo_root = Path(__file__).parent.resolve()
    sorted_dir = repo_root / "sorted"
    challenge_dir = sorted_dir / "challenge"
    challenge_dir.mkdir(parents=True, exist_ok=True)

    # Source paths
    train_root = sorted_dir / "training_data"
    val_images = sorted_dir / "validation_data" / "quadrant_enumeration_disease" / "xrays"
    test_images = sorted_dir / "disease" / "input"
    val_json = sorted_dir / "dentex_raw" / "DENTEX" / "validation_triple.json"

    # Tier1 JSON uses standard category_id (0-indexed, different order).
    # Must be converted to triple-label format with correct category remapping.
    _t1_raw = train_root / "quadrant" / "train_quadrant.json"
    t1_train_json = train_root / "quadrant" / "train_quadrant_triple.json"
    if not t1_train_json.exists() and _t1_raw.exists():
        _convert_tier1_json(_t1_raw, t1_train_json)
    t1_train_images = train_root / "quadrant" / "xrays"
    # Tier2 not released separately on HuggingFace — fall back to tier3 data.
    # Tier3 JSON has category_id_2 (enumeration) for 98% of annotations.
    # DEVIATION: disclose that tier2 uses tier3 training images/JSON.
    t2_train_json   = train_root / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
    t2_train_images = train_root / "quadrant-enumeration-disease" / "xrays"
    t3_train_json   = train_root / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
    t3_train_images = train_root / "quadrant-enumeration-disease" / "xrays"

    print("\n=== Checking source files ===")
    check_json(t1_train_json, "tier1 train JSON")
    check_json(t2_train_json, "tier2 train JSON")
    check_json(t3_train_json, "tier3 train JSON")
    check_json(val_json,      "validation JSON (triple)")
    print(f"  {'✓' if val_images.exists() else '✗'} Val images: {val_images} ({len(list(val_images.glob('*.png'))) if val_images.exists() else 0} files)")
    print(f"  {'✓' if test_images.exists() else '✗'} Test images: {test_images}")
    print(f"  {'✓' if t1_train_images.exists() else '✗'} Tier1 train images")
    print(f"  {'✓' if t2_train_images.exists() else '✗'} Tier2 train images")
    print(f"  {'✓' if t3_train_images.exists() else '✗'} Tier3 train images")

    print("\n=== Creating symlinks in sorted/challenge/ ===")

    # JSON symlinks
    symlink(t3_train_json, challenge_dir / "train_merged_disease_coco3class_onlyd_fixed.json")
    symlink(t2_train_json, challenge_dir / "train_enumeration_coco.json")
    symlink(t1_train_json, challenge_dir / "train_quadrant_coco.json")

    # Validation JSON serves all tiers (has category_id_1/2/3)
    for name in ["test_merged_disease_coco3class.json",
                 "test_enumeration_coco.json",
                 "test_quadrant_coco.json"]:
        symlink(val_json, challenge_dir / name)

    # Image directory symlinks
    symlink(t3_train_images, challenge_dir / "for_coco_disease_train")
    symlink(t2_train_images, challenge_dir / "for_coco_enumeration_train")
    symlink(t1_train_images, challenge_dir / "for_coco_quadrant_train")

    # All tiers share the same validation image set
    for name in ["for_coco_disease_test", "for_coco_enumeration_test", "for_coco_quadrant_test"]:
        symlink(val_images, challenge_dir / name)

    # Test images
    symlink(test_images, challenge_dir / "for_coco_test_input")

    # Verify
    print("\n=== Final challenge/ contents ===")
    for p in sorted(challenge_dir.iterdir()):
        target = p.resolve() if p.is_symlink() else p
        ok = "✓" if target.exists() else "✗"
        print(f"  {ok} {p.name}")

    print(f"""
=== Setup complete ===

Set these environment variables before running training:

  export DATA_ROOT={challenge_dir}
  export DENTEX_TIER=tier3   # or tier1, tier2

Then run:
  bash run_training.sh
""")


if __name__ == "__main__":
    main()
