"""
Normalize the DENTEX `quadrant` and `quadrant_enumeration` COCO annotation
files into the same 3-tier schema used by `quadrant-enumeration-disease`
(categories_1/2/3, category_id_1/2/3 per annotation), so all three tiers can
be loaded through the same COCO loading path and trained as stages of one
curriculum (the hierarchical noisy-box model has fixed-size heads for all
3 tiers; earlier stages simply have no supervision for the later heads).

Category ids are remapped by NAME, not by raw id: the raw `quadrant` tier's
own `categories` list uses a different id assignment than `categories_1` in
the other two tiers for the same 4 quadrant classes (verified empirically by
matching tooth boxes to their containing quadrant box across files -- see
docs/phase1_dataset_audit.md). A raw id-to-id copy would silently swap
quadrant classes 1 and 2 for a large fraction of annotations.
"""
import argparse
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def build_canonical_tiers(full_tier_json):
    return full_tier_json["categories_1"], full_tier_json["categories_2"], full_tier_json["categories_3"]


def name_to_id(categories):
    return {str(c["name"]): c["id"] for c in categories}


def normalize_quadrant(quadrant_json, canonical_cats_1, canonical_cats_2, canonical_cats_3):
    old_id_to_name = {c["id"]: str(c["name"]) for c in quadrant_json["categories"]}
    canonical_name_to_id = name_to_id(canonical_cats_1)

    out = dict(quadrant_json)
    del out["categories"]
    out["categories_1"] = canonical_cats_1
    out["categories_2"] = canonical_cats_2
    out["categories_3"] = canonical_cats_3

    new_anns = []
    for ann in quadrant_json["annotations"]:
        ann = dict(ann)
        old_cat_id = ann.pop("category_id")
        name = old_id_to_name[old_cat_id]
        ann["category_id_1"] = canonical_name_to_id[name]
        ann["category_id_2"] = None
        ann["category_id_3"] = None
        new_anns.append(ann)
    out["annotations"] = new_anns
    return out


def normalize_quadrant_enumeration(qe_json, canonical_cats_1, canonical_cats_2, canonical_cats_3):
    # categories_1 and categories_2 in quadrant_enumeration already share the
    # same id assignment (by id, not just by name) as the full 3-tier file --
    # verified directly by comparing the raw category lists. Replace with the
    # canonical lists anyway (identical ids) so `name` field types (str vs
    # int) are consistent across tiers, then just add the missing tier 3.
    out = dict(qe_json)
    out["categories_1"] = canonical_cats_1
    out["categories_2"] = canonical_cats_2
    out["categories_3"] = canonical_cats_3

    new_anns = []
    for ann in qe_json["annotations"]:
        ann = dict(ann)
        ann["category_id_3"] = None
        new_anns.append(ann)
    out["annotations"] = new_anns
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-tier-json", required=True, help="path to train_quadrant_enumeration_disease.json (canonical source of categories_1/2/3)")
    p.add_argument("--quadrant-json", required=True, help="path to train_quadrant.json")
    p.add_argument("--quadrant-enumeration-json", required=True, help="path to train_quadrant_enumeration.json")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    full_tier = load(args.full_tier_json)
    cats_1, cats_2, cats_3 = build_canonical_tiers(full_tier)

    os.makedirs(args.out_dir, exist_ok=True)

    quadrant = load(args.quadrant_json)
    normalized_quadrant = normalize_quadrant(quadrant, cats_1, cats_2, cats_3)
    out_path = os.path.join(args.out_dir, "train_quadrant_normalized.json")
    with open(out_path, "w") as f:
        json.dump(normalized_quadrant, f)
    print(f"wrote {out_path}: {len(normalized_quadrant['images'])} images, {len(normalized_quadrant['annotations'])} annotations")

    qe = load(args.quadrant_enumeration_json)
    normalized_qe = normalize_quadrant_enumeration(qe, cats_1, cats_2, cats_3)
    out_path = os.path.join(args.out_dir, "train_quadrant_enumeration_normalized.json")
    with open(out_path, "w") as f:
        json.dump(normalized_qe, f)
    print(f"wrote {out_path}: {len(normalized_qe['images'])} images, {len(normalized_qe['annotations'])} annotations")


if __name__ == "__main__":
    main()
