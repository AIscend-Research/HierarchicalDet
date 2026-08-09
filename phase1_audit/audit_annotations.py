import json, os
from collections import Counter

TIERS = {
    "quadrant (train)": {
        "json": "extracted/training_data/quadrant/train_quadrant.json",
        "img_dir": "extracted/training_data/quadrant/xrays",
        "cat_keys": ["categories"],
        "cat_id_keys": ["category_id"],
    },
    "quadrant-enumeration (train)": {
        "json": "extracted/training_data/quadrant_enumeration/train_quadrant_enumeration.json",
        "img_dir": "extracted/training_data/quadrant_enumeration/xrays",
        "cat_keys": ["categories_1", "categories_2"],
        "cat_id_keys": ["category_id_1", "category_id_2"],
    },
    "quadrant-enumeration-diagnosis (train)": {
        "json": "extracted/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        "img_dir": "extracted/training_data/quadrant-enumeration-disease/xrays",
        "cat_keys": ["categories_1", "categories_2", "categories_3"],
        "cat_id_keys": ["category_id_1", "category_id_2", "category_id_3"],
    },
    "quadrant-enumeration-diagnosis (validation)": {
        "json": "DENTEX/validation_triple.json",
        "img_dir": "extracted/validation_data/quadrant_enumeration_disease/xrays",
        "cat_keys": ["categories_1", "categories_2", "categories_3"],
        "cat_id_keys": ["category_id_1", "category_id_2", "category_id_3"],
    },
}

for name, cfg in TIERS.items():
    print(f"=== {name} ===")
    d = json.load(open(cfg["json"]))
    images = {im["id"]: im for im in d["images"]}
    annos = d["annotations"]

    # 1. image_id linkage: every annotation's image_id must exist in images
    bad_image_ids = [a["id"] for a in annos if a["image_id"] not in images]
    print(f"  images: {len(images)}  annotations: {len(annos)}")
    print(f"  annotations with invalid image_id: {len(bad_image_ids)}")

    # 2. file_name existence on disk
    missing_files = [im["file_name"] for im in d["images"] if not os.path.exists(os.path.join(cfg["img_dir"], im["file_name"]))]
    print(f"  image file_names missing on disk: {len(missing_files)}")
    if missing_files:
        print("   e.g.:", missing_files[:5])

    # 3. category_id validity: every annotation's category_id(s) must be in range of that tier's categories list
    cat_valid_ids = {}
    for ck in cfg["cat_keys"]:
        if ck in d:
            cat_valid_ids[ck] = {c["id"] for c in d[ck]}

    bad_cat_refs = Counter()
    null_cat_refs = Counter()
    for a in annos:
        for idk in cfg["cat_id_keys"]:
            if idk not in a:
                continue
            val = a[idk]
            corresponding_cat_key = cfg["cat_keys"][cfg["cat_id_keys"].index(idk)]
            if val is None:
                null_cat_refs[idk] += 1
            elif val not in cat_valid_ids.get(corresponding_cat_key, set()):
                bad_cat_refs[idk] += 1
    print(f"  annotations with out-of-range category ids: {dict(bad_cat_refs)}")
    print(f"  annotations with null category ids (partial annotation): {dict(null_cat_refs)}")

    # 4. images with zero annotations
    ann_counts = Counter(a["image_id"] for a in annos)
    n_images_no_annos = sum(1 for im_id in images if ann_counts.get(im_id, 0) == 0)
    print(f"  images with zero annotations: {n_images_no_annos}")

    # 5. bbox sanity: width/height must be positive, within image bounds
    bad_bbox = []
    for a in annos:
        im = images.get(a["image_id"])
        if im is None:
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            bad_bbox.append(("non-positive size", a["id"]))
        elif x < 0 or y < 0 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
            bad_bbox.append(("out of image bounds", a["id"]))
    print(f"  annotations with bad bboxes: {len(bad_bbox)}")
    if bad_bbox:
        print("   e.g.:", bad_bbox[:5])

    print()
