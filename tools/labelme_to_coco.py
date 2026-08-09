"""
Convert the DENTEX *test* split (250 images) from its per-image LabelMe polygon
format into the same normalized 3-tier COCO schema the training/validation
splits use, so the official dataloader and the official evaluator can read it.

Why this exists: `test_data.zip` is the only split not shipped as COCO. Each
image has a `disease/label/test_N.json` LabelMe file whose polygons carry a
single free-text `label` field with the quadrant, the diagnosis (as a Turkish
string) and the FDI tooth number packed together, e.g. `"1-curuk-15"`. Without
this conversion the test split cannot be loaded or evaluated at all --
`register_coco_instances` and `DiffusionDetDatasetMapper` both require COCO.

Two deliberate design choices:

1. **Category ids come from a canonical 3-tier COCO file** (pass the training
   diagnosis tier's JSON via --canonical-json), never from the strings in the
   test set. Class indices must be identical to the ones the model was trained
   with, and DENTEX's own per-file id numbering is not consistent across files
   (see docs/phase2_dataloader_fix.md).

2. **Unrecognized diagnosis strings are a hard error, not a guess.** Run with
   --report-only first: it prints every distinct raw label string and how it
   parsed, without writing anything. If a string is not covered by
   DIAGNOSIS_ALIASES below, extend that table rather than letting the script
   silently drop or mislabel annotations.

Boxes are the axis-aligned bounding box of each polygon, in COCO XYWH.
"""
import argparse
import json
import os
import re
from collections import Counter

# Turkish -> DENTEX English diagnosis class names. Keys are diacritic-stripped,
# lowercased, whitespace-collapsed. The four target names must match
# `categories_3` in the canonical file exactly.
DIAGNOSIS_ALIASES = {
    "curuk": "Caries",
    "caries": "Caries",
    "derin curuk": "Deep Caries",
    "derincuruk": "Deep Caries",
    "deep caries": "Deep Caries",
    "gomulu": "Impacted",
    "gomuk": "Impacted",
    "impacted": "Impacted",
    "periapikal lezyon": "Periapical Lesion",
    "periapikal": "Periapical Lesion",
    "periapical lesion": "Periapical Lesion",
    "lezyon": "Periapical Lesion",
}

TURKISH_FOLD = str.maketrans({
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
})


def fold(text):
    return re.sub(r"\s+", " ", text.translate(TURKISH_FOLD).lower().strip())


class LabelParseError(Exception):
    pass


def parse_label(raw_label):
    """
    Parse one LabelMe `label` string into (quadrant, tooth, diagnosis).

    quadrant is the FDI quadrant 1-4, tooth the FDI position 1-8 within that
    quadrant, diagnosis one of the four canonical English class names.

    Accepts the observed `quadrant-diagnosis-fdinumber` layout and the
    equivalent `quadrant-diagnosis-tooth` layout, and does not care about token
    order: the diagnosis is identified by name, and the numbers by value.
    """
    tokens = [t for t in re.split(r"[-_/|,]+", raw_label) if t.strip()]
    if not tokens:
        raise LabelParseError("empty label")

    numbers = []
    words = []
    for token in tokens:
        folded = fold(token)
        if re.fullmatch(r"\d+", folded):
            numbers.append(int(folded))
        else:
            words.append(folded)

    # Match the longest run of words first: some labels split a two-word
    # diagnosis on the same separator ("derin-curuk"), and matching token by
    # token would read that as plain "curuk" (Caries) and silently lose the
    # "derin" (Deep Caries) -- a wrong label, not a failed one.
    diagnosis = None
    unknown = list(words)
    for length in range(len(words), 0, -1):
        for start in range(0, len(words) - length + 1):
            candidate = fold(" ".join(words[start:start + length]))
            if candidate in DIAGNOSIS_ALIASES:
                diagnosis = DIAGNOSIS_ALIASES[candidate]
                unknown = words[:start] + words[start + length:]
                break
        if diagnosis is not None:
            break

    if diagnosis is None:
        raise LabelParseError(
            "no recognized diagnosis in {!r} (unmatched tokens: {}) -- "
            "extend DIAGNOSIS_ALIASES in tools/labelme_to_coco.py".format(raw_label, unknown)
        )

    quadrant = tooth = None
    for number in numbers:
        if 11 <= number <= 48 and 1 <= number % 10 <= 8:  # full FDI number
            quadrant, tooth = number // 10, number % 10
            break
    if quadrant is None:
        plain = [n for n in numbers if 1 <= n <= 8]
        if len(plain) >= 2:
            quadrant, tooth = plain[0], plain[1]
    if quadrant is None or not (1 <= quadrant <= 4) or not (1 <= tooth <= 8):
        raise LabelParseError(
            "could not read an FDI quadrant/tooth pair out of {!r} (numbers: {})".format(
                raw_label, numbers
            )
        )
    return quadrant, tooth, diagnosis


def polygon_to_bbox(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def name_to_id(categories):
    return {str(c["name"]): c["id"] for c in categories}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label-dir", required=True,
                   help="directory of LabelMe JSONs, e.g. .../test_data/disease/label")
    p.add_argument("--image-dir", required=True,
                   help="directory of test images, e.g. .../test_data/disease/input")
    p.add_argument("--canonical-json", required=True,
                   help="a normalized 3-tier COCO file to take categories_1/2/3 from "
                        "(train_quadrant_enumeration_disease.json)")
    p.add_argument("--out-json", help="where to write the converted COCO file")
    p.add_argument("--report-only", action="store_true",
                   help="parse and print a label report; write nothing")
    args = p.parse_args()

    with open(args.canonical_json) as f:
        canonical = json.load(f)
    cats_1, cats_2, cats_3 = canonical["categories_1"], canonical["categories_2"], canonical["categories_3"]
    q_name_to_id, t_name_to_id, d_name_to_id = (name_to_id(cats_1), name_to_id(cats_2), name_to_id(cats_3))

    missing = [n for n in DIAGNOSIS_ALIASES.values() if n not in d_name_to_id]
    if missing:
        raise SystemExit(
            "DIAGNOSIS_ALIASES maps to names absent from the canonical categories_3 "
            "({}): {}".format(sorted(d_name_to_id), sorted(set(missing)))
        )

    label_files = sorted(
        f for f in os.listdir(args.label_dir) if f.endswith(".json")
    )
    if not label_files:
        raise SystemExit("no .json label files found under {}".format(args.label_dir))

    images, annotations = [], []
    raw_label_counts = Counter()
    parsed_examples = {}
    failures = []
    per_image_counts = Counter()
    quadrant_field_mismatch = 0
    ann_id = 0

    for image_index, label_file in enumerate(label_files):
        with open(os.path.join(args.label_dir, label_file)) as f:
            labelme = json.load(f)

        stem = os.path.splitext(label_file)[0]
        image_name = labelme.get("imagePath") or (stem + ".png")
        image_name = os.path.basename(image_name)
        image_path = os.path.join(args.image_dir, image_name)
        if not os.path.exists(image_path):
            alternative = os.path.join(args.image_dir, stem + ".png")
            if os.path.exists(alternative):
                image_name, image_path = os.path.basename(alternative), alternative
            else:
                failures.append((label_file, "image file not found: {}".format(image_name)))
                continue

        height, width = labelme.get("imageHeight"), labelme.get("imageWidth")
        if not height or not width:
            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size

        image_id = image_index
        images.append({
            "id": image_id,
            "file_name": image_name,
            "height": int(height),
            "width": int(width),
        })

        for shape in labelme.get("shapes", []):
            raw = shape.get("label", "")
            raw_label_counts[raw] += 1
            try:
                quadrant, tooth, diagnosis = parse_label(raw)
            except LabelParseError as exc:
                failures.append((label_file, str(exc)))
                continue
            parsed_examples.setdefault(raw, (quadrant, tooth, diagnosis))

            # The leading token is redundant with the FDI number's quadrant
            # digit whenever both are present; count disagreements rather than
            # silently trusting one of them.
            leading = re.split(r"[-_/|,]+", raw)[0].strip()
            if leading.isdigit() and 1 <= int(leading) <= 4 and int(leading) != quadrant:
                quadrant_field_mismatch += 1

            bbox = polygon_to_bbox(shape["points"])
            if bbox[2] <= 0 or bbox[3] <= 0:
                failures.append((label_file, "degenerate polygon for label {!r}".format(raw)))
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "bbox": [round(v, 2) for v in bbox],
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "segmentation": [],
                "category_id_1": q_name_to_id[str(quadrant)],
                "category_id_2": t_name_to_id[str(tooth)],
                "category_id_3": d_name_to_id[diagnosis],
            })
            ann_id += 1
            per_image_counts[image_id] += 1

    print("images: {}".format(len(images)))
    print("annotations: {}".format(len(annotations)))
    print("images with zero annotations: {}".format(
        sum(1 for im in images if per_image_counts[im["id"]] == 0)))
    print("distinct raw label strings: {}".format(len(raw_label_counts)))
    print()
    print("{:<40} {:>6}  parsed as".format("raw label", "count"))
    for raw, count in raw_label_counts.most_common():
        parsed = parsed_examples.get(raw)
        print("{:<40} {:>6}  {}".format(
            raw[:40], count, "q{} t{} {}".format(*parsed) if parsed else "FAILED"))
    print()
    diagnosis_hist = Counter(a["category_id_3"] for a in annotations)
    id_to_diag = {c["id"]: c["name"] for c in cats_3}
    print("diagnosis distribution: {}".format(
        {id_to_diag[k]: v for k, v in sorted(diagnosis_hist.items())}))
    if quadrant_field_mismatch:
        print("WARNING: {} labels whose leading quadrant token disagreed with the FDI "
              "number's quadrant digit (the FDI number was used)".format(quadrant_field_mismatch))
    if failures:
        print()
        print("{} annotation(s)/file(s) could not be converted:".format(len(failures)))
        for label_file, reason in failures[:20]:
            print("  {}: {}".format(label_file, reason))
        if len(failures) > 20:
            print("  ... and {} more".format(len(failures) - 20))

    if args.report_only:
        return
    if failures:
        raise SystemExit(
            "refusing to write a partial test set: fix the failures above "
            "(usually by extending DIAGNOSIS_ALIASES) and re-run"
        )
    if not args.out_json:
        raise SystemExit("--out-json is required unless --report-only is given")

    out = {
        "info": {
            "description": "DENTEX test split converted from LabelMe by "
                           "tools/labelme_to_coco.py (MLRC HierarchicalDet reproduction)",
            "source_label_dir": os.path.abspath(args.label_dir),
            "canonical_categories_from": os.path.abspath(args.canonical_json),
        },
        "images": images,
        "annotations": annotations,
        "categories_1": cats_1,
        "categories_2": cats_2,
        "categories_3": cats_3,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print()
    print("wrote {}".format(args.out_json))


if __name__ == "__main__":
    main()
