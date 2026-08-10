"""
DENTEX -> the COCO-style JSONs the HierarchicalDet code actually expects.

The released code registers dataset paths that only existed on the authors'
machine, and the three DENTEX tiers ship in three *different* schemas, none of
which the vendored loader can read directly except the full 3-tier one. This
module owns the whole conversion:

* ``quadrant``            flat ``categories`` / ``category_id``      -> 3-tier
* ``quadrant_enumeration``  ``categories_1/2``                       -> 3-tier
* ``quadrant-enumeration-disease``  already 3-tier                   -> copied
* test split                per-image LabelMe polygons, Turkish text -> 3-tier

"3-tier" means the normalized schema the vendored ``pycocotools`` fork and
``detectron2/data/datasets/coco.py`` assume: top-level ``categories_1/2/3``,
and every annotation carrying ``category_id_1/2/3`` with ``null`` for tiers
that do not apply (which is exactly how the loss skips unsupervised heads).

Two traps this module exists to avoid, both verified against the real release:

1. ``quadrant``'s own category ids are **not** the same assignment as
   ``categories_1`` in the other two tiers for the same four classes, so a raw
   id-to-id copy silently swaps quadrants 1 and 2. Remapping is by *name*.
2. The diagnosis strings in the test split are Turkish free text packed
   together with the FDI number. Unrecognized strings are a hard error, never
   a guess or a silent drop.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from . import setup_env

HF_REPO_ID = "ibrahimhamamci/DENTEX"

#: Files to pull from the HuggingFace release, in the order they are consumed.
HF_FILES = (
    "DENTEX/validation_triple.json",
    "DENTEX/validation_data.zip",
    "DENTEX/test_data.zip",
    "DENTEX/training_data.zip",
)

#: Counts published by the paper and the DENTEX README. Notebook 01 asserts
#: these and hard-fails with the actual numbers if the release has changed.
PUBLISHED_COUNTS = {
    "quadrant_train": 693,
    "quadrant_enumeration_train": 634,
    "diagnosis_train": 705,
    "diagnosis_val": 50,
    "diagnosis_test": 250,
    "unlabelled": 1571,
}

#: Members inside training_data.zip.
TRAIN_TIERS = {
    "quadrant": (
        "training_data/quadrant/train_quadrant.json",
        "training_data/quadrant/xrays/",
    ),
    "quadrant_enumeration": (
        "training_data/quadrant_enumeration/train_quadrant_enumeration.json",
        "training_data/quadrant_enumeration/xrays/",
    ),
    "diagnosis": (
        "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        "training_data/quadrant-enumeration-disease/xrays/",
    ),
}
UNLABELLED_PREFIX = "training_data/unlabelled/xrays/"
VAL_IMAGE_PREFIX = "validation_data/quadrant_enumeration_disease/xrays/"
TEST_IMAGE_PREFIX = "disease/input/"
TEST_LABEL_PREFIX = "disease/label/"

TIER_NAMES = {0: "quadrant", 1: "enumeration", 2: "diagnosis"}


# --------------------------------------------------------------------------
# Layout of the converted dataset
# --------------------------------------------------------------------------
def layout(data_dir: str = None) -> Dict[str, str]:
    """Canonical paths inside the converted-dataset directory."""
    root = data_dir or setup_env.DATA_DIR
    return {
        "root": root,
        "coco": os.path.join(root, "coco"),
        "images": os.path.join(root, "images"),
        "subsets": os.path.join(root, "subsets"),
        "audit": os.path.join(root, "audit"),
        # annotations
        "train_quadrant": os.path.join(root, "coco", "train_quadrant.json"),
        "train_enumeration": os.path.join(root, "coco", "train_enumeration.json"),
        "train_diagnosis": os.path.join(root, "coco", "train_diagnosis.json"),
        "val_diagnosis": os.path.join(root, "coco", "val_diagnosis.json"),
        "test_diagnosis": os.path.join(root, "coco", "test_diagnosis.json"),
        # images
        "img_quadrant": os.path.join(root, "images", "quadrant"),
        "img_enumeration": os.path.join(root, "images", "quadrant_enumeration"),
        "img_diagnosis": os.path.join(root, "images", "diagnosis"),
        "img_val": os.path.join(root, "images", "validation"),
        "img_test": os.path.join(root, "images", "test"),
        "img_unlabelled": os.path.join(root, "images", "unlabelled"),
    }


def flat_json_path(tier: int, split: str, data_dir: str = None) -> str:
    """Single-label ("flat") view of one tier, for the base-DiffusionDet runs."""
    paths = layout(data_dir)
    return os.path.join(paths["coco"], "flat_tier{}_{}.json".format(tier, split))


# --------------------------------------------------------------------------
# Download + extract
# --------------------------------------------------------------------------
def download_and_extract(raw_dir: str, data_dir: str = None,
                         include_unlabelled: bool = False,
                         delete_archives: bool = True) -> Dict[str, object]:
    """
    Fetch DENTEX one archive at a time and extract only what is needed,
    deleting each archive as soon as it has been consumed.

    Kaggle's ``/kaggle/working`` is ~20 GB and the release is ~11.8 GB
    compressed; downloading everything first and extracting afterwards does not
    fit. Each step is marker-gated, so re-running after a session kill resumes
    instead of restarting.
    """
    from huggingface_hub import hf_hub_download

    paths = layout(data_dir)
    os.makedirs(raw_dir, exist_ok=True)
    for key in ("coco", "images", "subsets", "audit"):
        os.makedirs(paths[key], exist_ok=True)

    report: Dict[str, object] = {"raw_dir": raw_dir, "steps": []}

    def marker(name: str) -> str:
        return os.path.join(paths["root"], ".{}.done".format(name))

    def fetch(remote: str) -> str:
        return hf_hub_download(
            repo_id=HF_REPO_ID, repo_type="dataset", filename=remote, local_dir=raw_dir,
        )

    # --- validation annotations (a bare JSON, not zipped) -----------------
    if not os.path.exists(marker("validation_triple")):
        source = fetch("DENTEX/validation_triple.json")
        shutil.copy2(source, paths["val_diagnosis"])
        open(marker("validation_triple"), "w").close()
        report["steps"].append("validation_triple.json")

    # --- validation images -------------------------------------------------
    if not os.path.exists(marker("validation_images")):
        archive = fetch("DENTEX/validation_data.zip")
        _extract_prefix(archive, VAL_IMAGE_PREFIX, paths["img_val"])
        if delete_archives:
            os.remove(archive)
        open(marker("validation_images"), "w").close()
        report["steps"].append("validation_data.zip")

    # --- test images + LabelMe labels -------------------------------------
    test_label_dir = os.path.join(paths["root"], "test_labelme")
    if not os.path.exists(marker("test_data")):
        archive = fetch("DENTEX/test_data.zip")
        _extract_prefix(archive, TEST_IMAGE_PREFIX, paths["img_test"])
        _extract_prefix(archive, TEST_LABEL_PREFIX, test_label_dir)
        if delete_archives:
            os.remove(archive)
        open(marker("test_data"), "w").close()
        report["steps"].append("test_data.zip")
    report["test_label_dir"] = test_label_dir

    # --- training tiers ----------------------------------------------------
    raw_train_json = {}
    if not os.path.exists(marker("training_data")):
        archive = fetch("DENTEX/training_data.zip")
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            for tier, (json_member, image_prefix) in TRAIN_TIERS.items():
                if json_member not in names:
                    raise RuntimeError(
                        "training_data.zip does not contain {} — the DENTEX release "
                        "layout has changed and this converter must be updated."
                        .format(json_member)
                    )
                target = os.path.join(paths["root"], "raw_{}.json".format(tier))
                with open(target, "wb") as out:
                    out.write(handle.read(json_member))
                raw_train_json[tier] = target
                destination = {
                    "quadrant": paths["img_quadrant"],
                    "quadrant_enumeration": paths["img_enumeration"],
                    "diagnosis": paths["img_diagnosis"],
                }[tier]
                _extract_prefix_from_open(handle, image_prefix, destination)
            if include_unlabelled:
                _extract_prefix_from_open(handle, UNLABELLED_PREFIX, paths["img_unlabelled"])
            report["unlabelled_in_zip"] = sum(
                1 for n in names if n.startswith(UNLABELLED_PREFIX) and n.endswith(".png")
            )
        if delete_archives:
            os.remove(archive)
        open(marker("training_data"), "w").close()
        report["steps"].append("training_data.zip")
    else:
        for tier in TRAIN_TIERS:
            raw_train_json[tier] = os.path.join(paths["root"], "raw_{}.json".format(tier))

    report["raw_train_json"] = raw_train_json
    return report


def _extract_prefix(archive_path: str, prefix: str, destination: str) -> int:
    with zipfile.ZipFile(archive_path) as handle:
        return _extract_prefix_from_open(handle, prefix, destination)


def _extract_prefix_from_open(handle: zipfile.ZipFile, prefix: str, destination: str) -> int:
    """Extract every member under ``prefix`` *flat* into ``destination``."""
    os.makedirs(destination, exist_ok=True)
    count = 0
    for name in handle.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        target = os.path.join(destination, os.path.basename(name))
        if os.path.exists(target):
            count += 1
            continue
        with handle.open(name) as source, open(target, "wb") as out:
            shutil.copyfileobj(source, out)
        count += 1
    return count


# --------------------------------------------------------------------------
# Tier normalization
# --------------------------------------------------------------------------
def _load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _dump(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle)
    return path


def _name_to_id(categories: Sequence[dict]) -> Dict[str, int]:
    return {str(c["name"]): c["id"] for c in categories}


def normalize_quadrant(raw: dict, cats_1, cats_2, cats_3) -> dict:
    """
    Flat ``categories``/``category_id`` -> 3-tier, remapping **by name**.

    The name remap is the whole point: ``quadrant``'s own id 0 is named "2"
    while ``categories_1`` id 0 is named "1", so copying ids across would
    mislabel a large fraction of the annotations.
    """
    old_id_to_name = {c["id"]: str(c["name"]) for c in raw["categories"]}
    canonical = _name_to_id(cats_1)
    missing = sorted(set(old_id_to_name.values()) - set(canonical))
    if missing:
        raise ValueError(
            "quadrant categories {} have no counterpart in categories_1 {}"
            .format(missing, sorted(canonical))
        )

    out = {k: v for k, v in raw.items() if k != "categories"}
    out["categories_1"], out["categories_2"], out["categories_3"] = cats_1, cats_2, cats_3
    annotations = []
    for annotation in raw["annotations"]:
        annotation = dict(annotation)
        name = old_id_to_name[annotation.pop("category_id")]
        annotation["category_id_1"] = canonical[name]
        annotation["category_id_2"] = None
        annotation["category_id_3"] = None
        annotations.append(annotation)
    out["annotations"] = annotations
    return out


def normalize_quadrant_enumeration(raw: dict, cats_1, cats_2, cats_3) -> dict:
    """``categories_1/2`` -> 3-tier. Ids already agree; only tier 3 is added."""
    out = dict(raw)
    out["categories_1"], out["categories_2"], out["categories_3"] = cats_1, cats_2, cats_3
    annotations = []
    for annotation in raw["annotations"]:
        annotation = dict(annotation)
        annotation["category_id_3"] = None
        annotations.append(annotation)
    out["annotations"] = annotations
    return out


def flatten_tier(source: dict, tier: int) -> dict:
    """
    Collapse a 3-tier file into a **single-label** one for the requested tier:
    that tier's labels move into ``category_id_1`` and the other two tiers are
    nulled out.

    This is how the non-hierarchical, non-multilabel base DiffusionDet is
    trained without touching upstream code. The released head indexes
    ``MODEL.DiffusionDet.NUM_CLASSES`` as a 3-element list
    (``head.py:81-83``), so a genuinely single-head model cannot be configured;
    supervising only head 1 with the target tier's labels is the equivalent the
    data can express. ``categories_2/3`` are kept populated because the
    vendored loader takes ``min()``/``max()`` over every tier's category ids
    and raises on an empty list.
    """
    if tier not in (0, 1, 2):
        raise ValueError("tier must be 0, 1 or 2")
    key = "category_id_{}".format(tier + 1)
    out = dict(source)
    out["categories_1"] = source["categories_{}".format(tier + 1)]
    out["categories_2"] = source["categories_2"]
    out["categories_3"] = source["categories_3"]
    annotations = []
    for annotation in source["annotations"]:
        if annotation.get(key) is None:
            continue                      # no supervision for this tier
        annotation = dict(annotation)
        annotation["category_id_1"] = annotation[key]
        annotation["category_id_2"] = None
        annotation["category_id_3"] = None
        annotations.append(annotation)
    out["annotations"] = annotations
    return out


# --------------------------------------------------------------------------
# LabelMe (test split) -> COCO
# --------------------------------------------------------------------------
#: Turkish -> DENTEX English diagnosis names. Keys are diacritic-folded,
#: lowercased and whitespace-collapsed.
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

#: Labels in the test split that are real clinical annotations but OUTSIDE the
#: DENTEX 4-class diagnosis task. The raw test LabelMe files carry a 9-code
#: scheme (`<code>-<word>-<FDI>`; the leading number is a CLASS CODE, not a
#: quadrant) — verified against the full 250-file release, where every word maps
#: to exactly one code. Only codes 1/6/7 are task classes; the rest are excluded
#: *by name*, counted, and reported. An unknown word is still a hard error.
OUT_OF_TASK_LABELS = {
    "saglam": "healthy tooth (code 0)",
    "kuretaj": "curettage (code 2)",
    "kanal": "root canal treatment (code 3)",
    "cekim": "extraction (code 5)",
    "kirik": "fracture (code 8)",
}

#: Class code observed for each task word in the release, used as a consistency
#: check during conversion (a word appearing under two codes would mean the
#: labelling scheme is not what this converter assumes).
EXPECTED_WORD_CODES = {
    "saglam": 0, "curuk": 1, "kuretaj": 2, "kanal": 3,
    "cekim": 5, "gomulu": 6, "lezyon": 7, "kirik": 8,
}

#: The finding this table encodes, stated once and reused by notebook 01.
TEST_LABEL_FINDING = (
    "The released DENTEX test annotations use a 9-code clinical labelling scheme "
    "and contain NO 'Deep Caries' label: every carious tooth is plain 'çürük' "
    "(code 1). The public test release therefore cannot distinguish Caries from "
    "Deep Caries, so test-split evaluation covers 3 of the paper's 4 diagnosis "
    "classes and the Deep Caries column has no ground truth. This is a property "
    "of the data release, not of this conversion; the authors' own test file "
    "(test_merged_disease_coco3class.json) was never published."
)

_TURKISH_FOLD = str.maketrans({
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
})


class LabelParseError(Exception):
    pass


class OutOfTaskLabel(Exception):
    """A well-understood clinical label that is not one of the 4 task classes."""

    def __init__(self, word: str, description: str):
        super().__init__("{} = {}".format(word, description))
        self.word = word
        self.description = description


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TURKISH_FOLD).lower().strip())


def parse_labelme_label(raw_label: str) -> Tuple[int, int, str]:
    """
    Parse one LabelMe ``label`` string into ``(quadrant, tooth, diagnosis)``.

    Token order is irrelevant: the diagnosis is found by name and the FDI
    numbers by value. Multi-word diagnoses are matched longest-run-first, so
    ``"derin-curuk"`` reads as *Deep Caries* rather than silently degrading to
    *Caries*.
    """
    tokens = [t for t in re.split(r"[-_/|,]+", raw_label) if t.strip()]
    if not tokens:
        raise LabelParseError("empty label")

    numbers, words = [], []
    for token in tokens:
        folded = _fold(token)
        if re.fullmatch(r"\d+", folded):
            numbers.append(int(folded))
        else:
            words.append(folded)

    diagnosis, unknown = None, list(words)
    for length in range(len(words), 0, -1):
        for start in range(0, len(words) - length + 1):
            candidate = _fold(" ".join(words[start:start + length]))
            if candidate in DIAGNOSIS_ALIASES:
                diagnosis = DIAGNOSIS_ALIASES[candidate]
                unknown = words[:start] + words[start + length:]
                break
        if diagnosis is not None:
            break
    if diagnosis is None:
        # Not a task class — but maybe a *known* out-of-task clinical label.
        # Those are excluded by name and counted, never silently dropped.
        for word in words:
            if word in OUT_OF_TASK_LABELS:
                raise OutOfTaskLabel(word, OUT_OF_TASK_LABELS[word])
        raise LabelParseError(
            "no recognized diagnosis in {!r} (unmatched tokens: {}) — extend "
            "DIAGNOSIS_ALIASES or OUT_OF_TASK_LABELS rather than dropping the "
            "annotation".format(raw_label, unknown)
        )

    quadrant = tooth = None
    for number in numbers:
        if 11 <= number <= 48 and 1 <= number % 10 <= 8:      # full FDI number
            quadrant, tooth = number // 10, number % 10
            break
    if quadrant is None:
        plain = [n for n in numbers if 1 <= n <= 8]
        if len(plain) >= 2:
            quadrant, tooth = plain[0], plain[1]
    if quadrant is None or not (1 <= quadrant <= 4) or not (1 <= tooth <= 8):
        raise LabelParseError(
            "no FDI quadrant/tooth pair in {!r} (numbers: {})".format(raw_label, numbers)
        )
    return quadrant, tooth, diagnosis


def labelme_to_coco(label_dir: str, image_dir: str, canonical: dict,
                    strict: bool = True) -> Tuple[dict, dict]:
    """
    Convert the 250-image test split. Returns ``(coco_dict, parse_report)``.

    With ``strict`` (the default) any unparseable label raises, because a test
    split that quietly lost annotations is not a test split.
    """
    cats_1, cats_2, cats_3 = (canonical["categories_1"], canonical["categories_2"],
                              canonical["categories_3"])
    quadrant_ids, tooth_ids, diagnosis_ids = (_name_to_id(cats_1), _name_to_id(cats_2),
                                              _name_to_id(cats_3))
    unknown_targets = [n for n in set(DIAGNOSIS_ALIASES.values()) if n not in diagnosis_ids]
    if unknown_targets:
        raise ValueError(
            "DIAGNOSIS_ALIASES maps to names absent from categories_3 ({}): {}"
            .format(sorted(diagnosis_ids), sorted(unknown_targets))
        )

    label_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".json"))
    if not label_files:
        raise RuntimeError("no LabelMe .json files under {}".format(label_dir))

    images, annotations, failures = [], [], []
    raw_counts, parsed_examples = Counter(), {}
    excluded_counts: Counter = Counter()
    code_inconsistencies = 0
    annotation_id = 0

    for image_index, label_file in enumerate(label_files):
        with open(os.path.join(label_dir, label_file)) as handle:
            labelme = json.load(handle)
        stem = os.path.splitext(label_file)[0]
        image_name = os.path.basename(labelme.get("imagePath") or (stem + ".png"))
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            alternative = os.path.join(image_dir, stem + ".png")
            if os.path.exists(alternative):
                image_name, image_path = os.path.basename(alternative), alternative
            else:
                failures.append((label_file, "image not found: {}".format(image_name)))
                continue

        height, width = labelme.get("imageHeight"), labelme.get("imageWidth")
        if not height or not width:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size

        images.append({"id": image_index, "file_name": image_name,
                       "height": int(height), "width": int(width)})

        for shape in labelme.get("shapes", []):
            raw = shape.get("label", "")
            raw_counts[raw] += 1
            try:
                quadrant, tooth, diagnosis = parse_labelme_label(raw)
            except OutOfTaskLabel as excluded:
                excluded_counts[excluded.word] += 1
                continue
            except LabelParseError as error:
                failures.append((label_file, str(error)))
                continue
            parsed_examples.setdefault(raw, (quadrant, tooth, diagnosis))

            # The leading token is a CLASS CODE (`<code>-<word>-<FDI>`), not a
            # quadrant. Check it stays consistent with the word: an
            # inconsistency would mean the labelling scheme is not the one this
            # converter was verified against.
            leading = re.split(r"[-_/|,]+", raw)[0].strip()
            word = next((_fold(t) for t in re.split(r"[-_/|,]+", raw)
                         if not re.fullmatch(r"\d+", _fold(t))), None)
            if (leading.isdigit() and word in EXPECTED_WORD_CODES
                    and int(leading) != EXPECTED_WORD_CODES[word]):
                code_inconsistencies += 1

            xs = [float(p[0]) for p in shape["points"]]
            ys = [float(p[1]) for p in shape["points"]]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            if bbox[2] <= 0 or bbox[3] <= 0:
                failures.append((label_file, "degenerate polygon for {!r}".format(raw)))
                continue

            annotations.append({
                "id": annotation_id,
                "image_id": image_index,
                "bbox": [round(v, 2) for v in bbox],
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "segmentation": [],
                "category_id_1": quadrant_ids[str(quadrant)],
                "category_id_2": tooth_ids[str(tooth)],
                "category_id_3": diagnosis_ids[diagnosis],
            })
            annotation_id += 1

    diagnosis_names = [str(c["name"]) for c in cats_3]
    diagnosis_histogram = Counter()
    id_to_diagnosis = {c["id"]: str(c["name"]) for c in cats_3}
    for annotation in annotations:
        diagnosis_histogram[id_to_diagnosis[annotation["category_id_3"]]] += 1

    report = {
        "images": len(images),
        "annotations": len(annotations),
        "distinct_raw_labels": len(raw_counts),
        "raw_label_counts": dict(raw_counts.most_common()),
        "parsed_examples": {k: list(v) for k, v in parsed_examples.items()},
        "excluded_out_of_task": {
            word: {"count": count, "meaning": OUT_OF_TASK_LABELS[word]}
            for word, count in excluded_counts.most_common()
        },
        "excluded_total": sum(excluded_counts.values()),
        "class_code_inconsistencies": code_inconsistencies,
        "diagnosis_histogram": {name: diagnosis_histogram.get(name, 0)
                                for name in diagnosis_names},
        "diagnosis_classes_without_ground_truth": [
            name for name in diagnosis_names if diagnosis_histogram.get(name, 0) == 0
        ],
        "failures": [list(f) for f in failures],
    }
    if failures and strict:
        preview = "\n  ".join("{}: {}".format(*f) for f in failures[:10])
        raise RuntimeError(
            "refusing to write a partial test split: {} annotation(s)/file(s) failed "
            "to convert.\n  {}\nExtend DIAGNOSIS_ALIASES and re-run."
            .format(len(failures), preview)
        )

    coco = {
        "info": {
            "description": "DENTEX test split converted from LabelMe polygons",
            "source_label_dir": os.path.abspath(label_dir),
        },
        "images": images,
        "annotations": annotations,
        "categories_1": cats_1,
        "categories_2": cats_2,
        "categories_3": cats_3,
    }
    return coco, report


# --------------------------------------------------------------------------
# Full conversion
# --------------------------------------------------------------------------
def convert_all(raw_train_json: Dict[str, str], test_label_dir: str,
                data_dir: str = None) -> Dict[str, object]:
    """Write every COCO file the training/eval notebooks register."""
    paths = layout(data_dir)

    diagnosis = _load(raw_train_json["diagnosis"])
    cats_1 = diagnosis["categories_1"]
    cats_2 = diagnosis["categories_2"]
    cats_3 = diagnosis["categories_3"]

    _dump(diagnosis, paths["train_diagnosis"])
    _dump(normalize_quadrant(_load(raw_train_json["quadrant"]), cats_1, cats_2, cats_3),
          paths["train_quadrant"])
    _dump(normalize_quadrant_enumeration(
        _load(raw_train_json["quadrant_enumeration"]), cats_1, cats_2, cats_3),
        paths["train_enumeration"])

    # The validation annotations shipped as a bare JSON already in 3-tier form;
    # re-dump through the same path so category *lists* are byte-identical
    # across splits (class index assignment must not drift between splits).
    validation = _load(paths["val_diagnosis"])
    validation["categories_1"], validation["categories_2"], validation["categories_3"] = (
        cats_1, cats_2, cats_3)
    _dump(validation, paths["val_diagnosis"])

    test_coco, parse_report = labelme_to_coco(test_label_dir, paths["img_test"], diagnosis)
    _dump(test_coco, paths["test_diagnosis"])

    # Flat (single-label) views for the base-DiffusionDet baseline.
    flat_sources = {
        0: (paths["train_quadrant"], paths["test_diagnosis"]),
        1: (paths["train_enumeration"], paths["test_diagnosis"]),
        2: (paths["train_diagnosis"], paths["test_diagnosis"]),
    }
    for tier, (train_path, test_path) in flat_sources.items():
        _dump(flatten_tier(_load(train_path), tier), flat_json_path(tier, "train", data_dir))
        _dump(flatten_tier(_load(test_path), tier), flat_json_path(tier, "test", data_dir))

    return {"paths": paths, "test_parse_report": parse_report}


def dataset_hashes(data_dir: str = None) -> Dict[str, str]:
    """SHA-256 of every converted annotation file, for the repro checklist."""
    paths = layout(data_dir)
    digests = {}
    for name in sorted(os.listdir(paths["coco"])):
        if name.endswith(".json"):
            full = os.path.join(paths["coco"], name)
            digests[name] = setup_env.file_sha256(full)
    return digests


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def audit_split(json_path: str, image_dir: str, name: str,
                check_pixels: bool = True) -> Dict[str, object]:
    """
    Everything the dataset-audit table needs for one split: counts, tier
    coverage, resolutions, unreadable files, malformed boxes, class histograms.
    """
    from PIL import Image

    data = _load(json_path)
    images = data["images"]
    annotations = data["annotations"]
    by_image = defaultdict(list)
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)

    tier_coverage = {
        "tier0_quadrant": sum(1 for a in annotations if a.get("category_id_1") is not None),
        "tier1_enumeration": sum(1 for a in annotations if a.get("category_id_2") is not None),
        "tier2_diagnosis": sum(1 for a in annotations if a.get("category_id_3") is not None),
    }

    histograms = {}
    for tier in (0, 1, 2):
        key = "category_id_{}".format(tier + 1)
        id_to_name = {c["id"]: str(c["name"]) for c in data["categories_{}".format(tier + 1)]}
        counts = Counter(id_to_name.get(a[key], "?") for a in annotations if a.get(key) is not None)
        if counts:
            histograms[TIER_NAMES[tier]] = dict(sorted(counts.items()))

    malformed, missing_files, unreadable, resolutions = [], [], [], Counter()
    declared = {image["id"]: image for image in images}
    for image in images:
        path = os.path.join(image_dir, image["file_name"])
        if not os.path.exists(path):
            missing_files.append(image["file_name"])
            continue
        resolutions["{}x{}".format(image["width"], image["height"])] += 1
        if check_pixels:
            try:
                with Image.open(path) as handle:
                    handle.verify()
                with Image.open(path) as handle:
                    handle.load()
            except Exception as error:                    # noqa: BLE001 — reported, not swallowed
                unreadable.append({"file": image["file_name"], "error": repr(error)})

    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        image = declared.get(annotation["image_id"])
        problem = None
        if width <= 0 or height <= 0:
            problem = "non_positive_size"
        elif image and (x < -1 or y < -1
                        or x + width > image["width"] + 1
                        or y + height > image["height"] + 1):
            problem = "out_of_bounds"
        elif image is None:
            problem = "dangling_image_id"
        if problem:
            malformed.append({"annotation_id": annotation["id"], "problem": problem})

    annotation_counts = Counter(len(by_image[image["id"]]) for image in images)
    return {
        "split": name,
        "json": os.path.abspath(json_path),
        "image_dir": os.path.abspath(image_dir),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "images_with_zero_annotations": sum(1 for i in images if not by_image[i["id"]]),
        "tier_coverage": tier_coverage,
        "class_histograms": histograms,
        "resolutions": dict(resolutions.most_common()),
        "num_distinct_resolutions": len(resolutions),
        "missing_image_files": missing_files,
        "unreadable_images": unreadable,
        "malformed_boxes": malformed,
        "annotations_per_image_histogram": dict(sorted(annotation_counts.items())),
    }


def assert_published_counts(audits: Dict[str, Dict[str, object]],
                            unlabelled_count: Optional[int]) -> Dict[str, object]:
    """
    Hard-fail if the release no longer matches the published composition.
    Returns the actual counts either way, so they are recorded before the raise.
    """
    actual = {
        "quadrant_train": audits["quadrant_train"]["num_images"],
        "quadrant_enumeration_train": audits["quadrant_enumeration_train"]["num_images"],
        "diagnosis_train": audits["diagnosis_train"]["num_images"],
        "diagnosis_val": audits["diagnosis_val"]["num_images"],
        "diagnosis_test": audits["diagnosis_test"]["num_images"],
    }
    if unlabelled_count is not None:
        actual["unlabelled"] = unlabelled_count

    mismatches = {
        key: {"published": PUBLISHED_COUNTS[key], "actual": value}
        for key, value in actual.items() if PUBLISHED_COUNTS.get(key) != value
    }
    if mismatches:
        raise AssertionError(
            "the DENTEX release no longer matches the published composition: {}. "
            "Every count in the paper's comparison target assumes these splits; "
            "resolve this before training anything.".format(json.dumps(mismatches, indent=2))
        )
    return actual


def assert_test_ground_truth(test_audit: Dict[str, object]) -> None:
    """
    The test labels were withheld during the 2023 challenge and released later.
    Without them the whole comparison target changes, so this is checked
    explicitly and loudly rather than being discovered as an empty table.
    """
    if test_audit["num_annotations"] == 0:
        raise AssertionError(
            "the test split converted to ZERO annotations. Test ground truth appears "
            "to be absent from this DENTEX release. Every number in Table 1 is computed "
            "on the test split — stop here; the comparison target has changed."
        )
    covered = test_audit["tier_coverage"]["tier2_diagnosis"]
    if covered == 0:
        raise AssertionError(
            "the test split has boxes but no tier-2 (diagnosis) labels — the diagnosis "
            "row of the main table cannot be computed. Stop here."
        )


# --------------------------------------------------------------------------
# Clean / stress evaluation subsets
# --------------------------------------------------------------------------
#: Written verbatim into the audit output and the paper's scope statement.
SUBSET_RULE = (
    "clean = images whose annotation count is within 2 of the split median and "
    "non-zero, and whose annotations name at most 2 distinct diagnosis classes "
    "(ordinary anatomy, complete labelling). "
    "stress = the union of (a) images with zero annotations, (b) the top decile "
    "by annotation count, and (c) images carrying at least 3 distinct diagnosis "
    "classes (partial annotation and edge-case configurations). "
    "The two subsets are disjoint by construction: stress is selected first and "
    "removed from the clean candidate pool."
)


def build_clean_stress_subsets(json_path: str) -> Dict[str, object]:
    data = _load(json_path)
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    counts = {image["id"]: len(by_image[image["id"]]) for image in data["images"]}
    non_zero = sorted(v for v in counts.values() if v > 0)
    median = non_zero[len(non_zero) // 2] if non_zero else 0
    ordered = sorted(counts.values(), reverse=True)
    decile_index = max(0, len(ordered) // 10 - 1)
    decile_threshold = ordered[decile_index] if ordered else 0

    distinct_diagnoses = {
        image_id: len({a["category_id_3"] for a in annotations
                       if a.get("category_id_3") is not None})
        for image_id, annotations in by_image.items()
    }

    stress, clean = [], []
    for image in data["images"]:
        image_id = image["id"]
        count = counts[image_id]
        diagnoses = distinct_diagnoses.get(image_id, 0)
        if count == 0 or count >= decile_threshold or diagnoses >= 3:
            stress.append(image_id)
        elif abs(count - median) <= 2 and diagnoses <= 2:
            clean.append(image_id)

    id_to_name = {image["id"]: image["file_name"] for image in data["images"]}
    return {
        "rule": SUBSET_RULE,
        "source_json": os.path.abspath(json_path),
        "median_annotations_per_image": median,
        "top_decile_threshold": decile_threshold,
        "clean": {"image_ids": sorted(clean),
                  "file_names": [id_to_name[i] for i in sorted(clean)]},
        "stress": {"image_ids": sorted(stress),
                   "file_names": [id_to_name[i] for i in sorted(stress)]},
    }


def subset_json(source_json: str, image_ids: Sequence[int], out_path: str) -> str:
    """Write a COCO file restricted to ``image_ids`` (for subset evaluation)."""
    data = _load(source_json)
    keep = set(image_ids)
    data["images"] = [i for i in data["images"] if i["id"] in keep]
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep]
    return _dump(data, out_path)


#: The GitHub repo states CC BY-SA while the HuggingFace dataset card states
#: CC BY-NC-SA. Recorded in the audit output rather than silently picked.
LICENSE_NOTE = (
    "License discrepancy: the HierarchicalDet GitHub repository states CC BY-SA "
    "for DENTEX, while the HuggingFace dataset card states CC BY-NC-SA. The "
    "stricter reading (non-commercial) is assumed for this reproduction. This is "
    "reported, not resolved — it is a property of the release."
)
