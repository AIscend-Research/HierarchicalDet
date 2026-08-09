"""
Dataset registration for the converted DENTEX splits.

The released ``train_net.py`` registers two dataset names against paths that
only existed on the authors' machine. Rather than editing upstream, this module
owns registration for the whole suite and keeps the *names* the shipped configs
reference (``custom_train_class`` / ``custom_validation_class``), so
``DATASETS.TRAIN`` / ``DATASETS.TEST`` in ``configs_repro/`` stay meaningful.

Registration goes through the vendored ``register_coco_instances``, which
attaches one class list and one id map *per tier*
(``thing_classes1/2/3``, ``thing_dataset_id_to_contiguous_id_1/2/3``) rather
than the single pair stock detectron2 attaches.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from . import data_convert, setup_env

#: The names the configs reference. Keep these stable.
TRAIN_DATASET = "custom_train_class"
TEST_DATASET = "custom_validation_class"

#: Logical split -> (annotation path key, image dir key) in ``data_convert.layout``.
SPLITS: Dict[str, Tuple[str, str]] = {
    "quadrant_train": ("train_quadrant", "img_quadrant"),
    "quadrant_enumeration_train": ("train_enumeration", "img_enumeration"),
    "diagnosis_train": ("train_diagnosis", "img_diagnosis"),
    "diagnosis_val": ("val_diagnosis", "img_val"),
    "diagnosis_test": ("test_diagnosis", "img_test"),
}

#: Which tier each training split supervises (the highest tier its
#: annotations carry a label for).
SPLIT_TIER = {
    "quadrant_train": 0,
    "quadrant_enumeration_train": 1,
    "diagnosis_train": 2,
    "diagnosis_val": 2,
    "diagnosis_test": 2,
}


def split_paths(split: str, data_dir: Optional[str] = None) -> Tuple[str, str]:
    """``(annotation_json, image_dir)`` for a logical split."""
    if split not in SPLITS:
        raise KeyError("unknown split {!r}; known: {}".format(split, sorted(SPLITS)))
    layout = data_convert.layout(data_dir)
    json_key, image_key = SPLITS[split]
    return layout[json_key], layout[image_key]


def register(name: str, json_file: str, image_root: str):
    """
    Register (or re-register) a COCO split under ``name``.

    Idempotent: sweeps re-register the same name with different files (a
    degraded image directory, a subset JSON), so an existing registration
    pointing elsewhere is replaced rather than silently reused.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    for path in (json_file, image_root):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "cannot register {!r}: {} does not exist. Run notebook 01 first."
                .format(name, path)
            )

    if name in DatasetCatalog.list():
        metadata = MetadataCatalog.get(name)
        same = (getattr(metadata, "json_file", None) == json_file
                and getattr(metadata, "image_root", None) == image_root)
        if same:
            return metadata
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    register_coco_instances(name, {}, json_file, image_root)
    return MetadataCatalog.get(name)


def register_pair(train_json: str, train_images: str,
                  test_json: str, test_images: str) -> None:
    """Register the train/test pair under the names the configs reference."""
    register(TRAIN_DATASET, train_json, train_images)
    register(TEST_DATASET, test_json, test_images)


def register_split_pair(train_split: str, test_split: str = "diagnosis_test",
                        data_dir: Optional[str] = None) -> None:
    register_pair(*split_paths(train_split, data_dir), *split_paths(test_split, data_dir))


def training_env(train_split: str, val_split: str = "diagnosis_val",
                 data_dir: Optional[str] = None) -> Dict[str, str]:
    """
    The environment ``train_entry.py`` reads to register datasets inside the
    (possibly multi-process) training launcher, where in-process registration
    from the notebook would not survive ``launch()``.
    """
    train_json, train_images = split_paths(train_split, data_dir)
    val_json, val_images = split_paths(val_split, data_dir)
    return {
        "TRAIN_JSON": train_json,
        "TRAIN_IMG_DIR": train_images,
        "VAL_JSON": val_json,
        "VAL_IMG_DIR": val_images,
        "TIER": str(SPLIT_TIER[train_split]),
    }


def thing_classes(dataset_name: str, tier: int) -> List[str]:
    """Per-tier class-name list attached by the vendored loader."""
    from detectron2.data import MetadataCatalog

    metadata = MetadataCatalog.get(dataset_name)
    return list(metadata.get("thing_classes{}".format(tier + 1), []))


def contiguous_to_dataset_id(dataset_name: str, tier: int) -> Dict[int, int]:
    """Invert the vendored per-tier id map (predictions come out contiguous)."""
    from detectron2.data import MetadataCatalog

    metadata = MetadataCatalog.get(dataset_name)
    forward = getattr(
        metadata, "thing_dataset_id_to_contiguous_id_{}".format(tier + 1), {}
    )
    return {v: k for k, v in forward.items()}


def verify_registration(data_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Register every split once and confirm the per-tier class lists are
    *identical across splits*.

    This matters more than it looks: the model has fixed-size heads
    (``NUM_CLASSES = [4, 8, 4]``) shared across curriculum stages, so if the
    class index assignment drifted between the quadrant split and the diagnosis
    split, weight transfer would move a head onto a different label space
    without any error being raised.
    """
    setup_env.assert_vendored()
    report: Dict[str, object] = {"splits": {}}
    reference: Optional[Dict[int, List[str]]] = None
    for split in SPLITS:
        json_file, image_root = split_paths(split, data_dir)
        register("verify_" + split, json_file, image_root)
        classes = {tier: thing_classes("verify_" + split, tier) for tier in (0, 1, 2)}
        report["splits"][split] = {
            "json": json_file,
            "images": image_root,
            "thing_classes": {str(k): v for k, v in classes.items()},
        }
        if reference is None:
            reference = classes
        elif classes != reference:
            raise AssertionError(
                "per-tier class lists differ between splits ({} vs the first split): "
                "{} vs {}. Weight transfer across stages would silently remap labels."
                .format(split, classes, reference)
            )
    report["thing_classes"] = {str(k): v for k, v in (reference or {}).items()}
    return report
