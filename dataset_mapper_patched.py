"""
Patched DiffusionDetDatasetMapper for the HierarchicalDet reproduction.

Changes from the original hierarchialdet/dataset_mapper.py:
  1. Removes hardcoded noisy-box file paths
     (ibrahim/Diseasedataset_base_enumeration_m_t_inference_train/...)
  2. Accepts NOISY_BOX_TRAIN / NOISY_BOX_VAL environment variables (or None)
     to supply pre-computed inference boxes from a previously trained tier model.
  3. When no noisy-box files are available (tier-1 training or ablation),
     falls back to using only ground-truth boxes, matching the DiffusionDet
     baseline behaviour (usepretrainedboxes=False).

Drop this file into the HierarchicalDet_official/ root and import it from
train_net_patched.py by replacing:
    from hierarchialdet import DiffusionDetDatasetMapper
with:
    from dataset_mapper_patched import DiffusionDetDatasetMapper

Environment variables:
    NOISY_BOX_TRAIN   Path to tier-(k-1) inference JSON on the training set.
                      Format: COCO results list with 'image_id', 'bbox', 'score'.
    NOISY_BOX_VAL     Same for the validation set.
    NOISY_BOX_THRESH  Score threshold for filtering noisy boxes (default: 0.5).
    USE_NOISY_BOXES   Set to '0' to disable noisy boxes entirely (tier-1 mode).
"""

import copy
import logging
import os
import json
from pathlib import Path

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

logger = logging.getLogger(__name__)

__all__ = ["DiffusionDetDatasetMapper"]

# ── Environment-variable configuration ────────────────────────────────────────
NOISY_BOX_TRAIN = os.environ.get("NOISY_BOX_TRAIN", None)
NOISY_BOX_VAL   = os.environ.get("NOISY_BOX_VAL",   None)
NOISY_BOX_THRESH = float(os.environ.get("NOISY_BOX_THRESH", "0.5"))
USE_NOISY_BOXES  = os.environ.get("USE_NOISY_BOXES", "1") != "0"


def build_transform_gen(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2

    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def _load_noisy_boxes(path, thresh):
    """Load and filter a COCO results JSON into a dict {image_id: [bbox, ...]}."""
    if path is None or not Path(path).exists():
        return None
    logger.info(f"Loading noisy boxes from {path} (threshold={thresh})")
    with open(path) as f:
        preds = json.load(f)
    boxes_by_img = {}
    for p in preds:
        if p.get("score", 1.0) >= thresh:
            iid = p["image_id"]
            boxes_by_img.setdefault(iid, []).append(p["bbox"])
    logger.info(f"  Loaded noisy boxes for {len(boxes_by_img)} images")
    return boxes_by_img


class DiffusionDetDatasetMapper:
    """
    Dataset mapper for HierarchicalDet.
    Supports optional noisy-box injection from a previous tier's inference.
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        # Noisy box injection
        self.use_noisy_boxes = USE_NOISY_BOXES
        if self.use_noisy_boxes:
            train_path = NOISY_BOX_TRAIN
            val_path   = NOISY_BOX_VAL
            if train_path or val_path:
                self.train_boxes_by_img = _load_noisy_boxes(train_path, NOISY_BOX_THRESH)
                self.valid_boxes_by_img = _load_noisy_boxes(val_path,   NOISY_BOX_THRESH)
                if self.train_boxes_by_img is None and self.valid_boxes_by_img is None:
                    logger.warning(
                        "NOISY_BOX_TRAIN/NOISY_BOX_VAL set but files not found. "
                        "Falling back to GT-only mode (tier-1 behaviour)."
                    )
                    self.use_noisy_boxes = False
            else:
                logger.info(
                    "No NOISY_BOX_TRAIN/NOISY_BOX_VAL set — running in GT-only mode "
                    "(appropriate for tier-1 / quadrant training or ablation)."
                )
                self.use_noisy_boxes = False
                self.train_boxes_by_img = None
                self.valid_boxes_by_img = None
        else:
            logger.info("USE_NOISY_BOXES=0 — noisy boxes disabled.")
            self.train_boxes_by_img = None
            self.valid_boxes_by_img = None

    def _get_noisy_boxes(self, image_id):
        if not self.use_noisy_boxes:
            return []
        boxes_by_img = self.train_boxes_by_img if self.is_train else self.valid_boxes_by_img
        if boxes_by_img is None:
            return []
        return boxes_by_img.get(image_id, [])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            image_id = dataset_dict.get("image_id")
            bboxpre = self._get_noisy_boxes(image_id)

            if bboxpre:
                annos = [
                    utils.transform_instance_annotations(
                        obj, transforms, image_shape, bbox_pre=bboxpre
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
            else:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]

            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
