"""
Flat, single-tier dataset plumbing for the baseline detectors.

RetinaNet, Faster R-CNN and DETR predict one flat class set, so they cannot use
this repo's three-tier data path at all: the bundled `load_coco_json` requires
`categories_1/2/3`, and the bundled `annotations_to_instances` produces
`gt_classes_1/2/3` where a standard detector expects `gt_classes`. Rather than
patch those (they are the code HierarchicalDet itself depends on), the
baselines get their own loader and mapper here, reading the same DENTEX JSONs
and producing plain detectron2 dataset dicts.

One tier is selected per baseline run: tier 0 gives 4 quadrant classes, tier 1
gives 8 tooth classes, tier 2 gives 4 diagnosis classes. Annotations with no
label at the chosen tier are dropped, exactly as tools/coco_eval_standalone.py
drops them when scoring, so training and evaluation see the same boxes.
"""
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import numpy as np
import torch


def load_flat_dentex(json_file, image_root, tier):
    with open(json_file) as f:
        coco = json.load(f)

    key = "category_id_{}".format(tier + 1)
    categories = coco.get("categories_{}".format(tier + 1)) or coco["categories"]
    # Contiguous ids in the order the categories are declared, so class indices
    # match what tools/coco_eval_standalone.py reports.
    id_to_contiguous = {c["id"]: index for index, c in enumerate(sorted(categories, key=lambda c: c["id"]))}

    by_image = {}
    for image in coco["images"]:
        by_image[image["id"]] = {
            "file_name": os.path.join(image_root, image["file_name"]),
            "image_id": image["id"],
            "height": image["height"],
            "width": image["width"],
            "annotations": [],
        }
    for ann in coco["annotations"]:
        category_id = ann.get(key, ann.get("category_id"))
        if category_id is None or ann["image_id"] not in by_image:
            continue
        by_image[ann["image_id"]]["annotations"].append({
            "bbox": ann["bbox"],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": id_to_contiguous[category_id],
            "iscrowd": ann.get("iscrowd", 0),
        })
    return list(by_image.values()), [str(c["name"]) for c in sorted(categories, key=lambda c: c["id"])]


def register_flat(name, json_file, image_root, tier):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    dicts, class_names = load_flat_dentex(json_file, image_root, tier)
    DatasetCatalog.register(name, lambda d=dicts: d)
    MetadataCatalog.get(name).set(
        thing_classes=class_names, json_file=json_file, image_root=image_root,
        evaluator_type="coco", tier=tier)
    return dicts, class_names


class FlatDatasetMapper:
    """
    Minimal standard-detectron2 mapper.

    Deliberately does not call `detection_utils.annotations_to_instances`: this
    repo's copy of that function builds `gt_classes_1/2/3` for the three-tier
    model and never sets the plain `gt_classes` a baseline detector needs.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        if is_train:
            self.transforms = [
                T.RandomFlip(),
                T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                     cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
            ]
        else:
            self.transforms = [
                T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                                     cfg.INPUT.MAX_SIZE_TEST, "choice"),
            ]

    def __call__(self, dataset_dict):
        dataset_dict = dict(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image, transforms = T.apply_transform_gens(self.transforms, image)
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        annotations = dataset_dict.pop("annotations", [])
        if not self.is_train:
            return dataset_dict

        boxes, classes = [], []
        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue
            bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
            bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
            bbox = np.minimum(bbox, list(image_shape + image_shape)[::-1])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            boxes.append(bbox)
            classes.append(annotation["category_id"])

        instances = Instances(image_shape)
        instances.gt_boxes = Boxes(np.array(boxes, dtype=np.float32).reshape(-1, 4))
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        dataset_dict["instances"] = instances
        return dataset_dict
