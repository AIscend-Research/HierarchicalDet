import os, sys

sys.path.insert(0, "/Users/christopherhuang/Documents/GitHub/HierarchicalDet")
os.chdir("/Users/christopherhuang/Documents/GitHub/HierarchicalDet")

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.config import get_cfg
from hierarchialdet import add_diffusiondet_config
from hierarchialdet.dataset_mapper_patched import DiffusionDetDatasetMapper

AUDIT = "/Users/christopherhuang/Documents/GitHub/HierarchicalDet/phase1_audit"

SPLITS = {
    "quadrant_train": (
        f"{AUDIT}/extracted/training_data/quadrant/train_quadrant.json",
        f"{AUDIT}/extracted/training_data/quadrant/xrays",
    ),
    "quadrant_enumeration_train": (
        f"{AUDIT}/extracted/training_data/quadrant_enumeration/train_quadrant_enumeration.json",
        f"{AUDIT}/extracted/training_data/quadrant_enumeration/xrays",
    ),
    "quadrant_enumeration_disease_train": (
        f"{AUDIT}/extracted/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        f"{AUDIT}/extracted/training_data/quadrant-enumeration-disease/xrays",
    ),
    "quadrant_enumeration_disease_val": (
        f"{AUDIT}/DENTEX/validation_triple.json",
        f"{AUDIT}/extracted/validation_data/quadrant_enumeration_disease/xrays",
    ),
}

cfg = get_cfg()
add_diffusiondet_config(cfg)
cfg.merge_from_file("configs/diffdet.custom.swinbase.nonpretrain.yaml")
cfg.DATALOADER.NUM_WORKERS = 0

for name, (json_path, img_dir) in SPLITS.items():
    print(f"=== {name} ===")
    try:
        register_coco_instances(name, {}, json_path, img_dir)
        dicts = DatasetCatalog.get(name)
        print(f"  DatasetCatalog.get() returned {len(dicts)} records")
        print(f"  sample record keys: {list(dicts[0].keys())}")
        print(f"  sample annotation keys: {list(dicts[0]['annotations'][0].keys()) if dicts[0]['annotations'] else 'NO ANNOTATIONS'}")

        # Run every record through the official dataset mapper -- this is
        # the actual "does the official pipeline load this without errors"
        # check the roadmap asks for, not just JSON parsing.
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        n_ok = 0
        n_fail = 0
        failures = []
        for d in dicts:
            try:
                out = mapper(d)
                assert "image" in out and "instances" in out
                n_ok += 1
            except Exception as e:
                n_fail += 1
                failures.append((d.get("file_name", "?"), str(e)))
        print(f"  mapper success: {n_ok}/{len(dicts)}  failures: {n_fail}")
        if failures:
            for f, e in failures[:5]:
                print("   FAIL:", f, e)
    except Exception as e:
        print(f"  REGISTRATION/LOAD FAILED: {type(e).__name__}: {e}")
    print()
