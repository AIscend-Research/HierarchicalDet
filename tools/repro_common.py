"""
Shared helpers for the MLRC HierarchicalDet reproduction tooling.

Everything in tools/ builds on the official code paths: the official config
system (`add_diffusiondet_config` + the repo's own YAMLs), the official
dataset registration (`register_coco_instances` -> the bundled, 3-tier-aware
`load_coco_json`), and the official model (`build_model` on the registered
`DiffusionDet` meta-arch). This module only removes the boilerplate that would
otherwise be copy-pasted into every script.

Import note: this repo shadows the pip-installed `pycocotools` and `detectron2`
with its own bundled, modified copies, and that shadowing only works when the
repo root is on sys.path *first*. Scripts under tools/ are run as
`python tools/foo.py` from the repo root, so tools/ (not the repo root) lands
on sys.path[0] -- hence the explicit insert below.
"""
import hashlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# torch / detectron2 / hierarchialdet are imported inside the functions that
# need them, not at module scope: the orchestration scripts (run_curriculum,
# run_experiments, make_results_table) only want REPO_ROOT, file_md5 and the
# tier table, and should stay runnable -- and testable -- without a full deep
# learning environment.

# Tier index -> (short name, human-readable name). The model always carries all
# three classification heads; the tier selects how many are used.
TIERS = {
    0: ("quadrant", "quadrant detection"),
    1: ("quadrant_enumeration", "tooth enumeration (quadrant-enumeration)"),
    2: ("quadrant_enumeration_disease", "diagnosis (quadrant-enumeration-diagnosis)"),
}

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "diffdet.custom.swinbase.nonpretrain.yaml")

# Dataset names hardcoded in configs/diffdet.custom.swinbase.nonpretrain.yaml.
TRAIN_DATASET_NAME = "custom_train_class"
TEST_DATASET_NAME = "custom_validation_class"


def tier_name(tier):
    return TIERS[tier][0]


def setup_cfg(config_file=None, opts=None, weights=None, device=None, freeze=True):
    """Build a config exactly the way train_net.py / demo.py do."""
    import torch
    from detectron2.config import get_cfg
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file or DEFAULT_CONFIG)
    if opts:
        cfg.merge_from_list(list(opts))
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
    if device is not None:
        cfg.MODEL.DEVICE = device
    elif not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    if freeze:
        cfg.freeze()
    return cfg


def register_dataset(name, json_file, image_root):
    """
    Register a DENTEX split under `name`, tolerating repeated registration
    (scripts that sweep over several conditions re-register the same split).
    """
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    if name in MetadataCatalog.list():
        meta = MetadataCatalog.get(name)
        if getattr(meta, "json_file", None) == json_file:
            return meta
        MetadataCatalog.remove(name)
        from detectron2.data import DatasetCatalog

        DatasetCatalog.remove(name)
    register_coco_instances(name, {}, json_file, image_root)
    return MetadataCatalog.get(name)


def build_eval_model(cfg):
    """Build the model and load MODEL.WEIGHTS, honouring the EMA setup."""
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from hierarchialdet.util.model_ema import (
        may_build_model_ema,
        may_get_ema_checkpointer,
        EMADetectionCheckpointer,
    )

    model = build_model(cfg)
    may_build_model_ema(cfg, model)
    kwargs = may_get_ema_checkpointer(cfg, model)
    checkpointer_cls = EMADetectionCheckpointer if cfg.MODEL_EMA.ENABLED else DetectionCheckpointer
    checkpointer = checkpointer_cls(model, save_dir=cfg.OUTPUT_DIR, **kwargs)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    return model


def thing_classes(dataset_name, tier):
    """Per-tier class-name list as attached by the bundled load_coco_json."""
    from detectron2.data import MetadataCatalog

    meta = MetadataCatalog.get(dataset_name)
    return list(meta.get("thing_classes{}".format(tier + 1), []))


def file_md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pip_pycocotools():
    """
    Import the *pip-installed* pycocotools, bypassing this repo's bundled,
    3-tier-modified copy that shadows it at the repo root.

    Used for the independent, standard-COCO cross-check of the official
    evaluator's numbers, and for the flat single-tier baselines (RetinaNet /
    Faster R-CNN / DETR), whose predictions are plain COCO and must not go
    through the tier-aware fork. Returns (COCO, COCOeval).
    """
    # Drop the repo root (and the implicit cwd entry, which is the repo root
    # when scripts are launched from there) from sys.path, forget any already
    # imported pycocotools submodules, and import again -- so the import
    # machinery resolves the installed package instead of the fork. Everything
    # is restored afterwards, since the rest of the process still depends on
    # the fork.
    saved_modules = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "pycocotools"}
    saved_path = list(sys.path)
    try:
        for name in list(sys.modules):
            if name.split(".")[0] == "pycocotools":
                del sys.modules[name]
        sys.path = [
            p for p in sys.path
            if os.path.abspath(p or os.getcwd()) != REPO_ROOT
        ]
        importlib.invalidate_caches()
        try:
            coco_module = importlib.import_module("pycocotools.coco")
            cocoeval_module = importlib.import_module("pycocotools.cocoeval")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "no pip-installed pycocotools available ({}). Install it with "
                "`pip install pycocotools`; the copy bundled at {}/pycocotools is a "
                "3-tier fork and is deliberately not used here.".format(exc, REPO_ROOT)
            ) from exc
        if os.path.abspath(coco_module.__file__).startswith(REPO_ROOT + os.sep):
            raise ImportError(
                "only found the repo's bundled pycocotools fork at {}. Install the real "
                "one (`pip install pycocotools`) -- the fork assumes 3-tier categories and "
                "cannot evaluate flat COCO data.".format(coco_module.__file__)
            )
        return coco_module.COCO, cocoeval_module.COCOeval
    finally:
        sys.path = saved_path
        for name in list(sys.modules):
            if name.split(".")[0] == "pycocotools":
                del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()
