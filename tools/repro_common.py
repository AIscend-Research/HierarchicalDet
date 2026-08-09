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

import torch  # noqa: E402

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.data.datasets import register_coco_instances  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402

from hierarchialdet import add_diffusiondet_config  # noqa: E402
from hierarchialdet.util.model_ema import (  # noqa: E402
    add_model_ema_configs,
    may_build_model_ema,
    may_get_ema_checkpointer,
    EMADetectionCheckpointer,
)

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
    import site

    candidates = []
    for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
        candidates.append(os.path.join(base, "pycocotools"))
    for pkg_dir in candidates:
        init_py = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_py) or os.path.abspath(pkg_dir).startswith(REPO_ROOT + os.sep):
            continue
        modules = {}
        for mod_name in ("_pip_pycocotools", "_pip_pycocotools.mask", "_pip_pycocotools.coco",
                         "_pip_pycocotools.cocoeval"):
            file_name = {
                "_pip_pycocotools": "__init__.py",
                "_pip_pycocotools.mask": "mask.py",
                "_pip_pycocotools.coco": "coco.py",
                "_pip_pycocotools.cocoeval": "cocoeval.py",
            }[mod_name]
            spec = importlib.util.spec_from_file_location(mod_name, os.path.join(pkg_dir, file_name))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            modules[mod_name] = (spec, module)
        # Execute in dependency order: package, mask, coco, cocoeval.
        for mod_name in ("_pip_pycocotools", "_pip_pycocotools.mask", "_pip_pycocotools.coco",
                         "_pip_pycocotools.cocoeval"):
            spec, module = modules[mod_name]
            # coco.py / cocoeval.py do `from pycocotools import mask`, which
            # would resolve to the bundled fork. Point that name at the pip
            # copy for the duration of the import.
            saved = sys.modules.get("pycocotools"), sys.modules.get("pycocotools.mask")
            sys.modules["pycocotools"] = modules["_pip_pycocotools"][1]
            sys.modules["pycocotools.mask"] = modules["_pip_pycocotools.mask"][1]
            try:
                spec.loader.exec_module(module)
            finally:
                for key, value in zip(("pycocotools", "pycocotools.mask"), saved):
                    if value is None:
                        sys.modules.pop(key, None)
                    else:
                        sys.modules[key] = value
        return modules["_pip_pycocotools.coco"][1].COCO, modules["_pip_pycocotools.cocoeval"][1].COCOeval

    raise ImportError(
        "Could not find a pip-installed pycocotools outside the repo. "
        "Install it with `pip install pycocotools` -- the bundled copy at "
        "{}/pycocotools is a 3-tier fork and cannot evaluate flat COCO data.".format(REPO_ROOT)
    )
