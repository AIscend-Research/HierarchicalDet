"""
Patched train_net.py for the HierarchicalDet reproduction.

Changes from the original:
  - Replaces hardcoded ../sorted/challenge/... dataset paths with
    DATA_ROOT environment variable (default: ../sorted/challenge).
  - Adds DENTEX_TIER env variable to select which annotation tier to train on:
      tier1 = quadrant only
      tier2 = quadrant + enumeration
      tier3 = quadrant + enumeration + diagnosis  (default)
  - All other logic is identical to the original train_net.py.

Usage:
  export DATA_ROOT=/path/to/sorted/challenge
  export DENTEX_TIER=tier3   # or tier1, tier2
  python train_net_patched.py --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
      --num-gpus 1 MODEL.WEIGHTS models/swin_base_patch4_window7_224_22k.pkl

For eval only:
  python train_net_patched.py --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
      --eval-only MODEL.WEIGHTS output/model_final.pth
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import random
import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch,
    create_ddp_model, AMPTrainer, SimpleTrainer, hooks
)
from detectron2.evaluation import LVISEvaluator, verify_results, print_csv_format
from hierarchialdet.util.coco_3class_eval import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from evaluator import DatasetEvaluator, inference_on_dataset
from dataset_mapper_patched import DiffusionDetDatasetMapper
from hierarchialdet import add_diffusiondet_config, DiffusionDetWithTTA
from hierarchialdet.util.model_ema import (
    add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer,
    EMAHook, apply_model_ema_and_restore, EMADetectionCheckpointer
)


# ── Dataset path configuration ────────────────────────────────────────────────
# Set DATA_ROOT to the directory containing COCO JSONs and image folders.
# Default mirrors the original hardcoded path relative to the repo root.
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "sorted", "challenge"))
DATA_ROOT = os.path.abspath(DATA_ROOT)

# Tier-specific dataset registrations
TIER_CONFIGS = {
    "tier1": {
        "train_json": "train_quadrant_coco.json",
        "train_images": "for_coco_quadrant_train",
        "val_json": "test_quadrant_coco.json",
        "val_images": "for_coco_quadrant_test",
    },
    "tier2": {
        "train_json": "train_enumeration_coco.json",
        "train_images": "for_coco_enumeration_train",
        "val_json": "test_enumeration_coco.json",
        "val_images": "for_coco_enumeration_test",
    },
    "tier3": {
        # Full diagnosis tier — matches original hardcoded paths
        "train_json": "train_merged_disease_coco3class_onlyd_fixed.json",
        "train_images": "for_coco_disease_train",
        "val_json": "test_merged_disease_coco3class.json",
        "val_images": "for_coco_disease_test",
    },
}

DENTEX_TIER = os.environ.get("DENTEX_TIER", "tier3")
assert DENTEX_TIER in TIER_CONFIGS, f"DENTEX_TIER must be one of {list(TIER_CONFIGS.keys())}"


def _register_datasets():
    from detectron2.data.datasets import register_coco_instances
    tier = TIER_CONFIGS[DENTEX_TIER]

    train_json = os.path.join(DATA_ROOT, tier["train_json"])
    train_imgs = os.path.join(DATA_ROOT, tier["train_images"])
    val_json = os.path.join(DATA_ROOT, tier["val_json"])
    val_imgs = os.path.join(DATA_ROOT, tier["val_images"])

    logger = logging.getLogger(__name__)
    logger.info(f"Registering datasets for tier: {DENTEX_TIER}")
    logger.info(f"  Train JSON: {train_json}")
    logger.info(f"  Train images: {train_imgs}")
    logger.info(f"  Val JSON:   {val_json}")
    logger.info(f"  Val images: {val_imgs}")

    for path in [train_json, train_imgs, val_json, val_imgs]:
        if not os.path.exists(path):
            logger.warning(f"MISSING: {path}")

    register_coco_instances("custom_train_class", {}, train_json, train_imgs)
    register_coco_instances("custom_validation_class", {}, val_json, val_imgs)


# ── Trainer (identical to original) ──────────────────────────────────────────
class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        kwargs = {"trainer": weakref.proxy(self)}
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, **kwargs)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "lvis" in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)
            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, k=0, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators)
        results_1 = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn("No evaluator found.")
                    results_1[dataset_name] = {}
                    continue
            results_i_1 = inference_on_dataset(model, data_loader, k, evaluator)
            results_1[dataset_name] = results_i_1
            if comm.is_main_process():
                assert isinstance(results_i_1[0], dict)
                logger.info(f"Evaluation results for {dataset_name} (class {k+1}):")
                print_csv_format(results_i_1)
        if len(results_1) == 1:
            results_1 = list(results_1.values())[0]
        return results_1

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators, k=2)
        else:
            results = cls.test(cfg, model, evaluators=evaluators, k=2)
        return results

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ) if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) else None,
        ]
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results(k):
            self._last_eval_results = self.test(self.cfg, self.model, k)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results, 0))
        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"DENTEX_TIER: {DENTEX_TIER}")
    _register_datasets()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
