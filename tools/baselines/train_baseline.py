"""
Train and run one of the detectron2 baselines the paper claims to outperform:
RetinaNet or Faster R-CNN (DETR lives in tools/baselines/train_detr.py, which
needs a different framework).

Each baseline is trained on a single tier of DENTEX as a flat detection
problem, which is exactly the comparison the paper draws: a conventional
detector has no way to use the partially-annotated lower tiers, so it only sees
the fully-annotated diagnosis tier (--tier 2, the default). Predictions are
written in COCO-results format and scored with tools/coco_eval_standalone.py,
the same stock evaluator used for the HierarchicalDet dumps, so the comparison
does not depend on either model's own evaluation code.

The training loop is assembled directly from detectron2's SimpleTrainer rather
than DefaultTrainer: this repo's bundled detectron2 has a patched EvalHook whose
signature DefaultTrainer.build_hooks no longer matches, and reaching for the
patched class would drag the three-tier assumptions back into a flat baseline.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from detectron2 import model_zoo  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import build_detection_test_loader, build_detection_train_loader  # noqa: E402
from detectron2.engine import SimpleTrainer, hooks  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.solver import build_lr_scheduler, build_optimizer  # noqa: E402
from detectron2.structures import BoxMode  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from baselines.flat_data import FlatDatasetMapper, register_flat  # noqa: E402

MODELS = {
    "retinanet": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "faster_rcnn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
}


def build_cfg(args, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODELS[args.model]))
    cfg.MODEL.WEIGHTS = args.weights or model_zoo.get_checkpoint_url(MODELS[args.model])
    cfg.DATASETS.TRAIN = ("baseline_train",)
    cfg.DATASETS.TEST = ("baseline_test",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.7), int(args.max_iter * 0.9))
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def train(cfg, args):
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    data_loader = build_detection_train_loader(cfg, mapper=FlatDatasetMapper(cfg, is_train=True))

    trainer = SimpleTrainer(model, data_loader, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer)
    start_iter = 0
    if args.resume and checkpointer.has_checkpoint():
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1
    else:
        checkpointer.load(cfg.MODEL.WEIGHTS)

    # SimpleTrainer is a TrainerBase; register the hooks by hand.
    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.LRScheduler(optimizer, build_lr_scheduler(cfg, optimizer)),
        hooks.PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=cfg.SOLVER.MAX_ITER),
        hooks.PeriodicWriter([
            CommonMetricPrinter(cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ], period=20),
    ])
    trainer.train(start_iter, cfg.SOLVER.MAX_ITER)
    checkpointer.save("model_final")
    return model


def predict(cfg, model, dataset_name, output_path, tier):
    model.eval()
    data_loader = build_detection_test_loader(
        cfg, dataset_name, mapper=FlatDatasetMapper(cfg, is_train=False))

    results, per_image = [], []
    with torch.no_grad():
        for inputs in data_loader:
            start = time.perf_counter()
            outputs = model(inputs)
            if cfg.MODEL.DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            instances = outputs[0]["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            for box, score, category in zip(boxes, scores, classes):
                xywh = BoxMode.convert(box.tolist(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                results.append({
                    "image_id": inputs[0]["image_id"],
                    "category_id": int(category),
                    "category_id_{}".format(tier + 1): int(category),
                    "bbox": [round(float(v), 2) for v in xywh],
                    "score": float(score),
                })
            per_image.append({"image_id": inputs[0]["image_id"], "seconds": round(elapsed, 4),
                              "num_detections": int(len(scores))})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    times = [r["seconds"] for r in per_image][5:]
    runtime_path = os.path.splitext(output_path)[0] + "_runtime.json"
    with open(runtime_path, "w") as f:
        json.dump({"summary": {
            "device": cfg.MODEL.DEVICE,
            "images": len(per_image),
            "mean_seconds_per_image": round(sum(times) / max(len(times), 1), 4),
            "images_per_minute": round(60.0 * len(times) / max(sum(times), 1e-9), 2) if times else None,
        }, "per_image": per_image}, f, indent=2)
    return results, runtime_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--train-json", required=True)
    p.add_argument("--train-images", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--test-images", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--max-iter", type=int, default=20000)
    p.add_argument("--ims-per-batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.00025)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--checkpoint-period", type=int, default=5000)
    p.add_argument("--seed", type=int, default=40244023)
    p.add_argument("--weights", default=None, help="init weights; default is the COCO-pretrained model zoo entry")
    p.add_argument("--eval-only", action="store_true", help="skip training, use --weights as the model")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(output=args.output_dir, name="baseline")

    _, class_names = register_flat("baseline_train", args.train_json, args.train_images, args.tier)
    register_flat("baseline_test", args.test_json, args.test_images, args.tier)
    print("tier {}: {} classes {}".format(args.tier, len(class_names), class_names))

    cfg = build_cfg(args, len(class_names))
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    if args.eval_only:
        model = build_model(cfg)
        DetectionCheckpointer(model, cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    else:
        model = train(cfg, args)

    predictions_path = os.path.join(args.output_dir, "predictions.json")
    results, runtime_path = predict(cfg, model, "baseline_test", predictions_path, args.tier)
    print("wrote {} ({} predictions)".format(predictions_path, len(results)))
    print("wrote {}".format(runtime_path))
    print("\nScore it with:\n  python tools/coco_eval_standalone.py --gt-json {} "
          "--predictions {} --tier {} --output {}".format(
              args.test_json, predictions_path, args.tier,
              os.path.join(args.output_dir, "metrics.json")))


if __name__ == "__main__":
    main()
