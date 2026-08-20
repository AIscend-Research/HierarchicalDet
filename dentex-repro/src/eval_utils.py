"""
Evaluation: per-tier metrics, multi-seed inference, prediction dumps, runtime
and failure accounting.

The metric numbers come from the authors' own evaluator
(``hierarchialdet/util/coco_3class_eval.py``) driven by their own
``evaluator.inference_on_dataset``. This module wraps it; it does not
reimplement it. Two things are added on top, both of which are read off the
*same* ``COCOeval`` object the authors' code built:

* **AR**, which the paper's Table 1 reports but ``_derive_coco_results``
  discards (it keeps only ``stats[:6]``);
* **per-class AP**, kept as a labelled extension rather than a paper number.

Note on cost: ``inference_on_dataset(model, loader, k, evaluator)`` runs the
model **once** at tier ``k`` and then scores tiers ``0..k`` off the stored
predictions, so a single pass over the test split yields the quadrant,
enumeration and diagnosis rows together.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import statistics
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

from . import registration, setup_env

#: Metric names as they appear in the paper's Table 1, in column order.
PAPER_METRICS = ("AR", "AP", "AP50", "AP75", "APm", "APl")
#: Everything we record (APs is computed too; the paper omits it).
ALL_METRICS = ("AR", "AP", "AP50", "AP75", "APs", "APm", "APl")

TIER_KEY = {0: "quadrant", 1: "enumeration", 2: "diagnosis"}
#: Suffix the vendored evaluator appends to each metric name, per tier.
TIER_SUFFIX = {0: "Quadrant", 1: "Enumeration", 2: "Disease"}


# --------------------------------------------------------------------------
# Config / model
# --------------------------------------------------------------------------
def setup_cfg(config_file: str, weights: str, overrides: Sequence[str] = (),
              device: Optional[str] = None, sample_step: Optional[int] = None,
              output_dir: Optional[str] = None):
    from detectron2.config import get_cfg
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs
    import torch

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([str(o) for o in overrides])
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL_EMA.ENABLED = False          # held constant across every run
    if sample_step is not None:
        cfg.MODEL.DiffusionDet.SAMPLE_STEP = int(sample_step)
    cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.OUTPUT_DIR = output_dir or os.path.join(setup_env.RUNS_DIR, "_eval_tmp")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    return cfg


def build_eval_model(cfg):
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from hierarchialdet.util.model_ema import may_build_model_ema

    model = build_model(cfg)
    may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    model.eval()
    return model


def prior_box_coverage(predictions_json: Optional[str],
                       score_threshold: float = 0.5) -> Dict[str, object]:
    """
    How many prior-tier boxes would survive ``score_threshold``, and over how
    many images. Cheap enough to call before an experiment commits to a GPU.
    """
    empty = {"boxes_total": 0, "boxes_kept": 0, "images_with_boxes": 0,
             "score_threshold": score_threshold, "path": predictions_json}
    if not predictions_json or not os.path.exists(predictions_json):
        return empty
    with open(predictions_json) as handle:
        records = json.load(handle)
    kept_images = set()
    kept = 0
    for record in records:
        if record.get("score", 1.0) >= score_threshold:
            kept += 1
            kept_images.add(record.get("image_id"))
    return {"boxes_total": len(records), "boxes_kept": kept,
            "images_with_boxes": len(kept_images),
            "score_threshold": score_threshold, "path": predictions_json}


@contextlib.contextmanager
def noisy_box_inference(predictions_json: Optional[str], jitter: float = 0.0,
                        drop: float = 0.0, score_threshold: float = 0.5):
    """
    Turn on prior-tier box injection at *inference* time.

    The released code only injects prior-tier boxes during training (through the
    dataset mapper); the inference-time equivalent exists in ``detector.py`` only
    as commented-out code pointing at paths that were never published. The repo
    in this checkout re-enables it behind ``NOISY_BOX_INFER``, with
    ``_JITTER``/``_DROP`` to deliberately corrupt the injected boxes — which is
    exactly what the hierarchy fault-injection experiment needs.

    The detector reads these at construction, so the model must be built inside
    this context manager.

    Yields a dict describing what the detector will actually have to work with.
    If no prior box clears ``score_threshold`` the detector injects nothing and
    runs the *unperturbed* model, so every severity in a fault-injection sweep
    returns bit-identical numbers and the experiment reads as "the hierarchy is
    insensitive to prior-tier quality" when it in fact never tested that. That
    is a failed experiment, not a finding, so it raises here rather than
    producing results.
    """
    usable = prior_box_coverage(predictions_json, score_threshold)
    if predictions_json and not usable["boxes_kept"]:
        raise ValueError(
            "inference-time box injection would be a no-op: none of {} prior-tier "
            "predictions in {} scored >= {}. Running anyway would report the "
            "unperturbed model as a fault-injection result. Lower "
            "score_threshold, or use a prior-tier model that actually detects "
            "something.".format(usable["boxes_total"], predictions_json,
                                score_threshold))
    keys = {
        "NOISY_BOX_INFER": predictions_json or "",
        "NOISY_BOX_INFER_JITTER": str(jitter),
        "NOISY_BOX_INFER_DROP": str(drop),
        "NOISY_BOX_INFER_SCORE": str(score_threshold),
    }
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for key, value in keys.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        yield usable
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------
# Evaluator wrapper
# --------------------------------------------------------------------------
def _capturing_evaluator_class():
    """
    Subclass the vendored ``COCOEvaluator`` purely to keep the ``COCOeval``
    object its own code already built, so AR and the per-class APs come from
    the identical computation that produced the AP numbers.
    """
    from hierarchialdet.util.coco_3class_eval import COCOEvaluator

    class CapturingCOCOEvaluator(COCOEvaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured: Dict[int, Dict[str, object]] = {}

        def _derive_coco_results(self, coco_eval, iou_type, class_names=None, i=0):
            results = super()._derive_coco_results(coco_eval, iou_type, class_names, i)
            capture: Dict[str, object] = {"num_predictions": None}
            if coco_eval is not None and getattr(coco_eval, "stats", None) is not None:
                stats = list(coco_eval.stats)
                # COCOeval.stats layout: 0..5 AP family, 6..8 AR@{1,10,100},
                # 9..11 AR@100 for small/medium/large.
                capture.update({
                    "AR": float(stats[8]) if len(stats) > 8 else None,
                    "AR1": float(stats[6]) if len(stats) > 6 else None,
                    "AR10": float(stats[7]) if len(stats) > 7 else None,
                    "ARs": float(stats[9]) if len(stats) > 9 else None,
                    "ARm": float(stats[10]) if len(stats) > 10 else None,
                    "ARl": float(stats[11]) if len(stats) > 11 else None,
                    "num_predictions": int(len(coco_eval.cocoDt.anns)),
                })
            self.captured[i] = capture
            return results

    return CapturingCOCOEvaluator


def _unpack_tier_results(raw: Dict[str, object], tier: int,
                         captured: Dict[str, object]) -> Dict[str, object]:
    """
    Flatten the evaluator's tier-suffixed keys (``AP_Quadrant``, ...) into the
    plain metric names the tables use, and split out the per-class APs.
    """
    box = (raw or {}).get("bbox", {}) or {}
    suffix = TIER_SUFFIX[tier]
    metrics: Dict[str, Optional[float]] = {}
    for metric in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
        value = box.get("{}_{}".format(metric, suffix))
        metrics[metric] = float(value) if isinstance(value, (int, float)) else None
    metrics["AR"] = captured.get("AR")
    # The vendored evaluator names these "AP-" + class + tier
    # (coco_3class_eval.py:432), producing labels like "1Enumeration". Strip the
    # tier back off so per-class figures read "1".."8" rather than repeating the
    # tier on every tick.
    per_class = {}
    for key, value in box.items():
        if not key.startswith("AP-"):
            continue
        name = key[len("AP-"):]
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
        per_class[name] = float(value) if isinstance(value, (int, float)) else None
    return {
        "metrics": metrics,
        "per_class_AP": per_class,
        "recall_detail": {k: captured.get(k) for k in ("AR1", "AR10", "ARs", "ARm", "ARl")},
        "num_predictions": captured.get("num_predictions"),
        "raw": box,
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_checkpoint(weights: str, config_file: str, split: str = "diagnosis_test",
                        tier: int = 2, seed: int = 0, sample_step: Optional[int] = None,
                        overrides: Sequence[str] = (), device: Optional[str] = None,
                        data_dir: Optional[str] = None, limit: Optional[int] = None,
                        json_override: Optional[str] = None,
                        image_dir_override: Optional[str] = None,
                        dataset_name: str = registration.TEST_DATASET,
                        output_dir: Optional[str] = None) -> Dict[str, object]:
    """
    One evaluation pass: metrics for every tier from 0 up to ``tier``.

    ``seed`` seeds the *inference* RNG. DiffusionDet starts denoising from
    random boxes, so two seeds on identical weights give different numbers;
    that spread is a reported quantity, not noise to be hidden.
    """
    import torch

    from evaluator import inference_on_dataset

    json_file, image_dir = registration.split_paths(split, data_dir)
    json_file = json_override or json_file
    image_dir = image_dir_override or image_dir
    # `limit` must subset the GROUND TRUTH, not truncate the loader: the
    # evaluator scores against every image in the annotation file, so running
    # the model on 20 of 250 images while scoring against all 250 would count
    # 230 images as pure misses and report a meaningless AP.
    if limit is not None:
        json_file = subset_annotations(json_file, limit)
    registration.register(dataset_name, json_file, image_dir)

    cfg = setup_cfg(config_file, weights, overrides, device, sample_step, output_dir)
    seed_record = setup_env.seed_everything(seed, deterministic=True)
    model = build_eval_model(cfg)

    from train_net_patched import Trainer

    data_loader = Trainer.build_test_loader(cfg, dataset_name)

    evaluator = _capturing_evaluator_class()(dataset_name, cfg, True, None)
    if cfg.MODEL.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    raw_results = inference_on_dataset(model, data_loader, tier, evaluator)
    wall_seconds = time.time() - started

    tiers = {}
    for index in range(tier + 1):
        tiers[TIER_KEY[index]] = _unpack_tier_results(
            raw_results[index] if index < len(raw_results) else {},
            index, evaluator.captured.get(index, {}),
        )

    record = {
        "weights": os.path.abspath(weights),
        "weights_sha256": setup_env.file_sha256(weights),
        "config_file": os.path.abspath(config_file),
        "config_hash": setup_env.config_hash(config_file, list(overrides)),
        "split": split,
        "annotations": os.path.abspath(json_file),
        "image_dir": os.path.abspath(image_dir),
        "tier": tier,
        "inference_seed": seed,
        "determinism": seed_record,
        "sample_step": cfg.MODEL.DiffusionDet.SAMPLE_STEP,
        "num_proposals": cfg.MODEL.DiffusionDet.NUM_PROPOSALS,
        "device": cfg.MODEL.DEVICE,
        "limit": limit,
        "wall_seconds": round(wall_seconds, 2),
        "noisy_box_infer": os.environ.get("NOISY_BOX_INFER"),
        "noisy_box_infer_jitter": os.environ.get("NOISY_BOX_INFER_JITTER"),
        "noisy_box_infer_drop": os.environ.get("NOISY_BOX_INFER_DROP"),
        "tiers": tiers,
    }
    if cfg.MODEL.DEVICE == "cuda":
        record["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    del model
    _free_gpu()
    return record


def subset_annotations(json_file: str, limit: int) -> str:
    """
    Write (and cache) a COCO file restricted to the first ``limit`` images.

    Deterministic — the first N by file order, not a random sample — so a smoke
    run is comparable to itself across sessions.
    """
    from . import data_convert

    with open(json_file) as handle:
        data = json.load(handle)
    if len(data["images"]) <= limit:
        return json_file
    keep = [image["id"] for image in data["images"][:limit]]
    cache = os.path.join(setup_env.RUNS_DIR, "eval_subsets")
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, "{}_first{}.json".format(
        os.path.splitext(os.path.basename(json_file))[0], limit))
    if not os.path.exists(out):
        data_convert.subset_json(json_file, keep, out)
    return out


def _free_gpu():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_multi_seed(weights: str, config_file: str, seeds: Sequence[int],
                        **kwargs) -> Dict[str, object]:
    """Run :func:`evaluate_checkpoint` once per inference seed and aggregate."""
    runs = [evaluate_checkpoint(weights, config_file, seed=seed, **kwargs)
            for seed in seeds]
    return {"seeds": list(seeds), "runs": runs, "aggregate": aggregate_seeds(runs)}


def aggregate_seeds(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """mean / std / n per tier per metric across inference seeds."""
    aggregate: Dict[str, Dict[str, Dict[str, object]]] = {}
    if not runs:
        return aggregate
    for tier_key in runs[0]["tiers"]:
        aggregate[tier_key] = {}
        for metric in ALL_METRICS:
            raw = [run["tiers"][tier_key]["metrics"].get(metric) for run in runs]
            # NaN is a float, so an isinstance check alone let it through and
            # statistics.stdev died on it ("inf or nan encountered in data").
            # The evaluator legitimately returns NaN -- COCOeval reports -1 for
            # a metric with no ground truth in that size bucket, which
            # _derive_coco_results turns into NaN -- so this is a normal input,
            # not a corrupt one, and it must be dropped rather than crash a
            # 2.5-hour evaluation at the aggregation step.
            values = [v for v in raw
                      if isinstance(v, (int, float)) and math.isfinite(v)]
            if not values:
                # Dropping NaN loses the difference between "the seeds all
                # reported this metric as undefined" and "there were no seeds".
                # Both used to collapse to None, which the table formatter
                # prints as "not run" -- so every APm/APs cell claimed the
                # experiment had been skipped when it had run and simply found
                # no objects in that size bucket. Keep NaN for the first case;
                # it formats as "n/a".
                undefined = any(isinstance(v, float) and math.isnan(v) for v in raw)
                aggregate[tier_key][metric] = {
                    "mean": float("nan") if undefined else None,
                    "std": float("nan") if undefined else None,
                    "n": 0,
                }
                continue
            aggregate[tier_key][metric] = {
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "n": len(values),
                "values": [round(v, 4) for v in values],
            }
    return aggregate


# --------------------------------------------------------------------------
# Prediction dumps (noisy boxes, runtime, failure metrics)
# --------------------------------------------------------------------------
def classify_failures(records: Sequence[dict], image_info: dict) -> List[str]:
    problems = []
    if not records:
        problems.append("no_detections")
    width, height = image_info["width"], image_info["height"]
    for record in records:
        x, y, box_width, box_height = record["bbox"]
        if box_width <= 1 or box_height <= 1:
            problems.append("degenerate_box")
        elif x < -1 or y < -1 or x + box_width > width + 1 or y + box_height > height + 1:
            problems.append("box_out_of_image")
        elif box_width * box_height > 0.9 * width * height:
            problems.append("box_covers_whole_image")
        elif box_height > 0 and not (0.02 <= box_width / box_height <= 50):
            problems.append("extreme_aspect_ratio")
    return sorted(set(problems))


def dump_predictions(weights: str, config_file: str, split: str, tier: int,
                     output_json: str, seed: int = 0, device: Optional[str] = None,
                     sample_step: Optional[int] = None, limit: Optional[int] = None,
                     score_threshold: float = 0.0, data_dir: Optional[str] = None,
                     overrides: Sequence[str] = (),
                     json_override: Optional[str] = None,
                     image_dir_override: Optional[str] = None,
                     dataset_name: str = registration.TEST_DATASET) -> Dict[str, object]:
    """
    Run inference over a split and write COCO-format predictions, plus
    per-image runtime and a failure taxonomy.

    Used for three different jobs: producing the prior-tier boxes the
    manipulation switch consumes, measuring runtime per image, and counting
    degenerate detections.
    """
    import torch

    from hierarchialdet.util.coco_3class_eval import instances_to_coco_json

    json_file, image_dir = registration.split_paths(split, data_dir)
    json_file = json_override or json_file
    image_dir = image_dir_override or image_dir
    registration.register(dataset_name, json_file, image_dir)

    cfg = setup_cfg(config_file, weights, overrides, device, sample_step)
    setup_env.seed_everything(seed, deterministic=True)
    model = build_eval_model(cfg)

    from train_net_patched import Trainer

    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    with open(json_file) as handle:
        images = {image["id"]: image for image in json.load(handle)["images"]}
    contiguous_to_dataset = registration.contiguous_to_dataset_id(dataset_name, tier)

    results, per_image = [], []
    failure_counts: Counter = Counter()
    if cfg.MODEL.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for index, inputs in enumerate(data_loader):
            if limit is not None and index >= limit:
                break
            image_id = inputs[0]["image_id"]
            if cfg.MODEL.DEVICE == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(inputs, k=tier)
            if cfg.MODEL.DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            instances = outputs[0]["instances"].to("cpu")
            records = instances_to_coco_json(instances, image_id, tier)
            records = [r for r in records if r.get("score", 1.0) >= score_threshold]
            for record in records:
                predicted = record["category_id_{}".format(tier + 1)]
                record["category_id"] = contiguous_to_dataset.get(predicted, predicted)
            results.extend(records)

            problems = classify_failures(records, images[image_id])
            failure_counts.update(problems)
            per_image.append({
                "image_id": image_id,
                "file_name": os.path.basename(inputs[0]["file_name"]),
                "seconds": round(elapsed, 4),
                "num_detections": len(records),
                "problems": problems,
            })

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as handle:
        json.dump(results, handle)

    times = [row["seconds"] for row in per_image]
    warm = times[5:] if len(times) > 10 else times      # drop CUDA autotuning cost
    summary = {
        "weights": os.path.abspath(weights),
        "predictions": os.path.abspath(output_json),
        "split": split,
        "annotations": os.path.abspath(json_file),
        "tier": tier,
        "device": cfg.MODEL.DEVICE,
        "inference_seed": seed,
        "sample_step": cfg.MODEL.DiffusionDet.SAMPLE_STEP,
        "num_proposals": cfg.MODEL.DiffusionDet.NUM_PROPOSALS,
        "images": len(per_image),
        "total_detections": len(results),
        "mean_seconds_per_image": round(sum(warm) / len(warm), 4) if warm else None,
        "median_seconds_per_image": round(sorted(warm)[len(warm) // 2], 4) if warm else None,
        "images_per_minute": round(60.0 * len(warm) / sum(warm), 2) if warm and sum(warm) else None,
        "failure_counts": dict(failure_counts),
        "images_with_no_detections": sum(1 for r in per_image if r["num_detections"] == 0),
        "class_names": registration.thing_classes(dataset_name, tier),
    }
    if cfg.MODEL.DEVICE == "cuda":
        summary["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

    runtime_json = os.path.splitext(output_json)[0] + "_runtime.json"
    with open(runtime_json, "w") as handle:
        json.dump({"summary": summary, "per_image": per_image}, handle, indent=2)

    del model
    _free_gpu()
    summary["runtime_json"] = runtime_json
    return summary


# --------------------------------------------------------------------------
# Result persistence
# --------------------------------------------------------------------------
def save_result(name: str, payload: Dict[str, object]) -> str:
    """Every metric JSON lands in ``paper_assets/results_raw`` and is auditable."""
    os.makedirs(setup_env.RESULTS_RAW, exist_ok=True)
    path = os.path.join(setup_env.RESULTS_RAW, "{}.json".format(name))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def load_result(name: str) -> Optional[Dict[str, object]]:
    path = os.path.join(setup_env.RESULTS_RAW, "{}.json".format(name))
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return _repair_undefined_aggregates(json.load(handle))


def _repair_undefined_aggregates(payload):
    """
    Restore the NaN that older aggregates flattened to None.

    Results written before :func:`aggregate_seeds` kept the distinction store
    ``{"mean": null, "n": 0}`` for a metric every seed reported as undefined,
    which the table formatter renders as "not run" -- claiming an experiment was
    skipped when it ran. The per-run metrics in the same file still carry the
    NaN, so the truth is recoverable without re-evaluating anything.
    """
    if not isinstance(payload, dict):
        return payload
    aggregate = payload.get("aggregate")
    runs = payload.get("runs")
    if not isinstance(aggregate, dict) or not isinstance(runs, list):
        return payload
    for tier_key, metrics in aggregate.items():
        if not isinstance(metrics, dict):
            continue
        for metric, cell in metrics.items():
            if not isinstance(cell, dict) or cell.get("mean") is not None:
                continue
            observed = [(run.get("tiers", {}).get(tier_key, {})
                         .get("metrics", {}) or {}).get(metric)
                        for run in runs if isinstance(run, dict)]
            if any(isinstance(v, float) and math.isnan(v) for v in observed):
                cell["mean"] = float("nan")
                cell["std"] = float("nan")
    return payload


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------
def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two COCO XYWH boxes."""
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(predictions: Sequence[dict], ground_truth: Sequence[dict],
                      iou_threshold: float = 0.5) -> Dict[str, object]:
    """
    Greedy score-ordered matching of predictions to ground truth on one image.

    Returns matched pairs plus the unmatched predictions (false positives) and
    unmatched ground truth (misses). Localisation and classification are
    deliberately separated: a box that matches spatially but carries the wrong
    tier label is a *matched* pair with a class disagreement, not a miss plus a
    false positive. That distinction is the whole point of asking whether errors
    cluster at the diagnosis tier or the quadrant tier.
    """
    used = set()
    matches = []
    for prediction in sorted(predictions, key=lambda p: -p.get("score", 0.0)):
        best_index, best_iou = None, iou_threshold
        for index, truth in enumerate(ground_truth):
            if index in used:
                continue
            value = _iou(prediction["bbox"], truth["bbox"])
            if value >= best_iou:
                best_index, best_iou = index, value
        if best_index is None:
            matches.append({"prediction": prediction, "gt": None, "iou": None})
        else:
            used.add(best_index)
            matches.append({"prediction": prediction, "gt": ground_truth[best_index],
                            "iou": round(best_iou, 4)})
    missed = [truth for index, truth in enumerate(ground_truth) if index not in used]
    return {"matches": matches, "missed": missed}


def error_analysis(predictions_json: str, ground_truth_json: str, tier: int,
                   score_threshold: float = 0.5,
                   iou_threshold: float = 0.5) -> Dict[str, object]:
    """
    Per-tier confusion counts and a per-image error taxonomy.

    ``correct_lower_tier_wrong_upper`` is the number the hierarchy question
    turns on: boxes whose quadrant is right but whose enumeration or diagnosis
    is wrong.
    """
    with open(predictions_json) as handle:
        predictions = json.load(handle)
    with open(ground_truth_json) as handle:
        truth_data = json.load(handle)

    gt_by_image: Dict[int, List[dict]] = {}
    for annotation in truth_data["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation)
    predictions_by_image: Dict[int, List[dict]] = {}
    for record in predictions:
        if record.get("score", 1.0) >= score_threshold:
            predictions_by_image.setdefault(record["image_id"], []).append(record)

    id_to_name = {
        level: {c["id"]: str(c["name"])
                for c in truth_data["categories_{}".format(level + 1)]}
        for level in range(3)
    }
    confusion: Dict[int, Counter] = {level: Counter() for level in range(tier + 1)}
    per_image, totals = [], Counter()

    for image in truth_data["images"]:
        image_id = image["id"]
        truths = gt_by_image.get(image_id, [])
        preds = predictions_by_image.get(image_id, [])
        result = match_predictions(preds, truths, iou_threshold)

        counts = Counter()
        counts["gt"] = len(truths)
        counts["pred"] = len(preds)
        counts["missed"] = len(result["missed"])
        for match in result["matches"]:
            if match["gt"] is None:
                counts["false_positive"] += 1
                continue
            counts["localised"] += 1
            correct_so_far = True
            for level in range(tier + 1):
                predicted = match["prediction"].get("category_id_{}".format(level + 1))
                actual_dataset_id = match["gt"].get("category_id_{}".format(level + 1))
                if actual_dataset_id is None:
                    continue
                predicted_name = id_to_name[level].get(
                    _dataset_id_for(predicted, id_to_name[level]), str(predicted))
                actual_name = id_to_name[level].get(actual_dataset_id, str(actual_dataset_id))
                confusion[level][(actual_name, predicted_name)] += 1
                if predicted_name == actual_name:
                    counts["correct_tier{}".format(level)] += 1
                else:
                    counts["wrong_tier{}".format(level)] += 1
                    if correct_so_far and level > 0:
                        counts["correct_lower_tier_wrong_upper"] += 1
                    correct_so_far = False
            # A box spanning several teeth: much larger than the matched GT.
            gt_area = match["gt"]["bbox"][2] * match["gt"]["bbox"][3]
            pred_area = (match["prediction"]["bbox"][2] * match["prediction"]["bbox"][3])
            if gt_area > 0 and pred_area / gt_area > 2.5:
                counts["box_spans_multiple_teeth"] += 1

        totals.update(counts)
        per_image.append({"image_id": image_id, "file_name": image["file_name"],
                          **dict(counts)})

    return {
        "predictions": os.path.abspath(predictions_json),
        "ground_truth": os.path.abspath(ground_truth_json),
        "tier": tier,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "totals": dict(totals),
        "per_image": per_image,
        "confusion": {
            str(level): {"{} -> {}".format(actual, predicted): count
                         for (actual, predicted), count in pairs.most_common()}
            for level, pairs in confusion.items()
        },
        # None, not 0.0, when nothing was localised at this tier. The rate is a
        # fraction of correctly localised boxes, so with no such boxes it is 0/0
        # and undefined. Dividing by max(1, n) reported that case as a clean
        # zero, which reads as "no label errors" -- the opposite of the truth,
        # which is that there was nothing to get right or wrong. The denominator
        # ships alongside it so any consumer can tell the two apart.
        "error_rate_by_tier": {
            str(level): (
                round(sum(count for (a, p), count in confusion[level].items() if a != p)
                      / sum(confusion[level].values()), 4)
                if sum(confusion[level].values()) else None
            )
            for level in confusion
        },
        "labelled_boxes_by_tier": {
            str(level): sum(confusion[level].values()) for level in confusion
        },
    }


def _dataset_id_for(contiguous, id_to_name: Dict[int, str]):
    """
    Predictions carry *contiguous* class indices; ground truth carries dataset
    category ids. For DENTEX the two coincide because the category ids are
    already 0-based and dense, but map through the sorted id list rather than
    assuming it.
    """
    if contiguous is None:
        return None
    ordered = sorted(id_to_name)
    if isinstance(contiguous, int) and 0 <= contiguous < len(ordered):
        return ordered[contiguous]
    return contiguous


def select_gallery_cases(analysis: Dict[str, object], per_category: int = 2
                         ) -> Dict[str, List[dict]]:
    """
    Pick concrete image ids for the failure gallery, one bucket per failure
    mode, so the figure shows real named cases rather than cherry-picked ones.
    """
    rows = analysis["per_image"]
    buckets = {
        "missed_teeth": lambda r: r.get("missed", 0),
        "false_positives": lambda r: r.get("false_positive", 0),
        "boxes_spanning_multiple_teeth": lambda r: r.get("box_spans_multiple_teeth", 0),
        "wrong_upper_tier_correct_quadrant": lambda r: r.get("correct_lower_tier_wrong_upper", 0),
        "zero_detections": lambda r: 1 if r.get("pred", 0) == 0 else 0,
    }
    selected = {}
    for name, score in buckets.items():
        ranked = sorted((r for r in rows if score(r) > 0), key=score, reverse=True)
        selected[name] = ranked[:per_category]
    return selected


def model_size_report(weights: str) -> Dict[str, object]:
    """Parameter count and on-disk checkpoint size, for the low-resource table."""
    import torch

    state = torch.load(weights, map_location="cpu")
    tensors = state.get("model", state)
    parameters = sum(int(v.numel()) for v in tensors.values()
                     if hasattr(v, "numel"))
    return {
        "checkpoint": os.path.abspath(weights),
        "checkpoint_bytes": os.path.getsize(weights),
        "checkpoint_mb": round(os.path.getsize(weights) / (1024 ** 2), 1),
        "parameters": parameters,
        "parameters_millions": round(parameters / 1e6, 2),
    }
