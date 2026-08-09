"""
Run a trained HierarchicalDet checkpoint over a split and dump raw predictions
in COCO-results format, plus per-image runtime and failure records.

This is the single inference entry point the rest of the reproduction builds
on:

* the **curriculum** (tools/run_curriculum.py) uses it between stages, to turn
  the tier k-1 model's predictions into the noisy boxes tier k trains on --
  that is the paper's noisy box manipulation, and it needs predictions over the
  *next* tier's train and validation images;
* **evaluation** (tools/evaluate_tiers.py) scores these dumps;
* the **runtime / robustness / step-count** experiments all re-run this with
  different devices, images or SAMPLE_STEP values.

Predictions are produced through the official code path: the official test
loader (`DefaultTrainer.build_test_loader`), the official model called as
`model(inputs, k=tier)`, and the official `instances_to_coco_json` used by the
authors' own evaluator, so the dumped numbers are the same numbers the official
evaluator would see.

Output JSON is a flat list of COCO results, each with `image_id`, `bbox`
(absolute XYWH in original image coordinates), `score`, and `category_id_1..N`
for the tiers the model predicted at. `category_id` is set to the tier's
predicted class, which is what a standard COCO evaluator reads.
"""
import argparse
import json
import os
import time

from repro_common import (  # noqa: E402  (repo root is put on sys.path by this import)
    TEST_DATASET_NAME,
    build_eval_model,
    register_dataset,
    setup_cfg,
    thing_classes,
)

import torch  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402

from hierarchialdet.util.coco_3class_eval import instances_to_coco_json  # noqa: E402


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-file", default=None, help="defaults to configs/diffdet.custom.swinbase.nonpretrain.yaml")
    p.add_argument("--weights", required=True, help="checkpoint to run (.pth)")
    p.add_argument("--json", required=True, help="COCO annotation file of the split to run over")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--tier", type=int, required=True, choices=(0, 1, 2),
                   help="0=quadrant, 1=quadrant-enumeration, 2=diagnosis")
    p.add_argument("--output", required=True, help="path of the predictions JSON to write")
    p.add_argument("--dataset-name", default=TEST_DATASET_NAME,
                   help="detectron2 dataset name to register the split under; keep the "
                        "config's DATASETS.TEST name unless you know why you are changing it")
    p.add_argument("--device", default=None, choices=("cuda", "cpu"),
                   help="defaults to cuda when available")
    p.add_argument("--limit", type=int, default=None, help="only run the first N images")
    p.add_argument("--score-thresh", type=float, default=0.0,
                   help="drop predictions below this score before writing "
                        "(the noisy-box consumers apply their own 0.5 threshold)")
    p.add_argument("--runtime-json", default=None,
                   help="where to write per-image runtime + failure records "
                        "(default: alongside --output, with a _runtime suffix)")
    p.add_argument("--opts", default=[], nargs=argparse.REMAINDER,
                   help="config overrides, e.g. MODEL.DiffusionDet.SAMPLE_STEP 4")
    return p


def classify_failure(records, image_info):
    """
    Per-image failure taxonomy for the roadmap's failure metrics: how often
    inference produces nothing, or produces clearly invalid geometry.
    """
    problems = []
    if not records:
        problems.append("no_detections")
    width, height = image_info["width"], image_info["height"]
    for record in records:
        x, y, w, h = record["bbox"]
        if w <= 1 or h <= 1:
            problems.append("degenerate_box")
        elif x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1:
            problems.append("box_out_of_image")
        elif w * h > 0.9 * width * height:
            problems.append("box_covers_whole_image")
    return sorted(set(problems))


def main():
    args = get_parser().parse_args()

    cfg = setup_cfg(args.config_file, args.opts, weights=args.weights, device=args.device)
    register_dataset(args.dataset_name, args.json, args.image_dir)

    model = build_eval_model(cfg)

    from train_net_patched import Trainer

    data_loader = Trainer.build_test_loader(cfg, args.dataset_name)

    with open(args.json) as f:
        image_info = {im["id"]: im for im in json.load(f)["images"]}

    # The bundled loader stores one id map per tier
    # (thing_dataset_id_to_contiguous_id_1/2/3), not the single map stock
    # detectron2 uses. Predictions come out as contiguous class indices, so map
    # them back to this tier's dataset category ids for the `category_id` field
    # that standard COCO tooling reads.
    metadata = MetadataCatalog.get(args.dataset_name)
    tier_id_map = getattr(metadata, "thing_dataset_id_to_contiguous_id_{}".format(args.tier + 1), {})
    contiguous_to_dataset = {v: k for k, v in tier_id_map.items()}

    results = []
    per_image = []
    failure_counts = {}
    device = cfg.MODEL.DEVICE

    print("running tier {} inference on {} ({} images) on {}".format(
        args.tier, args.json, len(image_info), device))

    with torch.no_grad():
        for index, inputs in enumerate(data_loader):
            if args.limit is not None and index >= args.limit:
                break
            image_id = inputs[0]["image_id"]

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(inputs, k=args.tier)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            instances = outputs[0]["instances"].to("cpu")
            records = instances_to_coco_json(instances, image_id, args.tier)
            records = [r for r in records if r.get("score", 1.0) >= args.score_thresh]
            for record in records:
                tier_class = record["category_id_{}".format(args.tier + 1)]
                record["category_id"] = contiguous_to_dataset.get(tier_class, tier_class)
            results.extend(records)

            problems = classify_failure(records, image_info[image_id])
            for problem in problems:
                failure_counts[problem] = failure_counts.get(problem, 0) + 1
            per_image.append({
                "image_id": image_id,
                "file_name": os.path.basename(inputs[0]["file_name"]),
                "seconds": round(elapsed, 4),
                "num_detections": len(records),
                "problems": problems,
            })

            if (index + 1) % 25 == 0:
                print("  {}/{} images".format(index + 1, len(image_info)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)

    times = [r["seconds"] for r in per_image]
    # Skip the first few images: the first forward pass pays CUDA kernel
    # autotuning / lazy-init costs that are not representative of steady state.
    warm = times[5:] if len(times) > 10 else times
    summary = {
        "weights": os.path.abspath(args.weights),
        "annotations": os.path.abspath(args.json),
        "tier": args.tier,
        "device": device,
        "sample_step": cfg.MODEL.DiffusionDet.SAMPLE_STEP,
        "num_proposals": cfg.MODEL.DiffusionDet.NUM_PROPOSALS,
        "images": len(per_image),
        "total_detections": len(results),
        "mean_seconds_per_image": round(sum(warm) / max(len(warm), 1), 4),
        "median_seconds_per_image": round(sorted(warm)[len(warm) // 2], 4) if warm else None,
        "images_per_minute": round(60.0 * len(warm) / max(sum(warm), 1e-9), 2) if warm else None,
        "failure_counts": failure_counts,
        "class_names": thing_classes(args.dataset_name, args.tier),
    }
    if device == "cuda":
        summary["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

    runtime_path = args.runtime_json or (os.path.splitext(args.output)[0] + "_runtime.json")
    with open(runtime_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("wrote {} ({} predictions)".format(args.output, len(results)))
    print("wrote {}".format(runtime_path))


if __name__ == "__main__":
    main()
