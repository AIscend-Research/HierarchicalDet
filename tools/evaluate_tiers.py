"""
Evaluate a HierarchicalDet checkpoint per detection tier and write the numbers
the reproduction report needs, as JSON and as a Markdown table.

Runs the authors' own evaluation path -- `hierarchialdet.util.coco_3class_eval.COCOEvaluator`
driven by `evaluator.inference_on_dataset` -- which, when asked for tier k,
scores every tier from 0 up to k in one pass over the data. So a single run on
the tier-2 checkpoint yields quadrant, enumeration and diagnosis metrics
computed exactly the way the paper's code computes them.

Reported per tier: AP (0.5:0.95), AP50, AP75, size-split APs, and per-class AP
for every class in that tier (the quadrants, the eight tooth positions, the
four diagnoses). Runtime per image is measured separately by
tools/dump_predictions.py, which is also where the failure metrics come from.
"""
import argparse
import json
import os

from repro_common import (  # noqa: E402
    TEST_DATASET_NAME,
    TIERS,
    build_eval_model,
    file_md5,
    register_dataset,
    setup_cfg,
    thing_classes,
)

from evaluator import inference_on_dataset  # noqa: E402
from hierarchialdet.util.coco_3class_eval import COCOEvaluator  # noqa: E402


def to_markdown(results_by_tier, class_names_by_tier, title):
    lines = ["## {}".format(title), ""]
    lines.append("| Tier | AP@0.5:0.95 | AP@0.5 | AP@0.75 | APs | APm | APl |")
    lines.append("|---|---|---|---|---|---|---|")
    for tier in sorted(results_by_tier):
        box = results_by_tier[tier].get("bbox", {})

        def get(key):
            value = box.get(key)
            return "{:.2f}".format(value) if isinstance(value, (int, float)) else "n/a"

        lines.append("| {} ({}) | {} | {} | {} | {} | {} | {} |".format(
            tier, TIERS[tier][1], get("AP"), get("AP50"), get("AP75"),
            get("APs"), get("APm"), get("APl")))

    for tier in sorted(results_by_tier):
        box = results_by_tier[tier].get("bbox", {})
        per_class = {k[len("AP-"):]: v for k, v in box.items() if k.startswith("AP-")}
        if not per_class:
            continue
        lines += ["", "### Per-class AP -- tier {} ({})".format(tier, TIERS[tier][1]), "",
                  "| Class | AP |", "|---|---|"]
        for name in class_names_by_tier.get(tier, sorted(per_class)):
            value = per_class.get(str(name))
            lines.append("| {} | {} |".format(
                name, "{:.2f}".format(value) if isinstance(value, (int, float)) else "n/a"))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-file", default=None)
    p.add_argument("--weights", required=True)
    p.add_argument("--json", required=True, help="COCO annotations of the split to evaluate on")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2),
                   help="highest tier to evaluate; every lower tier is evaluated too")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default=TEST_DATASET_NAME)
    p.add_argument("--device", default=None, choices=("cuda", "cpu"))
    p.add_argument("--label", default=None, help="name for this run in the report (default: checkpoint name)")
    p.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = setup_cfg(args.config_file, args.opts, weights=args.weights, device=args.device, freeze=False)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()

    register_dataset(args.dataset_name, args.json, args.image_dir)
    model = build_eval_model(cfg)

    from train_net_patched import Trainer

    data_loader = Trainer.build_test_loader(cfg, args.dataset_name)
    evaluator = COCOEvaluator(args.dataset_name, cfg, True, os.path.join(args.output_dir, "inference"))

    # Returns one result dict per tier, 0..args.tier.
    results = inference_on_dataset(model, data_loader, args.tier, evaluator)

    results_by_tier = {tier: results[tier] for tier in range(args.tier + 1)}
    class_names_by_tier = {tier: thing_classes(args.dataset_name, tier) for tier in results_by_tier}

    label = args.label or os.path.basename(os.path.dirname(os.path.abspath(args.weights)))
    payload = {
        "label": label,
        "weights": os.path.abspath(args.weights),
        "weights_md5": file_md5(args.weights),
        "annotations": os.path.abspath(args.json),
        "config_file": os.path.abspath(args.config_file) if args.config_file else "configs/diffdet.custom.swinbase.nonpretrain.yaml",
        "config_overrides": list(args.opts),
        "device": cfg.MODEL.DEVICE,
        "sample_step": cfg.MODEL.DiffusionDet.SAMPLE_STEP,
        "num_proposals": cfg.MODEL.DiffusionDet.NUM_PROPOSALS,
        "seed": cfg.SEED,
        "results_by_tier": {str(k): v for k, v in results_by_tier.items()},
        "class_names_by_tier": {str(k): v for k, v in class_names_by_tier.items()},
    }
    json_path = os.path.join(args.output_dir, "tier_metrics.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(args.output_dir, "tier_metrics.md")
    with open(md_path, "w") as f:
        f.write(to_markdown(results_by_tier, class_names_by_tier,
                            "HierarchicalDet per-tier detection metrics -- {}".format(label)))

    print(open(md_path).read())
    print("wrote {}".format(json_path))
    print("wrote {}".format(md_path))


if __name__ == "__main__":
    main()
