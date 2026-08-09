"""
Run the Phase 4 experiment sweeps and collect them into one results table.

Each experiment is a set of (dump predictions -> score them) runs that differ
only in one controlled variable, so every row is directly comparable to the
clean baseline row:

  steps         MODEL.DiffusionDet.SAMPLE_STEP from 1 upwards -- accuracy vs.
                compute for the diffusion denoising process at inference time.
                Note SAMPLE_STEP > 1 only runs at all because of the ddim_sample
                fix in hierarchialdet/detector.py (the released code crashes
                there; see the comment at the fix).
  degradation   the conditions built by tools/degrade_images.py -- blur, JPEG
                compression, reduced resolution.
  subsets       clean vs. stress-test subsets from phase1_audit/clean_stress_subsets.json.
  device        cuda vs. cpu -- runtime and, as a check, identical detections.
  prior-noise   corrupt the tier k-1 predictions fed into tier k
                (NOISY_BOX_INFER_JITTER / NOISY_BOX_INFER_DROP) to test whether
                the hierarchical inference order tolerates imperfect early tiers.
                Requires --prior-predictions.

Scoring uses tools/coco_eval_standalone.py's stock COCO evaluator for all rows,
so no row depends on the repo's evaluation fork -- the point of these sweeps is
relative movement between rows, which a single consistent evaluator gives.
"""
import argparse
import json
import os
import subprocess
import sys

from repro_common import DEFAULT_CONFIG, REPO_ROOT, TIERS  # noqa: E402

PYTHON = sys.executable
DUMP = os.path.join(REPO_ROOT, "tools", "dump_predictions.py")


def dump_predictions(args, run_dir, split_json, image_dir, opts=None, env_extra=None, device=None):
    predictions = os.path.join(run_dir, "predictions.json")
    if os.path.exists(predictions):
        print("reusing existing {}".format(predictions))
        return predictions
    cmd = [
        PYTHON, DUMP,
        "--config-file", args.config_file,
        "--weights", args.weights,
        "--json", split_json,
        "--image-dir", image_dir,
        "--tier", str(args.tier),
        "--output", predictions,
        "--device", device or args.device,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if opts:
        cmd += ["--opts"] + list(opts)
    env = dict(os.environ, TIER=str(args.tier))
    env.pop("NOISY_BOX_INFER", None)
    env.pop("NOISY_BOX_INFER_JITTER", None)
    env.pop("NOISY_BOX_INFER_DROP", None)
    if env_extra:
        env.update(env_extra)
    print("\n$ {}".format(" ".join(cmd)), flush=True)
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit("prediction dump failed for {}".format(run_dir))
    return predictions


def score(gt_json, predictions, tier):
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from coco_eval_standalone import evaluate, flatten_gt, flatten_predictions

    with open(gt_json) as f:
        gt = json.load(f)
    with open(predictions) as f:
        preds = json.load(f)
    metrics = evaluate(flatten_gt(gt, tier), flatten_predictions(preds, tier), quiet=True)
    metrics.pop("summary_text", None)
    return metrics


def runtime_of(predictions):
    runtime_path = os.path.splitext(predictions)[0] + "_runtime.json"
    if not os.path.exists(runtime_path):
        return {}
    with open(runtime_path) as f:
        return json.load(f)["summary"]


def subset_json(source_json, image_names, out_path):
    """Write a COCO file restricted to the named images (clean/stress subsets)."""
    with open(source_json) as f:
        coco = json.load(f)
    wanted = set(image_names)
    images = [im for im in coco["images"] if im["file_name"] in wanted]
    keep_ids = {im["id"] for im in images}
    coco = dict(coco)
    coco["images"] = images
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)
    return out_path, len(images)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--json", required=True, help="clean evaluation split (COCO, 3-tier)")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--config-file", default=DEFAULT_CONFIG)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--limit", type=int, default=None, help="cap images per run (for smoke tests)")
    p.add_argument("--experiments", nargs="*",
                   default=["steps", "degradation", "subsets", "device", "prior-noise"])
    p.add_argument("--steps", nargs="*", type=int, default=[1, 2, 4, 8],
                   help="SAMPLE_STEP values for the diffusion-step sweep")
    p.add_argument("--degraded-root", default=None,
                   help="output-root of tools/degrade_images.py (required for 'degradation')")
    p.add_argument("--subsets-json", default=os.path.join(REPO_ROOT, "phase1_audit", "clean_stress_subsets.json"))
    p.add_argument("--cpu-limit", type=int, default=10,
                   help="images to run in the CPU-vs-GPU comparison (CPU inference is slow)")
    p.add_argument("--prior-predictions", default=None,
                   help="tier k-1 predictions JSON to inject at inference for the "
                        "'prior-noise' experiment (from tools/dump_predictions.py)")
    p.add_argument("--prior-noise-levels", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.2],
                   help="NOISY_BOX_INFER_JITTER values (normalized-coordinate std)")
    p.add_argument("--prior-drop-levels", nargs="*", type=float, default=[0.0, 0.25, 0.5],
                   help="NOISY_BOX_INFER_DROP values (fraction of prior boxes dropped)")
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    rows = []

    def add_row(experiment, condition, predictions, gt_json=None, notes=None):
        metrics = score(gt_json or args.json, predictions, args.tier)
        summary = runtime_of(predictions)
        rows.append({
            "experiment": experiment,
            "condition": condition,
            "AP": round(metrics["AP"], 2),
            "AP50": round(metrics["AP50"], 2),
            "AP75": round(metrics["AP75"], 2),
            "num_predictions": metrics["num_predictions"],
            "seconds_per_image": summary.get("mean_seconds_per_image"),
            "images_per_minute": summary.get("images_per_minute"),
            "device": summary.get("device"),
            "sample_step": summary.get("sample_step"),
            "images": summary.get("images"),
            "failures": summary.get("failure_counts"),
            "per_class_AP": {k: round(v, 2) for k, v in metrics["per_class_AP"].items()},
            "notes": notes,
        })
        print("  -> AP {:.2f} / AP50 {:.2f} ({})".format(metrics["AP"], metrics["AP50"], condition))

    if "steps" in args.experiments:
        print("\n########## diffusion step sensitivity ##########")
        for step in args.steps:
            run_dir = os.path.join(args.output_root, "steps", "step_{}".format(step))
            predictions = dump_predictions(
                args, run_dir, args.json, args.image_dir,
                opts=["MODEL.DiffusionDet.SAMPLE_STEP", str(step)])
            add_row("steps", "SAMPLE_STEP={}".format(step), predictions)

    if "degradation" in args.experiments:
        print("\n########## image degradation ##########")
        if not args.degraded_root:
            print("skipped: --degraded-root not given (run tools/degrade_images.py first)")
        else:
            with open(os.path.join(args.degraded_root, "degradation_index.json")) as f:
                index = json.load(f)
            baseline = dump_predictions(args, os.path.join(args.output_root, "degradation", "clean"),
                                        args.json, args.image_dir)
            add_row("degradation", "clean", baseline)
            for condition, spec in index["conditions"].items():
                run_dir = os.path.join(args.output_root, "degradation", condition)
                predictions = dump_predictions(args, run_dir, spec["json"], spec["image_dir"])
                add_row("degradation", condition, predictions, gt_json=spec["json"])

    if "subsets" in args.experiments:
        print("\n########## clean vs stress-test subsets ##########")
        if not os.path.exists(args.subsets_json):
            print("skipped: {} not found".format(args.subsets_json))
        else:
            with open(args.subsets_json) as f:
                subsets = json.load(f)
            full = dump_predictions(args, os.path.join(args.output_root, "subsets", "all"),
                                    args.json, args.image_dir)
            for subset_name, entry in subsets.items():
                names = entry if isinstance(entry, list) else entry.get("images", [])
                names = [os.path.basename(n) for n in names]
                if not names:
                    continue
                subset_dir = os.path.join(args.output_root, "subsets", subset_name)
                os.makedirs(subset_dir, exist_ok=True)
                gt_path, count = subset_json(args.json, names, os.path.join(subset_dir, "gt.json"))
                # Reuse the single full-split dump: restricting the GT is enough,
                # and re-running inference per subset would only add noise.
                add_row("subsets", subset_name, full, gt_json=gt_path,
                        notes="{} images".format(count))

    if "device" in args.experiments:
        print("\n########## CPU vs GPU ##########")
        for device in ("cuda", "cpu"):
            run_args = argparse.Namespace(**vars(args))
            run_args.limit = args.cpu_limit
            run_dir = os.path.join(args.output_root, "device", device)
            try:
                predictions = dump_predictions(run_args, run_dir, args.json, args.image_dir, device=device)
            except SystemExit as exc:
                print("{} run failed: {}".format(device, exc))
                continue
            add_row("device", device, predictions,
                    notes="first {} images only".format(args.cpu_limit))

    if "prior-noise" in args.experiments:
        print("\n########## imperfect prior-tier predictions ##########")
        if not args.prior_predictions:
            print("skipped: --prior-predictions not given "
                  "(dump tier {} predictions over this split first)".format(max(args.tier - 1, 0)))
        elif args.tier == 0:
            print("skipped: tier 0 has no prior tier")
        else:
            for jitter in args.prior_noise_levels:
                for drop in args.prior_drop_levels:
                    condition = "jitter={} drop={}".format(jitter, drop)
                    run_dir = os.path.join(args.output_root, "prior_noise",
                                           "j{}_d{}".format(jitter, drop))
                    predictions = dump_predictions(
                        args, run_dir, args.json, args.image_dir,
                        env_extra={
                            "NOISY_BOX_INFER": os.path.abspath(args.prior_predictions),
                            "NOISY_BOX_INFER_JITTER": str(jitter),
                            "NOISY_BOX_INFER_DROP": str(drop),
                        })
                    add_row("prior-noise", condition, predictions)

    results_path = os.path.join(args.output_root, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump({"weights": os.path.abspath(args.weights), "tier": args.tier,
                   "tier_name": TIERS[args.tier][1], "rows": rows}, f, indent=2)

    md_path = os.path.join(args.output_root, "experiment_results.md")
    with open(md_path, "w") as f:
        f.write("# Phase 4 experiments -- tier {} ({})\n\n".format(args.tier, TIERS[args.tier][1]))
        f.write("| Experiment | Condition | AP | AP50 | AP75 | s/image | img/min | preds | notes |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                row["experiment"], row["condition"], row["AP"], row["AP50"], row["AP75"],
                row["seconds_per_image"], row["images_per_minute"], row["num_predictions"],
                row["notes"] or ""))

    print("\nwrote {}".format(results_path))
    print("wrote {}".format(md_path))


if __name__ == "__main__":
    main()
