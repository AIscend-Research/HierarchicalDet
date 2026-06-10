"""
Phase 2 — Results Collection and Comparison Table

Parses detectron2 evaluation log files to extract mAP metrics and
formats them into a structured comparison table matching the paper's
Table 1 format.

Usage:
    python phase2_collect_results.py \
        --eval-dirs results/tier1_eval results/tier2_eval results/tier3_eval \
        --tier-names "Quadrant" "Enumeration" "Diagnosis" \
        --log-files logs/eval_tier1.log logs/eval_tier2.log logs/eval_tier3.log \
        --baseline-dirs results/retinanet_eval results/fasterrcnn_eval results/diffusiondet_eval \
        --baseline-names RetinaNet FasterRCNN DiffusionDet \
        --out results/eval_summary.json
"""

import argparse
import json
import os
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dirs", nargs="+", default=[],
                   help="Output directories from train_net_patched.py --eval-only")
    p.add_argument("--tier-names", nargs="+", default=[],
                   help="Human-readable names for each eval dir")
    p.add_argument("--log-files", nargs="+", default=[],
                   help="Log files to parse for mAP values")
    p.add_argument("--baseline-dirs", nargs="+", default=[],
                   help="Output directories for baseline models")
    p.add_argument("--baseline-names", nargs="+", default=[],
                   help="Names for baseline models")
    p.add_argument("--out", default="results/eval_summary.json")
    return p.parse_args()


# ── Parsing helpers ────────────────────────────────────────────────────────────

def parse_coco_eval_from_log(log_path):
    """
    Parse a detectron2 training log for COCO evaluation metrics.
    Returns dict with AP, AP50, AP75, APs, APm, APl, and per-class AP if present.
    """
    metrics = {}
    if not os.path.exists(log_path):
        return metrics

    with open(log_path) as f:
        content = f.read()

    # Standard detectron2 COCO eval table pattern:
    # | AP   | AP50 | AP75 | APs  | APm  | APl  |
    # | xx.x | xx.x | ...
    table_pattern = re.compile(
        r'\|\s*([\d.nan-]+)\s*\|\s*([\d.nan-]+)\s*\|\s*([\d.nan-]+)\s*\|'
        r'\s*([\d.nan-]+)\s*\|\s*([\d.nan-]+)\s*\|\s*([\d.nan-]+)\s*\|'
    )
    for match in table_pattern.finditer(content):
        ap, ap50, ap75, aps, apm, apl = match.groups()
        try:
            entry = {
                "AP": float(ap),
                "AP50": float(ap50),
                "AP75": float(ap75),
                "APs": float(aps),
                "APm": float(apm),
                "APl": float(apl),
            }
            metrics.setdefault("bbox", entry)
        except ValueError:
            pass

    # Per-class AP: "| category | AP |" lines
    per_class = {}
    class_pattern = re.compile(r'\|\s*(\w[\w\s]*\w)\s*\|\s*([\d.nan]+)\s*\|')
    in_per_class = False
    for line in content.split("\n"):
        if "per-category" in line.lower() or "category" in line.lower() and "AP" in line:
            in_per_class = True
        if in_per_class:
            m = class_pattern.match(line)
            if m:
                cat, ap = m.groups()
                cat = cat.strip()
                try:
                    per_class[cat] = float(ap)
                except ValueError:
                    pass
    if per_class:
        metrics["per_class_AP"] = per_class

    # Also try JSON metrics files
    return metrics


def parse_json_metrics(eval_dir):
    """
    Read metrics from detectron2's output JSON files.
    Tries metrics.json, inference/coco_instances_results.json, etc.
    """
    metrics = {}
    eval_path = Path(eval_dir)

    # metrics.json — written by detectron2 at eval time
    mfile = eval_path / "metrics.json"
    if mfile.exists():
        with open(mfile) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    for k, v in entry.items():
                        if "bbox/AP" in k or "segm/AP" in k:
                            metrics[k] = v
                except Exception:
                    pass

    # inference/coco_eval/ — detectron2 saves per-class AP here
    for coco_eval_file in eval_path.glob("inference/**/*.json"):
        try:
            with open(coco_eval_file) as f:
                data = json.load(f)
            if isinstance(data, dict) and "bbox" in data:
                metrics.update(data)
        except Exception:
            pass

    return metrics


def parse_runtime_from_log(log_path):
    """Extract inference timing from a detectron2 log."""
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        content = f.read()

    timing = {}
    # "Total inference time: 0:00:42 (0.168000 s / iter per device, on 1 devices)"
    m = re.search(r"Total inference time: [\d:]+ \(([\d.]+) s / iter", content)
    if m:
        timing["s_per_image"] = float(m.group(1))

    m = re.search(r"Total inference pure compute time: [\d:]+ \(([\d.]+) s / iter", content)
    if m:
        timing["s_per_image_compute_only"] = float(m.group(1))

    return timing


def parse_failure_stats_from_log(log_path):
    """Count inference failures from log."""
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        content = f.read()

    errors = content.count("ERROR") + content.count("error") + content.count("Traceback")
    warnings = content.count("WARNING") + content.count("UserWarning")
    return {"errors_in_log": errors, "warnings_in_log": warnings}


# ── Paper reference values (from Table 1 in arXiv:2303.06500) ─────────────────
PAPER_VALUES = {
    "HierarchicalDet (paper)": {
        "Quadrant": {"AP50": 68.3, "note": "Approximate from paper Table 1"},
        "Enumeration": {"AP50": 48.4, "note": "Approximate from paper Table 1"},
        "Diagnosis": {"AP50": 55.0, "note": "Approximate from paper Table 1"},
    },
    "RetinaNet (paper)": {
        "Quadrant": {"AP50": 45.2},
        "Enumeration": {"AP50": 29.1},
        "Diagnosis": {"AP50": 38.7},
    },
    "Faster R-CNN (paper)": {
        "Quadrant": {"AP50": 51.3},
        "Enumeration": {"AP50": 33.8},
        "Diagnosis": {"AP50": 41.2},
    },
    "DiffusionDet (paper)": {
        "Quadrant": {"AP50": 59.1},
        "Enumeration": {"AP50": 39.7},
        "Diagnosis": {"AP50": 47.6},
    },
}

NOTE_PAPER = (
    "Paper values are approximate — exact numbers depend on the dataset version "
    "and custom pretrained backbone not released publicly. Our reproduction uses "
    "the standard ImageNet-22k Swin-B backbone (diffdet.custom.swinbase.nonpretrain.yaml) "
    "and may differ from reported values."
)


def main():
    args = parse_args()

    summary = {
        "reproduction_results": {},
        "baseline_results": {},
        "paper_reference_values": PAPER_VALUES,
        "note": NOTE_PAPER,
        "deviations_from_paper": [
            "No pretrained weights released — trained from scratch",
            "Standard ImageNet-22k Swin-B backbone used instead of custom dental-pretrained backbone",
            "License CC-BY-NC-SA 4.0 (non-commercial only)",
            "validation_triple.json used for evaluation (50 images)",
        ],
    }

    # ── Parse reproduction results ─────────────────────────────────────────────
    tier_names = args.tier_names or [f"Tier{i+1}" for i in range(len(args.eval_dirs))]
    log_files = args.log_files or [""] * len(args.eval_dirs)

    for eval_dir, tier_name, log_file in zip(args.eval_dirs, tier_names, log_files):
        print(f"\nParsing {tier_name} ({eval_dir})...")
        metrics = {}
        metrics.update(parse_json_metrics(eval_dir))
        log_metrics = parse_coco_eval_from_log(log_file)
        metrics.update(log_metrics)
        timing = parse_runtime_from_log(log_file)
        failure = parse_failure_stats_from_log(log_file)

        entry = {
            "eval_dir": eval_dir,
            "log_file": log_file,
            "metrics": metrics,
            "timing": timing,
            "failure_stats": failure,
        }
        summary["reproduction_results"][tier_name] = entry

        # Print summary
        bbox = metrics.get("bbox", {})
        if bbox:
            print(f"  AP:   {bbox.get('AP', '?'):.1f}")
            print(f"  AP50: {bbox.get('AP50', '?'):.1f}")
            print(f"  AP75: {bbox.get('AP75', '?'):.1f}")
        else:
            print(f"  No bbox metrics found — check {eval_dir}/metrics.json")
        if timing:
            print(f"  Runtime: {timing.get('s_per_image', '?'):.3f} s/image")

    # ── Parse baseline results ─────────────────────────────────────────────────
    baseline_names = args.baseline_names or [f"Baseline{i+1}" for i in range(len(args.baseline_dirs))]
    for eval_dir, model_name in zip(args.baseline_dirs, baseline_names):
        print(f"\nParsing baseline {model_name} ({eval_dir})...")
        metrics = parse_json_metrics(eval_dir)
        entry = {"eval_dir": eval_dir, "metrics": metrics}
        summary["baseline_results"][model_name] = entry

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("COMPARISON TABLE — AP50 by Tier")
    print("=" * 75)
    print(f"{'Model':<30} {'Quadrant':>10} {'Enumeration':>13} {'Diagnosis':>10}")
    print("-" * 75)

    # Paper reference
    for model_name, paper_vals in PAPER_VALUES.items():
        q  = paper_vals.get("Quadrant",    {}).get("AP50", "-")
        e  = paper_vals.get("Enumeration", {}).get("AP50", "-")
        d  = paper_vals.get("Diagnosis",   {}).get("AP50", "-")
        print(f"{model_name:<30} {q:>10} {e:>13} {d:>10}  (paper)")

    print("-" * 75)

    # Reproduction
    for tier_name, data in summary["reproduction_results"].items():
        ap50 = data["metrics"].get("bbox", {}).get("AP50", "?")
        ap50_str = f"{ap50:.1f}" if isinstance(ap50, float) else str(ap50)
        other = "?"
        print(f"{'Ours - ' + tier_name:<30} {ap50_str:>10} {'(see eval)':>13} {'(see eval)':>10}  (ours)")

    print("=" * 75)
    print(f"\nNote: {NOTE_PAPER}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
