"""
Phase 1 — Dataloader Verification Script
Smoke-tests that the official data loading pipeline can read the DENTEX dataset
without errors, using the patched dataset mapper.

Run AFTER:
  - environment is set up (conda activate hierarchicaldet)
  - dataset is downloaded and extracted
  - DATA_ROOT env var is set

Usage:
    export DATA_ROOT=/path/to/sorted/challenge
    export DENTEX_TIER=tier3
    # Optionally set noisy box paths if testing diagnosis tier:
    # export NOISY_BOX_TRAIN=/path/to/enumeration_inference_train.json
    # export NOISY_BOX_VAL=/path/to/enumeration_inference_val.json
    python phase1_verify_dataloader.py --n-samples 10

Outputs a per-image pass/fail table and summary statistics.
"""

import os
import sys
import json
import argparse
import traceback
import time
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of samples to load and verify")
    p.add_argument("--config-file", default="configs/diffdet.custom.swinbase.nonpretrain.yaml")
    p.add_argument("--out", default="phase1_dataloader_report.txt")
    return p.parse_args()


def register_datasets():
    """Register DENTEX datasets using the patched train_net logic."""
    from detectron2.data.datasets import register_coco_instances

    DATA_ROOT = os.environ.get(
        "DATA_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "sorted", "challenge")
    )
    DATA_ROOT = os.path.abspath(DATA_ROOT)
    DENTEX_TIER = os.environ.get("DENTEX_TIER", "tier3")

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
            "train_json": "train_merged_disease_coco3class_onlyd_fixed.json",
            "train_images": "for_coco_disease_train",
            "val_json": "test_merged_disease_coco3class.json",
            "val_images": "for_coco_disease_test",
        },
    }

    tier = TIER_CONFIGS[DENTEX_TIER]
    train_json  = os.path.join(DATA_ROOT, tier["train_json"])
    train_imgs  = os.path.join(DATA_ROOT, tier["train_images"])
    val_json    = os.path.join(DATA_ROOT, tier["val_json"])
    val_imgs    = os.path.join(DATA_ROOT, tier["val_images"])

    print(f"Registering tier: {DENTEX_TIER}")
    for label, path in [("train_json", train_json), ("train_imgs", train_imgs),
                        ("val_json", val_json), ("val_imgs", val_imgs)]:
        exists = os.path.exists(path)
        print(f"  {'✓' if exists else '✗'} {label}: {path}")
        if not exists:
            print(f"    WARNING: path missing — dataloader will fail")

    register_coco_instances("custom_train_class", {}, train_json, train_imgs)
    register_coco_instances("custom_validation_class", {}, val_json, val_imgs)
    return DENTEX_TIER


def main():
    args = parse_args()

    # ── Import detectron2 from bundled copy ───────────────────────────────────
    # Ensure we're using the repo's bundled detectron2, not a pip-installed one
    repo_root = Path(__file__).parent
    sys.path.insert(0, str(repo_root))

    try:
        from detectron2.config import get_cfg
        from detectron2.data import build_detection_train_loader, build_detection_test_loader
        from hierarchialdet import add_diffusiondet_config
        from hierarchialdet.util.model_ema import add_model_ema_configs
        from dataset_mapper_patched import DiffusionDetDatasetMapper
        print("✓ Imports OK")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Make sure you are running from inside HierarchicalDet_official/")
        print("  and the conda environment is activated.")
        sys.exit(1)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        print(f"WARNING: config file not found: {args.config_file}")
        print("  Using default config — this may not match your dataset.")
    cfg.freeze()

    # ── Dataset registration ──────────────────────────────────────────────────
    tier = register_datasets()

    # ── Build mapper and loader ───────────────────────────────────────────────
    print("\nBuilding dataset mapper...")
    try:
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        print("✓ DiffusionDetDatasetMapper initialized (train mode)")
    except Exception as e:
        print(f"✗ Mapper init failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Building training dataloader...")
    try:
        loader = build_detection_train_loader(cfg, mapper=mapper)
        print("✓ Training dataloader built")
    except Exception as e:
        print(f"✗ Dataloader build failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Sample verification ───────────────────────────────────────────────────
    print(f"\nLoading {args.n_samples} samples...")
    results = []
    loader_iter = iter(loader)

    for i in range(args.n_samples):
        t0 = time.time()
        try:
            batch = next(loader_iter)
            # batch is a list of dicts
            for sample in batch:
                img = sample.get("image")
                instances = sample.get("instances")

                checks = {
                    "has_image": img is not None,
                    "image_is_tensor": isinstance(img, torch.Tensor),
                    "image_3d": img is not None and img.ndim == 3,
                    "image_shape_ok": img is not None and img.shape[0] == 3,
                    "has_instances": instances is not None,
                    "instances_has_boxes": instances is not None and instances.has("gt_boxes"),
                    "instances_has_classes": instances is not None and instances.has("gt_classes"),
                    "no_nan_in_image": img is not None and not torch.isnan(img.float()).any().item(),
                }

                n_boxes = len(instances) if instances is not None else 0
                has_bbox_pre = instances is not None and instances.has("gt_boxes_pre")

                record = {
                    "sample_idx": i,
                    "file_name": sample.get("file_name", "?"),
                    "image_id": sample.get("image_id", "?"),
                    "image_shape": list(img.shape) if img is not None else None,
                    "n_gt_boxes": n_boxes,
                    "has_noisy_boxes": has_bbox_pre,
                    "load_time_s": round(time.time() - t0, 3),
                    "checks": checks,
                    "all_checks_pass": all(checks.values()),
                    "error": None,
                }
                results.append(record)

                status = "✓" if record["all_checks_pass"] else "✗"
                print(f"  [{i+1}/{args.n_samples}] {status} "
                      f"img_id={record['image_id']} "
                      f"shape={record['image_shape']} "
                      f"n_boxes={n_boxes} "
                      f"noisy_boxes={has_bbox_pre} "
                      f"({record['load_time_s']}s)")

        except StopIteration:
            print(f"  Dataset exhausted after {i} samples")
            break
        except Exception as e:
            results.append({
                "sample_idx": i, "error": str(e),
                "all_checks_pass": False,
                "load_time_s": round(time.time() - t0, 3),
            })
            print(f"  [{i+1}/{args.n_samples}] ✗ ERROR: {e}")

    # ── Validation loader ─────────────────────────────────────────────────────
    print("\nBuilding validation dataloader (no noisy boxes)...")
    val_results = []
    try:
        val_mapper = DiffusionDetDatasetMapper(cfg, is_train=False)
        val_loader = build_detection_test_loader(cfg, "custom_validation_class", mapper=val_mapper)
        val_iter = iter(val_loader)
        for i in range(min(3, args.n_samples)):
            t0 = time.time()
            try:
                batch = next(val_iter)
                for sample in batch:
                    img = sample.get("image")
                    val_results.append({
                        "file_name": sample.get("file_name", "?"),
                        "image_id": sample.get("image_id", "?"),
                        "image_shape": list(img.shape) if img is not None else None,
                        "load_time_s": round(time.time() - t0, 3),
                        "error": None,
                    })
                    print(f"  ✓ val img_id={sample.get('image_id')} shape={list(img.shape) if img is not None else None}")
            except Exception as e:
                val_results.append({"error": str(e)})
                print(f"  ✗ ERROR: {e}")
    except Exception as e:
        print(f"  ✗ Val loader build failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r.get("all_checks_pass"))
    n_fail = len(results) - n_pass
    avg_time = (sum(r.get("load_time_s", 0) for r in results) / len(results)) if results else 0

    summary = [
        "",
        "=" * 60,
        "DATALOADER VERIFICATION SUMMARY",
        "=" * 60,
        f"Tier: {tier}",
        f"Samples tested: {len(results)}",
        f"Pass: {n_pass}  Fail: {n_fail}",
        f"Avg load time: {avg_time:.3f}s per sample",
        f"Noisy boxes active: {os.environ.get('NOISY_BOX_TRAIN') is not None}",
        "",
        "INDIVIDUAL RESULTS",
    ]
    for r in results:
        if r.get("error"):
            summary.append(f"  ✗ sample {r['sample_idx']}: {r['error']}")
        else:
            failed = [k for k, v in r.get("checks", {}).items() if not v]
            line = f"  {'✓' if r['all_checks_pass'] else '✗'} img_id={r['image_id']} n_boxes={r['n_gt_boxes']}"
            if failed:
                line += f" FAILED: {failed}"
            summary.append(line)

    print("\n".join(summary))
    with open(args.out, "w") as f:
        f.write("\n".join(summary))
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
