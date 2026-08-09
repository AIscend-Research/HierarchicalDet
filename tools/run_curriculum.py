"""
Drive the full 3-stage HierarchicalDet curriculum end to end.

The paper trains one model in hierarchical order, transferring weights forward
and seeding each stage's diffusion process with the previous stage's
predictions ("noisy box manipulation"):

    stage 0  quadrant                       (tier-1 head supervised)
      |  run the stage-0 model over stage 1's train + val images
      v  -> noisy boxes
    stage 1  quadrant-enumeration           (tier-1 + tier-2 heads supervised)
      |  run the stage-1 model over stage 2's train + val images
      v  -> noisy boxes
    stage 2  quadrant-enumeration-diagnosis (all three heads supervised)

Which heads get supervision is decided by the data, not by a flag: the
normalized tier JSONs carry `category_id_2/3 = null` for tiers that do not
apply, and the loss skips them (verified in docs/phase2_dataloader_fix.md).
Weight transfer is just MODEL.WEIGHTS pointing at the previous stage's
model_final.pth. The noisy boxes are passed through NOISY_BOX_TRAIN /
NOISY_BOX_VAL, which the patched dataset mapper reads.

Every stage is skipped if its model_final.pth already exists, so the run can be
resumed across Kaggle sessions (~10h per stage on a T4 at the config's 40k
iterations -- see the Phase 0 benchmark). Use --max-iter to run a shorter
budget, and --dry-run to print the exact commands without executing them.

--mode single-tier trains stage 2 alone with no prior-tier boxes and no weight
transfer: that is the non-hierarchical DiffusionDet baseline the paper's
central claim is measured against, so the curriculum-vs-single-tier gap can be
attributed to the hierarchy rather than to the architecture.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from repro_common import REPO_ROOT, DEFAULT_CONFIG, file_md5  # noqa: E402

PYTHON = sys.executable


class Stage:
    def __init__(self, tier, train_json, train_images, name):
        self.tier = tier
        self.train_json = train_json
        self.train_images = train_images
        self.name = name


def run(cmd, env=None, dry_run=False, log_path=None):
    printable = " ".join(cmd)
    print("\n$ {}".format(printable), flush=True)
    if dry_run:
        return {"command": printable, "skipped": "dry-run"}
    started = time.time()
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as log:
            process = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            returncode = process.wait()
    else:
        returncode = subprocess.run(cmd, env=env, cwd=REPO_ROOT).returncode
    elapsed = time.time() - started
    if returncode != 0:
        raise SystemExit("command failed with exit code {}:\n  {}".format(returncode, printable))
    return {"command": printable, "seconds": round(elapsed, 1), "log": log_path}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="directory holding the extracted DENTEX training tiers "
                        "(the parent of quadrant/, quadrant_enumeration/, quadrant-enumeration-disease/)")
    p.add_argument("--normalized-dir", required=True,
                   help="output dir of tools/normalize_dentex_tiers.py")
    p.add_argument("--val-json", required=True, help="validation_triple.json")
    p.add_argument("--val-images", required=True, help="validation image dir")
    p.add_argument("--output-root", required=True, help="where stage output dirs are created")
    p.add_argument("--init-weights", required=True,
                   help="backbone init for stage 0, e.g. models/swin_base_patch4_window7_224_22k.pth")
    p.add_argument("--config-file", default=DEFAULT_CONFIG)
    p.add_argument("--mode", default="hierarchical", choices=("hierarchical", "single-tier"))
    p.add_argument("--max-iter", type=int, default=None, help="override SOLVER.MAX_ITER per stage")
    p.add_argument("--ims-per-batch", type=int, default=1,
                   help="1 by default: batch 2 (the config default) OOMs on a 16GB T4, "
                        "measured on real hardware in the Phase 0 benchmark")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--eval-period", type=int, default=2000)
    p.add_argument("--seed", type=int, default=40244023, help="the config's own SEED")
    p.add_argument("--start-stage", type=int, default=0, choices=(0, 1, 2))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stages = [
        Stage(0, os.path.join(args.normalized_dir, "train_quadrant_normalized.json"),
              os.path.join(args.data_root, "quadrant", "xrays"), "stage0_quadrant"),
        Stage(1, os.path.join(args.normalized_dir, "train_quadrant_enumeration_normalized.json"),
              os.path.join(args.data_root, "quadrant_enumeration", "xrays"), "stage1_enumeration"),
        Stage(2, os.path.join(args.data_root, "quadrant-enumeration-disease",
                              "train_quadrant_enumeration_disease.json"),
              os.path.join(args.data_root, "quadrant-enumeration-disease", "xrays"), "stage2_diagnosis"),
    ]
    if args.mode == "single-tier":
        stages = stages[2:]

    for stage in stages:
        for path in (stage.train_json, stage.train_images):
            if not os.path.exists(path):
                raise SystemExit("missing input for {}: {}".format(stage.name, path))

    os.makedirs(args.output_root, exist_ok=True)
    manifest = {
        "mode": args.mode,
        "config_file": os.path.abspath(args.config_file),
        "seed": args.seed,
        "ims_per_batch": args.ims_per_batch,
        "max_iter": args.max_iter,
        "stages": [],
    }
    manifest_path = os.path.join(args.output_root, "curriculum_manifest.json")

    previous_weights = os.path.abspath(args.init_weights)
    previous_tier = None

    for stage in stages:
        stage_dir = os.path.join(args.output_root, stage.name)
        final_weights = os.path.join(stage_dir, "model_final.pth")
        record = {"stage": stage.name, "tier": stage.tier, "output_dir": stage_dir,
                  "train_json": stage.train_json, "init_weights": previous_weights, "steps": []}

        if stage.tier < args.start_stage:
            print("\n=== skipping {} (--start-stage {}) ===".format(stage.name, args.start_stage))
            if os.path.exists(final_weights):
                previous_weights, previous_tier = final_weights, stage.tier
            continue

        # --- noisy boxes from the previous stage, over THIS stage's images ---
        noisy_train = noisy_val = None
        if previous_tier is not None and args.mode == "hierarchical":
            noisy_dir = os.path.join(stage_dir, "noisy_boxes")
            noisy_train = os.path.join(noisy_dir, "train_boxes.json")
            noisy_val = os.path.join(noisy_dir, "val_boxes.json")
            for out_path, split_json, split_images in (
                (noisy_train, stage.train_json, stage.train_images),
                (noisy_val, args.val_json, args.val_images),
            ):
                if os.path.exists(out_path):
                    print("noisy boxes already present: {}".format(out_path))
                    continue
                record["steps"].append(run([
                    PYTHON, os.path.join("tools", "dump_predictions.py"),
                    "--config-file", args.config_file,
                    "--weights", previous_weights,
                    "--json", split_json,
                    "--image-dir", split_images,
                    "--tier", str(previous_tier),
                    "--output", out_path,
                ], env=dict(os.environ, TIER=str(previous_tier)), dry_run=args.dry_run,
                    log_path=os.path.join(noisy_dir, os.path.basename(out_path) + ".log")))
            record["noisy_box_train"] = noisy_train
            record["noisy_box_val"] = noisy_val

        # --- train this stage ---
        if os.path.exists(final_weights):
            print("\n=== {} already trained ({}), skipping ===".format(stage.name, final_weights))
        else:
            env = dict(
                os.environ,
                TIER=str(stage.tier),
                TRAIN_JSON=stage.train_json,
                TRAIN_IMG_DIR=stage.train_images,
                VAL_JSON=args.val_json,
                VAL_IMG_DIR=args.val_images,
            )
            if noisy_train and noisy_val:
                env["NOISY_BOX_TRAIN"] = noisy_train
                env["NOISY_BOX_VAL"] = noisy_val
            else:
                env.pop("NOISY_BOX_TRAIN", None)
                env.pop("NOISY_BOX_VAL", None)

            cmd = [
                PYTHON, "train_net_patched.py",
                "--config-file", args.config_file,
                "--num-gpus", str(args.num_gpus),
                "--resume",
                "MODEL.WEIGHTS", previous_weights,
                "OUTPUT_DIR", stage_dir,
                "SOLVER.IMS_PER_BATCH", str(args.ims_per_batch),
                "TEST.EVAL_PERIOD", str(args.eval_period),
                "SEED", str(args.seed),
            ]
            if args.max_iter is not None:
                cmd += ["SOLVER.MAX_ITER", str(args.max_iter)]
            record["steps"].append(run(cmd, env=env, dry_run=args.dry_run,
                                       log_path=os.path.join(stage_dir, "train.log")))

        if os.path.exists(final_weights):
            record["final_weights"] = final_weights
            record["final_weights_md5"] = file_md5(final_weights)
            previous_weights, previous_tier = final_weights, stage.tier
        elif not args.dry_run:
            raise SystemExit("{} finished but {} was not written".format(stage.name, final_weights))

        manifest["stages"].append(record)
        if not args.dry_run:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    print("\ncurriculum complete.")
    if not args.dry_run:
        print("manifest: {}".format(manifest_path))
        print("final checkpoint: {}".format(previous_weights))


if __name__ == "__main__":
    main()
