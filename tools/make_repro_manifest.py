"""
Collect the reproducibility checklist the report has to ship with: environment,
package versions, git state, dataset fingerprints, checkpoint hashes, configs,
seeds, and the commands that were actually run.

Run this at the end of a session, pointing it at whatever exists. Anything
missing is recorded as missing rather than silently omitted -- a checklist that
quietly drops the parts that were not done is worse than one that says so.
"""
import argparse
import json
import os
import platform
import subprocess
import sys

from repro_common import REPO_ROOT, file_md5  # noqa: E402


def shell(cmd):
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - a manifest must never fail the run
        return "unavailable: {}".format(exc)


def describe_file(path):
    if not path or not os.path.exists(path):
        return {"path": path, "present": False}
    return {
        "path": os.path.abspath(path),
        "present": True,
        "size_bytes": os.path.getsize(path),
        "md5": file_md5(path),
    }


def describe_dataset(path):
    info = describe_file(path)
    if not info["present"]:
        return info
    try:
        with open(path) as f:
            coco = json.load(f)
    except (ValueError, OSError) as exc:
        info["error"] = str(exc)
        return info
    info["images"] = len(coco.get("images", []))
    info["annotations"] = len(coco.get("annotations", []))
    for tier in range(3):
        key = "categories_{}".format(tier + 1)
        if key in coco:
            info[key] = [str(c["name"]) for c in coco[key]]
            info["annotations_labelled_tier{}".format(tier + 1)] = sum(
                1 for a in coco["annotations"] if a.get("category_id_{}".format(tier + 1)) is not None)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="manifest JSON to write (a .md is written alongside)")
    p.add_argument("--checkpoints", nargs="*", default=[], help="checkpoints to hash")
    p.add_argument("--datasets", nargs="*", default=[], help="COCO annotation files to fingerprint")
    p.add_argument("--configs", nargs="*", default=[], help="config YAMLs used")
    p.add_argument("--curriculum-manifest", default=None,
                   help="curriculum_manifest.json from tools/run_curriculum.py (records the commands)")
    p.add_argument("--extra-notes", default=None, help="free text to include verbatim")
    args = p.parse_args()

    import torch

    manifest = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        "git": {
            "commit": shell(["git", "rev-parse", "HEAD"]),
            "branch": shell(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_porcelain": shell(["git", "status", "--porcelain"]),
            "remote": shell(["git", "remote", "-v"]),
        },
        "packages": shell([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "configs": {os.path.basename(c): describe_file(c) for c in args.configs},
        "checkpoints": {os.path.basename(c): describe_file(c) for c in args.checkpoints},
        "datasets": {os.path.basename(d): describe_dataset(d) for d in args.datasets},
        "deviations_from_official_readme": [
            "train_net_patched.py replaces train_net.py: dataset paths come from "
            "TRAIN_JSON/TRAIN_IMG_DIR/VAL_JSON/VAL_IMG_DIR, the patched dataset mapper is used, "
            "and the evaluation tier comes from TIER instead of being hardcoded.",
            "hierarchialdet/dataset_mapper_patched.py replaces the released mapper, whose "
            "hardcoded noisy-box paths are not part of the public release.",
            "hierarchialdet/detector.py: fixed ddim_sample's box replenishment, which as "
            "released raises AttributeError/RuntimeError for any SAMPLE_STEP > 1; added "
            "opt-in inference-time prior-tier box injection (NOISY_BOX_INFER*), which exists "
            "in the released code only as commented-out lines.",
            "tools/normalize_dentex_tiers.py: the raw quadrant / quadrant_enumeration JSONs "
            "do not match the 3-tier schema the loader and the model config assume, and are "
            "normalized before use (see docs/phase2_dataloader_fix.md).",
            "tools/labelme_to_coco.py: the DENTEX test split ships as LabelMe polygons with "
            "Turkish diagnosis strings and is converted to COCO before evaluation.",
            "No pretrained HierarchicalDet weights are publicly released, so all reported "
            "numbers come from training carried out in this reproduction, not from the "
            "authors' checkpoints.",
        ],
    }

    if args.curriculum_manifest and os.path.exists(args.curriculum_manifest):
        with open(args.curriculum_manifest) as f:
            manifest["curriculum"] = json.load(f)
    else:
        manifest["curriculum"] = {"present": False, "path": args.curriculum_manifest}

    if args.extra_notes:
        manifest["notes"] = args.extra_notes

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w") as f:
        f.write("# Reproducibility checklist\n\n")
        env = manifest["environment"]
        f.write("## Environment\n\n")
        for key, value in env.items():
            f.write("- **{}**: {}\n".format(key, value))
        f.write("\n## Git\n\n")
        for key, value in manifest["git"].items():
            f.write("- **{}**: `{}`\n".format(key, (value or "").replace("\n", " ")[:300]))
        f.write("\n## Checkpoints\n\n| File | Size (MB) | MD5 |\n|---|---|---|\n")
        for name, info in manifest["checkpoints"].items():
            f.write("| {} | {} | {} |\n".format(
                name,
                round(info["size_bytes"] / (1024 ** 2), 1) if info["present"] else "missing",
                info.get("md5", "-")))
        f.write("\n## Datasets\n\n| File | Images | Annotations | MD5 |\n|---|---|---|---|\n")
        for name, info in manifest["datasets"].items():
            f.write("| {} | {} | {} | {} |\n".format(
                name, info.get("images", "-"), info.get("annotations", "-"), info.get("md5", "-")))
        f.write("\n## Deviations from the official README\n\n")
        for item in manifest["deviations_from_official_readme"]:
            f.write("- {}\n".format(item))
        f.write("\n## Package versions\n\n```\n{}\n```\n".format("\n".join(manifest["packages"])))

    print("wrote {}".format(args.output))
    print("wrote {}".format(md_path))


if __name__ == "__main__":
    main()
