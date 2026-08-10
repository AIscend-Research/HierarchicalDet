#!/usr/bin/env python3
"""
Generate ``notebooks/*.ipynb`` from the cell definitions below.

The notebooks are generated rather than hand-edited on purpose. This project's
history contains a concrete failure mode — an updated cell pasted as a *new*
cell, leaving a stale duplicate further down that kept executing instead of the
fix. With a generator there is exactly one source for each cell, and
regenerating is a diffable operation.

There are **three** notebooks, split on the only boundaries that matter on
Kaggle: what accelerator a session needs, and what can finish in one session.

    01_setup_and_data              CPU only  — quota-free
    02_train_all                   GPU       — the only notebook that spends quota
    03_evaluate_and_build_assets   GPU or CPU — adapts to whichever it gets

Anything finer than that would only add cross-notebook state to lose.

Usage:  python tools/build_notebooks.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")


# --------------------------------------------------------------------------
# Shared cells
# --------------------------------------------------------------------------
PARAMS = '''\
# ============================================================================
# PARAMETERS — this cell is identical in all three notebooks.
# ============================================================================
RUN_MODE = "micro"          # "smoke" | "micro" (default) | "budget" | "full"
NUM_GPUS = None             # None = use every visible GPU; set 1 to force single-GPU
PUBLISH_KAGGLE_DATASET = True
CKPT_DATASET_SLUG = "dentex-repro-ckpts"
DATA_DATASET_SLUG = "dentex-repro-data"
REPO_URL = "https://github.com/AIscend-Research/dental-repro.git"

import os, subprocess, sys

# On Kaggle the repo is cloned into /kaggle/working (the only writable place
# that survives "Save Version"); locally the notebook already sits inside it.
if os.path.isdir("/kaggle/working"):
    CLONE = "/kaggle/working/repo"
    if os.path.isdir(os.path.join(CLONE, ".git")):
        subprocess.run(["git", "-C", CLONE, "pull", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "50", REPO_URL, CLONE], check=True)
    PROJECT_ROOT = os.path.join(CLONE, "dentex-repro")
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["RUN_MODE"] = RUN_MODE
print("project root:", PROJECT_ROOT)
'''

ENVIRONMENT = '''\
# ---- Environment: install, pin, and prove the VENDORED code is what loaded ----
# Kaggle reverts to its base image every session, so this runs every time.
import json
from src import setup_env

setup_env.install_dependencies()
# The vendored pycocotools ships Python sources only; its compiled `_mask`
# extension is grafted in here and VERIFIED BY IMPORT. It is compiled against
# numpy's C ABI, so a mismatch surfaces as "numpy.dtype size changed" deep
# inside detectron2.structures — which reads as a detectron2 problem and is not.
import numpy
print("numpy {} | pycocotools _mask -> {}".format(
    numpy.__version__, setup_env.ensure_pycocotools_mask()))
run = setup_env.bootstrap(RUN_MODE@REQUIRE_GPU@)

from src import manifest, train_utils

NUM_GPUS = NUM_GPUS if NUM_GPUS is not None else max(1, train_utils.visible_gpus())
lock = setup_env.write_requirements_lock()
environment = setup_env.env_report()
manifest.record_environment(environment)

# The repo vendors MODIFIED detectron2 / pycocotools (multi-label partial
# annotations, 3-tier category schema). A pip-installed copy silently shadows
# them and every number changes, so this is an assertion, not a warning.
found = setup_env.assert_vendored()
for module, path in found.items():
    print("{:14s} -> {}".format(module, path))
import detectron2, pycocotools, evaluator                              # noqa: F401
from hierarchialdet.util.coco_3class_eval import COCOEvaluator         # noqa: F401
from hierarchialdet.dataset_mapper_patched import DiffusionDetDatasetMapper  # noqa: F401
print("full import chain OK | commit {} | {} GPU process(es)".format(
    environment["repo_commit"][:12], NUM_GPUS))
'''


def environment_cell(require_gpu: bool = False) -> str:
    # Token replacement, not str.format: the cell body is full of Python format
    # placeholders of its own.
    return ENVIRONMENT.replace("@REQUIRE_GPU@", ", require_gpu=True" if require_gpu else "")


def summary_cell(notebook: str, body: str) -> str:
    return (
        "# ---- Notebook summary (the only cross-notebook contract) ----\n"
        + body
        + '\npath = setup_env.write_notebook_summary("{}", summary)\n'
          'print("wrote", path)\nprint(json.dumps(summary, indent=2, default=str)[:4000])\n'
        .format(notebook)
    )


# --------------------------------------------------------------------------
# Notebook 1 — setup and data (CPU only)
# --------------------------------------------------------------------------
def notebook_01():
    return [
        ("md", """\
# 01 — Setup and data (CPU only, quota-free)

Run this with **Accelerator = None**. It costs no GPU quota and produces
everything the training notebook needs.

1. installs and pins the dependency set, and proves `detectron2` /
   `pycocotools` resolved to the **vendored** copies in the repo rather than
   anything pip installed;
2. downloads DENTEX one archive at a time, extracting and deleting each before
   fetching the next (the release is ~11.8 GB compressed and `/kaggle/working`
   is ~20 GB);
3. converts all four schemas into the normalized 3-tier COCO the vendored
   loader expects;
4. audits what actually arrived, and **stops the study** if it disagrees with
   what was published;
5. builds the clean / stress evaluation subsets and the ground-truth sanity
   figure;
6. publishes everything as the `dentex-repro-data` Kaggle Dataset.

Three conditions abort here rather than being discovered later as a strange
number: the split composition not matching (693 / 634 / 705 / 50 / 250 / 1571),
**the test split having no ground truth** (it was withheld during the 2023
challenge and released later — without it the comparison target changes
entirely), and any diagnosis string in the test labels the alias table does not
cover.

**Settings:** Accelerator → *None*, Internet → *On*, Persistence → *Variables
and files*. Expected wall time: ~50 min, dominated by the download.
"""),
        ("code", PARAMS),
        ("code", environment_cell()),
        ("code", '''\
# ---- Download and extract ----
# Marker-gated at every step, so a session kill resumes rather than restarts.
from src import data_convert

RAW = os.path.join(setup_env.PROJECT_ROOT, "dentex_raw")
download = data_convert.download_and_extract(
    RAW, include_unlabelled=False, delete_archives=True)
print(json.dumps({k: v for k, v in download.items() if k != "raw_train_json"}, indent=2))
'''),
        ("code", '''\
# ---- Convert every split into the normalized 3-tier schema ----
# quadrant            flat categories/category_id      -> 3-tier (remapped BY NAME:
#                                                         a raw id copy would swap
#                                                         quadrants 1 and 2)
# quadrant_enumeration  categories_1/2                 -> 3-tier
# diagnosis             already 3-tier                 -> copied
# test split            LabelMe polygons, Turkish text -> 3-tier
conversion = data_convert.convert_all(download["raw_train_json"], download["test_label_dir"])
paths = conversion["paths"]
report = conversion["test_parse_report"]
print("test split parsed: {} images, {} annotations kept, {} distinct raw labels"
      .format(report["images"], report["annotations"], report["distinct_raw_labels"]))

# The raw test labels use a 9-code clinical scheme; only codes 1/6/7 are task
# classes. Everything else is excluded BY NAME and counted — never silently.
print("\\nexcluded out-of-task annotations ({} total):".format(report["excluded_total"]))
for word, info in report["excluded_out_of_task"].items():
    print("  {:10s} {:4d}  ({})".format(word, info["count"], info["meaning"]))
print("class-code inconsistencies:", report["class_code_inconsistencies"])

print("\\ntest-split diagnosis ground truth:")
for name, count in report["diagnosis_histogram"].items():
    print("  {:20s} {:4d}".format(name, count))

missing = report["diagnosis_classes_without_ground_truth"]
if missing:
    print("\\n*** FINDING: no test ground truth for {} ***".format(missing))
    print(data_convert.TEST_LABEL_FINDING)
    setup_env.log_deviation(
        "test-split ground truth missing entirely for {}".format(missing),
        data_convert.TEST_LABEL_FINDING,
        "01_setup_and_data",
        impact="test-split evaluation covers {} of 4 diagnosis classes; the "
               "missing class contributes no ground truth, so its per-class AP "
               "is undefined and the diagnosis-tier mean AP averages over the "
               "classes that have ground truth".format(4 - len(missing)))

for raw, parsed in list(report["parsed_examples"].items())[:12]:
    print("  {!r:26s} -> q{} t{} {}".format(raw, *parsed))
'''),
        ("code", '''\
# ---- Audit each split ----
audits = {}
for name, (json_key, image_key) in [
    ("quadrant_train", ("train_quadrant", "img_quadrant")),
    ("quadrant_enumeration_train", ("train_enumeration", "img_enumeration")),
    ("diagnosis_train", ("train_diagnosis", "img_diagnosis")),
    ("diagnosis_val", ("val_diagnosis", "img_val")),
    ("diagnosis_test", ("test_diagnosis", "img_test")),
]:
    audits[name] = data_convert.audit_split(paths[json_key], paths[image_key], name)
    audit = audits[name]
    print("{:28s} {:>5} images {:>6} annotations  tier coverage {}  zero-ann {}".format(
        name, audit["num_images"], audit["num_annotations"],
        audit["tier_coverage"], audit["images_with_zero_annotations"]))
    if audit["unreadable_images"] or audit["malformed_boxes"] or audit["missing_image_files"]:
        print("    PROBLEMS: unreadable={} malformed_boxes={} missing_files={}".format(
            len(audit["unreadable_images"]), len(audit["malformed_boxes"]),
            len(audit["missing_image_files"])))
'''),
        ("code", '''\
# ---- Hard assertions: composition, and the presence of test ground truth ----
actual_counts = data_convert.assert_published_counts(
    audits, download.get("unlabelled_in_zip"))
print("published composition confirmed:", json.dumps(actual_counts, indent=2))

data_convert.assert_test_ground_truth(audits["diagnosis_test"])
print("test ground truth present: {} annotations, {} with a diagnosis label".format(
    audits["diagnosis_test"]["num_annotations"],
    audits["diagnosis_test"]["tier_coverage"]["tier2_diagnosis"]))

print()
print(data_convert.LICENSE_NOTE)
setup_env.log_deviation(
    "DENTEX license discrepancy (CC BY-SA on GitHub vs CC BY-NC-SA on HuggingFace)",
    "the two published statements disagree; the stricter non-commercial reading is assumed",
    "01_setup_and_data")
'''),
        ("code", '''\
# ---- Registration check: identical class indices across every split ----
# The model has fixed-size heads shared across curriculum stages, so drifting
# class indices between splits would let weight transfer silently remap labels
# with no error raised.
from src import registration

registration_report = registration.verify_registration()
print(json.dumps(registration_report["thing_classes"], indent=2))
'''),
        ("code", '''\
# ---- Clean and stress evaluation subsets (rule written down, ids saved) ----
subsets = data_convert.build_clean_stress_subsets(paths["test_diagnosis"])
os.makedirs(paths["subsets"], exist_ok=True)
subset_path = os.path.join(paths["subsets"], "clean_stress.json")
with open(subset_path, "w") as handle:
    json.dump(subsets, handle, indent=2)

for which in ("clean", "stress"):
    out = os.path.join(paths["subsets"], "test_{}.json".format(which))
    data_convert.subset_json(paths["test_diagnosis"], subsets[which]["image_ids"], out)
    print("{:6s} {:>3} images -> {}".format(which, len(subsets[which]["image_ids"]), out))
print()
print(subsets["rule"])
'''),
        ("code", '''\
# ---- Ground-truth sanity figure: 5 random images per tier, boxes drawn ----
import random
from src import figures

random.seed(setup_env.BASE_SEED)
panels = []
for name, (json_key, image_key, tier) in {
    "quadrant": ("train_quadrant", "img_quadrant", 0),
    "enumeration": ("train_enumeration", "img_enumeration", 1),
    "diagnosis": ("test_diagnosis", "img_test", 2),
}.items():
    with open(paths[json_key]) as handle:
        data = json.load(handle)
    by_image = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    names = {level: {c["id"]: str(c["name"])
                     for c in data["categories_{}".format(level + 1)]} for level in range(3)}
    chosen = random.sample([i for i in data["images"] if by_image.get(i["id"])], 5)
    for image in chosen:
        boxes = []
        for annotation in by_image[image["id"]]:
            label = "/".join(
                names[level].get(annotation.get("category_id_{}".format(level + 1)), "")
                for level in range(tier + 1)
                if annotation.get("category_id_{}".format(level + 1)) is not None)
            boxes.append((annotation["bbox"], label))
        panels.append({"image_path": os.path.join(paths[image_key], image["file_name"]),
                       "title": "{} — {}".format(name, image["file_name"]),
                       "gt": boxes, "pred": []})

figure = figures.overlay_grid(panels, columns=5, panel_height=1.7,
                              title="Ground-truth sanity check (5 images per tier)")
figures.save_figure(figure, "gt_sanity_check",
                    "Ground-truth annotations overlaid on five randomly sampled images "
                    "from each DENTEX tier. Labels read quadrant/enumeration/diagnosis.",
                    "01_setup_and_data", run.mode, "figure:gt_sanity",
                    inputs=[paths["train_quadrant"], paths["train_enumeration"],
                            paths["test_diagnosis"]])
'''),
        ("code", '''\
# ---- Dataset audit table, hashes, and publication ----
from src import tables

tables.write_table(
    "dataset_audit", tables.audit_rows(audits),
    ["split", "images", "annotations", "images_without_annotations",
     "quadrant_labels", "enumeration_labels", "diagnosis_labels",
     "distinct_resolutions", "unreadable_images", "missing_image_files",
     "malformed_boxes"],
    "DENTEX composition and integrity, as received and converted by this study.",
    "01_setup_and_data", run.mode, "table:dataset_audit", inputs=[paths["coco"]])

hashes = data_convert.dataset_hashes()
with open(os.path.join(paths["root"], "dataset_hashes.json"), "w") as handle:
    json.dump(hashes, handle, indent=2)
for name, digest in hashes.items():
    print("{:34s} {}".format(name, digest[:16]))
'''),
        ("code", summary_cell("01_setup_and_data", '''\
summary = {
    "run_mode": run.mode,
    "environment": environment,
    "requirements_lock": lock,
    "vendored_modules": found,
    "download": {k: v for k, v in download.items() if k != "raw_train_json"},
    "paths": paths,
    "audits": audits,
    "published_counts": actual_counts,
    "test_parse_report": {k: v for k, v in report.items() if k != "raw_label_counts"},
    "registration": registration_report,
    "subsets": {"path": subset_path, "rule": subsets["rule"],
                "clean": len(subsets["clean"]["image_ids"]),
                "stress": len(subsets["stress"]["image_ids"])},
    "dataset_hashes": hashes,
    "license_note": data_convert.LICENSE_NOTE,
}
''')),
        ("code", '''\
# ---- Publish, last: the summary above must be inside what gets published ----
# paper_assets/ ships with the data because it holds this notebook's summary
# (dataset hashes, audit counts), which notebook 03's reproducibility checklist
# reads back — and a fresh Kaggle session re-clones paper_assets/ empty.
publish = {"status": "disabled"}
if PUBLISH_KAGGLE_DATASET:
    publish = train_utils.publish_kaggle_dataset(
        DATA_DATASET_SLUG, [paths["root"], setup_env.PAPER_ASSETS],
        "converted DENTEX (run mode {})".format(run.mode))
    summary["kaggle_publish"] = {k: v for k, v in publish.items()
                                 if k not in ("stdout", "stderr")}
    setup_env.write_notebook_summary("01_setup_and_data", summary)
print(json.dumps({k: v for k, v in publish.items() if k not in ("stdout", "stderr")},
                 indent=2))
print("\\nAttach this dataset to notebook 02 as: {}".format(DATA_DATASET_SLUG))
'''),
    ]


# --------------------------------------------------------------------------
# Notebook 2 — all training (GPU)
# --------------------------------------------------------------------------
def notebook_02():
    return [
        ("md", """\
# 02 — All training (the only notebook that spends GPU quota)

**Settings:** Accelerator → *GPU T4 x2*, Internet → *On*, Persistence →
*Variables and files*. Attach `dentex-repro-data`, and `dentex-repro-ckpts` on
any re-run.

```
quadrant (tier 0)
   └─ inference over the enumeration split ──► noisy boxes ─┐
enumeration (tier 1) ◄─────────── weight transfer ──────────┘
   └─ inference over the diagnosis split ───► noisy boxes ──┐
diagnosis × 4 variants ◄────────── weight transfer ─────────┘
base DiffusionDet (flat, no hierarchy, no multi-label)
```

## The ablation

| variant | weight transfer | noisy-box manipulation | multi-label |
|---|---|---|---|
| `full` | ✓ enumeration checkpoint | ✓ | ✓ |
| `wo_transfer` | ✗ ImageNet Swin-B | ✓ | ✓ |
| `wo_manipulation` | ✓ | ✗ | ✓ |
| `wo_manip_transfer` | ✗ | ✗ | ✓ |

Those two switches are the *only* difference between the runs. Transfer is
`MODEL.WEIGHTS`; manipulation is the `NOISY_BOX_TRAIN` / `NOISY_BOX_VAL`
environment variables the patched dataset mapper reads. No upstream file is
edited and no config differs between variants.

## This notebook will not finish in one session — that is the design

`micro` needs ~12.25 GPU-h against a 12 h session limit. Every run checkpoints
every 2,500 iterations, is skipped once complete, and resumes from its last
checkpoint otherwise. **Re-run this same notebook in a fresh session with
`dentex-repro-ckpts` attached** until it reaches the end.

In hour-capped modes it first measures real throughput over 500 genuine
training iterations. The measurement is cached and shared, so all diagnosis
variants receive the *same* `MAX_ITER` — identical budgets by construction, not
approximately equal. `assert_matched_budgets` aborts if they ever diverge.
"""),
        ("code", PARAMS),
        ("code", environment_cell(require_gpu=True)),
        ("code", '''\
# ---- Inputs, weights, configs ----
from src import data_convert, eval_utils, registration

paths = data_convert.layout()
for key in ("train_quadrant", "train_enumeration", "train_diagnosis", "val_diagnosis",
            "test_diagnosis"):
    assert os.path.exists(paths[key]), (
        "{} is missing — run notebook 01, or attach the {} dataset"
        .format(paths[key], DATA_DATASET_SLUG))

IMAGENET_WEIGHTS = train_utils.ensure_swin_weights()
CFG = {name: os.path.join(setup_env.CONFIGS_REPRO, "diffdet.dentex.{}.yaml".format(name))
       for name in ("quadrant", "enumeration", "diagnosis", "base_diffusiondet")}
NOISY_DIR = os.path.join(setup_env.RUNS_DIR, "noisy_boxes")
os.makedirs(NOISY_DIR, exist_ok=True)
print("ImageNet Swin-B:", IMAGENET_WEIGHTS)
print("GPU-hours already recorded on disk: {:.2f}".format(setup_env.gpu_hours_spent()))
'''),
        ("code", '''\
# ---- Does multi-GPU actually work in this Kaggle session? ----
# Answered by running a real 20-iteration job, not assumed. A failure falls back
# to one GPU and is recorded, rather than being fought for the rest of the study.
import shutil

ddp = {"requested": NUM_GPUS, "works": None, "error": None}
probe_dir = os.path.join(setup_env.RUNS_DIR, "ddp_probe")
if NUM_GPUS > 1 and not os.path.exists(os.path.join(setup_env.RUNS_DIR, ".ddp_ok")):
    try:
        train_utils.launch_training(
            CFG["diagnosis"],
            train_utils.base_overrides(run, probe_dir, 20, IMAGENET_WEIGHTS, NUM_GPUS),
            registration.training_env("diagnosis_train"), probe_dir, NUM_GPUS,
            resume=False, log_name="ddp_probe.log")
        ddp["works"] = True
        open(os.path.join(setup_env.RUNS_DIR, ".ddp_ok"), "w").close()
    except RuntimeError as error:
        ddp["works"] = False
        ddp["error"] = str(error)[-1500:]
        NUM_GPUS = 1
        setup_env.log_deviation(
            "multi-GPU training disabled (DDP failed in this Kaggle session)",
            "launch(num_gpus=2) failed during the probe; the study runs single-GPU "
            "rather than fighting a flaky DDP setup", "02_train_all",
            impact="effective batch size halves relative to a 2-GPU run")
else:
    ddp["works"] = "already verified" if NUM_GPUS > 1 else "single GPU"
shutil.rmtree(probe_dir, ignore_errors=True)
print(json.dumps(ddp, indent=2), "-> NUM_GPUS =", NUM_GPUS)
'''),
        ("code", '''\
# ---- Pre-flight: 200 iterations + a real evaluation, on 10 images ----
# Proves the whole chain before committing hours of quota to it. Skipped when
# RUN_MODE="smoke", where the real runs below already are exactly this.
smoke = {"skipped": run.is_smoke}
if not run.is_smoke and not train_utils.is_complete("preflight"):
    smoke_dir = train_utils.run_dir("preflight")
    train_utils.launch_training(
        CFG["diagnosis"],
        train_utils.base_overrides(run, smoke_dir, 200, IMAGENET_WEIGHTS, NUM_GPUS),
        registration.training_env("diagnosis_train"), smoke_dir, NUM_GPUS, resume=False)
    smoke_weights = os.path.join(smoke_dir, "model_final.pth")
    assert os.path.exists(smoke_weights), "pre-flight produced no model_final.pth"
    smoke_eval = eval_utils.evaluate_checkpoint(
        smoke_weights, CFG["diagnosis"], split="diagnosis_test", tier=2, seed=0, limit=10)
    # 200 iterations cannot detect anything useful; the assertion is that the
    # metric pipeline RAN, not that it scored well.
    assert set(smoke_eval["tiers"]) == {"quadrant", "enumeration", "diagnosis"}
    smoke = {"weights": smoke_weights,
             "metrics": {t: p["metrics"] for t, p in smoke_eval["tiers"].items()}}
    shutil.rmtree(smoke_dir, ignore_errors=True)
print(json.dumps(smoke, indent=2, default=str))
'''),
        ("code", '''\
# ---- Throughput calibration (hour-capped modes only) ----
# 500 real training iterations, 100 warm-up discarded. Cached, so it is paid
# once for the whole study and charged to the budget once, not once per stage.
calibration = None
if run.quadrant.max_iter is None or run.diagnosis.max_iter is None:
    calibration = train_utils.calibrate_rate(
        "swinb_gpu{}".format(NUM_GPUS), CFG["quadrant"], run,
        "quadrant_train", IMAGENET_WEIGHTS, NUM_GPUS)
    print(json.dumps(calibration, indent=2))
'''),
        ("md", "## Stage 0–1 — prerequisites, trained once and shared by every variant"),
        ("code", '''\
# ---- Stage 0: quadrant ----
training_records = []
quadrant_record = train_utils.train_stage(
    "quadrant_stage", CFG["quadrant"], run, "quadrant_train",
    IMAGENET_WEIGHTS, NUM_GPUS, run.quadrant, calibration=calibration)
quadrant_record["kind"] = "prerequisite"
training_records.append(quadrant_record)
QUADRANT_WEIGHTS = train_utils.final_weights("quadrant_stage")
print(json.dumps({k: v for k, v in quadrant_record.items() if k != "launch"},
                 indent=2, default=str)[:2000])
'''),
        ("code", '''\
# ---- Quadrant model -> noisy boxes for the enumeration stage ----
# This is the manipulation signal: tier k-1's detections over tier k's own train
# and validation images, filtered at score >= 0.5 by the dataset mapper.
enum_boxes = {}
for key, split in (("NOISY_BOX_TRAIN", "quadrant_enumeration_train"),
                   ("NOISY_BOX_VAL", "diagnosis_val")):
    output = os.path.join(NOISY_DIR, "quadrant_over_{}.json".format(split))
    if not os.path.exists(output):
        print(json.dumps(eval_utils.dump_predictions(
            QUADRANT_WEIGHTS, CFG["quadrant"], split, 0, output, seed=0), indent=2)[:900])
    enum_boxes[key] = output
print(enum_boxes)
'''),
        ("code", '''\
# ---- Stage 1: enumeration (transfer + manipulation from stage 0) ----
enumeration_record = train_utils.train_stage(
    "enumeration_stage", CFG["enumeration"], run, "quadrant_enumeration_train",
    QUADRANT_WEIGHTS, NUM_GPUS, run.enumeration, calibration=calibration,
    noisy_boxes=enum_boxes)
enumeration_record["kind"] = "prerequisite"
training_records.append(enumeration_record)
ENUM_WEIGHTS = train_utils.final_weights("enumeration_stage")
print(json.dumps({k: v for k, v in enumeration_record.items() if k != "launch"},
                 indent=2, default=str)[:2000])
'''),
        ("code", '''\
# ---- Enumeration model -> noisy boxes for the diagnosis stage ----
from src import degradations

diagnosis_boxes = {}
for key, split in (("NOISY_BOX_TRAIN", "diagnosis_train"),
                   ("NOISY_BOX_VAL", "diagnosis_val")):
    output = os.path.join(NOISY_DIR, "enumeration_over_{}.json".format(split))
    if not os.path.exists(output):
        print(json.dumps(eval_utils.dump_predictions(
            ENUM_WEIGHTS, CFG["enumeration"], split, 1, output, seed=0), indent=2)[:900])
    diagnosis_boxes[key] = output

# Also produced here because notebook 03's fault-injection experiment needs the
# prior tier's detections over the *test* split.
prior_test = os.path.join(NOISY_DIR, "enumeration_over_diagnosis_test.json")
if not os.path.exists(prior_test):
    eval_utils.dump_predictions(ENUM_WEIGHTS, CFG["enumeration"], "diagnosis_test", 1,
                                prior_test, seed=0)

box_stats = {key: degradations.summarize_prediction_file(path)
             for key, path in diagnosis_boxes.items()}
print(json.dumps(box_stats, indent=2))
'''),
        ("md", "## Stage 2 — the diagnosis variants, at identical budgets"),
        ("code", '''\
# ---- Train every variant ----
plans = train_utils.variant_plan(run, {"enumeration": ENUM_WEIGHTS},
                                 IMAGENET_WEIGHTS, diagnosis_boxes)
variant_records = []
for plan in plans:
    print("\\n" + "=" * 72)
    print("{}  (transfer={}, manipulation={})".format(
        plan["variant"], plan["switches"]["transfer"], plan["switches"]["manipulation"]))
    print("=" * 72)
    record = train_utils.train_stage(
        plan["run_name"], CFG["diagnosis"], run, "diagnosis_train",
        plan["weights"], NUM_GPUS, run.diagnosis, calibration=calibration,
        noisy_boxes=plan["noisy_boxes"],
        trajectory_fractions=run.trajectory_fractions)
    record.update({"variant": plan["variant"], "label": plan["label"],
                   "switches": plan["switches"], "kind": "diagnosis_variant"})
    variant_records.append(record)
    training_records.append(record)
    print("iterations {} | wall {} s | stopped on time budget: {}".format(
        record.get("max_iter"), record.get("wall_seconds"),
        record.get("stopped_on_time_budget")))

# Different wall time between variants is fine (manipulation costs extra
# dataloading). Different iteration counts, seeds or batch sizes are not — that
# would make the comparison measure the budget instead of the switch.
matched = train_utils.assert_matched_budgets(variant_records)
print("\\n" + json.dumps(matched, indent=2, default=str))
'''),
        ("code", '''\
# ---- Base DiffusionDet: flat, no hierarchy, no multi-label ----
# INTERPRETATION (ambiguous in the released code, logged as a deviation): the
# vendored head indexes NUM_CLASSES as a 3-list (head.py:81-83), so a genuinely
# single-head DiffusionDet cannot be configured — the repo's own enumeration
# config (NUM_CLASSES: 32, scalar) would raise on construction. The flat label
# space is therefore expressed in the DATA: the target tier's labels move into
# category_id_1 and the other tiers are nulled, so exactly one head is
# supervised. Everything else matches our models, which makes the difference
# against Ours_wo_Manip_Transfer exactly the multi-label head structure.
TIER_CLASSES = {0: 4, 1: 8, 2: 4}
TIER_SPLIT = {0: "quadrant_train", 1: "quadrant_enumeration_train", 2: "diagnosis_train"}
base_records = []
for tier in run.base_tiers:
    flat_train = data_convert.flat_json_path(tier, "train")
    assert os.path.exists(flat_train), "run notebook 01 first: {}".format(flat_train)
    print("\\n=== base_diffusiondet_tier{} ===".format(tier))
    record = train_utils.train_stage(
        "base_diffusiondet_tier{}".format(tier), CFG["base_diffusiondet"], run,
        TIER_SPLIT[tier], IMAGENET_WEIGHTS, NUM_GPUS, run.base_diffusiondet,
        extra_overrides=["MODEL.DiffusionDet.NUM_CLASSES",
                         "[{}, 8, 4]".format(TIER_CLASSES[tier])],
        env_override={"TRAIN_JSON": flat_train,
                      "VAL_JSON": data_convert.flat_json_path(tier, "test"),
                      "VAL_IMG_DIR": paths["img_test"], "TIER": "0"})
    record.update({"tier": tier, "flat_train_json": flat_train, "kind": "base_diffusiondet"})
    base_records.append(record)
    training_records.append(record)
if not run.base_tiers:
    print("base DiffusionDet skipped in RUN_MODE={}".format(run.mode))
'''),
        ("code", '''\
# ---- Deviations this notebook establishes ----
setup_env.log_deviation(
    "AMP (SOLVER.AMP.ENABLED=True) enabled for all training",
    "the released configs leave AMP off; T4s need it to fit Swin-B in 16 GB at a "
    "usable throughput, and AMPTrainer is the repo's own code path", "02_train_all",
    impact="mixed-precision reduction order is a residual source of nondeterminism")
setup_env.log_deviation(
    "EMA (MODEL_EMA.ENABLED) left OFF for every run",
    "the repo ships EMA hooks but no config enables them and the paper does not "
    "mention EMA; held constant across variants so it cannot explain any difference",
    "02_train_all")
setup_env.log_deviation(
    "Swin backbone loaded from a .pth, not the config's .pkl filename",
    "DetectionCheckpointer dispatches on file extension: a torch checkpoint named "
    ".pkl is parsed as a Caffe2 blob and the backbone silently stays random",
    "02_train_all",
    impact="fixes a silent failure; without it every number would come from a "
           "randomly initialized backbone")
setup_env.log_deviation(
    "Swin-B used where the released nonpretrain config said SWIN.SIZE: L-22k",
    "that config's own MODEL.WEIGHTS points at a Swin-B checkpoint, so the released "
    "file is internally inconsistent; Swin-B is also what fits a 16 GB T4",
    "02_train_all",
    impact="a Swin-L run would have more capacity; this is a lower bound")
setup_env.log_deviation(
    "SimMIM pretraining on the 1,571 unlabelled X-rays skipped",
    "the authors' SimMIM checkpoint is not published and the pretraining lives in a "
    "separate repository; initialization follows the repo's own nonpretrain config",
    "02_train_all",
    impact="our backbone has never seen a panoramic radiograph; any gap against the "
           "paper is confounded by this")

skipped = [v for v in setup_env.ALL_VARIANTS if v not in run.variants]
if skipped:
    setup_env.log_deviation(
        "diagnosis variants {} not trained".format(skipped),
        "RUN_MODE={} trades variant coverage for matched per-variant compute".format(run.mode),
        "02_train_all",
        impact="those rows carry the original's numbers, clearly marked, and no "
               "reproduced value")
if not run.base_tiers:
    setup_env.log_deviation(
        "base DiffusionDet not trained",
        "RUN_MODE={} spends its budget on the diagnosis ablation, which is the "
        "paper's central claim".format(run.mode), "02_train_all",
        impact="the DiffusionDet_base row carries the original's numbers as "
               "untested context")
print(open(setup_env.DEVIATIONS_MD).read())
'''),
        ("code", summary_cell("02_train_all", '''\
summary = {
    "run_mode": run.mode,
    "num_gpus": NUM_GPUS,
    "multi_gpu": ddp,
    "preflight": smoke,
    "calibration": calibration,
    "records": training_records,
    "variants_trained": list(run.variants),
    "variants_skipped": skipped,
    "matched_budgets": matched,
    "weights": {
        "imagenet": IMAGENET_WEIGHTS,
        "quadrant": QUADRANT_WEIGHTS,
        "enumeration": ENUM_WEIGHTS,
        "variants": {r["variant"]: train_utils.final_weights(r["name"])
                     for r in variant_records},
        "base": {str(r["tier"]): train_utils.final_weights(r["name"])
                 for r in base_records},
    },
    "noisy_boxes": {"for_enumeration": enum_boxes, "for_diagnosis": diagnosis_boxes,
                    "prior_over_test": prior_test, "stats": box_stats},
    "trajectory_checkpoints": {r["variant"]: r.get("trajectory", {})
                               for r in variant_records},
    "gpu_hours_spent": setup_env.gpu_hours_spent(),
}
''')),
        ("code", '''\
# ---- Publish, last: the summary above must be inside what gets published ----
# Notebook 03 reads that summary to find the checkpoints, the noisy-box dumps
# and the trajectory snapshot names, so it has to be in the dataset.
publish = {"status": "disabled"}
if PUBLISH_KAGGLE_DATASET:
    publish = train_utils.publish_kaggle_dataset(
        CKPT_DATASET_SLUG, [setup_env.RUNS_DIR, setup_env.PAPER_ASSETS],
        "training ({} mode)".format(run.mode))
    summary["kaggle_publish"] = {k: v for k, v in publish.items()
                                 if k not in ("stdout", "stderr")}
    setup_env.write_notebook_summary("02_train_all", summary)
print(json.dumps({k: v for k, v in publish.items() if k not in ("stdout", "stderr")},
                 indent=2))
print("\\nGPU-hours recorded so far: {:.2f}".format(setup_env.gpu_hours_spent()))
print("Attach to the next session as: {} (plus {})".format(
    CKPT_DATASET_SLUG, DATA_DATASET_SLUG))
'''),
    ]


# --------------------------------------------------------------------------
# Notebook 3 — evaluation, robustness, error analysis, paper assets
# --------------------------------------------------------------------------
def notebook_03():
    return [
        ("md", """\
# 03 — Evaluation, robustness, error analysis, and `paper_assets/`

**Adapts to whatever accelerator it gets.** Attach `dentex-repro-data` and
`dentex-repro-ckpts`.

* **Accelerator = GPU T4** → runs everything: main multi-seed evaluation, the
  checkpoint-trajectory sweep, sampling-step sweep, degradation grid,
  clean/stress subsets, hierarchy fault injection, GPU throughput/VRAM, error
  analysis and the qualitative figures — then builds all the assets. ~2.5 h in
  `micro`.
* **Accelerator = None** → skips the GPU sections, measures **CPU-only**
  inference (quota-free), and rebuilds every table, figure and document from
  `paper_assets/results_raw/`. ~5 min.

So the normal sequence is: run it once on a T4, then run it again on CPU to add
the CPU benchmark and the CPU-vs-GPU agreement check. After that, any CPU run
regenerates every asset in minutes without touching the quota.

DiffusionDet denoises from **random** boxes, so inference is stochastic even at
fixed weights. Every model is evaluated under three inference seeds and reported
as mean ± std; that spread is a result, not noise to hide.

*All outputs are research artifacts. Nothing here is a clinical claim, and
nothing here is validated for clinical use.*
"""),
        ("code", PARAMS),
        ("code", environment_cell()),
        ("code", '''\
# ---- What can this session do, and what was trained? ----
from src import data_convert, degradations, eval_utils, figures, registration, tables

HAS_GPU = setup_env.gpu_report().get("available", False)
print("GPU available:", HAS_GPU, "->", "full pass" if HAS_GPU
      else "CPU pass (benchmark + asset build only)")

paths = data_convert.layout()
CFG = {name: os.path.join(setup_env.CONFIGS_REPRO, "diffdet.dentex.{}.yaml".format(name))
       for name in ("quadrant", "enumeration", "diagnosis", "base_diffusiondet")}
training = setup_env.read_notebook_summary("02_train_all") or {}
weights = training.get("weights") or {}
FULL_WEIGHTS = (weights.get("variants") or {}).get("full")

TIER_CLASSES = {0: 4, 1: 8, 2: 4}
models = {}
for label, key, config in (("Stage_quadrant", "quadrant", CFG["quadrant"]),
                           ("Stage_enumeration", "enumeration", CFG["enumeration"])):
    path = weights.get(key)
    if path and os.path.exists(path):
        models[label] = {"weights": path, "config": config, "tier": 2,
                         "overrides": [], "json_override": None}
for variant, path in (weights.get("variants") or {}).items():
    if os.path.exists(path):
        models[setup_env.VARIANT_LABELS[variant]] = {
            "weights": path, "config": CFG["diagnosis"], "tier": 2,
            "overrides": [], "json_override": None}
for tier_str, path in (weights.get("base") or {}).items():
    if os.path.exists(path):
        tier = int(tier_str)
        models["DiffusionDet_base_tier{}".format(tier)] = {
            "weights": path, "config": CFG["base_diffusiondet"], "tier": 0,
            "overrides": ["MODEL.DiffusionDet.NUM_CLASSES",
                          "[{}, 8, 4]".format(TIER_CLASSES[tier])],
            "json_override": data_convert.flat_json_path(tier, "test"),
            "flat_tier": tier}
for label, spec in models.items():
    print("{:26s} {}".format(label, spec["weights"]))
if HAS_GPU:
    assert models, "no trained checkpoints found — run notebook 02, or attach its dataset"
'''),
        ("md", "## GPU sections — skipped automatically on a CPU session"),
        ("code", '''\
# ---- Main evaluation: every model, every tier, three inference seeds ----
# One inference pass at tier 2 scores all three tiers, because the authors'
# inference_on_dataset stores the predictions once and evaluates tiers 0..k.
if HAS_GPU:
    for label, spec in models.items():
        print("\\n=== {} ({} seeds) ===".format(label, len(run.eval_seeds)))
        payload = eval_utils.evaluate_multi_seed(
            spec["weights"], spec["config"], run.eval_seeds,
            split="diagnosis_test", tier=spec["tier"], overrides=spec["overrides"],
            limit=run.eval_limit, json_override=spec["json_override"],
            image_dir_override=paths["img_test"] if spec["json_override"] else None)
        payload["label"] = label
        payload["flat_tier"] = spec.get("flat_tier")
        eval_utils.save_result(manifest.result_name("eval_main", model=label), payload)
        for tier, cells in payload["aggregate"].items():
            ap = cells.get("AP", {})
            if ap.get("mean") is not None:
                print("  {:12s} AP {:.2f} ± {:.2f}   AR {}".format(
                    tier, ap["mean"], ap["std"],
                    "{:.3f}".format(cells["AR"]["mean"])
                    if cells["AR"]["mean"] is not None else "n/a"))
'''),
        ("code", '''\
# ---- Runtime, zero-detection and degenerate-box accounting ----
if HAS_GPU:
    for label, spec in models.items():
        output = os.path.join(setup_env.RUNS_DIR, "predictions", "{}_test.json".format(label))
        summary_i = eval_utils.dump_predictions(
            spec["weights"], spec["config"], "diagnosis_test", spec["tier"], output,
            seed=run.eval_seeds[0], limit=run.eval_limit, overrides=spec["overrides"],
            json_override=spec["json_override"],
            image_dir_override=paths["img_test"] if spec["json_override"] else None)
        eval_utils.save_result(
            manifest.result_name("runtime", model=label, device=summary_i["device"],
                                 step=summary_i["sample_step"]), summary_i)
        print("{:26s} {:.3f} s/image  {} images with no detections  failures {}".format(
            label, summary_i["mean_seconds_per_image"] or float("nan"),
            summary_i["images_with_no_detections"], summary_i["failure_counts"]))
'''),
        ("code", '''\
# ---- Checkpoint trajectory: is the ablation ordering stable over training? ----
# Each variant at ~1/3, ~2/3 and 1/1 of its budget. Stable ordering across
# checkpoints is evidence the ablation conclusion is not an artifact of the
# shortened schedule; unstable ordering is itself the finding.
if HAS_GPU:
    for record in training.get("records", []):
        if record.get("kind") != "diagnosis_variant":
            continue
        variant = record["variant"]
        max_iter = record.get("max_iter") or record.get("plan", {}).get("max_iter")
        points = []
        for iteration, name in sorted((record.get("trajectory") or {}).items(),
                                      key=lambda kv: int(kv[0])):
            checkpoint = os.path.join(train_utils.run_dir(record["name"]), name + ".pth")
            if not os.path.exists(checkpoint):
                print("missing trajectory checkpoint (skipped):", checkpoint)
                continue
            payload = eval_utils.evaluate_checkpoint(
                checkpoint, CFG["diagnosis"], split="diagnosis_test", tier=2,
                seed=run.eval_seeds[0], limit=run.eval_limit)
            points.append({"iteration": int(iteration), "checkpoint": name,
                           "progress": round(int(iteration) / max_iter, 3),
                           "result": payload})
            print("{:20s} @{:>6} iters  diagnosis AP {}".format(
                variant, iteration, payload["tiers"]["diagnosis"]["metrics"]["AP"]))
        final = eval_utils.load_result(
            manifest.result_name("eval_main", model=setup_env.VARIANT_LABELS[variant]))
        if final:
            points.append({"iteration": max_iter, "checkpoint": "model_final",
                           "progress": 1.0, "result": final["runs"][0],
                           "aggregate": final["aggregate"]})
        eval_utils.save_result(manifest.result_name("trajectory", variant=variant),
                               {"variant": variant, "max_iter": max_iter, "points": points})
'''),
        ("code", '''\
# ---- Robustness 1: diffusion sampling steps (accuracy AND measured latency) ----
if HAS_GPU and FULL_WEIGHTS:
    for step in run.step_sweep:
        payload = eval_utils.evaluate_multi_seed(
            FULL_WEIGHTS, CFG["diagnosis"], run.robustness_seeds,
            split="diagnosis_test", tier=2, sample_step=step, limit=run.eval_limit)
        eval_utils.save_result(manifest.result_name("steps", step=step), payload)
        latency = eval_utils.dump_predictions(
            FULL_WEIGHTS, CFG["diagnosis"], "diagnosis_test", 2,
            os.path.join(setup_env.RUNS_DIR, "predictions", "full_step{}.json".format(step)),
            seed=run.robustness_seeds[0], sample_step=step, limit=run.eval_limit)
        eval_utils.save_result(manifest.result_name("steps_latency", step=step), latency)
        print("steps={:2d}  diagnosis AP {:.2f}  {:.3f} s/image".format(
            step, payload["aggregate"]["diagnosis"]["AP"]["mean"],
            latency["mean_seconds_per_image"]))
'''),
        ("code", '''\
# ---- Robustness 2: degradation grid ----
# Downscaled images are resampled back to the original resolution, so the
# ground-truth boxes stay valid and this measures lost detail, not a changed
# coordinate system.
if HAS_GPU and FULL_WEIGHTS:
    DEGRADED_ROOT = os.path.join(setup_env.RUNS_DIR, "degraded")
    for condition in degradations.degradation_grid(run):
        image_dir = paths["img_test"] if condition["kind"] == "none" else \\
            degradations.build_degraded_images(paths["img_test"], condition["kind"],
                                               condition["severity"], DEGRADED_ROOT)
        payload = eval_utils.evaluate_multi_seed(
            FULL_WEIGHTS, CFG["diagnosis"], run.robustness_seeds,
            split="diagnosis_test", tier=2, limit=run.eval_limit,
            image_dir_override=image_dir)
        payload["condition"] = condition
        eval_utils.save_result(
            manifest.result_name("degradation", condition=condition["label"]), payload)
        print("{:20s} diagnosis AP {:.2f}".format(
            condition["label"], payload["aggregate"]["diagnosis"]["AP"]["mean"]))
'''),
        ("code", '''\
# ---- Robustness 3: clean vs stress subsets, for every trained model ----
if HAS_GPU:
    for which in ("clean", "stress"):
        subset_json = os.path.join(paths["subsets"], "test_{}.json".format(which))
        assert os.path.exists(subset_json), "run notebook 01 first: {}".format(subset_json)
        for label, spec in models.items():
            if spec["json_override"]:
                continue          # flat baselines have their own ground truth
            payload = eval_utils.evaluate_multi_seed(
                spec["weights"], spec["config"], run.robustness_seeds,
                split="diagnosis_test", tier=2, json_override=subset_json,
                limit=run.eval_limit)
            payload.update({"subset": which, "label": label})
            eval_utils.save_result(
                manifest.result_name("subset", model=label, subset=which), payload)
            print("{:24s} {:6s} diagnosis AP {:.2f}".format(
                label, which, payload["aggregate"]["diagnosis"]["AP"]["mean"]))
'''),
        ("code", '''\
# ---- Robustness 4: hierarchy fault injection ----
# The prior-tier boxes the diagnosis model consumes are deliberately corrupted.
# Inference-time injection is OFF for every other experiment (that reproduces the
# released behaviour, which never injects at inference), so the jitter=0/drop=0
# condition — not the main table — is this experiment's reference point.
if HAS_GPU and FULL_WEIGHTS:
    prior_test = (training.get("noisy_boxes") or {}).get("prior_over_test")
    assert prior_test and os.path.exists(prior_test), (
        "notebook 02 must produce the enumeration model's predictions over the test split")
    for condition in degradations.fault_grid(run):
        with eval_utils.noisy_box_inference(prior_test, jitter=condition["jitter"],
                                            drop=condition["drop"]):
            payload = eval_utils.evaluate_multi_seed(
                FULL_WEIGHTS, CFG["diagnosis"], run.robustness_seeds,
                split="diagnosis_test", tier=2, limit=run.eval_limit)
        payload["condition"] = condition
        eval_utils.save_result(
            manifest.result_name("fault", condition=condition["label"]), payload)
        print("{:16s} ({:s}) diagnosis AP {:.2f}".format(
            condition["label"], condition["axis"],
            payload["aggregate"]["diagnosis"]["AP"]["mean"]))
    setup_env.log_deviation(
        "inference-time prior-tier box injection used for the fault-injection experiment",
        "the released code injects prior-tier boxes only during training; the "
        "inference-time path exists in detector.py but is entirely commented out and "
        "referenced unpublished paths. It is off for every other experiment, so no "
        "main-table number depends on it",
        "03_evaluate_and_build_assets")
'''),
        ("code", '''\
# ---- Error analysis and the qualitative figures (need the images themselves) ----
if HAS_GPU and FULL_WEIGHTS:
    predictions_path = os.path.join(setup_env.RUNS_DIR, "predictions", "Ours_full_test.json")
    if not os.path.exists(predictions_path):
        eval_utils.dump_predictions(FULL_WEIGHTS, CFG["diagnosis"], "diagnosis_test", 2,
                                    predictions_path, seed=0, limit=run.eval_limit)
    # Localisation and classification stay separate: a box that overlaps a real
    # tooth but carries the wrong diagnosis is a matched detection with a class
    # disagreement, not a miss plus a false positive.
    analysis = eval_utils.error_analysis(predictions_path, paths["test_diagnosis"], tier=2)
    eval_utils.save_result(manifest.result_name("errors", model="Ours_full"), analysis)
    print(json.dumps(analysis["totals"], indent=2))
    print("\\nerror rate by tier (wrong label among correctly localised boxes):")
    for tier, rate in analysis["error_rate_by_tier"].items():
        print("  tier {}: {:.3f}".format(tier, rate))
    for tier, pairs in analysis["confusion"].items():
        print("  tier {} top confusions: {}".format(tier, list(pairs.items())[:5]))

    figure = figures.grouped_bars(
        ["quadrant", "enumeration", "diagnosis"],
        {"wrong label": [analysis["error_rate_by_tier"][str(t)] for t in range(3)]},
        "fraction of localised boxes with the wrong label",
        "Where errors cluster across the hierarchy",
        width=figures.SINGLE_COLUMN, rotate=0)
    figures.save_figure(figure, "error_clusters",
                        "Per-tier classification error rate among correctly localised "
                        "detections (IoU >= 0.5), full model, test split.",
                        "03_evaluate_and_build_assets", run.mode, "figure:error_clusters",
                        inputs=[predictions_path, paths["test_diagnosis"]])
'''),
        ("code", '''\
# ---- Failure gallery and curated overlays ----
if HAS_GPU and FULL_WEIGHTS:
    with open(paths["test_diagnosis"]) as handle:
        truth = json.load(handle)
    gt_by_image = {}
    for annotation in truth["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation)
    with open(predictions_path) as handle:
        predictions = json.load(handle)
    pred_by_image = {}
    for record in predictions:
        if record.get("score", 0) >= 0.5:
            pred_by_image.setdefault(record["image_id"], []).append(record)
    names = {level: {c["id"]: str(c["name"])
                     for c in truth["categories_{}".format(level + 1)]} for level in range(3)}
    by_id = {image["id"]: image for image in truth["images"]}

    def panel(image_id, title):
        gt = [(a["bbox"], names[2].get(a.get("category_id_3"), ""))
              for a in gt_by_image.get(image_id, [])]
        pred = [(r["bbox"], "{}({:.2f})".format(
            names[2].get(r.get("category_id_3"), r.get("category_id_3")), r.get("score", 0)))
            for r in pred_by_image.get(image_id, [])]
        return {"image_path": os.path.join(paths["img_test"], by_id[image_id]["file_name"]),
                "title": title, "gt": gt, "pred": pred}

    gallery = eval_utils.select_gallery_cases(analysis, per_category=2)
    gallery_ids, panels = {}, []
    for bucket, rows in gallery.items():
        gallery_ids[bucket] = [r["image_id"] for r in rows]
        for row in rows:
            panels.append(panel(row["image_id"], "{}\\n{}".format(bucket, row["file_name"])))
    if panels:
        figure = figures.overlay_grid(panels, columns=4, panel_height=2.0,
                                      title="Failure gallery — blue solid: ground truth, "
                                            "orange dashed: prediction")
        figures.save_figure(figure, "failure_gallery",
                            "Representative failures of the full model on the DENTEX "
                            "test split, one row per failure mode.",
                            "03_evaluate_and_build_assets", run.mode, "figure:qualitative",
                            inputs=[predictions_path], note=json.dumps(gallery_ids))

    with open(os.path.join(paths["subsets"], "clean_stress.json")) as handle:
        subsets = json.load(handle)
    curated = subsets["clean"]["image_ids"][:6] + subsets["stress"]["image_ids"][:6]
    panels = [panel(i, "{} ({})".format(
        by_id[i]["file_name"],
        "clean" if i in subsets["clean"]["image_ids"] else "stress"))
        for i in curated if i in by_id]
    figure = figures.overlay_grid(panels, columns=4, panel_height=2.0,
                                  title="Prediction vs ground truth, 12 curated test images")
    figures.save_figure(figure, "qualitative_overlays",
                        "Full-model predictions (orange, dashed) against ground truth "
                        "(blue, solid) on six clean and six stress-subset test images.",
                        "03_evaluate_and_build_assets", run.mode, "figure:qualitative",
                        inputs=[predictions_path], note=json.dumps(curated))
    print(json.dumps(gallery_ids, indent=2))
'''),
        ("md", "## Low-resource benchmark — the CPU half is what a CPU session is for"),
        ("code", '''\
# ---- CPU or GPU inference cost, whichever this session can measure ----
CPU_IMAGES = 20
if FULL_WEIGHTS and HAS_GPU:
    for step in (1, 4):
        output = os.path.join(setup_env.RUNS_DIR, "predictions",
                              "full_gpu_step{}.json".format(step))
        measured = eval_utils.dump_predictions(
            FULL_WEIGHTS, CFG["diagnosis"], "diagnosis_test", 2, output, seed=0,
            device="cuda", sample_step=step, limit=run.eval_limit)
        measured.update(eval_utils.model_size_report(FULL_WEIGHTS))
        eval_utils.save_result(
            manifest.result_name("lowresource", device="cuda", step=step), measured)
        print("step {}: {:.3f} s/image, {:.1f} img/min, peak VRAM {} MB, {} M params".format(
            step, measured["mean_seconds_per_image"], measured["images_per_minute"],
            measured.get("peak_gpu_memory_mb"), measured["parameters_millions"]))
elif FULL_WEIGHTS:
    output = os.path.join(setup_env.RUNS_DIR, "predictions", "full_cpu20.json")
    measured = eval_utils.dump_predictions(
        FULL_WEIGHTS, CFG["diagnosis"], "diagnosis_test", 2, output, seed=0,
        device="cpu", limit=CPU_IMAGES, sample_step=1)
    measured.update(eval_utils.model_size_report(FULL_WEIGHTS))
    eval_utils.save_result(manifest.result_name("lowresource", device="cpu"), measured)
    print("CPU: {:.2f} s/image over {} images".format(
        measured["mean_seconds_per_image"], measured["images"]))
'''),
        ("code", '''\
# ---- CPU vs GPU: same weights, same seed — do the detections agree? ----
agreement = None
cpu_predictions = os.path.join(setup_env.RUNS_DIR, "predictions", "full_cpu20.json")
gpu_predictions = os.path.join(setup_env.RUNS_DIR, "predictions", "full_gpu_step1.json")
if os.path.exists(cpu_predictions) and os.path.exists(gpu_predictions):
    with open(cpu_predictions) as handle:
        cpu_records = json.load(handle)
    with open(gpu_predictions) as handle:
        gpu_records = json.load(handle)
    shared = {r["image_id"] for r in cpu_records} & {r["image_id"] for r in gpu_records}
    matched_count = total = 0
    for image_id in shared:
        left = [r for r in cpu_records if r["image_id"] == image_id and r["score"] >= 0.5]
        right = [r for r in gpu_records if r["image_id"] == image_id and r["score"] >= 0.5]
        for record in left:
            total += 1
            if any(eval_utils._iou(record["bbox"], other["bbox"]) >= 0.5
                   and record.get("category_id_3") == other.get("category_id_3")
                   for other in right):
                matched_count += 1
    agreement = {"images_compared": len(shared), "cpu_detections": total,
                 "matched_on_gpu": matched_count,
                 "agreement_rate": round(matched_count / total, 4) if total else None,
                 "note": "residual disagreement is CPU/GPU kernel nondeterminism plus "
                         "the stochastic noisy-box start"}
    print(json.dumps(agreement, indent=2))
else:
    print("run this notebook once on GPU and once on CPU to get the agreement check")
'''),
        ("md", """\
## Asset build — runs in every session, GPU or not

Everything below reads only `paper_assets/results_raw/`, so a CPU session
regenerates every table and figure in minutes without re-running an experiment.
"""),
        ("code", '''\
# ---- Load every raw result ----
def collect(kind, key):
    out = {}
    for name in manifest.list_results(kind):
        parts = manifest.parse_result_name(name)
        if parts["kind"] != kind:
            continue
        out[parts.get(key, name)] = eval_utils.load_result(name)
    return out

main_results = collect("eval_main", "model")
step_results = collect("steps", "step")
step_latency = collect("steps_latency", "step")
degradation_results = collect("degradation", "condition")
fault_results = collect("fault", "condition")
runtime_results = collect("runtime", "model")
lowresource = collect("lowresource", "device")
trajectories = collect("trajectory", "variant")
subset_results = {}
for name in manifest.list_results("subset"):
    parts = manifest.parse_result_name(name)
    subset_results["{}|{}".format(parts.get("model"), parts.get("subset"))] = \\
        eval_utils.load_result(name)
print({k: len(v) for k, v in {
    "main": main_results, "steps": step_results, "degradation": degradation_results,
    "fault": fault_results, "subsets": subset_results, "runtime": runtime_results,
    "trajectory": trajectories, "lowresource": lowresource}.items()})
'''),
        ("code", '''\
# ---- Tables ----
NB = "03_evaluate_and_build_assets"
written = {}

if main_results:
    written["main_results"] = tables.write_table(
        "main_results", tables.main_results_rows(main_results),
        ["tier", "method"] + list(tables.METRIC_COLUMNS) + ["seeds"],
        "Per-tier detection metrics on the DENTEX test split, mean +/- std over "
        "{} inference seeds (RUN_MODE={}).".format(len(run.eval_seeds), run.mode),
        NB, run.mode, "table:main_results")
    written["original_vs_reproduced"] = tables.write_table(
        "original_vs_reproduced", tables.comparison_rows(main_results),
        ["tier", "method", "metric", "original", "reproduced", "reproduced_std",
         "delta", "delta_pct"],
        "Reproduced values against the originals reported in HierarchicalDet "
        "(MICCAI 2023, Table 1). 'not run' marks rows this run mode did not train.",
        NB, run.mode, "table:original_vs_reproduced",
        note="original column is quoted from the paper, not produced here")
    written["seed_variance"] = tables.write_table(
        "seed_variance", tables.seed_variance_rows(main_results),
        ["method", "tier", "metric", "n_seeds", "mean", "std", "min", "max", "values"],
        "Spread across inference seeds. DiffusionDet denoises from random boxes, so "
        "this is inherent inference variance at fixed weights.",
        NB, run.mode, "table:seed_variance")
    written["per_class_ap"] = tables.write_table(
        "per_class_ap", tables.per_class_rows(main_results),
        ["method", "tier", "class", "AP_mean", "n_seeds", "note"],
        "Per-class AP. OUR EXTENSION: the original paper reports tier-level "
        "aggregates only, so there is no reference column.",
        NB, run.mode, "table:main_results")
else:
    for asset_class in ("table:main_results", "table:original_vs_reproduced",
                        "table:seed_variance"):
        tables.record_not_run(asset_class, NB, run.mode, "no evaluation results found")

for name, payloads, axis, asset_class, caption in (
    ("step_sweep", step_results, "sampling_steps", "table:step_sweep",
     "Accuracy against diffusion sampling steps, full model."),
    ("degradation", degradation_results, "condition", "table:degradation",
     "Accuracy under image degradation, full model, test split."),
    ("clean_vs_stress", subset_results, "model_subset", "table:clean_vs_stress",
     "Clean versus stress evaluation subsets (selection rule in the dataset audit)."),
    ("fault_injection", fault_results, "condition", "table:fault_injection",
     "Diagnosis-tier accuracy when the prior-tier boxes feeding noisy-box "
     "manipulation are jittered or dropped."),
):
    if not payloads:
        tables.record_not_run(asset_class, NB, run.mode, "experiment not run in this mode")
        continue
    extra = None
    columns = [axis, "tier"] + list(tables.METRIC_COLUMNS)
    if name == "step_sweep" and step_latency:
        extra = {k: {"seconds_per_image":
                     (step_latency.get(k) or {}).get("mean_seconds_per_image")}
                 for k in payloads}
        columns.append("seconds_per_image")
    written[name] = tables.write_table(name, tables.sweep_rows(payloads, axis, extra),
                                       columns, caption, NB, run.mode, asset_class)

if runtime_results:
    written["failure_counts"] = tables.write_table(
        "failure_counts", tables.failure_rows(runtime_results),
        ["method", "images", "detections", "images_with_no_detections", "degenerate_box",
         "box_out_of_image", "box_covers_whole_image", "extreme_aspect_ratio", "crashes"],
        "Inference failure accounting on the test split.",
        NB, run.mode, "table:failure_counts")
else:
    tables.record_not_run("table:failure_counts", NB, run.mode, "no runtime results")

low_rows = [{"method": "Ours_full", **payload} for payload in lowresource.values() if payload]
if low_rows:
    written["low_resource"] = tables.write_table(
        "low_resource", tables.low_resource_rows(low_rows),
        ["method", "device", "sample_step", "seconds_per_image", "images_per_minute",
         "peak_gpu_memory_mb", "parameters_millions", "checkpoint_mb", "images"],
        "Low-resource benchmark: CPU and GPU inference cost of the full model.",
        NB, run.mode, "table:low_resource")
else:
    tables.record_not_run("table:low_resource", NB, run.mode, "no benchmark measured yet")
'''),
        ("code", '''\
# ---- Figures ----
if step_results:
    series = []
    ordered_steps = sorted(step_results, key=lambda s: int(s))
    for tier in tables.TIER_ORDER:
        points = [(float(s), (step_results[s]["aggregate"].get(tier, {}).get("AP", {}) or {}))
                  for s in ordered_steps]
        points = [(x, c["mean"], c.get("std") or 0.0) for x, c in points
                  if c.get("mean") is not None]
        if points:
            series.append({"label": tier, "x": [p[0] for p in points],
                           "y": [p[1] for p in points], "yerr": [p[2] for p in points]})
    latency = {int(k): (v or {}).get("mean_seconds_per_image")
               for k, v in step_latency.items() if v}
    figure = figures.ap_vs_steps_figure(series, latency or None)
    figures.save_figure(figure, "ap_vs_steps",
                        "AP against diffusion sampling steps for the full model, with "
                        "measured per-image latency on the right axis. Error bars are "
                        "the std over inference seeds.",
                        NB, run.mode, "figure:ap_vs_steps")
else:
    figures.record_not_run("figure:ap_vs_steps", NB, run.mode, "step sweep not run")

if degradation_results:
    baseline = ((degradation_results.get("clean") or {}).get("aggregate", {})
                .get("diagnosis", {}).get("AP", {}) or {}).get("mean")
    series = []
    for kind in degradations.KINDS:
        points = []
        for payload in degradation_results.values():
            condition = payload.get("condition") or {}
            if condition.get("kind") != kind:
                continue
            cell = payload["aggregate"]["diagnosis"]["AP"]
            points.append((float(condition["severity"]), cell["mean"], cell.get("std") or 0.0))
        if points:
            points.sort()
            series.append({"label": kind, "x": [p[0] for p in points],
                           "y": [p[1] for p in points], "yerr": [p[2] for p in points]})
    figure = figures.sweep_figure(series, "severity (blur sigma / JPEG quality / scale)",
                                  "diagnosis AP [0.5:0.95]",
                                  "Robustness to image degradation",
                                  width=figures.DOUBLE_COLUMN * 0.6)
    figures.save_figure(figure, "degradation_curves",
                        "Diagnosis-tier AP under Gaussian blur, JPEG recompression and "
                        "downscaling. Severity axes are per-type and not comparable "
                        "across lines; the clean baseline is AP={}.".format(baseline),
                        NB, run.mode, "figure:degradation")
else:
    figures.record_not_run("figure:degradation", NB, run.mode, "degradation grid not run")

if fault_results:
    series = []
    for axis in ("jitter", "drop"):
        points = []
        for payload in fault_results.values():
            condition = payload.get("condition") or {}
            if condition.get("axis") != axis:
                continue
            cell = payload["aggregate"]["diagnosis"]["AP"]
            points.append((condition.get(axis, 0.0), cell["mean"], cell.get("std") or 0.0))
        if points:
            points.sort()
            series.append({"label": "prior-tier {}".format(axis),
                           "x": [p[0] for p in points], "y": [p[1] for p in points],
                           "yerr": [p[2] for p in points]})
    figure = figures.sweep_figure(series, "perturbation magnitude",
                                  "diagnosis AP [0.5:0.95]",
                                  "Sensitivity to an imperfect prior tier")
    figures.save_figure(figure, "fault_injection",
                        "Diagnosis-tier AP as the enumeration model's detections are "
                        "jittered (normalized-coordinate sigma) or randomly dropped "
                        "before being used as noisy boxes.",
                        NB, run.mode, "figure:fault_injection")
else:
    figures.record_not_run("figure:fault_injection", NB, run.mode, "fault injection not run")
'''),
        ("code", '''\
# ---- Per-class bars and the checkpoint-trajectory figure ----
per_class = [r for r in tables.per_class_rows(main_results)
             if r["tier"] in ("enumeration", "diagnosis")]
if per_class:
    for tier in ("enumeration", "diagnosis"):
        rows = [r for r in per_class if r["tier"] == tier]
        if not rows:
            continue
        classes = sorted({r["class"] for r in rows})
        groups = {method: [next((r["AP_mean"] for r in rows
                                 if r["method"] == method and r["class"] == klass), 0.0)
                           for klass in classes]
                  for method in sorted({r["method"] for r in rows})}
        figure = figures.grouped_bars(classes, groups, "AP [0.5:0.95]",
                                      "Per-class AP — {} tier (our extension)".format(tier),
                                      rotate=45 if tier == "diagnosis" else 90)
        figures.save_figure(figure, "per_class_ap_{}".format(tier),
                            "Per-class AP at the {} tier. OUR EXTENSION: the original "
                            "paper reports tier-level aggregates only.".format(tier),
                            NB, run.mode, "figure:per_class_ap")
else:
    figures.record_not_run("figure:per_class_ap", NB, run.mode, "no evaluation results")

ordering, stable = {}, {}
if trajectories:
    shaped = {}
    for variant, payload in trajectories.items():
        per_tier = {}
        for point in payload.get("points", []):
            aggregate = point.get("aggregate")
            tiers = (point.get("result") or {}).get("tiers", {})
            for tier in tables.TIER_ORDER:
                if aggregate:
                    cell = aggregate.get(tier, {}).get("AP", {})
                    value, spread = cell.get("mean"), cell.get("std")
                else:
                    value, spread = tiers.get(tier, {}).get("metrics", {}).get("AP"), 0.0
                if value is not None:
                    per_tier.setdefault(tier, []).append(
                        {"progress": point["progress"], "AP": value, "AP_std": spread})
        for tier in per_tier:
            per_tier[tier].sort(key=lambda p: p["progress"])
        shaped[setup_env.VARIANT_LABELS.get(variant, variant)] = per_tier
    figure = figures.trajectory_figure(shaped, list(tables.TIER_ORDER))
    figures.save_figure(figure, "checkpoint_trajectory",
                        "Per-tier AP against training progress for each diagnosis "
                        "variant. Stable ordering across checkpoints is evidence the "
                        "ablation conclusion is not an artifact of the shortened "
                        "schedule; unstable ordering is itself a finding.",
                        NB, run.mode, "figure:checkpoint_trajectory")
    for tier in tables.TIER_ORDER:
        by_progress = {}
        for variant, per_tier in shaped.items():
            for point in per_tier.get(tier, []):
                by_progress.setdefault(point["progress"], []).append((variant, point["AP"]))
        ordering[tier] = {str(p): [v for v, _ in sorted(vals, key=lambda kv: -kv[1])]
                          for p, vals in sorted(by_progress.items())}
    stable = {tier: len({tuple(order) for order in per_progress.values()}) == 1
              for tier, per_progress in ordering.items()}
    print(json.dumps({"ordering": ordering, "ordering_stable": stable}, indent=2))
else:
    figures.record_not_run("figure:checkpoint_trajectory", NB, run.mode,
                           "no trajectory checkpoints evaluated")
'''),
        ("code", '''\
# ---- repro_checklist.md ----
environment = manifest.load_manifest().get("environment", {})
data_summary = setup_env.read_notebook_summary("01_setup_and_data") or {}
records = [r for r in (training.get("records") or []) if r]

lines = ["# Reproducibility checklist", "",
         "Generated from executed-run records only.", "",
         "## Environment", "",
         "- repo commit: `{}`".format(environment.get("repo_commit")),
         "- python: {}".format(environment.get("python")),
         "- platform: {}".format(environment.get("platform")),
         "- GPU: {}".format(json.dumps(environment.get("gpu", {}).get("devices", []))),
         "- CUDA / cuDNN: {} / {}".format(environment.get("gpu", {}).get("cuda"),
                                          environment.get("gpu", {}).get("cudnn")),
         "- run mode: **{}**".format(run.mode),
         "- training seed: {} (the repo's own SEED)".format(setup_env.BASE_SEED),
         "- inference seeds: {}".format(list(run.eval_seeds)),
         "- multi-GPU: {}".format(json.dumps(training.get("multi_gpu", {}))), "",
         "## Converted dataset", ""]
for name, digest in (data_summary.get("dataset_hashes") or {}).items():
    lines.append("- `{}`: `{}`".format(name, digest))

lines += ["", "## Runs", "",
          "| run | config hash | iterations | seed | batch | wall (s) | GPUs | stopped on budget |",
          "|---|---|---|---|---|---|---|---|"]
for record in records:
    lines.append("| {} | `{}` | {} | {} | {} | {} | {} | {} |".format(
        record.get("name"), (record.get("config_hash") or "n/a")[:12],
        record.get("max_iter"), record.get("seed"), record.get("ims_per_batch"),
        record.get("wall_seconds"), record.get("num_gpus"),
        record.get("stopped_on_time_budget")))

lines += ["", "## Exact commands", ""]
for record in records:
    command = (record.get("launch") or {}).get("command")
    if command:
        lines.append("```\\n{}\\n```".format(" ".join(str(c) for c in command)))

lines += ["", "## Remaining nondeterminism", "",
          "- mixed-precision (AMP) reduction order during training",
          "- atomics in the torchvision ROIAlign / NMS CUDA kernels",
          "- DataLoader worker interleaving (NUM_WORKERS=2)",
          "- inference starts from random noisy boxes, which is why every reported "
          "number is a mean over {} inference seeds".format(len(run.eval_seeds)), ""]

checklist = os.path.join(setup_env.PAPER_ASSETS, "repro_checklist.md")
with open(checklist, "w") as handle:
    handle.write("\\n".join(lines) + "\\n")
manifest.record_asset(checklist, "doc:repro_checklist", NB, run.mode)
manifest.record_asset(setup_env.DEVIATIONS_MD, "doc:deviations", NB, run.mode)
print("\\n".join(lines[:35]))
'''),
        ("code", '''\
# ---- scope_and_claims.md: the claim-coverage matrix ----
tested_models = set(main_results)
coverage = []
for claim in tables.PAPER_CLAIMS:
    status, evidence = claim["default_status"], claim["evidence"]
    if claim.get("requires_models"):
        missing = [m for m in claim["requires_models"] if m not in tested_models]
        if missing:
            status = "cited, untested"
            evidence = "not trained in RUN_MODE={} (missing: {})".format(
                run.mode, ", ".join(missing))
    coverage.append({**claim, "status": status, "evidence": evidence})

gpu_hours = setup_env.gpu_hours_spent()
lines = ["# Scope, claims and what a reviewer can re-verify", "",
         "Run mode: **{}**. Measured GPU-hours actually spent: **{:.2f}**."
         .format(run.mode, gpu_hours), "",
         "## Claim coverage", "",
         "| # | Claim in the original paper | Status | Evidence |", "|---|---|---|---|"]
for index, claim in enumerate(coverage, 1):
    lines.append("| {} | {} | {} | {} |".format(
        index, claim["claim"], claim["status"], claim["evidence"]))

lines += ["", "## Compute actually spent", "", "| run | wall (s) | GPUs |", "|---|---|---|"]
for record in records:
    lines.append("| {} | {} | {} |".format(
        record.get("name"), record.get("wall_seconds"), record.get("num_gpus")))

lines += ["", "## Re-verifying without retraining", "",
          "Attach `{}` and `{}`, open `notebooks/03_evaluate_and_build_assets.ipynb` "
          "with Accelerator = GPU T4, set `RUN_MODE = \\"{}\\"`, and Run All."
          .format(CKPT_DATASET_SLUG, DATA_DATASET_SLUG, run.mode), "",
          "That reproduces every number in `tables/main_results.csv` from the released "
          "checkpoints, with no training. Re-running the same notebook on a CPU session "
          "rebuilds every table and figure for free.", "",
          "## Not a clinical claim", "",
          "Every artifact here is a research artifact produced under a constrained "
          "compute budget. None of it is validated for, or intended for, clinical use.",
          ""]
scope = os.path.join(setup_env.PAPER_ASSETS, "scope_and_claims.md")
with open(scope, "w") as handle:
    handle.write("\\n".join(lines) + "\\n")
manifest.record_asset(scope, "doc:scope_and_claims", NB, run.mode)
print("\\n".join(lines[:30]))
'''),
        ("code", '''\
# ---- Contract check: nothing may be cited that is not in the manifest ----
contract = manifest.assert_asset_classes()
print(json.dumps(contract, indent=2))
print()
for row in manifest.summary_table():
    print("{:34s} {:8s} {}".format(row["asset_class"], row["status"], row["path"]))
'''),
        ("code", summary_cell("03_evaluate_and_build_assets", '''\
summary = {
    "run_mode": run.mode,
    "had_gpu": HAS_GPU,
    "eval_seeds": list(run.eval_seeds),
    "models_evaluated": sorted(main_results),
    "aggregate": {k: v.get("aggregate") for k, v in main_results.items()},
    "cpu_gpu_agreement": agreement,
    "tables": written,
    "contract": contract,
    "ablation_ordering": ordering,
    "ablation_ordering_stable": stable,
    "gpu_hours_spent": gpu_hours,
    "manifest": manifest.MANIFEST_PATH,
}
''')),
        ("code", '''\
# ---- Publish, last ----
# Round-trips paper_assets/ back into the checkpoint dataset so the next
# session (GPU or CPU) starts with every result this one produced.
publish = {"status": "disabled"}
if PUBLISH_KAGGLE_DATASET:
    publish = train_utils.publish_kaggle_dataset(
        CKPT_DATASET_SLUG, [setup_env.RUNS_DIR, setup_env.PAPER_ASSETS],
        "evaluation + paper assets ({} mode)".format(run.mode))
    summary["kaggle_publish"] = {k: v for k, v in publish.items()
                                 if k not in ("stdout", "stderr")}
    setup_env.write_notebook_summary("03_evaluate_and_build_assets", summary)
print(json.dumps({k: v for k, v in publish.items() if k not in ("stdout", "stderr")},
                 indent=2))
print("\\npaper_assets/ is at:", setup_env.PAPER_ASSETS)
'''),
    ]


# --------------------------------------------------------------------------
# Optional extras
# --------------------------------------------------------------------------
def notebook_opt():
    return [
        ("md", """\
# opt — Baselines and SimMIM pretraining (OPTIONAL, outside the main sequence)

Neither part is needed for the reproduction's central claim, and nothing in
`paper_assets/` depends on this notebook. Both are opt-in via the flags in the
first code cell.

**Baselines** (RetinaNet / Faster R-CNN): single-tier detectors trained on the
flattened tier data. The repo ships no baseline configs, so these reproduce the
*comparison*, not the authors' specific baseline runs. ~1 GPU-h each.

**SimMIM pretraining**: the paper pretrains the Swin backbone on the 1,571
unlabelled DENTEX X-rays, but that checkpoint is not public and the code lives
in a separate repository. The default everywhere else is to skip it and
initialize from public ImageNet-22k Swin-B — which is what the repo's own
`nonpretrain` config does. Running this closes that deviation. 10+ GPU-h.

To use a SimMIM checkpoint afterwards, point notebook 02's `IMAGENET_WEIGHTS` at
it and retrain from scratch — partial adoption would confound every variant.
"""),
        ("code", PARAMS),
        ("code", environment_cell(require_gpu=True)),
        ("code", '''\
# ---- Opt-in flags ----
from src import data_convert

RUN_BASELINES = False
RUN_SIMMIM = False
BASELINE_MODELS = ("retinanet", "faster_rcnn")
BASELINE_TIERS = (2,)
BASELINE_MAX_ITER = 20000 if run.mode == "full" else 10000
SIMMIM_EPOCHS = 100

paths = data_convert.layout()
print("baselines:", RUN_BASELINES, "| SimMIM:", RUN_SIMMIM)
'''),
        ("code", '''\
# ---- Baselines, via the existing tools/baselines driver ----
baseline_records = []
if RUN_BASELINES:
    for model in BASELINE_MODELS:
        for tier in BASELINE_TIERS:
            output = os.path.join(setup_env.RUNS_DIR,
                                  "baseline_{}_tier{}".format(model, tier))
            command = [sys.executable, os.path.join("tools", "baselines", "train_baseline.py"),
                       "--model", model,
                       "--train-json", data_convert.flat_json_path(tier, "train"),
                       "--train-images", paths["img_diagnosis"],
                       "--test-json", data_convert.flat_json_path(tier, "test"),
                       "--test-images", paths["img_test"],
                       "--output-dir", output, "--tier", "0",
                       "--max-iter", str(BASELINE_MAX_ITER),
                       "--seed", str(setup_env.BASE_SEED)]
            print("$", " ".join(command))
            result = subprocess.run(command, cwd=setup_env.REPO_ROOT)
            baseline_records.append({"model": model, "tier": tier, "output_dir": output,
                                     "returncode": result.returncode,
                                     "command": " ".join(command)})
            if result.returncode != 0:
                raise RuntimeError("baseline {} tier {} failed".format(model, tier))
    setup_env.log_deviation(
        "baselines are our own training runs, not the authors' configurations",
        "the repository ships no RetinaNet / Faster R-CNN / DETR config, so these "
        "reproduce the comparison rather than the authors' specific baseline runs",
        "opt_baselines_and_simmim")
print(json.dumps(baseline_records, indent=2))
'''),
        ("code", '''\
# ---- SimMIM pretraining on the 1,571 unlabelled X-rays ----
simmim = {"ran": False}
if RUN_SIMMIM:
    unlabelled = paths["img_unlabelled"]
    if not (os.path.isdir(unlabelled) and os.listdir(unlabelled)):
        data_convert.download_and_extract(
            os.path.join(setup_env.PROJECT_ROOT, "dentex_raw"),
            include_unlabelled=True, delete_archives=True)
    print("unlabelled images:", len(os.listdir(unlabelled)))

    clone = os.path.join(setup_env.PROJECT_ROOT, "Swin-Transformer")
    if not os.path.isdir(os.path.join(clone, ".git")):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/microsoft/Swin-Transformer.git", clone],
                       check=True)
    output = os.path.join(setup_env.RUNS_DIR, "simmim_pretrain")
    os.makedirs(output, exist_ok=True)
    command = [sys.executable, os.path.join(clone, "main_simmim.py"),
               "--cfg", os.path.join(clone, "configs", "simmim",
                                     "simmim_pretrain__swin_base__img192_window6__100ep.yaml"),
               "--data-path", os.path.dirname(unlabelled),
               "--batch-size", "16", "--output", output,
               "--opts", "TRAIN.EPOCHS", str(SIMMIM_EPOCHS)]
    print("$", " ".join(command))
    result = subprocess.run(command, cwd=clone)
    simmim = {"ran": True, "returncode": result.returncode, "output_dir": output,
              "command": " ".join(command), "epochs": SIMMIM_EPOCHS}
    if result.returncode == 0:
        setup_env.log_deviation(
            "SimMIM pretraining WAS performed in this run",
            "the optional notebook was executed, so backbone initialization follows the "
            "paper's description rather than the released nonpretrain config",
            "opt_baselines_and_simmim")
print(json.dumps(simmim, indent=2))
'''),
        ("code", summary_cell("opt_baselines_and_simmim", '''\
summary = {"run_mode": run.mode, "baselines": baseline_records, "simmim": simmim}
''')),
    ]


NOTEBOOKS = {
    "01_setup_and_data": notebook_01,
    "02_train_all": notebook_02,
    "03_evaluate_and_build_assets": notebook_03,
    "opt_baselines_and_simmim": notebook_opt,
}


def build(name, cells):
    payload = {
        "cells": [
            {"cell_type": kind,
             "metadata": {},
             "source": source.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, source in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    path = os.path.join(NOTEBOOKS_DIR, name + ".ipynb")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    return path


def main():
    import ast

    for stale in sorted(os.listdir(NOTEBOOKS_DIR)) if os.path.isdir(NOTEBOOKS_DIR) else []:
        if stale.endswith(".ipynb") and stale[:-6] not in NOTEBOOKS:
            os.remove(os.path.join(NOTEBOOKS_DIR, stale))
            print("removed stale notebook {}".format(stale))
    for name, factory in NOTEBOOKS.items():
        cells = factory()
        for kind, source in cells:
            if kind == "code":
                ast.parse(source)          # a generated notebook must at least parse
        path = build(name, cells)
        print("wrote {} ({} cells)".format(path, len(cells)))


if __name__ == "__main__":
    main()
