"""
Training orchestration: budgets, calibration, resume, the ablation matrix, and
Kaggle-session persistence.

The single idea this module exists to enforce: **the three diagnosis-stage
ablation variants must differ only in the two switches under test.** Same data,
same seed, same iteration count, same hardware. Anything else and the ablation
comparison is measuring the wrong thing, so :func:`assert_matched_budgets`
aborts rather than letting an unmatched comparison reach a table.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

from . import registration, setup_env
from .setup_env import VARIANT_SWITCHES, StageBudget

TRAIN_ENTRY = os.path.join(setup_env.SRC_DIR, "train_entry.py")

#: Public ImageNet-22k Swin-B. The `.pth` extension is load-bearing:
#: detectron2's DetectionCheckpointer dispatches on extension and would parse a
#: `.pkl` as a Caffe2 blob, silently leaving the backbone random.
SWIN_B_URL = ("https://github.com/SwinTransformer/storage/releases/download/"
              "v1.0.0/swin_base_patch4_window7_224_22k.pth")
SWIN_B_FILENAME = "swin_base_patch4_window7_224_22k.pth"

#: Checkpoint cadence asked for by the runbook (survives a session kill).
CHECKPOINT_PERIOD = 2500

#: Refuse to keep training below this much free space. A checkpoint is
#: ~2.0 GB, so this leaves room to finish the one in flight.
DISK_FLOOR_GB = 3.0


def checkpoint_period(max_iter: int) -> int:
    """
    Checkpoint often enough that a session kill cannot cost a whole stage.

    A flat 2,500 was fine against the paper's 30k–40k schedules, but measured
    throughput on Kaggle T4 x2 is 9.13 s/iter, so an hour-capped stage is only
    ~600 iterations — the periodic checkpointer would never fire once, and a
    session killed at 95% of a stage would restart it from zero. Cap the period
    at a quarter of the run so there are always ~4 resume points.
    """
    return max(25, min(CHECKPOINT_PERIOD, int(max_iter) // 4))

#: Throughput calibration is bounded by **wall time, not iteration count**.
#:
#: A fixed 500-iteration probe was measured at 9.13 s/iter on Kaggle's T4 x2
#: (Swin-B, 1000 proposals, batch 2) — that is 76 minutes, or 72% of the entire
#: 1.75 h quadrant budget, spent measuring instead of training. A time bound
#: costs the same on any hardware: fast GPUs contribute more samples, slow ones
#: fewer, and neither overruns.
CALIBRATION_SECONDS = 300
#: Iterations discarded before measuring: the first step pays CUDA autotuning
#: and dataloader spin-up (measured data_time 10.5 s on iteration 1, then
#: 0.005 s), so a handful is enough.
CALIBRATION_WARMUP = 5
#: Upper bound on probe iterations; the wall-clock hook is what actually stops
#: it, so this only needs to be unreachable on plausible hardware.
CALIBRATION_MAX_ITERS = 100000


# --------------------------------------------------------------------------
# Backbone weights
# --------------------------------------------------------------------------
def ensure_swin_weights(models_dir: Optional[str] = None) -> str:
    """Download the public Swin-B ImageNet-22k checkpoint once."""
    models_dir = models_dir or os.path.join(setup_env.REPO_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, SWIN_B_FILENAME)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path

    import urllib.request

    tmp = path + ".part"
    urllib.request.urlretrieve(SWIN_B_URL, tmp)
    size = os.path.getsize(tmp)
    if size < 1_000_000:
        os.remove(tmp)
        raise RuntimeError(
            "Swin-B download produced {} bytes — that is an error page, not a "
            "checkpoint.".format(size)
        )
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------
def run_dir(name: str) -> str:
    path = os.path.join(setup_env.RUNS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def final_weights(name: str) -> str:
    return os.path.join(run_dir(name), "model_final.pth")


def is_complete(name: str) -> bool:
    """A run is complete iff it wrote both its final weights and its record."""
    directory = run_dir(name)
    return (os.path.exists(os.path.join(directory, "model_final.pth"))
            and os.path.exists(os.path.join(directory, "run_record.json")))


def read_run_record(name: str) -> Optional[Dict[str, object]]:
    path = os.path.join(run_dir(name), "run_record.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def disk_free_gb(path: str = None) -> float:
    """Free space on the filesystem holding the run outputs."""
    usage = shutil.disk_usage(path or setup_env.RUNS_ROOT)
    return usage.free / (1024 ** 3)


def slim_checkpoint(path: str) -> Dict[str, object]:
    """
    Strip the optimizer state out of a checkpoint, keeping only model weights.

    A Swin-B + DiffusionDet checkpoint is ~2.0 GB, and **two thirds of that is
    AdamW's two moment tensors**. Those are needed only to *resume* a run — a
    completed stage is never resumed, and weight transfer and evaluation both
    load model weights alone. Keeping them cost a real run its whole session:
    six checkpoints across three stages filled ``/kaggle/working`` (~20 GB) and
    training died at iteration 899 of 900 with ``No space left on device``.

    Only ever call this on checkpoints that are final for their purpose
    (``model_final``, trajectory snapshots). Numbered checkpoints of a run still
    in flight must keep their optimizer state or the resume is worthless.
    """
    import torch

    before = os.path.getsize(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        return {"path": path, "skipped": "not a checkpoint dict"}
    if "optimizer" not in checkpoint and "scheduler" not in checkpoint:
        return {"path": path, "skipped": "already slim", "bytes": before}
    slim = {"model": checkpoint["model"]}
    for key in ("iteration", "matching_heuristics"):
        if key in checkpoint:
            slim[key] = checkpoint[key]
    temporary = path + ".slim"
    torch.save(slim, temporary)
    os.replace(temporary, path)
    after = os.path.getsize(path)
    return {"path": path, "bytes_before": before, "bytes_after": after,
            "freed_gb": round((before - after) / (1024 ** 3), 2)}


def prune_checkpoints(name: str, keep: Sequence[str] = (),
                      complete: bool = False) -> Dict[str, object]:
    """
    Reclaim disk after a stage.

    ``complete=True`` (the stage produced ``model_final.pth``) is aggressive:
    every numbered checkpoint goes, because a finished run is never resumed,
    and ``model_final`` plus the trajectory snapshots are slimmed to weights
    only. That takes a finished stage from ~4 GB to ~0.7 GB.

    Otherwise the newest numbered checkpoint is kept intact — it is the resume
    point for an interrupted run, and it needs its optimizer state.
    """
    directory = run_dir(name)
    numbered = sorted(glob.glob(os.path.join(directory, "model_[0-9]*.pth")))
    removed, slimmed = [], []

    survivors = numbered if complete else numbered[:-1]
    for path in survivors:
        os.remove(path)
        removed.append(os.path.basename(path))

    if complete:
        finals = [os.path.join(directory, "model_final.pth")]
        finals += [os.path.join(directory, k + ".pth") for k in keep]
        for path in finals:
            if os.path.exists(path):
                slimmed.append(slim_checkpoint(path))
        # last_checkpoint points at a file that may no longer exist; a completed
        # run is skipped rather than resumed, so it is dead weight either way.
        marker = os.path.join(directory, "last_checkpoint")
        if os.path.exists(marker):
            os.remove(marker)

    return {"removed": removed, "slimmed": slimmed,
            "disk_free_gb": round(disk_free_gb(), 2)}


# --------------------------------------------------------------------------
# Cross-session resume
# --------------------------------------------------------------------------
def locate_weights(run_name: str, filename: str = "model_final.pth",
                   search_roots: Sequence[str] = ()) -> Optional[str]:
    """
    Find a run's weights, whether they were produced here or arrived attached.

    Notebook 02 records absolute paths under ``/kaggle/working/runs/...``. A
    later session starts with that directory empty and the checkpoints mounted
    read-only under ``/kaggle/input/``, so those recorded paths do not resolve
    and every model silently drops out of the evaluation.

    Returns the **attached path directly** rather than copying. Evaluation only
    reads weights, and the checkpoint set is several GB -- copying it in would
    reintroduce the disk pressure that has already cost three sessions.
    """
    local = os.path.join(run_dir(run_name), filename)
    if os.path.exists(local) and checkpoint_is_readable(local):
        return local

    roots = list(search_roots) or (
        sorted(glob.glob(os.path.join(setup_env.KAGGLE_INPUT, "*")))
        if setup_env.ON_KAGGLE else []
    )
    mode = setup_env.ACTIVE_MODE
    for root in roots:
        for pattern in (os.path.join(root, "**", "runs", mode, run_name, filename),
                        os.path.join(root, "**", mode, run_name, filename)):
            for candidate in sorted(glob.glob(pattern, recursive=True)):
                if checkpoint_is_readable(candidate):
                    return candidate
    return None


def checkpoint_is_readable(path: str) -> bool:
    """
    Cheap integrity check for a ``.pth`` checkpoint.

    A torch checkpoint is a zip archive, so a write cut short by a full disk or
    a killed session leaves a file whose central directory (written last) is
    missing. ``torch.load`` then dies with "PytorchStreamReader ... failed
    finding central directory" -- but only after the caller has already spent
    the time getting there. ``zipfile.is_zipfile`` reads the end-of-archive
    record, so it rejects a truncated file in milliseconds without loading
    2 GB of tensors.
    """
    import zipfile

    try:
        return os.path.getsize(path) > 1024 and zipfile.is_zipfile(path)
    except OSError:
        return False


def purge_corrupt_checkpoints(name: str) -> List[str]:
    """
    Delete unreadable checkpoints from a run directory and repair
    ``last_checkpoint`` if it names one of them.

    detectron2 resumes from whatever ``last_checkpoint`` points at, so a single
    truncated file left by an interrupted session poisons every later attempt
    until it is removed.
    """
    directory = run_dir(name)
    removed = []
    for path in sorted(glob.glob(os.path.join(directory, "*.pth"))):
        if not checkpoint_is_readable(path):
            os.remove(path)
            removed.append(os.path.basename(path))

    marker = os.path.join(directory, "last_checkpoint")
    if os.path.exists(marker):
        with open(marker) as handle:
            target = handle.read().strip()
        if not os.path.exists(os.path.join(directory, os.path.basename(target))):
            survivors = sorted(glob.glob(os.path.join(directory, "model_[0-9]*.pth")),
                               key=_iteration_of)
            if survivors:
                with open(marker, "w") as handle:
                    handle.write(os.path.basename(survivors[-1]))
            else:
                os.remove(marker)
    if removed:
        print("[{}] removed truncated checkpoint(s): {}".format(name, removed))
    return removed


def _iteration_of(path: str) -> int:
    match = re.search(r"model_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def seed_from_attached_datasets(name: str,
                                search_roots: Sequence[str] = ()) -> Optional[str]:
    """
    Copy the newest checkpoint for ``name`` out of an attached Kaggle Dataset
    into this session's run directory, so ``--resume`` picks up where the last
    session stopped.

    Kaggle datasets mount read-only under ``/kaggle/input/<slug>/``; the layout
    inside is whatever ``publish_kaggle_dataset`` uploaded, i.e. ``runs/<name>/``.
    """
    roots = list(search_roots) or (
        sorted(glob.glob(os.path.join(setup_env.KAGGLE_INPUT, "*")))
        if setup_env.ON_KAGGLE else []
    )
    destination = run_dir(name)
    if os.path.exists(os.path.join(destination, "model_final.pth")):
        return None

    # Only ever resume from a checkpoint produced under the SAME run mode: run
    # directories are mode-scoped precisely so a 200-iteration smoke checkpoint
    # can never be picked up as the starting point of a real run.
    mode = setup_env.ACTIVE_MODE
    candidates: List[str] = []
    for root in roots:
        candidates.extend(glob.glob(
            os.path.join(root, "**", "runs", mode, name, "model_*.pth"), recursive=True))
        candidates.extend(glob.glob(
            os.path.join(root, "**", mode, name, "model_*.pth"), recursive=True))
    # The two globs above overlap, so the same file can appear twice.
    candidates = sorted(set(os.path.abspath(c) for c in candidates))
    if not candidates:
        return None

    # Pick the newest checkpoint that is actually READABLE. A crash mid-save
    # (a full disk, a killed session) leaves a truncated .pth behind, and the
    # newest file is exactly the one most likely to be the casualty: a run that
    # died writing model_0000899.pth left it half-written, and resuming from it
    # failed with "PytorchStreamReader ... failed finding central directory"
    # after the session had already spent an hour getting there.
    finals = [c for c in candidates
              if c.endswith("model_final.pth") and checkpoint_is_readable(c)]
    numbered = sorted((c for c in candidates if not c.endswith("model_final.pth")),
                      key=_iteration_of, reverse=True)
    rejected = [os.path.basename(c) for c in candidates if not checkpoint_is_readable(c)]
    if rejected:
        print("[{}] ignoring truncated checkpoint(s) from an interrupted "
              "session: {}".format(name, rejected))
    best = finals[0] if finals else next(
        (c for c in numbered if checkpoint_is_readable(c)), None)
    if best is None:
        print("[{}] no readable checkpoint in the attached datasets; "
              "training from scratch".format(name))
        return None

    local = os.path.join(destination, os.path.basename(best))
    if not os.path.exists(local):
        shutil.copy2(best, local)
    # detectron2 resumes from whatever `last_checkpoint` names.
    with open(os.path.join(destination, "last_checkpoint"), "w") as handle:
        handle.write(os.path.basename(best))
    # Attached datasets are read-only under /kaggle/input and cost nothing
    # against the ~20 GB working quota -- but this copy lands in /kaggle/working
    # and does. A finished stage restored from an older session still carries
    # its optimizer state (~2/3 of ~2.0 GB), which nothing downstream needs, so
    # slim it on arrival rather than paying for it all session.
    if best.endswith("model_final.pth"):
        print("[{}] restored from attached dataset; {}".format(
            name, slim_checkpoint(local)))

    sibling = os.path.join(os.path.dirname(best), "run_record.json")
    if os.path.exists(sibling):
        if best.endswith("model_final.pth"):
            # The stage FINISHED in an earlier session. `is_complete` requires
            # both model_final.pth and run_record.json, so restoring only the
            # weights would leave the stage looking unfinished and retrain it
            # from scratch -- hours of GPU quota to reproduce a checkpoint that
            # was sitting right there.
            target = os.path.join(destination, "run_record.json")
            if not os.path.exists(target):
                shutil.copy2(sibling, target)
        elif not os.path.exists(os.path.join(destination, "run_record.prev.json")):
            # Interrupted run: keep the record for budget accounting, but do
            # NOT make the stage look complete -- it has to resume and finish.
            shutil.copy2(sibling, os.path.join(destination, "run_record.prev.json"))
    return local


def publish_kaggle_dataset(slug: str, directories: Sequence[str], message: str,
                           staging: Optional[str] = None,
                           dry_run: bool = False) -> Dict[str, object]:
    """
    Version a Kaggle Dataset from the given directories using the ``kaggle``
    CLI, authenticating from ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (Kaggle
    Secrets). Never writes credentials anywhere.

    Returns a record including the manual fallback, which is what the runbook
    tells the reader to do if the CLI is unavailable.
    """
    staging = staging or os.path.join(setup_env.KAGGLE_WORKING
                                      if setup_env.ON_KAGGLE else setup_env.PROJECT_ROOT,
                                      "kaggle_upload", slug)
    fallback = ("Save Version -> Save & Run All, then add the produced output as "
                "a Data source on the next notebook (Add Data -> Your Work).")
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    record: Dict[str, object] = {
        "slug": slug, "staging": staging, "message": message,
        "directories": [os.path.abspath(d) for d in directories],
        "manual_fallback": fallback,
    }
    if not (username and key):
        record["status"] = "skipped"
        record["reason"] = ("KAGGLE_USERNAME / KAGGLE_KEY not set — attach them via "
                            "Add-ons -> Secrets, or publish manually.")
        return record

    os.makedirs(staging, exist_ok=True)
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        target = os.path.join(staging, os.path.basename(os.path.normpath(directory)))
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.copytree(directory, target)

    metadata_path = os.path.join(staging, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as handle:
            json.dump({"title": slug, "id": "{}/{}".format(username, slug),
                       "licenses": [{"name": "CC0-1.0"}]}, handle, indent=2)
        command = ["kaggle", "datasets", "create", "-p", staging, "-r", "zip"]
    else:
        command = ["kaggle", "datasets", "version", "-p", staging,
                   "-m", message, "-r", "zip"]
    record["command"] = " ".join(command)
    if dry_run:
        record["status"] = "dry-run"
        return record

    result = subprocess.run(command, capture_output=True, text=True)
    record["status"] = "ok" if result.returncode == 0 else "failed"
    record["stdout"] = result.stdout[-4000:]
    record["stderr"] = result.stderr[-4000:]
    if result.returncode != 0:
        raise RuntimeError(
            "kaggle datasets publish failed (exit {}):\n{}\nFall back to: {}"
            .format(result.returncode, result.stderr.strip(), fallback)
        )
    return record


# --------------------------------------------------------------------------
# Config assembly
# --------------------------------------------------------------------------
def base_overrides(cfg_run, output_dir: str, max_iter: int, weights: str,
                   num_gpus: int, seed: int = setup_env.BASE_SEED,
                   amp: bool = True, extra: Sequence[str] = ()) -> List[str]:
    """
    Command-line overrides shared by every training run.

    ``IMS_PER_BATCH`` is the *total* batch across processes (detectron2 divides
    it by world size), so it is scaled with the GPU count to keep the
    per-GPU batch — and therefore the memory profile — constant.
    ``TEST.EVAL_PERIOD 0`` disables in-training evaluation: every number in the
    paper comes from notebook 05, and mid-run evaluation would spend the
    wall-clock budget that is supposed to buy training iterations.
    """
    overrides = [
        "OUTPUT_DIR", output_dir,
        "MODEL.WEIGHTS", weights,
        "SOLVER.MAX_ITER", str(int(max_iter)),
        "SOLVER.IMS_PER_BATCH", str(cfg_run.ims_per_batch * max(1, num_gpus)),
        "SOLVER.CHECKPOINT_PERIOD", str(checkpoint_period(max_iter)),
        "SOLVER.AMP.ENABLED", "True" if amp else "False",
        "MODEL_EMA.ENABLED", "False",
        "TEST.EVAL_PERIOD", "0",
        "SEED", str(int(seed)),
    ]
    return overrides + [str(x) for x in extra]


def visible_gpus() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


# --------------------------------------------------------------------------
# Launching
# --------------------------------------------------------------------------
def launch_training(config_file: str, overrides: Sequence[str], env_extra: Dict[str, str],
                    output_dir: str, num_gpus: int, resume: bool = True,
                    budget_seconds: float = 0.0,
                    trajectory: Optional[Dict[int, str]] = None,
                    rate_probe: str = "", log_name: str = "train.log",
                    timeout: Optional[float] = None) -> Dict[str, object]:
    """Run ``train_entry.py`` as a subprocess, streaming its log to disk."""
    os.makedirs(output_dir, exist_ok=True)
    command = [sys.executable, TRAIN_ENTRY,
               "--config-file", config_file,
               "--num-gpus", str(max(0, num_gpus))]
    if resume:
        command.append("--resume")
    if budget_seconds:
        command += ["--budget-seconds", str(float(budget_seconds))]
    if trajectory:
        command += ["--trajectory", json.dumps({str(k): v for k, v in trajectory.items()})]
    if rate_probe:
        command += ["--rate-probe", rate_probe,
                    "--rate-probe-warmup", str(CALIBRATION_WARMUP)]
    # Stop cleanly before the volume fills; a crash mid-save leaves a truncated
    # checkpoint that poisons the next resume.
    command += ["--disk-floor-gb", str(DISK_FLOOR_GB)]
    command += ["--heartbeat", os.path.join(output_dir, "heartbeat.json")]
    command += list(overrides)

    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    # Anything not explicitly requested must be *unset*, not inherited: a stale
    # NOISY_BOX_TRAIN from a previous cell would silently turn the
    # "w/o Manipulation" variant back into the full model.
    for key in ("NOISY_BOX_TRAIN", "NOISY_BOX_VAL", "NOISY_BOX_INFER"):
        if key not in env_extra:
            env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [setup_env.REPO_ROOT, setup_env.PROJECT_ROOT, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    log_path = os.path.join(output_dir, log_name)
    started = time.time()
    with open(log_path, "a") as log:
        log.write("\n\n===== {} =====\n$ {}\n".format(
            time.strftime("%Y-%m-%dT%H:%M:%S%z"), " ".join(command)))
        log.flush()
        process = subprocess.Popen(command, cwd=setup_env.REPO_ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait(timeout=timeout)
    elapsed = time.time() - started

    if returncode != 0:
        raise RuntimeError(
            "training failed with exit code {}. Full log: {}".format(returncode, log_path)
        )
    return {"command": command, "log": log_path, "wall_seconds": round(elapsed, 1)}


# --------------------------------------------------------------------------
# Throughput calibration (hour-capped modes)
# --------------------------------------------------------------------------
def calibration_path(tag: str) -> str:
    return os.path.join(setup_env.RUNS_DIR, "calibration", "{}.json".format(tag))


def calibrate_rate(tag: str, config_file: str, cfg_run, train_split: str,
                   weights: str, num_gpus: int, data_dir: Optional[str] = None,
                   seconds: float = CALIBRATION_SECONDS,
                   force: bool = False) -> Dict[str, object]:
    """
    Measure real iterations/second for this stage on this hardware, by running
    genuine training steps for ``seconds`` of wall clock and discarding warm-up.

    Cached per ``tag`` so the three diagnosis variants share one measurement —
    which is the point: they then also share one ``MAX_ITER``, making their
    budgets identical by construction rather than approximately equal.
    """
    path = calibration_path(tag)
    if os.path.exists(path) and not force:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("iters_per_second"):
            cached["cached"] = True
            return cached

    probe_dir = os.path.join(setup_env.RUNS_DIR, "calibration", tag)
    if os.path.isdir(probe_dir):
        shutil.rmtree(probe_dir)
    os.makedirs(probe_dir, exist_ok=True)

    overrides = base_overrides(
        cfg_run, probe_dir, CALIBRATION_MAX_ITERS, weights, num_gpus,
        # No periodic checkpoints: the probe's weights are thrown away, and
        # writing a ~1 GB Swin-B checkpoint mid-probe would corrupt the very
        # throughput number being measured.
        extra=["SOLVER.CHECKPOINT_PERIOD", str(CALIBRATION_MAX_ITERS * 10)])
    launch = launch_training(
        config_file, overrides,
        registration.training_env(train_split, data_dir=data_dir),
        probe_dir, num_gpus, resume=False, rate_probe=path,
        budget_seconds=seconds, log_name="calibration.log",
    )
    with open(path) as handle:
        measured = json.load(handle)
    if not measured.get("iters_per_second"):
        raise RuntimeError("calibration produced no throughput measurement: {}".format(path))
    measured.update({
        "tag": tag, "config_file": os.path.abspath(config_file),
        "num_gpus": num_gpus, "ims_per_batch": cfg_run.ims_per_batch * max(1, num_gpus),
        "cached": False,
        # Charge the probe's TRUE cost (model build + steps), not an idealised
        # iterations/rate figure, so the hour cap accounts for real time spent.
        "probe_seconds": launch["wall_seconds"],
        "probe_budget_seconds": seconds,
    })
    with open(path, "w") as handle:
        json.dump(measured, handle, indent=2)
    # The probe's checkpoints are dead weight on a 20 GB quota.
    shutil.rmtree(probe_dir, ignore_errors=True)
    return measured


def resolve_max_iter(budget: StageBudget, calibration: Optional[Dict[str, object]],
                     round_to: int = 300) -> Dict[str, object]:
    """
    Turn a :class:`StageBudget` into a concrete iteration count.

    Iteration-capped modes pass through. Hour-capped modes multiply the
    *remaining* budget by the measured rate and round **down** to a multiple of
    ``round_to`` — 300 by default so the 1/3 and 2/3 trajectory snapshots land
    on whole iterations.

    The calibration probe is real GPU time and is charged to the budget, but
    **once for the whole study**, not once per stage: the measurement is shared
    by every stage, so charging it repeatedly would silently shrink each
    stage. The "already charged" flag lives in the calibration file so it
    survives a session kill.
    """
    if budget.max_iter is not None:
        return {"max_iter": int(budget.max_iter), "source": "fixed schedule",
                "budget_seconds": 0.0}
    if not calibration or not calibration.get("iters_per_second"):
        raise ValueError("an hour-capped stage needs a throughput calibration first")

    tag = calibration.get("tag")
    path = calibration_path(tag) if tag else None
    charged = bool(calibration.get("probe_charged"))
    if path and os.path.exists(path):
        with open(path) as handle:
            charged = bool(json.load(handle).get("probe_charged", charged))

    total_seconds = float(budget.hours) * 3600.0
    probe_seconds = 0.0 if charged else float(calibration.get("probe_seconds") or 0.0)
    remaining = max(0.0, total_seconds - probe_seconds)
    if probe_seconds and path and os.path.exists(path):
        with open(path) as handle:
            stored = json.load(handle)
        stored["probe_charged"] = True
        with open(path, "w") as handle:
            json.dump(stored, handle, indent=2)
    calibration["probe_charged"] = True
    raw = remaining * float(calibration["iters_per_second"])
    max_iter = max(round_to, int(raw // round_to) * round_to)
    return {
        "max_iter": max_iter,
        "source": "calibrated from {:.4f} it/s over {} probe iterations".format(
            calibration["iters_per_second"], calibration.get("iterations_timed")),
        "budget_hours": budget.hours,
        "budget_seconds": remaining,
        "probe_seconds_charged": probe_seconds,
        "raw_estimate": round(raw, 1),
        "round_to": round_to,
    }


def trajectory_map(max_iter: int, fractions: Sequence[float]) -> Dict[int, str]:
    """``{iteration: checkpoint name}`` for the trajectory snapshots."""
    out: Dict[int, str] = {}
    for fraction in fractions:
        iteration = int(round(max_iter * fraction))
        if 0 < iteration < max_iter:
            out[iteration] = "traj_{:03d}".format(int(round(fraction * 100)))
    return out


# --------------------------------------------------------------------------
# Stage / variant training
# --------------------------------------------------------------------------
def train_stage(name: str, config_file: str, cfg_run, train_split: str,
                weights: str, num_gpus: int, budget: StageBudget,
                calibration: Optional[Dict[str, object]] = None,
                noisy_boxes: Optional[Dict[str, str]] = None,
                trajectory_fractions: Sequence[float] = (),
                data_dir: Optional[str] = None,
                extra_overrides: Sequence[str] = (),
                seed: int = setup_env.BASE_SEED,
                env_override: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """
    Train one stage or variant, resuming across sessions.

    ``weights`` is the weight-transfer switch (previous stage's checkpoint vs
    the ImageNet Swin-B) and ``noisy_boxes`` is the manipulation switch
    (``{"NOISY_BOX_TRAIN": ..., "NOISY_BOX_VAL": ...}`` or ``None``). Nothing
    else about the run changes between variants.
    """
    directory = run_dir(name)
    # Clear any truncated checkpoint left in the local run directory by an
    # interrupted session BEFORE restoring, so a poisoned `last_checkpoint`
    # cannot survive into this run.
    purge_corrupt_checkpoints(name)
    seed_from_attached_datasets(name)

    if is_complete(name):
        record = read_run_record(name) or {}
        record["skipped"] = "already complete"
        record["output_dir"] = directory
        record["name"] = name
        return record

    # A stage needs room for one in-flight numbered checkpoint plus its final
    # and trajectory snapshots. Checking here turns "died at iteration 899 of
    # 900 after 3.4 hours" into a message before any GPU time is spent.
    free = disk_free_gb()
    needed = 2.5 + 0.8 * (1 + len(trajectory_fractions))
    if free < needed:
        raise RuntimeError(
            "only {:.1f} GB free on {} but {} needs ~{:.1f} GB (one in-flight "
            "checkpoint plus its final and trajectory snapshots).\n"
            "Free space first: completed stages under {} can be slimmed with "
            "train_utils.prune_checkpoints(<name>, complete=True), and old run "
            "modes' directories can be deleted outright."
            .format(free, setup_env.RUNS_ROOT, name, needed, setup_env.RUNS_ROOT)
        )

    plan = resolve_max_iter(budget, calibration)
    trajectory = trajectory_map(plan["max_iter"], trajectory_fractions)

    environment = registration.training_env(train_split, data_dir=data_dir)
    # Used by the base-DiffusionDet runs to point TRAIN_JSON at the flattened
    # single-label view of a tier instead of its normalized 3-tier file.
    environment.update(env_override or {})
    if noisy_boxes:
        for key, path in noisy_boxes.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "manipulation is ON for {} but {} = {} does not exist"
                    .format(name, key, path)
                )
        environment.update(noisy_boxes)

    overrides = base_overrides(cfg_run, directory, plan["max_iter"], weights,
                               num_gpus, seed=seed, extra=extra_overrides)
    launch = launch_training(
        config_file, overrides, environment, directory, num_gpus,
        resume=True, budget_seconds=plan.get("budget_seconds", 0.0),
        trajectory=trajectory,
    )

    record = read_run_record(name) or {}
    record.update({
        "name": name,
        "output_dir": directory,
        "train_split": train_split,
        "plan": plan,
        "trajectory": {str(k): v for k, v in trajectory.items()},
        "manipulation": bool(noisy_boxes),
        "weights_init": weights,
        "launch": launch,
        "config_hash": setup_env.config_hash(config_file, overrides),
    })
    with open(os.path.join(directory, "run_record.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    # The stage is done, so its optimizer state is dead weight: strip it and
    # drop every numbered checkpoint. Without this a 3-stage run overruns
    # /kaggle/working (~20 GB) partway through the third stage.
    record["disk"] = prune_checkpoints(
        name, keep=list(trajectory.values()),
        complete=os.path.exists(os.path.join(directory, "model_final.pth")))
    print("[{}] disk reclaimed; {:.1f} GB free".format(
        name, record["disk"]["disk_free_gb"]))
    return record


def variant_plan(cfg_run, prerequisite_weights: Dict[str, str],
                 imagenet_weights: str,
                 noisy_boxes: Optional[Dict[str, str]]) -> List[Dict[str, object]]:
    """
    Expand the run mode's variant list into concrete training specs.

    ``transfer`` picks the initial weights (the enumeration stage's checkpoint
    vs ImageNet Swin-B); ``manipulation`` decides whether the enumeration
    stage's inferred boxes are fed to the dataset mapper.
    """
    plans = []
    for variant in cfg_run.variants:
        switches = VARIANT_SWITCHES[variant]
        plans.append({
            "variant": variant,
            "label": setup_env.VARIANT_LABELS[variant],
            "run_name": "diagnosis_{}".format(variant),
            "weights": (prerequisite_weights["enumeration"] if switches["transfer"]
                        else imagenet_weights),
            "noisy_boxes": noisy_boxes if switches["manipulation"] else None,
            "switches": switches,
        })
    return plans


def assert_matched_budgets(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """
    Refuse to let an unmatched ablation comparison reach a table.

    Every diagnosis variant must have trained for the same number of iterations
    with the same seed, batch size and data. Different *wall time* is fine (and
    expected — manipulation costs a little extra dataloading); different
    *iterations* is not.
    """
    if len(records) < 2:
        return {"matched": True, "reason": "fewer than two variants"}

    def key(record):
        return (
            int(record.get("max_iter") or record.get("plan", {}).get("max_iter") or -1),
            int(record.get("seed", -1)),
            int(record.get("ims_per_batch", -1)),
            str(record.get("train_split")),
        )

    keys = {record.get("name", "?"): key(record) for record in records}
    distinct = set(keys.values())
    if len(distinct) != 1:
        raise AssertionError(
            "diagnosis variants were trained at UNMATCHED budgets, so the ablation "
            "comparison is invalid: {}. Re-run the odd one out at the shared "
            "(max_iter, seed, ims_per_batch, split).".format(json.dumps(keys, indent=2))
        )
    early = {name: record.get("stopped_on_time_budget")
             for name, record in zip(keys, records)}
    return {
        "matched": True,
        "shared": {"max_iter": list(distinct)[0][0], "seed": list(distinct)[0][1],
                   "ims_per_batch": list(distinct)[0][2], "split": list(distinct)[0][3]},
        "stopped_on_time_budget": early,
        "wall_seconds": {record.get("name"): record.get("wall_seconds")
                         for record in records},
    }
