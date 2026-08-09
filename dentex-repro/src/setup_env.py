"""
Environment bootstrap, run-mode resolution and deviation logging.

Every notebook in this suite starts by calling :func:`bootstrap`, which

1. puts the HierarchicalDet repo root on ``sys.path`` **first**, so the
   vendored (multi-label-modified) ``detectron2`` and ``pycocotools`` win over
   anything pip may have installed, and asserts that this actually happened;
2. resolves ``RUN_MODE`` into a :class:`RunConfig` holding every budget, seed
   and subset size the rest of the suite reads;
3. prints the header the runbook asks for (mode, seeds, config hash, GPU,
   estimated remaining GPU-time).

Nothing here imports torch at module scope: notebook 09 (asset build) and the
CPU-only benchmark session must import this module without a CUDA stack.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# src/ lives at <repo>/dentex-repro/src, so the HierarchicalDet checkout (the
# one holding the vendored detectron2/) is two levels up.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)          # .../dentex-repro
REPO_ROOT = os.path.dirname(PROJECT_ROOT)        # .../ (HierarchicalDet checkout)

CONFIGS_REPRO = os.path.join(PROJECT_ROOT, "configs_repro")
PAPER_ASSETS = os.path.join(PROJECT_ROOT, "paper_assets")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
REQUIREMENTS_LOCK = os.path.join(SRC_DIR, "requirements.lock.txt")

ON_KAGGLE = os.path.isdir("/kaggle/working")
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"

#: Where runs write checkpoints, predictions and metric JSONs. On Kaggle this
#: must be under /kaggle/working to survive "Save Version"; locally it is a
#: sibling of the project so a `git status` stays clean.
RUNS_DIR = os.environ.get(
    "DENTEX_RUNS_DIR",
    os.path.join(KAGGLE_WORKING, "runs") if ON_KAGGLE else os.path.join(PROJECT_ROOT, "runs"),
)

#: Converted DENTEX (notebook 01's output, republished as a Kaggle Dataset).
DATA_DIR = os.environ.get(
    "DENTEX_DATA_DIR",
    os.path.join(KAGGLE_WORKING, "dentex_converted") if ON_KAGGLE
    else os.path.join(PROJECT_ROOT, "dentex_converted"),
)

RESULTS_RAW = os.path.join(PAPER_ASSETS, "results_raw")
DEVIATIONS_MD = os.path.join(PAPER_ASSETS, "deviations.md")

#: The repo's own seed (configs/Base-DiffusionDet.yaml). Used as the training
#: seed for every primary run so variants differ only in the two switches.
BASE_SEED = 40244023


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class StageBudget:
    """How long / how far one training stage runs, in the active run mode."""

    #: Wall-clock budget in hours. ``None`` means "iteration-capped instead".
    hours: Optional[float]
    #: Fixed iteration count. ``None`` means "derive it from ``hours``".
    max_iter: Optional[int]

    def describe(self) -> str:
        if self.max_iter is not None:
            return "{} iters".format(self.max_iter)
        return "{:.2f} h (iteration count calibrated at runtime)".format(self.hours)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    mode: str
    quadrant: StageBudget
    enumeration: StageBudget
    diagnosis: StageBudget
    base_diffusiondet: Optional[StageBudget]
    #: Diagnosis-stage ablation variants to train, in order.
    variants: Sequence[str]
    #: Tiers to train a flat (non-multilabel) DiffusionDet baseline for.
    base_tiers: Sequence[int]
    #: Inference seeds for the main evaluation (DiffusionDet samples its
    #: starting boxes from noise, so inference is stochastic).
    eval_seeds: Sequence[int]
    #: Inference seeds for the cheaper robustness sweeps.
    robustness_seeds: Sequence[int]
    #: Cap on eval images (``None`` = the whole split). Only smoke shrinks it.
    eval_limit: Optional[int]
    #: Diffusion sampling steps to sweep in notebook 06.
    step_sweep: Sequence[int]
    #: (kind, severity) pairs for the degradation grid in notebook 06.
    degradations: Sequence[tuple]
    #: Prior-tier box jitter magnitudes for hierarchy fault injection.
    fault_jitters: Sequence[float]
    #: Prior-tier detection drop fractions for hierarchy fault injection.
    fault_drops: Sequence[float]
    #: Fractions of each diagnosis run at which to snapshot for the
    #: checkpoint-trajectory figure.
    trajectory_fractions: Sequence[float]
    #: Images per training batch (per process). Scaled by GPU count at launch.
    ims_per_batch: int
    #: Rough GPU-hour cost, printed in the header and recorded in the manifest.
    est_gpu_hours: float

    @property
    def is_smoke(self) -> bool:
        return self.mode == "smoke"


#: Full degradation grid (kind, severity) from the paper-asset spec.
_FULL_DEGRADATIONS = (
    ("blur", 1.0), ("blur", 2.0), ("blur", 4.0),
    ("jpeg", 80), ("jpeg", 50), ("jpeg", 20),
    ("downscale", 0.75), ("downscale", 0.50), ("downscale", 0.25),
)
#: micro keeps every corruption *type* but only mid + extreme severities.
_MICRO_DEGRADATIONS = (
    ("blur", 2.0), ("blur", 4.0),
    ("jpeg", 50), ("jpeg", 20),
    ("downscale", 0.50), ("downscale", 0.25),
)

ALL_VARIANTS = ("full", "wo_manipulation", "wo_transfer", "wo_manip_transfer")

RUN_MODES: Dict[str, RunConfig] = {
    # Prove the whole chain runs. Every notebook, every experiment class, tiny.
    "smoke": RunConfig(
        mode="smoke",
        quadrant=StageBudget(hours=None, max_iter=200),
        enumeration=StageBudget(hours=None, max_iter=200),
        diagnosis=StageBudget(hours=None, max_iter=200),
        base_diffusiondet=StageBudget(hours=None, max_iter=200),
        variants=ALL_VARIANTS,
        base_tiers=(2,),
        eval_seeds=(0,),
        robustness_seeds=(0,),
        eval_limit=20,
        step_sweep=(1, 2),
        degradations=(("blur", 2.0), ("jpeg", 50), ("downscale", 0.50)),
        fault_jitters=(0.0, 0.1),
        fault_drops=(0.0, 0.5),
        trajectory_fractions=(1 / 3, 2 / 3),
        ims_per_batch=1,
        est_gpu_hours=0.6,
    ),
    # Default. Hour-capped: the cap holds by construction on any hardware.
    "micro": RunConfig(
        mode="micro",
        quadrant=StageBudget(hours=1.75, max_iter=None),
        enumeration=StageBudget(hours=1.75, max_iter=None),
        diagnosis=StageBudget(hours=2.75, max_iter=None),
        base_diffusiondet=None,                      # skipped: see RUNBOOK.md
        variants=("full", "wo_manipulation", "wo_transfer"),
        base_tiers=(),
        eval_seeds=(0, 1, 2),
        robustness_seeds=(0, 1),
        eval_limit=None,
        step_sweep=(1, 4),
        degradations=_MICRO_DEGRADATIONS,
        fault_jitters=(0.0, 0.05, 0.1, 0.2),
        fault_drops=(0.0, 0.25, 0.5),
        trajectory_fractions=(1 / 3, 2 / 3),
        ims_per_batch=1,
        est_gpu_hours=13.0,
    ),
    # Half the paper's schedules.
    "budget": RunConfig(
        mode="budget",
        quadrant=StageBudget(hours=None, max_iter=15000),
        enumeration=StageBudget(hours=None, max_iter=15000),
        diagnosis=StageBudget(hours=None, max_iter=20000),
        base_diffusiondet=StageBudget(hours=None, max_iter=20000),
        variants=ALL_VARIANTS,
        base_tiers=(2,),
        eval_seeds=(0, 1, 2),
        robustness_seeds=(0, 1, 2),
        eval_limit=None,
        step_sweep=(1, 2, 4, 8),
        degradations=_FULL_DEGRADATIONS,
        fault_jitters=(0.0, 0.05, 0.1, 0.2),
        fault_drops=(0.0, 0.25, 0.5),
        trajectory_fractions=(1 / 3, 2 / 3),
        ims_per_batch=1,
        est_gpu_hours=30.0,
    ),
    # The repo's own schedules.
    "full": RunConfig(
        mode="full",
        quadrant=StageBudget(hours=None, max_iter=30000),
        enumeration=StageBudget(hours=None, max_iter=30000),
        diagnosis=StageBudget(hours=None, max_iter=40000),
        base_diffusiondet=StageBudget(hours=None, max_iter=40000),
        variants=ALL_VARIANTS,
        base_tiers=(0, 1, 2),
        eval_seeds=(0, 1, 2),
        robustness_seeds=(0, 1, 2),
        eval_limit=None,
        step_sweep=(1, 2, 4, 8),
        degradations=_FULL_DEGRADATIONS,
        fault_jitters=(0.0, 0.05, 0.1, 0.2),
        fault_drops=(0.0, 0.25, 0.5),
        trajectory_fractions=(1 / 3, 2 / 3),
        ims_per_batch=1,
        est_gpu_hours=60.0,
    ),
}

#: Human-readable variant descriptions; also used as table row labels.
VARIANT_LABELS = {
    "full": "Ours_full",
    "wo_transfer": "Ours_wo_Transfer",
    "wo_manipulation": "Ours_wo_Manipulation",
    "wo_manip_transfer": "Ours_wo_Manip_Transfer",
    "base_diffusiondet": "DiffusionDet_base",
}

#: (transfer, manipulation) switch settings per variant. These are the *only*
#: two things that differ between the four diagnosis-stage runs.
VARIANT_SWITCHES = {
    "full": {"transfer": True, "manipulation": True},
    "wo_transfer": {"transfer": False, "manipulation": True},
    "wo_manipulation": {"transfer": True, "manipulation": False},
    "wo_manip_transfer": {"transfer": False, "manipulation": False},
}


def resolve_run_mode(mode: Optional[str] = None) -> RunConfig:
    """Resolve ``RUN_MODE`` (argument > env var > 'micro')."""
    mode = (mode or os.environ.get("RUN_MODE") or "micro").strip().lower()
    if mode not in RUN_MODES:
        raise ValueError(
            "unknown RUN_MODE {!r}; expected one of {}".format(mode, sorted(RUN_MODES))
        )
    return RUN_MODES[mode]


# --------------------------------------------------------------------------
# Vendored-import guard
# --------------------------------------------------------------------------
def assert_vendored() -> Dict[str, str]:
    """
    Import ``detectron2`` and ``pycocotools`` and prove they came from the
    checked-out repo, not from site-packages.

    The repo ships *modified* copies of both (multi-label partial annotations,
    3-tier category schema). A pip-installed detectron2 silently shadows them
    and produces numbers that are not this paper's numbers, so this is a hard
    assertion, not a warning.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    elif sys.path[0] != REPO_ROOT:
        sys.path.remove(REPO_ROOT)
        sys.path.insert(0, REPO_ROOT)

    import detectron2
    import pycocotools

    found = {}
    for module in (detectron2, pycocotools):
        path = os.path.abspath(module.__file__)
        found[module.__name__] = path
        if not path.startswith(REPO_ROOT + os.sep):
            raise ImportError(
                "{} was imported from {}, not from the vendored copy under {}. "
                "The vendored copies are modified for multi-label partial "
                "annotations; a pip-installed one produces different numbers. "
                "Uninstall it (`pip uninstall -y detectron2 pycocotools`) or fix "
                "sys.path ordering.".format(module.__name__, path, REPO_ROOT)
            )
    return found


def _pycocotools_mask_extension_present() -> bool:
    import glob

    return bool(glob.glob(os.path.join(REPO_ROOT, "pycocotools", "_mask*.so")))


def ensure_pycocotools_mask() -> str:
    """
    The vendored ``pycocotools`` package ships Python sources only -- its
    compiled ``_mask`` extension is not in the repo. Copy the compiled
    extension out of the pip-installed pycocotools (installed for its binary,
    then shadowed for its Python) into the vendored package.

    Returns the path of the extension now sitting in the vendored package.
    """
    import glob

    existing = glob.glob(os.path.join(REPO_ROOT, "pycocotools", "_mask*.so"))
    if existing:
        return existing[0]

    import shutil
    import site
    import sysconfig

    candidates: List[str] = []
    site_dirs = list(getattr(site, "getsitepackages", lambda: [])())
    site_dirs.append(sysconfig.get_paths()["purelib"])
    site_dirs.append(sysconfig.get_paths()["platlib"])
    for directory in site_dirs:
        candidates.extend(glob.glob(os.path.join(directory, "pycocotools", "_mask*.so")))
    candidates = [c for c in candidates if not os.path.abspath(c).startswith(REPO_ROOT + os.sep)]
    if not candidates:
        raise RuntimeError(
            "no compiled pycocotools _mask extension found in site-packages. "
            "`pip install pycocotools` first: the vendored fork needs its binary."
        )
    destination = os.path.join(REPO_ROOT, "pycocotools", os.path.basename(candidates[0]))
    shutil.copy2(candidates[0], destination)
    return destination


# --------------------------------------------------------------------------
# Environment reporting
# --------------------------------------------------------------------------
def git_commit(path: str = REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def gpu_report() -> Dict[str, object]:
    """GPU inventory. Returns ``{"available": False}`` on a CPU-only session."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is False"}
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "total_memory_gb": round(properties.total_memory / (1024 ** 3), 2),
            "capability": "{}.{}".format(properties.major, properties.minor),
        })
    return {
        "available": True,
        "count": len(devices),
        "devices": devices,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


#: Exact pins for everything the vendored code needs that a Kaggle image may
#: not already carry. torch / torchvision / numpy / Pillow / matplotlib /
#: pandas / scipy are deliberately NOT forced: Kaggle ships them built against
#: its own CUDA, and overriding them is the reliable way to break a session.
#: Their resolved versions are captured in requirements.lock.txt instead, which
#: is what actually pins the environment for reproduction.
PINNED_REQUIREMENTS = (
    "timm==0.9.16",
    "fvcore==0.1.5.post20221221",
    "iopath==0.1.9",
    "omegaconf==2.3.0",
    "einops==0.8.0",
    "cloudpickle==3.0.0",
    "termcolor==2.4.0",
    "tabulate==0.9.0",
    "opencv-python-headless==4.10.0.84",
    "pycocotools==2.0.7",
    "huggingface_hub==0.24.6",
)


def install_dependencies(lock_path: Optional[str] = None,
                         extra: Sequence[str] = ()) -> Dict[str, object]:
    """
    Install the dependency set. Prefers an existing lock file (exact
    reproduction); falls back to :data:`PINNED_REQUIREMENTS` the first time.

    Kaggle reverts to its base image on every fresh session, so this runs at the
    top of every notebook, not just once.
    """
    lock_path = lock_path or REQUIREMENTS_LOCK
    if os.path.exists(lock_path):
        with open(lock_path) as handle:
            specs = [line.strip() for line in handle
                     if line.strip() and not line.startswith("#")]
        source = lock_path
        # A full freeze includes Kaggle-image packages we must not force.
        specs = [s for s in specs if s.split("==")[0].lower() in
                 {p.split("==")[0].lower() for p in PINNED_REQUIREMENTS}]
    else:
        specs, source = list(PINNED_REQUIREMENTS), "PINNED_REQUIREMENTS"
    specs = list(specs) + list(extra)

    command = [sys.executable, "-m", "pip", "install", "-q"] + specs
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and "--break-system-packages" not in result.stderr:
        retry = subprocess.run(command + ["--break-system-packages"],
                               capture_output=True, text=True)
        if retry.returncode != 0:
            raise RuntimeError(
                "dependency install failed:\n{}\n{}".format(result.stderr, retry.stderr)
            )
        result = retry
    elif result.returncode != 0:
        raise RuntimeError("dependency install failed:\n{}".format(result.stderr))
    return {"source": source, "specs": specs, "stdout": result.stdout[-2000:]}


def pip_freeze() -> List[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def write_requirements_lock(path: str = REQUIREMENTS_LOCK) -> str:
    """Freeze the resolved environment. Notebook 00 writes it; the rest read it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = pip_freeze()
    header = [
        "# Resolved dependency set for the HierarchicalDet reproduction.",
        "# Generated by src.setup_env.write_requirements_lock in notebook 00.",
        "# Python {} on {} ({}).".format(
            platform.python_version(), platform.platform(), platform.machine()
        ),
        "# detectron2 and pycocotools are VENDORED in the repo and deliberately",
        "# absent from this list except for pycocotools' binary wheel, which is",
        "# installed only so its compiled _mask extension can be copied into the",
        "# vendored package (see setup_env.ensure_pycocotools_mask).",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(header + lines) + "\n")
    return path


def env_report() -> Dict[str, object]:
    """Everything ``paper_assets/repro_checklist.md`` needs about this session."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "on_kaggle": ON_KAGGLE,
        "repo_root": REPO_ROOT,
        "repo_commit": git_commit(),
        "gpu": gpu_report(),
        "pip_freeze": pip_freeze(),
    }


# --------------------------------------------------------------------------
# Hashing / determinism
# --------------------------------------------------------------------------
def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config_file: str, overrides: Sequence[str] = ()) -> str:
    """
    Stable identity of "the exact configuration this run used": the YAML bytes
    plus the ordered command-line overrides. Recorded next to every asset so a
    number can never be attributed to the wrong config.
    """
    digest = hashlib.sha256()
    with open(config_file, "rb") as handle:
        digest.update(handle.read())
    digest.update(b"\0".join(str(o).encode() for o in overrides))
    return digest.hexdigest()[:16]


def seed_everything(seed: int, deterministic: bool = False) -> Dict[str, object]:
    """
    Seed python/numpy/torch. Returns a record of what remains nondeterministic,
    which goes verbatim into the repro checklist.

    ``deterministic=True`` additionally asks cuDNN for deterministic kernels.
    It is off by default for training because it costs throughput on the
    convolutional stem, and this suite is wall-clock-budgeted; it is on for
    evaluation, where the cost is negligible.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    return {
        "seed": seed,
        "cudnn_deterministic": bool(deterministic),
        "cudnn_benchmark": not deterministic,
        "remaining_nondeterminism": [
            "AMP autocast reduction order (SOLVER.AMP.ENABLED)",
            "atomics in torchvision ROIAlign / NMS CUDA kernels",
            "DataLoader worker interleaving when NUM_WORKERS > 0",
        ] if not deterministic else [
            "atomics in torchvision ROIAlign / NMS CUDA kernels",
        ],
    }


# --------------------------------------------------------------------------
# Deviation log
# --------------------------------------------------------------------------
def log_deviation(what: str, why: str, notebook: str, impact: str = "") -> None:
    """
    Append one departure from the paper/repo defaults to
    ``paper_assets/deviations.md``. Idempotent: the same (notebook, what) pair
    is written once, so re-running a notebook does not duplicate entries.
    """
    os.makedirs(PAPER_ASSETS, exist_ok=True)
    key = "**{}** — {}".format(notebook, what)
    if os.path.exists(DEVIATIONS_MD):
        with open(DEVIATIONS_MD) as handle:
            body = handle.read()
        if key in body:
            return
    else:
        body = (
            "# Deviations from the original paper and released code\n\n"
            "Auto-accumulated by `src.setup_env.log_deviation`. Every entry was\n"
            "written by a notebook that actually executed; nothing here is planned\n"
            "or hypothetical.\n\n"
        )
    entry = ["- {}".format(key), "  - Why: {}".format(why)]
    if impact:
        entry.append("  - Impact on results: {}".format(impact))
    with open(DEVIATIONS_MD, "w") as handle:
        handle.write(body.rstrip("\n") + "\n" + "\n".join(entry) + "\n")


# --------------------------------------------------------------------------
# Notebook summaries
# --------------------------------------------------------------------------
def write_notebook_summary(notebook: str, payload: Dict[str, object]) -> str:
    """
    Every notebook ends by writing one of these. Notebook 09 builds the paper
    assets purely out of them plus ``results_raw/``, so figures can be
    regenerated on a CPU session without re-running any experiment.
    """
    os.makedirs(RESULTS_RAW, exist_ok=True)
    path = os.path.join(RESULTS_RAW, "summary_{}.json".format(notebook))
    payload = dict(payload)
    payload.setdefault("notebook", notebook)
    payload.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def read_notebook_summary(notebook: str) -> Optional[Dict[str, object]]:
    path = os.path.join(RESULTS_RAW, "summary_{}.json".format(notebook))
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def gpu_hours_spent() -> float:
    """
    Total GPU-hours recorded by training/eval runs so far, read back from the
    run records on disk. Printed in every notebook header so a session never
    silently overruns the weekly Kaggle quota.
    """
    total = 0.0
    if not os.path.isdir(RUNS_DIR):
        return 0.0
    for root, _dirs, files in os.walk(RUNS_DIR):
        if "run_record.json" not in files:
            continue
        try:
            with open(os.path.join(root, "run_record.json")) as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        if record.get("device") == "cpu":
            continue
        total += float(record.get("wall_seconds", 0.0)) / 3600.0
    return total


def hydrate_from_attached_datasets(search_roots: Sequence[str] = ()) -> Dict[str, object]:
    """
    Copy accumulated results out of attached Kaggle Datasets into this session's
    ``paper_assets/``.

    This is load-bearing, not a convenience. ``paper_assets/`` lives inside the
    *cloned repo*, and a fresh Kaggle session re-clones it empty — so without
    this, a later session would see no prior results at all: the asset build
    would find nothing to assemble, and the reproducibility checklist would lose
    the dataset hashes recorded by the data notebook.

    Only files missing locally are restored, so a re-run never clobbers output
    it has just produced with an older copy from an attached dataset.
    """
    import glob
    import shutil

    roots = list(search_roots) or (
        sorted(glob.glob(os.path.join(KAGGLE_INPUT, "*"))) if ON_KAGGLE else []
    )
    restored: List[str] = []
    for root in roots:
        for source in glob.glob(os.path.join(root, "**", "paper_assets"), recursive=True):
            if os.path.abspath(source) == os.path.abspath(PAPER_ASSETS):
                continue
            for relative in ("results_raw", "manifest.json", "deviations.md"):
                origin = os.path.join(source, relative)
                if not os.path.exists(origin):
                    continue
                target = os.path.join(PAPER_ASSETS, relative)
                if os.path.isdir(origin):
                    os.makedirs(target, exist_ok=True)
                    for name in os.listdir(origin):
                        if not os.path.exists(os.path.join(target, name)):
                            shutil.copy2(os.path.join(origin, name),
                                         os.path.join(target, name))
                            restored.append(os.path.join(relative, name))
                elif not os.path.exists(target):
                    shutil.copy2(origin, target)
                    restored.append(relative)
    return {"searched": roots, "restored": restored}


def bootstrap(mode: Optional[str] = None, require_gpu: bool = False,
              quiet: bool = False) -> RunConfig:
    """
    The first cell of every notebook. Returns the resolved :class:`RunConfig`.
    """
    found = assert_vendored()
    run = resolve_run_mode(mode)

    for directory in (RUNS_DIR, DATA_DIR, PAPER_ASSETS, RESULTS_RAW):
        os.makedirs(directory, exist_ok=True)
    hydrated = hydrate_from_attached_datasets()

    gpu = gpu_report()
    if require_gpu and not gpu.get("available"):
        raise RuntimeError(
            "this notebook needs a GPU but none is visible ({}). On Kaggle: "
            "Settings -> Accelerator -> GPU T4 x2 (phone verification required)."
            .format(gpu.get("reason"))
        )

    if not quiet:
        spent = gpu_hours_spent()
        print("=" * 72)
        print("HierarchicalDet reproduction — RUN_MODE = {}".format(run.mode))
        print("=" * 72)
        print("repo root        : {}".format(REPO_ROOT))
        print("repo commit      : {}".format(git_commit()))
        print("detectron2 from  : {}".format(found["detectron2"]))
        print("pycocotools from : {}".format(found["pycocotools"]))
        print("runs dir         : {}".format(RUNS_DIR))
        print("data dir         : {}".format(DATA_DIR))
        print("training seed    : {}".format(BASE_SEED))
        print("eval seeds       : {}".format(list(run.eval_seeds)))
        print("variants         : {}".format(list(run.variants)))
        print("quadrant stage   : {}".format(run.quadrant.describe()))
        print("enumeration stage: {}".format(run.enumeration.describe()))
        print("diagnosis stage  : {}  (per variant)".format(run.diagnosis.describe()))
        if gpu.get("available"):
            names = ", ".join(d["name"] for d in gpu["devices"])
            print("GPU              : {}x {} (torch {}, CUDA {})".format(
                gpu["count"], names, gpu["torch"], gpu["cuda"]))
        else:
            print("GPU              : none — {}".format(gpu.get("reason")))
        print("GPU-hours spent  : {:.2f} recorded / {:.1f} estimated for this mode"
              .format(spent, run.est_gpu_hours))
        if hydrated["restored"]:
            print("restored {} result file(s) from attached datasets"
                  .format(len(hydrated["restored"])))
        print("=" * 72)
    return run
