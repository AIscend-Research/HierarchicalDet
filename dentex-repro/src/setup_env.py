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

#: The active run mode, read at import time. The notebooks' parameter cell sets
#: ``RUN_MODE`` in the environment *before* importing this package, so every
#: mode-scoped path below resolves consistently across a session.
ACTIVE_MODE = (os.environ.get("RUN_MODE") or "micro").strip().lower()

#: Root of all run outputs. On Kaggle this must be under /kaggle/working to
#: survive "Save Version"; locally it is a sibling of the project so a
#: `git status` stays clean.
RUNS_ROOT = os.environ.get(
    "DENTEX_RUNS_DIR",
    os.path.join(KAGGLE_WORKING, "runs") if ON_KAGGLE else os.path.join(PROJECT_ROOT, "runs"),
)

#: Where THIS run mode writes checkpoints, predictions and metric JSONs.
#:
#: Scoping by mode is a correctness requirement, not tidiness. Runs are skipped
#: when their ``model_final.pth`` already exists, so an unscoped directory means
#: a 200-iteration ``smoke`` checkpoint would make the real ``micro`` run skip
#: training entirely and silently build the paper from smoke-mode models. Mode
#: in the path makes that impossible.
RUNS_DIR = os.path.join(RUNS_ROOT, ACTIVE_MODE)

#: Converted DENTEX (notebook 01's output, republished as a Kaggle Dataset).
DATA_DIR = os.environ.get(
    "DENTEX_DATA_DIR",
    os.path.join(KAGGLE_WORKING, "dentex_converted") if ON_KAGGLE
    else os.path.join(PROJECT_ROOT, "dentex_converted"),
)

#: Raw metric JSONs, scoped by mode for the same reason ``RUNS_DIR`` is: result
#: files are addressed by name, so an unscoped directory would let a stale
#: smoke-mode metric survive under a micro-mode filename and be cited as a real
#: number. Notebook 03 therefore only ever sees the active mode's results.
RESULTS_RAW = os.path.join(PAPER_ASSETS, "results_raw", ACTIVE_MODE)
DEVIATIONS_MD = os.path.join(PAPER_ASSETS, "deviations.md")

#: The repo's own seed (configs/Base-DiffusionDet.yaml). Used as the training
#: seed for every primary run so variants differ only in the two switches.
BASE_SEED = 40244023

#: Build fingerprints of the generated notebooks, written by
#: ``tools/build_notebooks.py``.
NOTEBOOK_BUILDS = os.path.join(SRC_DIR, "notebook_build.json")


def assert_notebook_current(notebook: str, build: str) -> Dict[str, str]:
    """
    Fail if the notebook running these cells is older than the cloned repo.

    This exists because of a real, silent divergence: the parameter cell
    ``git pull``s the repo, so ``src/`` and ``configs_repro/`` are always fresh
    — but the **notebook cells themselves are whatever was uploaded to Kaggle**
    and never update. A session was observed running ``src/`` at one commit
    while its cells were three commits behind, quietly using a superseded DDP
    probe and a pre-flight four times longer than intended.

    ``build`` is a fingerprint baked into the notebook at generation time; the
    matching value ships in the repo, so a mismatch means "re-import this
    notebook".
    """
    if not os.path.exists(NOTEBOOK_BUILDS):
        return {"status": "unknown", "reason": "no notebook_build.json in this checkout"}
    with open(NOTEBOOK_BUILDS) as handle:
        expected = json.load(handle).get(notebook)
    if expected is None:
        return {"status": "unknown", "reason": "no fingerprint recorded for " + notebook}
    if expected != build:
        raise RuntimeError(
            "STALE NOTEBOOK: the cells you are running were generated as {} but the "
            "repo now ships {} for {}.\n"
            "`git pull` refreshes src/ and configs_repro/, but NOT the notebook cells "
            "-- those are the copy uploaded to Kaggle.\n"
            "Fix: File -> Import Notebook, re-upload notebooks/{}.ipynb from the repo "
            "(or copy its cells in), then re-run.".format(build, expected, notebook, notebook)
        )
    return {"status": "current", "build": build}


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
    #
    # 50 iterations, not 200: measured throughput on Kaggle T4 x2 is 9.13
    # s/iter, so 200 iterations is 30 MINUTES per training run and smoke has
    # seven of them -- 3.6 h to prove plumbing. 50 keeps every code path
    # (checkpointing, trajectory snapshots at 1/3 and 2/3, evaluation) while
    # costing ~8 min per run.
    "smoke": RunConfig(
        mode="smoke",
        quadrant=StageBudget(hours=None, max_iter=50),
        enumeration=StageBudget(hours=None, max_iter=50),
        diagnosis=StageBudget(hours=None, max_iter=50),
        base_diffusiondet=StageBudget(hours=None, max_iter=50),
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
        est_gpu_hours=1.2,
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


def _vendored_mask_imports() -> bool:
    """
    Can the vendored ``pycocotools.mask`` actually be imported?

    Checked in a **subprocess**: a failed extension import cannot be retried in
    the same interpreter (the module is cached and the shared object is already
    mapped), so an in-process check would report stale results after a repair.
    """
    code = (
        "import sys; sys.path.insert(0, {!r}); "
        "import pycocotools.mask".format(REPO_ROOT)
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, cwd=os.sep)
    return result.returncode == 0


def _site_package_masks() -> List[str]:
    import glob
    import site
    import sysconfig

    directories = list(getattr(site, "getsitepackages", lambda: [])())
    directories += [sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]]
    candidates: List[str] = []
    for directory in directories:
        candidates.extend(glob.glob(os.path.join(directory, "pycocotools", "_mask*.so")))
    return [c for c in candidates if not os.path.abspath(c).startswith(REPO_ROOT + os.sep)]


def ensure_pycocotools_mask() -> str:
    """
    Graft a **working** compiled ``_mask`` extension into the vendored
    ``pycocotools`` package, which ships Python sources only.

    The subtlety this function exists for: ``_mask`` is a Cython extension
    compiled against numpy's C ABI. If it is built against a different numpy
    major version than the one installed at runtime, importing it dies with
    ``ValueError: numpy.dtype size changed ... Expected 96 ... got 88`` — deep
    inside ``detectron2.structures``, which reads as a detectron2 problem and is
    not one. Pinning a pycocotools version makes this *more* likely, because pip
    then builds from source in an isolated environment that pulls the newest
    numpy regardless of what is installed.

    So: prefer whatever already works, verify by actually importing, and only as
    a last resort rebuild against the installed numpy with build isolation off.
    """
    import glob
    import shutil

    destination_dir = os.path.join(REPO_ROOT, "pycocotools")
    existing = glob.glob(os.path.join(destination_dir, "_mask*.so"))
    if existing and _vendored_mask_imports():
        return existing[0]
    for stale in existing:                       # ABI-mismatched; do not keep it
        os.remove(stale)

    def try_candidates() -> Optional[str]:
        for candidate in _site_package_masks():
            target = os.path.join(destination_dir, os.path.basename(candidate))
            shutil.copy2(candidate, target)
            if _vendored_mask_imports():
                return target
            os.remove(target)
        return None

    grafted = try_candidates()
    if grafted:
        return grafted

    # Nothing usable is installed. Every remaining attempt passes the
    # constraints file, because pip must not be allowed to "solve" an ABI
    # mismatch by upgrading numpy — that would break torch and every other
    # compiled package in the image. Each attempt is verified by import.
    constraints = _write_constraints()
    before = {name: _distribution_version(name) for name in ABI_CRITICAL}
    attempts = (
        # A published wheel first: no compiler needed, and often correct.
        ["--force-reinstall", PYCOCOTOOLS_SPEC],
        # Otherwise build from source so the compile sees the installed numpy:
        # --no-build-isolation stops pip creating a newest-numpy build env, and
        # --no-binary refuses a wheel built against some other numpy.
        ["--force-reinstall", "--no-build-isolation", "--no-binary", ":all:",
         PYCOCOTOOLS_SPEC],
    )
    result = None
    for index, specs in enumerate(attempts):
        if index == 1:
            _pip_install(["cython", "setuptools", "wheel"], constraints)
        result = _pip_install(specs, constraints)
        after = {name: _distribution_version(name) for name in ABI_CRITICAL}
        moved = {k: (before[k], after[k]) for k in ABI_CRITICAL
                 if before[k] and before[k] != after[k]}
        if moved:
            raise RuntimeError(
                "installing pycocotools moved an ABI-critical package ({}), which "
                "breaks torch and every other compiled extension in the image. "
                "Restart the session and report this — the constraints file should "
                "have prevented it.".format(moved)
            )
        if result.returncode == 0:
            grafted = try_candidates()
            if grafted:
                return grafted

    raise RuntimeError(
        "could not produce a pycocotools _mask extension compatible with the "
        "installed numpy ({}). The vendored pycocotools fork needs one.\n"
        "pip said:\n{}\n"
        "Try in a cell, then re-run this notebook:\n"
        "  !pip install --force-reinstall --no-build-isolation --no-binary :all: "
        "pycocotools".format(_numpy_version(), (result.stderr or "")[-2000:])
    )


def _numpy_version() -> str:
    try:
        import numpy

        return numpy.__version__
    except ImportError:
        return "not installed"


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


#: pycocotools is handled apart from everything else. It is the one dependency
#: with a compiled extension we graft into the vendored package, and pinning a
#: version forces pip to build it from source in an isolated environment — which
#: pulls the newest numpy, producing a `.so` binary-incompatible with the numpy
#: actually installed ("numpy.dtype size changed, Expected 96 ... got 88").
#: `ensure_pycocotools_mask` owns it end to end, and verifies by import.
PYCOCOTOOLS_SPEC = "pycocotools"


#: Packages whose compiled extensions define the ABI everything else links
#: against. If pip changes any of these, previously-compiled wheels elsewhere in
#: the image (torch, torchvision, pycocotools, opencv) start failing with
#: "binary incompatibility" errors far from the cause. They are pinned to
#: whatever the image already has, via a pip constraints file, on every install.
ABI_CRITICAL = ("numpy", "torch", "torchvision")

#: ``(import name, pip spec)``. The import name is what gets probed; the spec is
#: used **only if that import fails**. Deliberately unversioned: the Kaggle image
#: already satisfies most of these, and forcing a version is how the numpy ABI
#: gets broken. Exact resolved versions are captured in ``requirements.lock.txt``
#: after the fact — that file, not these specs, is what pins the environment.
#:
#: This list is DERIVED, not curated: it is the union of every module-level
#: third-party import reachable from the vendored ``detectron2`` /
#: ``hierarchialdet`` / ``pycocotools`` trees (found by AST scan — see
#: ``tools/check_import_chain.py``, which asserts the list stays complete),
#: plus what training needs at runtime (tensorboard for detectron2's writers,
#: psutil for its memory logging) and what ``src/`` itself uses. A hand-picked
#: subset of this list is exactly how a Kaggle run died on
#: ``ModuleNotFoundError: fairscale`` four imports deep.
REQUIREMENTS = (
    ("timm", "timm"),
    ("fvcore", "fvcore"),
    ("iopath", "iopath"),
    ("omegaconf", "omegaconf"),
    ("einops", "einops"),
    ("fairscale", "fairscale"),
    ("cloudpickle", "cloudpickle"),
    ("termcolor", "termcolor"),
    ("tabulate", "tabulate"),
    ("yaml", "pyyaml"),
    ("cv2", "opencv-python-headless"),
    ("huggingface_hub", "huggingface_hub"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("seaborn", "seaborn"),
    ("PIL", "pillow"),
    ("tqdm", "tqdm"),
    ("packaging", "packaging"),
    # pkg_resources was REMOVED from setuptools 81+. Only detectron2/model_zoo
    # (the optional baselines path) imports it; the spec targets the last line
    # that still ships it, and installs only when the module is missing.
    ("pkg_resources", "setuptools<81"),
    ("tensorboard", "tensorboard"),
    ("psutil", "psutil"),
)


def _distribution_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _module_importable(module: str) -> bool:
    """Probe in a subprocess: a broken compiled module poisons this interpreter."""
    result = subprocess.run([sys.executable, "-c", "import {}".format(module)],
                            capture_output=True, text=True, cwd=os.sep)
    return result.returncode == 0


def _write_constraints() -> Optional[str]:
    """Pin the ABI-critical packages to the versions already installed."""
    lines = []
    for name in ABI_CRITICAL:
        found = _distribution_version(name)
        if found:
            lines.append("{}=={}".format(name, found))
    if not lines:
        return None
    path = os.path.join(PROJECT_ROOT, ".pip-constraints.txt")
    with open(path, "w") as handle:
        handle.write("# Generated by src.setup_env: do not let pip move the ABI.\n")
        handle.write("\n".join(lines) + "\n")
    return path


def _pip_install(specs: Sequence[str], constraints: Optional[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "pip", "install", "-q"]
    if constraints:
        command += ["-c", constraints]
    command += list(specs)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # Newer pip refuses to touch a distro-managed environment without this.
        result = subprocess.run(command + ["--break-system-packages"],
                                capture_output=True, text=True)
    return result


def install_dependencies(extra: Sequence[str] = ()) -> Dict[str, object]:
    """
    Make every dependency importable, without moving anything already working.

    Two rules, both learned the hard way on a real Kaggle session:

    1. **Install only what is missing.** The Kaggle image already satisfies most
       of this list. Reinstalling a package to a pinned version drags its build
       requirements in with it, and that is how ``pycocotools`` ended up compiled
       against numpy 2 while the image ran numpy 1 — surfacing four imports later
       as ``numpy.dtype size changed`` inside ``detectron2.structures``.
    2. **Never let pip move numpy / torch / torchvision.** They are pinned to the
       installed versions through a constraints file on every call, and the
       versions are re-checked afterwards.

    Kaggle reverts to its base image every session, so this runs at the top of
    every notebook.
    """
    before = {name: _distribution_version(name) for name in ABI_CRITICAL}
    constraints = _write_constraints()

    missing, already = [], []
    for module, spec in REQUIREMENTS:
        (already if _module_importable(module) else missing).append((module, spec))

    result = None
    specs = [spec for _module, spec in missing] + list(extra)
    if specs:
        result = _pip_install(specs, constraints)
        if result.returncode != 0:
            raise RuntimeError(
                "dependency install failed for {}:\n{}".format(specs, result.stderr[-4000:])
            )

    still_missing = [module for module, _spec in missing if not _module_importable(module)]
    if still_missing:
        raise RuntimeError(
            "these modules are still not importable after installing {}: {}. "
            "Check the pip output above.".format(specs, still_missing)
        )

    after = {name: _distribution_version(name) for name in ABI_CRITICAL}
    moved = {name: (before[name], after[name])
             for name in ABI_CRITICAL if before[name] and before[name] != after[name]}
    if moved:
        raise RuntimeError(
            "pip changed an ABI-critical package despite the constraints file: {}. "
            "Every compiled extension in the image was built against the old "
            "version; restart the session before continuing.".format(moved)
        )

    return {
        "installed": specs,
        "already_present": [module for module, _spec in already],
        "constraints": constraints,
        "abi_versions": after,
        "stdout": (result.stdout[-2000:] if result else ""),
    }


def pip_freeze() -> List[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def write_requirements_lock(path: str = REQUIREMENTS_LOCK) -> str:
    """
    Freeze the fully resolved environment.

    **This file, not the specs in :data:`REQUIREMENTS`, is what pins the
    environment for reproduction.** The specs are unversioned on purpose: forcing
    versions on top of a Kaggle image is what breaks the numpy ABI. Recording
    exactly what resolved gives reproducibility without that risk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = pip_freeze()
    header = [
        "# Resolved dependency set for the HierarchicalDet reproduction.",
        "# Generated by src.setup_env.write_requirements_lock.",
        "# Python {} on {} ({}).".format(
            platform.python_version(), platform.platform(), platform.machine()
        ),
        "# This is a RECORD of what resolved, not an input: the notebooks install",
        "# only modules that are missing, and never force a version on top of the",
        "# host image (that is how the pycocotools/numpy ABI break happened).",
        "# detectron2 and pycocotools are VENDORED in the repo; the pip-installed",
        "# pycocotools exists only to supply a compiled _mask extension, which is",
        "# grafted in and verified by setup_env.ensure_pycocotools_mask.",
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
    if not os.path.isdir(RUNS_ROOT):
        return 0.0
    for root, _dirs, files in os.walk(RUNS_ROOT):   # every mode, not just this one
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


def discover_converted_data(search_roots: Sequence[str] = ()) -> Optional[str]:
    """
    Point :data:`DATA_DIR` at the converted DENTEX dataset when it arrives as an
    *attached* Kaggle Dataset instead of having been produced in this session.

    Notebook 01 writes to ``/kaggle/working/dentex_converted``; every later
    notebook reads through ``data_convert.layout()``, which resolves
    ``setup_env.DATA_DIR`` at call time. But attached datasets mount read-only
    under ``/kaggle/input/<slug>/...``, so without this remap notebook 02 would
    fail its "run notebook 01 first" assertion even with the data attached.
    A read-only root is fine: everything ever written into the data dir is
    written by notebook 01 itself; later notebooks only read from it.

    The sentinel is ``coco/train_diagnosis.json`` — present in every complete
    conversion, regardless of whether the attachment is the published dataset
    (``.../dentex_converted/coco/...``) or a raw notebook-output attachment.
    """
    global DATA_DIR

    import glob

    if os.path.exists(os.path.join(DATA_DIR, "coco", "train_diagnosis.json")):
        return None                      # local copy exists (this session ran 01)

    roots = list(search_roots) or (
        sorted(glob.glob(os.path.join(KAGGLE_INPUT, "*"))) if ON_KAGGLE else []
    )
    for root in roots:
        for sentinel in glob.glob(
                os.path.join(root, "**", "coco", "train_diagnosis.json"),
                recursive=True):
            DATA_DIR = os.path.dirname(os.path.dirname(sentinel))
            os.environ["DENTEX_DATA_DIR"] = DATA_DIR
            return DATA_DIR
    return None


def disk_report() -> Dict[str, object]:
    """
    Where the disk actually went.

    Three sessions were lost to ``No space left on device`` while a
    hand-maintained model of checkpoint sizes said there was room. The model was
    wrong somewhere it could not see -- most likely because attached datasets
    are not free, they occupy the same volume as ``/kaggle/working``. So stop
    modelling and report: total/used/free on the volume, plus the size of every
    attached dataset and of each run directory.
    """
    import glob
    import shutil

    def tree_bytes(path: str) -> int:
        total = 0
        for directory, _subdirs, names in os.walk(path):
            for name in names:
                try:
                    total += os.path.getsize(os.path.join(directory, name))
                except OSError:
                    pass
        return total

    usage = shutil.disk_usage(RUNS_ROOT if os.path.isdir(RUNS_ROOT) else PROJECT_ROOT)
    inputs = {}
    if ON_KAGGLE:
        for root in sorted(glob.glob(os.path.join(KAGGLE_INPUT, "*"))):
            inputs[os.path.basename(root)] = round(tree_bytes(root) / (1024 ** 3), 2)
    runs = {}
    if os.path.isdir(RUNS_ROOT):
        for mode_dir in sorted(glob.glob(os.path.join(RUNS_ROOT, "*"))):
            for run in sorted(glob.glob(os.path.join(mode_dir, "*"))):
                if os.path.isdir(run):
                    key = os.path.join(os.path.basename(mode_dir), os.path.basename(run))
                    runs[key] = round(tree_bytes(run) / (1024 ** 3), 2)
    return {
        "volume_total_gb": round(usage.total / (1024 ** 3), 2),
        "volume_used_gb": round(usage.used / (1024 ** 3), 2),
        "volume_free_gb": round(usage.free / (1024 ** 3), 2),
        "attached_inputs_gb": inputs,
        "attached_inputs_total_gb": round(sum(inputs.values()), 2),
        "runs_gb": runs,
        "runs_total_gb": round(sum(runs.values()), 2),
    }


def print_disk_report(prefix: str = "") -> Dict[str, object]:
    report = disk_report()
    print("{}disk: {:.1f} GB free of {:.1f} GB total ({:.1f} used)".format(
        prefix, report["volume_free_gb"], report["volume_total_gb"],
        report["volume_used_gb"]))
    if report["attached_inputs_gb"]:
        print("{}  attached inputs ({:.1f} GB): {}".format(
            prefix, report["attached_inputs_total_gb"],
            ", ".join("{}={:.1f}".format(k, v)
                      for k, v in report["attached_inputs_gb"].items())))
    if report["runs_gb"]:
        print("{}  runs ({:.1f} GB): {}".format(
            prefix, report["runs_total_gb"],
            ", ".join("{}={:.1f}".format(k, v)
                      for k, v in sorted(report["runs_gb"].items()) if v > 0.01)))
    return report


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
                    # results_raw is nested by run mode, so walk it rather than
                    # copying one flat level.
                    for directory, _subdirs, names in os.walk(origin):
                        local = os.path.join(target, os.path.relpath(directory, origin))
                        os.makedirs(local, exist_ok=True)
                        for name in names:
                            if not os.path.exists(os.path.join(local, name)):
                                shutil.copy2(os.path.join(directory, name),
                                             os.path.join(local, name))
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

    attached_data = discover_converted_data()
    for directory in (RUNS_DIR, PAPER_ASSETS, RESULTS_RAW):
        os.makedirs(directory, exist_ok=True)
    if attached_data is None:            # local (writable) data dir, maybe empty
        os.makedirs(DATA_DIR, exist_ok=True)
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
        print("data dir         : {}{}".format(
            DATA_DIR, "  (attached dataset, read-only)" if attached_data else ""))
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
        # est_gpu_hours is the projected TOTAL for the whole study in this run
        # mode (nearly all of it notebook 02's training) — NOT what the current
        # notebook will consume. Say so, or a CPU notebook printing "13.0
        # estimated" reads as if the download were about to bill 13 GPU-hours.
        print("GPU-hours        : {:.2f} spent so far across the study; "
              "~{:.1f} projected total for {} mode (almost entirely notebook "
              "02's training{})".format(
                  spent, run.est_gpu_hours, run.mode,
                  "" if gpu.get("available") else "; THIS session is CPU-only "
                  "and spends none"))
        if hydrated["restored"]:
            print("restored {} result file(s) from attached datasets"
                  .format(len(hydrated["restored"])))
        print_disk_report()
        print("=" * 72)
    return run
