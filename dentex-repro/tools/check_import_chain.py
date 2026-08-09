#!/usr/bin/env python3
"""
Prove the environment cell's import chain works — by running it.

Two Kaggle sessions in a row died inside that cell, both on problems a local
execution of the same imports would have caught (an ABI-broken pycocotools
extension; a hand-curated requirements list missing ``fairscale``). This script
exists so that never has to be discovered on Kaggle again:

1. **Coverage check (static).** AST-scan the vendored ``detectron2`` /
   ``hierarchialdet`` / ``pycocotools`` trees for module-level third-party
   imports and assert ``setup_env.REQUIREMENTS`` covers every one that is not
   explicitly exempted (with a reason) — so the next vendored import added
   upstream fails HERE, not four imports deep on Kaggle.

2. **Import chain (dynamic).** In a subprocess with the repo root first on
   ``sys.path`` — exactly how the notebooks run — execute the same imports as
   the notebooks' environment cell, plus the deeper modules training and
   evaluation pull in.

Run it inside any environment that has the requirements installed:

    python tools/check_import_chain.py

CI-style usage before pushing: create a fresh venv, let
``setup_env.install_dependencies()`` + ``ensure_pycocotools_mask()`` populate
it, then run this. If it exits 0 there, the environment cell passes on Kaggle
up to Kaggle-specific state (GPU presence, image contents), because these are
the same imports in the same order.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.setup_env import REQUIREMENTS  # noqa: E402

#: Modules found by the scan whose importing files are NOT on any import path
#: this suite executes. Each exemption names the file and the reason — a module
#: may only be exempted if importing it is genuinely unreachable from
#: ``hierarchialdet`` / ``train_net_patched`` / ``evaluator`` / ``src``.
EXEMPT = {
    "caffe2": "detectron2/export/* only; the export package is never imported",
    "onnx": "detectron2/export/* only; the export package is never imported",
    "panopticapi": "detectron2/projects/panoptic_deeplab only; projects are never imported",
    "mask": "pycocotools/coco_edited.py is a dead vendored variant, never imported",
    "torch": "supplied by the host image; never installed by this suite",
    "torchvision": "supplied by the host image; never installed by this suite",
    "numpy": "supplied by the host image; never installed by this suite",
}

LOCAL = {"detectron2", "hierarchialdet", "pycocotools", "evaluator", "train_net",
         "train_net_patched", "demo", "src", "tools", "baselines"}

#: The dynamic proof: the environment cell's imports verbatim, then the deeper
#: modules the training/eval code paths pull in at import time.
IMPORT_CHAIN = [
    "import detectron2, pycocotools, evaluator",
    "from hierarchialdet.util.coco_3class_eval import COCOEvaluator",
    "from hierarchialdet.dataset_mapper_patched import DiffusionDetDatasetMapper",
    "from hierarchialdet import add_diffusiondet_config, DiffusionDetWithTTA",
    "from hierarchialdet.util.model_ema import add_model_ema_configs",
    "import train_net_patched",
    "from detectron2.engine import launch, default_argument_parser",
    "from detectron2.checkpoint import DetectionCheckpointer",
    "from detectron2.data.datasets import register_coco_instances",
    "from src import setup_env, data_convert, registration, train_utils",
    "from src import eval_utils, degradations, figures, tables, manifest",
]


def module_level_imports() -> dict:
    """{module: [files]} for every module-level third-party import."""
    stdlib = set(sys.stdlib_module_names)
    found: dict = {}

    def scan(path: str) -> None:
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            return
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in stdlib or name in LOCAL or name.startswith("_"):
                    continue
                found.setdefault(name, []).append(os.path.relpath(path, REPO_ROOT))

    for tree_root in ("detectron2", "hierarchialdet", "pycocotools"):
        for dirpath, _dirs, files in os.walk(os.path.join(REPO_ROOT, tree_root)):
            for fname in files:
                if fname.endswith(".py"):
                    scan(os.path.join(dirpath, fname))
    for fname in ("evaluator.py", "train_net_patched.py"):
        scan(os.path.join(REPO_ROOT, fname))
    return found


def check_coverage() -> bool:
    provided = {module for module, _spec in REQUIREMENTS}
    needed = module_level_imports()
    ok = True
    for module in sorted(needed):
        if module in provided:
            status = "covered"
        elif module in EXEMPT:
            status = "exempt: {}".format(EXEMPT[module])
        else:
            status = "** NOT COVERED — add to setup_env.REQUIREMENTS **"
            ok = False
        print("  {:14s} {:3d} file(s)  {}".format(module, len(needed[module]), status))
    stale = provided - set(needed) - {"huggingface_hub", "tensorboard", "psutil",
                                      "tqdm", "pkg_resources", "packaging"}
    for module in sorted(stale):
        print("  {:14s} note: not found in vendored module-level imports "
              "(kept for src/ or runtime use)".format(module))
    return ok


def check_import_chain() -> bool:
    """Run the chain in a subprocess with the notebooks' exact sys.path setup."""
    code = "import sys; sys.path.insert(0, {root!r}); sys.path.insert(1, {proj!r})\n".format(
        root=REPO_ROOT, proj=PROJECT_ROOT)
    code += "\n".join(IMPORT_CHAIN)
    code += "\nprint('IMPORT CHAIN OK')\n"
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, cwd=os.sep)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stdout.write(result.stderr[-4000:] + "\n")
    return result.returncode == 0


def main() -> int:
    print("== 1/2 static coverage: vendored imports vs setup_env.REQUIREMENTS ==")
    coverage_ok = check_coverage()
    print("\n== 2/2 dynamic proof: executing the environment cell's import chain ==")
    chain_ok = check_import_chain()
    print("\ncoverage: {}   import chain: {}".format(
        "OK" if coverage_ok else "FAIL", "OK" if chain_ok else "FAIL"))
    return 0 if (coverage_ok and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
