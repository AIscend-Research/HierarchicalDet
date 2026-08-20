"""
The machine-readable index of everything the paper is allowed to cite.

Rule enforced here: **nothing lands in the paper unless it is in the
manifest.** Every table, figure and raw metric file is registered with the
notebook that produced it, the run mode, the config hash, the timestamp and its
upstream inputs. :func:`assert_asset_classes` is what notebook 09 calls to
prove the output contract was met, and it fails loudly on a missing class
rather than quietly shipping a thinner paper.

This module also owns the naming convention for raw result files, so producers
(notebooks 05–08) and the consumer (notebook 09) cannot drift apart.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

from . import setup_env

MANIFEST_PATH = os.path.join(setup_env.PAPER_ASSETS, "manifest.json")

#: Asset classes the output contract requires. Notebook 09 asserts each is
#: present (or explicitly marked "not run" for the active run mode).
REQUIRED_ASSET_CLASSES = (
    "table:main_results",
    "table:original_vs_reproduced",
    "table:seed_variance",
    "table:step_sweep",
    "table:degradation",
    "table:clean_vs_stress",
    "table:fault_injection",
    "table:low_resource",
    "table:dataset_audit",
    "table:failure_counts",
    "figure:ap_vs_steps",
    "figure:degradation",
    "figure:fault_injection",
    "figure:per_class_ap",
    "figure:qualitative",
    "figure:gt_sanity",
    "figure:error_clusters",
    "figure:checkpoint_trajectory",
    "doc:repro_checklist",
    "doc:deviations",
    "doc:scope_and_claims",
)


# --------------------------------------------------------------------------
# Naming convention for raw result files
# --------------------------------------------------------------------------
def result_name(kind: str, **parts) -> str:
    """
    Canonical name of a raw result file (without extension).

    ``result_name("eval_main", model="diagnosis_full")`` -> ``eval_main__model-diagnosis_full``
    """
    suffix = "__".join(
        "{}-{}".format(key, str(value).replace(os.sep, "_"))
        for key, value in sorted(parts.items())
    )
    return "{}__{}".format(kind, suffix) if suffix else kind


def list_results(kind: Optional[str] = None) -> List[str]:
    """Names (without extension) of raw result files, optionally by kind."""
    if not os.path.isdir(setup_env.RESULTS_RAW):
        return []
    names = [f[:-5] for f in sorted(os.listdir(setup_env.RESULTS_RAW))
             if f.endswith(".json")]
    if kind:
        names = [n for n in names if n == kind or n.startswith(kind + "__")]
    return names


def parse_result_name(name: str) -> Dict[str, str]:
    """Inverse of :func:`result_name`."""
    head, _, tail = name.partition("__")
    parts = {"kind": head}
    if tail:
        for chunk in tail.split("__"):
            key, _, value = chunk.partition("-")
            parts[key] = value
    return parts


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def load_manifest() -> Dict[str, object]:
    if not os.path.exists(MANIFEST_PATH):
        return {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "assets": []}
    with open(MANIFEST_PATH) as handle:
        return json.load(handle)


def _save(manifest: Dict[str, object]) -> None:
    os.makedirs(setup_env.PAPER_ASSETS, exist_ok=True)
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def record_asset(path: str, asset_class: str, notebook: str, run_mode: str,
                 inputs: Sequence[str] = (), config_hash: Optional[str] = None,
                 note: str = "", status: str = "produced") -> Dict[str, object]:
    """
    Register one produced asset. Re-registering the same path replaces the
    previous entry, so re-running a notebook updates rather than duplicates.

    ``status="not run"`` records that an asset class was deliberately skipped in
    this run mode — which is how a table cell is allowed to say "not run"
    instead of carrying a guessed number.
    """
    manifest = load_manifest()
    absolute = os.path.abspath(path)
    entry = {
        "path": os.path.relpath(absolute, setup_env.PROJECT_ROOT),
        "asset_class": asset_class,
        "notebook": notebook,
        "run_mode": run_mode,
        "status": status,
        "config_hash": config_hash,
        "inputs": [os.path.relpath(os.path.abspath(i), setup_env.PROJECT_ROOT)
                   if os.path.exists(str(i)) else str(i) for i in inputs],
        "note": note,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sha256": (setup_env.file_sha256(absolute)
                   if status == "produced" and os.path.exists(absolute) else None),
        "bytes": (os.path.getsize(absolute)
                  if status == "produced" and os.path.exists(absolute) else None),
    }
    manifest["assets"] = [a for a in manifest.get("assets", [])
                          if a.get("path") != entry["path"]]
    manifest["assets"].append(entry)
    _save(manifest)
    return entry


def missing_assets() -> Dict[str, List[str]]:
    """
    Asset classes the manifest claims to have produced whose files are absent.

    A manifest that cites a figure nobody can open is worse than one that says
    the figure was not produced: the citation is what a reader trusts.
    """
    missing: Dict[str, List[str]] = {}
    for asset in load_manifest().get("assets", []):
        if asset.get("status") != "produced":
            continue
        path = asset.get("path", "")
        if not os.path.exists(os.path.join(setup_env.PROJECT_ROOT, path)):
            missing.setdefault(asset.get("asset_class", "?"), []).append(path)
    return missing


def drop_missing(asset_class: str) -> int:
    """
    Forget entries of ``asset_class`` whose file is no longer on disk.

    Needed when an asset class stops being produced: ``record_asset`` replaces
    entries by path, so a class that goes from "produced" to "not run" keeps its
    old rows, and the manifest goes on claiming a figure that is not there. Only
    entries whose file is actually absent are dropped, so this can never discard
    a genuine record. Returns how many were removed.
    """
    manifest = load_manifest()
    assets = manifest.get("assets", [])
    kept = [a for a in assets
            if a.get("asset_class") != asset_class
            or a.get("status") != "produced"
            or os.path.exists(os.path.join(setup_env.PROJECT_ROOT, a.get("path", "")))]
    removed = len(assets) - len(kept)
    if removed:
        manifest["assets"] = kept
        _save(manifest)
    return removed


def record_environment(payload: Dict[str, object]) -> None:
    manifest = load_manifest()
    manifest["environment"] = payload
    _save(manifest)


def record_run(name: str, payload: Dict[str, object]) -> None:
    """Register a training/evaluation run so assets can point back at it."""
    manifest = load_manifest()
    runs = manifest.setdefault("runs", {})
    runs[name] = payload
    _save(manifest)


def assert_asset_classes(required: Sequence[str] = REQUIRED_ASSET_CLASSES) -> Dict[str, object]:
    """
    Fail if any required asset class is absent from the manifest entirely.
    A class explicitly recorded with ``status="not run"`` counts as present:
    the contract is that every class is *accounted for*, not that every
    experiment ran in every mode.
    """
    manifest = load_manifest()
    present = {}
    for asset in manifest.get("assets", []):
        present.setdefault(asset["asset_class"], []).append(asset)
    missing = [name for name in required if name not in present]
    if missing:
        raise AssertionError(
            "manifest is missing required asset classes: {}. Nothing may be cited "
            "in the paper that is not in the manifest, so either produce these or "
            "record them explicitly with status='not run'.".format(missing)
        )
    return {
        "classes": {name: len(present[name]) for name in required},
        "not_run": sorted({a["asset_class"] for a in manifest.get("assets", [])
                           if a.get("status") == "not run"}),
        "total_assets": len(manifest.get("assets", [])),
    }


def summary_table() -> List[Dict[str, object]]:
    """Flat view of the manifest, for printing at the end of notebook 09."""
    manifest = load_manifest()
    return [
        {
            "asset_class": asset["asset_class"],
            "path": asset["path"],
            "notebook": asset["notebook"],
            "run_mode": asset["run_mode"],
            "status": asset["status"],
        }
        for asset in sorted(manifest.get("assets", []),
                            key=lambda a: (a["asset_class"], a["path"]))
    ]
