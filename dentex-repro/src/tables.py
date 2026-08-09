"""
Table construction for ``paper_assets/tables/``.

Every table is emitted twice — CSV (auditable, diffable) and booktabs LaTeX
(paste-ready) — and every emission registers itself in the manifest.

The original paper's Table 1 is embedded verbatim below. It is the *only*
place in this repository where a number that was not produced by an executed
run is allowed to live, and it is labelled as such everywhere it is used. Cells
we did not run say ``not run``; they are never interpolated, extrapolated or
inferred from a neighbouring mode.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Dict, List, Optional, Sequence

from . import manifest, setup_env

TABLES_DIR = os.path.join(setup_env.PAPER_ASSETS, "tables")

#: Sentinel for a cell whose experiment did not run in this mode.
NOT_RUN = "not run"

# --------------------------------------------------------------------------
# Reference numbers — HierarchicalDet, MICCAI 2023, Table 1 (test split).
# Columns: AR, AP[0.5:0.95], AP50, AP75, APm, APl.
# --------------------------------------------------------------------------
REFERENCE_TABLE_CSV = """\
tier,method,AR,AP,AP50,AP75,APm,APl
quadrant,RetinaNet,0.604,25.1,41.7,28.8,32.9,25.1
quadrant,FasterRCNN,0.588,29.5,48.6,33.0,39.9,29.5
quadrant,DETR,0.659,39.1,60.5,47.6,55.0,39.1
quadrant,DiffusionDet_base,0.677,38.8,60.7,46.1,39.1,39.0
quadrant,Ours_wo_Transfer,0.699,42.7,64.7,52.4,50.5,42.8
quadrant,Ours_wo_Manipulation,0.727,40.0,60.7,48.2,59.3,40.0
quadrant,Ours_wo_Manip_Transfer,0.658,38.1,60.1,45.3,45.1,38.1
quadrant,Ours_full,0.717,43.2,65.1,51.0,68.3,43.1
enumeration,RetinaNet,0.560,25.4,41.5,28.5,55.1,25.2
enumeration,FasterRCNN,0.496,25.6,43.7,27.0,53.3,25.2
enumeration,DETR,0.440,23.1,37.3,26.6,43.4,23.0
enumeration,DiffusionDet_base,0.617,29.9,47.4,34.2,48.6,29.7
enumeration,Ours_wo_Transfer,0.648,32.8,49.4,39.4,60.1,32.9
enumeration,Ours_wo_Manipulation,0.662,30.4,46.5,36.6,58.4,30.5
enumeration,Ours_wo_Manip_Transfer,0.557,26.8,42.4,29.5,51.4,26.5
enumeration,Ours_full,0.668,30.5,47.6,37.1,51.8,30.4
diagnosis,RetinaNet,0.587,32.5,54.2,35.6,41.7,32.5
diagnosis,FasterRCNN,0.533,33.2,54.3,38.0,24.2,33.3
diagnosis,DETR,0.514,33.4,52.8,41.7,48.3,33.4
diagnosis,DiffusionDet_base,0.644,37.0,58.1,42.6,31.8,37.2
diagnosis,Ours_wo_Transfer,0.669,39.4,61.3,47.9,49.7,39.5
diagnosis,Ours_wo_Manipulation,0.688,36.3,55.5,43.1,45.6,37.4
diagnosis,Ours_wo_Manip_Transfer,0.648,37.3,59.5,42.8,33.6,36.4
diagnosis,Ours_full,0.691,37.6,60.2,44.0,36.0,37.7
"""

#: Metric column order used everywhere a per-tier result is tabulated.
METRIC_COLUMNS = ("AR", "AP", "AP50", "AP75", "APm", "APl")
TIER_ORDER = ("quadrant", "enumeration", "diagnosis")


def reference_rows() -> List[Dict[str, object]]:
    """The paper's Table 1 as records, with numeric metric values."""
    rows = []
    for row in csv.DictReader(io.StringIO(REFERENCE_TABLE_CSV)):
        parsed = {"tier": row["tier"], "method": row["method"]}
        for metric in METRIC_COLUMNS:
            parsed[metric] = float(row[metric])
        rows.append(parsed)
    return rows


def reference_lookup() -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{tier: {method: {metric: value}}}``."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in reference_rows():
        out.setdefault(row["tier"], {})[row["method"]] = {
            metric: row[metric] for metric in METRIC_COLUMNS
        }
    return out


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------
def _format_cell(value) -> str:
    if value is None:
        return NOT_RUN
    if isinstance(value, float):
        return "{:.3f}".format(value) if abs(value) < 10 else "{:.2f}".format(value)
    return str(value)


def _latex_escape(text: str) -> str:
    for character, replacement in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                                   ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                                   ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                                   ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(character, replacement)
    return text


def write_table(name: str, rows: Sequence[Dict[str, object]], columns: Sequence[str],
                caption: str, notebook: str, run_mode: str, asset_class: str,
                inputs: Sequence[str] = (), label: Optional[str] = None,
                note: str = "") -> Dict[str, str]:
    """Write ``<name>.csv`` and ``<name>.tex`` and register both."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, name + ".csv")
    tex_path = os.path.join(TABLES_DIR, name + ".tex")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})

    # Right-align a column only if every cell in it reads as a number; text
    # columns (tier, method, metric, ...) stay left-aligned.
    def numeric(column: str) -> bool:
        values = [_format_cell(row.get(column)) for row in rows]
        values = [v for v in values if v not in ("", NOT_RUN)]
        if not values:
            return False
        for value in values:
            try:
                float(value.split(" ")[0])
            except ValueError:
                return False
        return True

    label = label or "tab:" + name
    lines = [
        "% Generated by src/tables.py — do not edit by hand.",
        "% Source rows: {}".format(csv_path),
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + _latex_escape(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + "".join("r" if numeric(c) else "l" for c in columns) + "}",
        r"\toprule",
        " & ".join(_latex_escape(str(c)) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_cell(row.get(c))) for c in columns)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(tex_path, "w") as handle:
        handle.write("\n".join(lines))

    for path in (csv_path, tex_path):
        manifest.record_asset(path, asset_class, notebook, run_mode,
                              inputs=inputs, note=note)
    return {"csv": csv_path, "tex": tex_path}


def record_not_run(asset_class: str, notebook: str, run_mode: str, reason: str) -> None:
    """Account for an asset class that this run mode deliberately skips."""
    placeholder = os.path.join(TABLES_DIR, "_not_run_{}.txt".format(
        asset_class.replace(":", "_")))
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(placeholder, "w") as handle:
        handle.write("{} was not produced in RUN_MODE={}.\nReason: {}\n".format(
            asset_class, run_mode, reason))
    manifest.record_asset(placeholder, asset_class, notebook, run_mode,
                          status="not run", note=reason)


# --------------------------------------------------------------------------
# Builders — each takes already-loaded raw results and returns table rows
# --------------------------------------------------------------------------
def _agg(payload: Dict[str, object], tier: str, metric: str) -> Optional[Dict[str, float]]:
    try:
        cell = payload["aggregate"][tier][metric]
    except (KeyError, TypeError):
        return None
    if cell.get("mean") is None:
        return None
    return cell


def _mean_std(cell: Optional[Dict[str, float]], digits: int = 2) -> object:
    if cell is None:
        return None
    if cell.get("n", 0) <= 1:
        return round(cell["mean"], digits)
    return "{:.{d}f} ± {:.{d}f}".format(cell["mean"], cell["std"], d=digits)


def main_results_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    ``{model_label: multi-seed payload}`` -> one row per (tier, model) with
    mean ± std across inference seeds.
    """
    rows = []
    for tier in TIER_ORDER:
        for label, payload in results.items():
            row: Dict[str, object] = {"tier": tier, "method": label}
            any_value = False
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric)
                digits = 3 if metric == "AR" else 2
                row[metric] = _mean_std(cell, digits)
                any_value = any_value or cell is not None
            row["seeds"] = len(payload.get("seeds", []) or [])
            if any_value:
                rows.append(row)
    return rows


def comparison_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Reproduced vs original, with absolute and relative deltas.

    A method the original reports but we did not run appears with the original
    value and ``not run`` reproduced — never the other way round.
    """
    reference = reference_lookup()
    rows = []
    for tier in TIER_ORDER:
        for method, original in reference.get(tier, {}).items():
            payload = results.get(method)
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric) if payload else None
                ours = cell["mean"] if cell else None
                delta = None if ours is None else round(ours - original[metric], 3)
                relative = (None if ours is None or original[metric] == 0
                            else round(100.0 * (ours - original[metric]) / original[metric], 1))
                rows.append({
                    "tier": tier,
                    "method": method,
                    "metric": metric,
                    "original": original[metric],
                    "reproduced": None if cell is None else round(cell["mean"], 3),
                    "reproduced_std": None if cell is None else round(cell.get("std", 0.0), 3),
                    "delta": delta,
                    "delta_pct": relative,
                })
    return rows


def seed_variance_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for label, payload in results.items():
        for tier in TIER_ORDER:
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric)
                if cell is None:
                    continue
                rows.append({
                    "method": label, "tier": tier, "metric": metric,
                    "n_seeds": cell.get("n"),
                    "mean": round(cell["mean"], 4),
                    "std": round(cell.get("std", 0.0), 4),
                    "min": round(min(cell.get("values", [cell["mean"]])), 4),
                    "max": round(max(cell.get("values", [cell["mean"]])), 4),
                    "values": ";".join("{:.3f}".format(v) for v in cell.get("values", [])),
                })
    return rows


def sweep_rows(payloads: Dict[str, Dict[str, object]], axis_name: str,
               extra: Optional[Dict[str, Dict[str, object]]] = None
               ) -> List[Dict[str, object]]:
    """
    Generic one-axis sweep table (steps, degradation, fault injection,
    clean-vs-stress): one row per (axis value, tier), metrics mean ± std.
    """
    rows = []
    for axis_value, payload in payloads.items():
        for tier in TIER_ORDER:
            row: Dict[str, object] = {axis_name: axis_value, "tier": tier}
            any_value = False
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric)
                row[metric] = _mean_std(cell, 3 if metric == "AR" else 2)
                any_value = any_value or cell is not None
            for key, value in (extra or {}).get(axis_value, {}).items():
                row[key] = value
            if any_value:
                rows.append(row)
    return rows


def audit_rows(audits: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for name, audit in audits.items():
        coverage = audit.get("tier_coverage", {})
        rows.append({
            "split": name,
            "images": audit.get("num_images"),
            "annotations": audit.get("num_annotations"),
            "images_without_annotations": audit.get("images_with_zero_annotations"),
            "quadrant_labels": coverage.get("tier0_quadrant"),
            "enumeration_labels": coverage.get("tier1_enumeration"),
            "diagnosis_labels": coverage.get("tier2_diagnosis"),
            "distinct_resolutions": audit.get("num_distinct_resolutions"),
            "unreadable_images": len(audit.get("unreadable_images", [])),
            "missing_image_files": len(audit.get("missing_image_files", [])),
            "malformed_boxes": len(audit.get("malformed_boxes", [])),
        })
    return rows


def failure_rows(runtimes: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for label, summary in runtimes.items():
        counts = summary.get("failure_counts", {}) or {}
        rows.append({
            "method": label,
            "images": summary.get("images"),
            "detections": summary.get("total_detections"),
            "images_with_no_detections": summary.get("images_with_no_detections"),
            "degenerate_box": counts.get("degenerate_box", 0),
            "box_out_of_image": counts.get("box_out_of_image", 0),
            "box_covers_whole_image": counts.get("box_covers_whole_image", 0),
            "extreme_aspect_ratio": counts.get("extreme_aspect_ratio", 0),
            "crashes": summary.get("crashes", 0),
        })
    return rows


def low_resource_rows(entries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for entry in entries:
        rows.append({
            "method": entry.get("method"),
            "device": entry.get("device"),
            "sample_step": entry.get("sample_step"),
            "seconds_per_image": entry.get("mean_seconds_per_image"),
            "images_per_minute": entry.get("images_per_minute"),
            "peak_gpu_memory_mb": entry.get("peak_gpu_memory_mb"),
            "parameters_millions": entry.get("parameters_millions"),
            "checkpoint_mb": entry.get("checkpoint_mb"),
            "images": entry.get("images"),
        })
    return rows


def per_class_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Per-class AP for every tier. Explicitly OUR extension: the paper reports
    only tier-level aggregates, so these numbers have no reference column.
    """
    rows = []
    for label, payload in results.items():
        runs = payload.get("runs") or []
        if not runs:
            continue
        for tier in TIER_ORDER:
            per_class_by_name: Dict[str, List[float]] = {}
            for run in runs:
                for name, value in (run["tiers"].get(tier, {})
                                    .get("per_class_AP", {}) or {}).items():
                    if isinstance(value, (int, float)):
                        per_class_by_name.setdefault(name, []).append(value)
            for name, values in sorted(per_class_by_name.items()):
                rows.append({
                    "method": label, "tier": tier, "class": name,
                    "AP_mean": round(sum(values) / len(values), 2),
                    "n_seeds": len(values),
                    "note": "OUR EXTENSION — not reported in the original paper",
                })
    return rows


# --------------------------------------------------------------------------
# Claim-coverage matrix (consumed by paper_assets/scope_and_claims.md)
# --------------------------------------------------------------------------
#: Every claim the original paper makes, mapped to how this study treats it.
#: ``requires_models`` downgrades a claim to "cited, untested" automatically
#: when the run mode did not train the models the claim needs — so the scope
#: statement can never overstate what was actually tested.
PAPER_CLAIMS = (
    {
        "claim": "The hierarchical multi-label model beats base DiffusionDet at every tier.",
        "default_status": "tested",
        "evidence": "notebook 05, tables/original_vs_reproduced",
        "requires_models": ["Ours_full", "DiffusionDet_base_tier2"],
    },
    {
        "claim": "Noisy-box manipulation is the component that contributes the accuracy gain.",
        "default_status": "tested",
        "evidence": "notebook 03 (matched budgets) + notebook 05; "
                    "Ours_full vs Ours_wo_Manipulation",
        "requires_models": ["Ours_full", "Ours_wo_Manipulation"],
    },
    {
        "claim": "Weight transfer alone does not improve accuracy.",
        "default_status": "tested",
        "evidence": "Ours_wo_Transfer vs Ours_wo_Manip_Transfer at matched budgets",
        "requires_models": ["Ours_wo_Transfer", "Ours_wo_Manip_Transfer"],
    },
    {
        "claim": "The full model outperforms RetinaNet, Faster R-CNN and DETR at every tier.",
        "default_status": "tested",
        "evidence": "notebook opt-10 baselines",
        "requires_models": ["RetinaNet", "FasterRCNN", "DETR"],
    },
    {
        "claim": "SimMIM pretraining on 1,571 unlabelled X-rays contributes to the result.",
        "default_status": "out of scope",
        "evidence": "the authors' SimMIM checkpoint is not published; notebook opt-11 "
                    "exists for anyone with spare quota",
    },
    {
        "claim": "The reported numbers are obtained on the DENTEX test split (250 images).",
        "default_status": "tested",
        "evidence": "notebook 01 converts and verifies the test ground truth; every "
                    "number in main_results is computed on that split",
    },
    {
        "claim": "The approach is robust enough for panoramic X-ray analysis in practice.",
        "default_status": "extended beyond the paper",
        "evidence": "notebook 06: degradation grid, clean/stress subsets and hierarchy "
                    "fault injection — none of which the original paper tests",
    },
    {
        "claim": "Diffusion sampling steps trade accuracy against latency.",
        "default_status": "extended beyond the paper",
        "evidence": "notebook 06 step sweep with both axes measured",
    },
)


def load_all(kind: str) -> Dict[str, Dict[str, object]]:
    """Load every raw result of a kind, keyed by its distinguishing part."""
    out = {}
    for name in manifest.list_results(kind):
        path = os.path.join(setup_env.RESULTS_RAW, name + ".json")
        with open(path) as handle:
            out[name] = json.load(handle)
    return out
