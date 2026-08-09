"""
Image degradations and hierarchy fault injection — the two robustness axes.

**Image degradations.** Blur, JPEG recompression and downscaling are applied to
the *test images only*; annotations are untouched. Downscaling resamples back
up to the original resolution afterwards, so the ground-truth boxes stay valid
and the experiment measures loss of detail rather than a coordinate mismatch.

**Hierarchy fault injection.** The paper's contribution is a chain: the
enumeration model's detections steer the diagnosis model's noisy boxes. This
module perturbs that intermediate signal — Gaussian jitter on the box
coordinates and random dropping of detections — so the downstream cost of an
imperfect earlier tier can be measured instead of assumed. The perturbation
itself lives in the detector (behind ``NOISY_BOX_INFER_*``); here we only
enumerate the grid and name the conditions.
"""
from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional, Sequence

from . import setup_env

KINDS = ("blur", "jpeg", "downscale")


def condition_label(kind: str, severity) -> str:
    if kind == "blur":
        return "blur_sigma{:g}".format(severity)
    if kind == "jpeg":
        return "jpeg_q{:d}".format(int(severity))
    if kind == "downscale":
        return "downscale_{:d}pct".format(int(round(float(severity) * 100)))
    raise ValueError("unknown degradation kind {!r}".format(kind))


def pretty_label(kind: str, severity) -> str:
    if kind == "blur":
        return "Gaussian blur σ={:g}".format(severity)
    if kind == "jpeg":
        return "JPEG quality {}".format(int(severity))
    if kind == "downscale":
        return "downscale to {:.0f}%".format(float(severity) * 100)
    raise ValueError("unknown degradation kind {!r}".format(kind))


def degrade_image(image, kind: str, severity):
    """Apply one degradation to a PIL image, returning a new PIL image."""
    from PIL import Image, ImageFilter

    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        return Image.open(buffer).convert(image.mode)
    if kind == "downscale":
        factor = float(severity)
        if not 0 < factor <= 1:
            raise ValueError("downscale severity must be in (0, 1]")
        width, height = image.size
        small = image.resize((max(1, int(width * factor)), max(1, int(height * factor))),
                             Image.BICUBIC)
        # Resample back so annotations remain valid in original coordinates:
        # this measures lost detail, not a changed coordinate system.
        return small.resize((width, height), Image.BICUBIC)
    raise ValueError("unknown degradation kind {!r}".format(kind))


def build_degraded_images(source_dir: str, kind: str, severity, out_root: str,
                          file_names: Optional[Sequence[str]] = None) -> str:
    """
    Write a degraded copy of a split's images. Idempotent and marker-gated, so
    a rerun after a session kill does not redo finished conditions.
    """
    from PIL import Image

    destination = os.path.join(out_root, condition_label(kind, severity))
    marker = destination + ".done"
    if os.path.exists(marker):
        return destination
    os.makedirs(destination, exist_ok=True)

    names = list(file_names) if file_names else sorted(
        f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    for name in names:
        target = os.path.join(destination, name)
        if os.path.exists(target):
            continue
        with Image.open(os.path.join(source_dir, name)) as image:
            image.load()
            degraded = degrade_image(image, kind, severity)
            # Always PNG on disk: the JPEG condition's loss is already baked in
            # by the round-trip above, and re-encoding here would compound it.
            degraded.save(target, format="PNG")
    open(marker, "w").close()
    return destination


def degradation_grid(run_config) -> List[Dict[str, object]]:
    """The (kind, severity) conditions this run mode evaluates, plus the baseline."""
    grid = [{"kind": "none", "severity": None, "label": "clean",
             "pretty": "no degradation"}]
    for kind, severity in run_config.degradations:
        grid.append({
            "kind": kind, "severity": severity,
            "label": condition_label(kind, severity),
            "pretty": pretty_label(kind, severity),
        })
    return grid


# --------------------------------------------------------------------------
# Hierarchy fault injection
# --------------------------------------------------------------------------
def fault_grid(run_config) -> List[Dict[str, object]]:
    """
    Conditions for "how brittle is the hierarchical chain?".

    Jitter and drop are swept one at a time from a shared zero point, so each
    curve isolates one failure mode of the intermediate tier: displaced boxes
    versus missing boxes.
    """
    conditions = []
    for jitter in run_config.fault_jitters:
        conditions.append({
            "axis": "jitter", "jitter": float(jitter), "drop": 0.0,
            "label": "jitter_{:.2f}".format(float(jitter)),
            "pretty": "prior-tier box jitter σ={:.2f} (normalized coords)".format(float(jitter)),
        })
    for drop in run_config.fault_drops:
        if float(drop) == 0.0:
            continue                      # already covered by the jitter=0 point
        conditions.append({
            "axis": "drop", "jitter": 0.0, "drop": float(drop),
            "label": "drop_{:.2f}".format(float(drop)),
            "pretty": "prior-tier detections randomly dropped: {:.0f}%".format(float(drop) * 100),
        })
    return conditions


def summarize_prediction_file(path: str) -> Dict[str, object]:
    """Box-count and score summary of an injected prior-tier prediction file."""
    with open(path) as handle:
        records = json.load(handle)
    per_image: Dict[int, int] = {}
    for record in records:
        per_image[record["image_id"]] = per_image.get(record["image_id"], 0) + 1
    scores = sorted(r.get("score", 1.0) for r in records)
    return {
        "path": os.path.abspath(path),
        "sha256": setup_env.file_sha256(path),
        "boxes": len(records),
        "images_covered": len(per_image),
        "median_boxes_per_image": (sorted(per_image.values())[len(per_image) // 2]
                                   if per_image else 0),
        "median_score": round(scores[len(scores) // 2], 4) if scores else None,
    }
