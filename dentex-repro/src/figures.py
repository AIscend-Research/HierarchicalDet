"""
Figures for ``paper_assets/figures/``.

House rules, enforced by :func:`use_paper_style` and :func:`save_figure`:

* every figure is written as **PDF (vector) and PNG at 300 dpi**;
* colours come from the Okabe–Ito colourblind-safe palette, and every encoding
  by colour is also encoded by marker or hatch, so the figures survive
  greyscale printing;
* base font is 9 pt at the final size (single-column width 3.4 in), so nothing
  falls below the 8 pt floor after typesetting;
* every saved figure registers itself in the manifest.
"""
from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional, Sequence

from . import manifest, setup_env

FIGURES_DIR = os.path.join(setup_env.PAPER_ASSETS, "figures")

#: Okabe–Ito, colourblind-safe.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

SINGLE_COLUMN = 3.4      # inches
DOUBLE_COLUMN = 7.0


def use_paper_style() -> None:
    import matplotlib

    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,      # embed TrueType, not Type 3 - required by most venues
        "ps.fonttype": 42,
        # Matplotlib renders negative ticks with U+2212 by default, which is one
        # of the glyphs assert_no_banned_glyphs rejects. Emit ASCII instead, so
        # the axis and the guard agree.
        "axes.unicode_minus": False,
    })


#: Characters banned from anything a figure renders or ships. Em and en dashes
#: are the ones that keep reappearing; a few fonts drop them entirely, and they
#: are trivially confusable with a minus sign on an axis.
BANNED_GLYPHS = {"—": " - ", "–": "-", "−": "-"}


def scrub(text: str) -> str:
    """Replace banned glyphs in any string destined for a figure."""
    for bad, good in BANNED_GLYPHS.items():
        text = text.replace(bad, good)
    return text


def assert_no_banned_glyphs(fig) -> None:
    """
    Fail if any rendered text in the figure contains a banned glyph.

    Checked on the figure itself rather than on the source strings, so titles,
    axis labels, tick labels, legend entries and annotations are all covered no
    matter where they were set.
    """
    offenders = []
    for axis in fig.get_axes():
        texts = [axis.title, axis.xaxis.label, axis.yaxis.label]
        texts += list(axis.texts) + list(axis.get_xticklabels()) + list(axis.get_yticklabels())
        legend = axis.get_legend()
        if legend is not None:
            texts += list(legend.get_texts())
        for item in texts:
            content = item.get_text() or ""
            if any(bad in content for bad in BANNED_GLYPHS):
                offenders.append(content)
    for item in getattr(fig, "texts", []):
        if any(bad in (item.get_text() or "") for bad in BANNED_GLYPHS):
            offenders.append(item.get_text())
    if offenders:
        raise ValueError(
            "figure text contains banned glyphs (em/en dash or unicode minus): {}. "
            "Use figures.scrub() or plain ASCII.".format(offenders)
        )


def save_figure(fig, name: str, caption: str, notebook: str, run_mode: str,
                asset_class: str, inputs: Sequence[str] = (),
                note: str = "") -> Dict[str, str]:
    assert_no_banned_glyphs(fig)
    caption = scrub(caption)
    note = scrub(note)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, name + ".pdf")
    png_path = os.path.join(FIGURES_DIR, name + ".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    caption_path = os.path.join(FIGURES_DIR, name + ".caption.txt")
    with open(caption_path, "w") as handle:
        handle.write(caption.strip() + "\n")
    for path in (pdf_path, png_path):
        manifest.record_asset(path, asset_class, notebook, run_mode,
                              inputs=inputs, note=note or caption)
    return {"pdf": pdf_path, "png": png_path, "caption": caption_path}


def record_not_run(asset_class: str, notebook: str, run_mode: str, reason: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    placeholder = os.path.join(FIGURES_DIR, "_not_run_{}.txt".format(
        asset_class.replace(":", "_")))
    with open(placeholder, "w") as handle:
        handle.write("{} was not produced in RUN_MODE={}.\nReason: {}\n".format(
            asset_class, run_mode, reason))
    manifest.drop_missing(asset_class)
    manifest.record_asset(placeholder, asset_class, notebook, run_mode,
                          status="not run", note=reason)


# --------------------------------------------------------------------------
# Tick labels
# --------------------------------------------------------------------------
def fit_xticklabels(ax, labels: Sequence[str], rotation: int = 0,
                    fallback: int = 45) -> int:
    """
    Set x tick labels, rotating them only if flat ones would collide.

    The enumeration tier has eight labels of the form ``1Enumeration``; laid out
    flat they ran into each other and the axis read as one unbroken word. A
    rotation hard-coded per tier is a guess about how wide the labels will be at
    the final figure size, so measure the rendered extents and rotate only when
    they actually overlap. Returns the rotation used.
    """
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="center" if rotation in (0, 90) else "right")
    if rotation != 0:
        return rotation
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = sorted((item.get_window_extent(renderer)
                    for item in ax.get_xticklabels() if item.get_text()),
                   key=lambda box: box.x0)
    collides = any(later.x0 < earlier.x1 + 2.0
                   for earlier, later in zip(boxes, boxes[1:]))
    if collides:
        ax.set_xticklabels(labels, rotation=fallback, ha="right")
        return fallback
    return rotation


# --------------------------------------------------------------------------
# Legend placement
# --------------------------------------------------------------------------
def legend_below(axes, handles=None, labels=None, ncol: int = 3, pad: float = 0.04):
    """
    Put the legend under ``axes``, clear of whatever is already down there.

    A fixed offset in axes fractions cannot do this. ``bbox_to_anchor=(0.5,
    -0.28)`` is measured from the top of the axes box, so it lands on the tick
    labels as soon as they are rotated or long, and it lands *inside* the data
    whenever the axes are short -- which is how every legend in the shipped
    figures ended up sitting on the curves it was labelling.

    So measure instead: draw once, find the lowest rendered extent of the tick
    labels and the axis label, and anchor the legend below that in figure
    coordinates. ``savefig(bbox_inches="tight")`` then crops to include it.

    ``axes`` may be a single axes or a sequence, in which case the legend is
    centred under the whole row.
    """
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    anchor = axes[0]
    figure = anchor.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()

    bottom = min(axis.get_position().y0 for axis in axes)
    for axis in axes:
        below = list(axis.get_xticklabels()) + [axis.xaxis.label]
        for item in below:
            if not item.get_text():
                continue
            box = item.get_window_extent(renderer).transformed(to_figure)
            bottom = min(bottom, box.y0)

    left = min(axis.get_position().x0 for axis in axes)
    right = max(axis.get_position().x1 for axis in axes)
    if handles is None:
        handles, labels = anchor.get_legend_handles_labels()
    return anchor.legend(handles, labels, loc="upper center",
                         bbox_to_anchor=((left + right) / 2.0, bottom - pad),
                         bbox_transform=figure.transFigure,
                         ncol=ncol, borderaxespad=0.0)


# --------------------------------------------------------------------------
# Generic series plotting
# --------------------------------------------------------------------------
def line_series(ax, series: Sequence[Dict[str, object]], xlabel: str, ylabel: str,
                logx: bool = False) -> None:
    """
    ``series`` items: ``{"label": str, "x": [...], "y": [...], "yerr": [...]|None}``.
    Colour and marker vary together so the plot reads in greyscale.
    """
    for index, entry in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        marker = MARKERS[index % len(MARKERS)]
        yerr = entry.get("yerr")
        if yerr and any(e for e in yerr):
            ax.errorbar(entry["x"], entry["y"], yerr=yerr, label=entry["label"],
                        color=colour, marker=marker, capsize=2, elinewidth=0.8)
        else:
            ax.plot(entry["x"], entry["y"], label=entry["label"],
                    color=colour, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)


def sweep_figure(series: Sequence[Dict[str, object]], xlabel: str, ylabel: str,
                 title: str, logx: bool = False, width: float = SINGLE_COLUMN):
    import matplotlib.pyplot as plt

    use_paper_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    line_series(ax, series, xlabel, ylabel, logx=logx)
    ax.set_title(title)
    # Outside the axes: with error bars these curves fill the panel, and an
    # in-plot legend sat on top of the data in every rendered figure.
    fig.tight_layout()
    legend_below(ax, ncol=min(3, len(series)))
    return fig


def ap_vs_steps_figure(series: Sequence[Dict[str, object]],
                       latency: Optional[Dict[int, float]] = None):
    """
    AP against diffusion sampling steps, with measured latency on a twin axis.

    Both axes are measured: the latency line is per-image seconds recorded in
    the same runs that produced the AP points, never an assumed cost model.
    """
    import matplotlib.pyplot as plt

    use_paper_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.78))
    line_series(ax, series, "diffusion sampling steps", "AP [0.5:0.95]", logx=True)
    handles, labels = ax.get_legend_handles_labels()
    if latency:
        twin = ax.twinx()
        steps = sorted(latency)
        twin.plot(steps, [latency[s] for s in steps], color="#666666",
                  linestyle="--", marker="x", label="latency")
        twin.set_ylabel("seconds / image (measured)")
        twin.grid(False)
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        handles, labels = handles + twin_handles, labels + twin_labels
    ax.set_xticks(sorted({x for entry in series for x in entry["x"]}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    fig.tight_layout()
    legend_below(ax, handles, labels, ncol=2)
    return fig


def grouped_bars(categories: Sequence[str], groups: Dict[str, Sequence[float]],
                 ylabel: str, title: str, width: float = DOUBLE_COLUMN,
                 rotate: int = 90):
    """
    Grouped bar chart (per-class AP, error clusters).

    A category with **no ground truth at all** is not a category the model
    failed on, and it has no bar to draw at any height. COCOeval reports NaN for
    such a class; callers drop those categories before plotting and name them in
    the caption, because an empty slot mid-axis reads as a measured zero.

    Values may still be NaN or None for an individual series, in which case that
    one bar is simply not drawn.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    use_paper_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.34))
    positions = np.arange(len(categories))
    # None means "not measured" and must not be plotted as a value; NaN is how
    # matplotlib says "draw nothing here".
    groups = {label: [float("nan") if value is None else float(value)
                      for value in values]
              for label, values in groups.items()}
    count = max(1, len(groups))
    bar_width = 0.8 / count
    hatches = ("", "//", "..", "xx", "\\\\", "++")
    for index, (label, values) in enumerate(groups.items()):
        offsets = positions + index * bar_width - 0.4 + bar_width / 2
        colour = PALETTE[index % len(PALETTE)]
        ax.bar(offsets, values, width=bar_width, label=label, color=colour,
               hatch=hatches[index % len(hatches)], edgecolor="white", linewidth=0.4)
        # A bar of height exactly zero is invisible, and an invisible bar reads
        # as an absent one -- the very distinction this figure is drawn to make.
        # Mark measured zeros with a rule on the axis so "we scored it and it is
        # zero" cannot be confused with "there was nothing to score".
        for offset, value in zip(offsets, values):
            if value == 0:
                ax.plot([offset - bar_width / 2, offset + bar_width / 2], [0, 0],
                        color=colour, linewidth=1.6, solid_capstyle="butt",
                        zorder=4, clip_on=False)
    ax.set_xticks(positions)
    fit_xticklabels(ax, categories, rotate)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    drawn = [value for values in groups.values() for value in values if value == value]
    if not drawn or not any(drawn):
        # An empty or all-zero chart renders as a blank box with no explanation,
        # and a reader fills that in with the flattering reading. Say which it
        # is: nothing measurable, or measured and zero.
        message = "no values to plot" if not drawn else "all values are zero"
        if not drawn:
            # Otherwise the y axis autoscales to a meaningless +/- 0.05 around
            # nothing at all.
            ax.set_ylim(0.0, 1.0)
        # Wrapped by hand: matplotlib's wrap= measures against the figure width,
        # so the note ran out over the tick labels and past the panel edge.
        ax.text(0.5, 0.5, textwrap.fill(message, 38), transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="#666666")
    fig.tight_layout()
    if len(groups) > 1:
        # Below the axes, not inside them: with five methods the in-plot legend
        # overlapped the title and the tallest bars, and entries collided with
        # each other. Measured, because the rotated class names are tall.
        legend_below(ax, ncol=min(3, len(groups)))
    return fig


def degradation_figure(by_kind: Dict[str, List[Dict[str, float]]],
                       baseline: Optional[float], ylabel: str):
    """
    One panel per corruption type, each with its own severity axis.

    Blur sigma, JPEG quality and downscale factor are different quantities in
    different units and opposite directions (higher sigma is worse, higher JPEG
    quality is *better*). Drawing them against one shared x put JPEG quality 50
    and blur sigma 4 at the same tick and squeezed the downscale points into the
    left margin -- a figure whose own caption had to warn that its axis meant
    nothing. Separate panels with a shared y let the curves be compared on the
    axis that is actually common: accuracy.
    """
    import matplotlib.pyplot as plt

    use_paper_style()
    kinds = [k for k in KIND_ORDER if by_kind.get(k)]
    if not kinds:
        kinds = sorted(by_kind)
    fig, axes = plt.subplots(1, len(kinds), figsize=(DOUBLE_COLUMN, 2.3), sharey=True)
    if len(kinds) == 1:
        axes = [axes]
    for index, (axis, kind) in enumerate(zip(axes, kinds)):
        points = sorted(by_kind[kind], key=lambda p: p["severity"])
        axis.errorbar([p["severity"] for p in points], [p["mean"] for p in points],
                      yerr=[p.get("std") or 0.0 for p in points],
                      color=PALETTE[index % len(PALETTE)],
                      marker=MARKERS[index % len(MARKERS)], capsize=2, elinewidth=0.8)
        if baseline is not None:
            axis.axhline(baseline, color="#666666", linestyle=":", linewidth=1.0,
                         label="clean" if index == 0 else None)
        axis.set_title(KIND_TITLES.get(kind, kind))
        axis.set_xlabel(KIND_XLABEL.get(kind, "severity"))
        if index == 0:
            axis.set_ylabel(ylabel)
    fig.tight_layout()
    if baseline is not None:
        # Under the row rather than loc="best" inside panel 0: "best" only
        # avoids the data it can see, and the baseline it labels is a rule
        # spanning the full width of every panel.
        legend_below(list(axes), ncol=1)
    return fig


#: Panel order and axis wording for the degradation figure.
KIND_ORDER = ("blur", "jpeg", "downscale")
KIND_TITLES = {"blur": "Gaussian blur", "jpeg": "JPEG recompression",
               "downscale": "Downscaling"}
KIND_XLABEL = {"blur": "sigma (higher = blurrier)",
               "jpeg": "quality (lower = worse)",
               "downscale": "scale factor (lower = worse)"}


def trajectory_figure(trajectories: Dict[str, Dict[str, List]], tiers: Sequence[str]):
    """
    Per-tier AP against training progress, one line per ablation variant.

    Stable ordering across checkpoints is direct evidence that the ablation
    conclusion is not an artifact of the shortened schedule; unstable ordering
    is itself the finding.
    """
    import matplotlib.pyplot as plt

    use_paper_style()
    fig, axes = plt.subplots(1, len(tiers), figsize=(DOUBLE_COLUMN, 2.4), sharey=False)
    if len(tiers) == 1:
        axes = [axes]
    for axis, tier in zip(axes, tiers):
        series = []
        for variant, data in trajectories.items():
            points = data.get(tier) or []
            if not points:
                continue
            series.append({
                "label": variant,
                "x": [p["progress"] for p in points],
                "y": [p["AP"] for p in points],
                "yerr": [p.get("AP_std") or 0.0 for p in points],
            })
        line_series(axis, series, "fraction of training budget", "AP [0.5:0.95]")
        axis.set_title(tier)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_below(list(axes), handles, labels, ncol=min(3, len(labels)))
    return fig


# --------------------------------------------------------------------------
# Image overlays
# --------------------------------------------------------------------------
def draw_boxes(ax, boxes: Sequence[Sequence[float]], labels: Sequence[str],
               colour: str, linestyle: str = "-", fontsize: int = 5) -> None:
    """Draw COCO XYWH boxes with short labels."""
    from matplotlib.patches import Rectangle

    for box, label in zip(boxes, labels):
        x, y, width, height = box
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=colour,
                               linewidth=0.8, linestyle=linestyle))
        if label:
            ax.text(x, max(0, y - 2), label, color=colour, fontsize=fontsize,
                    va="bottom", ha="left",
                    bbox=dict(facecolor="white", alpha=0.6, pad=0.4, linewidth=0))


def overlay_grid(panels: Sequence[Dict[str, object]], columns: int = 4,
                 title: str = "", panel_height: float = 1.9):
    """
    Grid of annotated X-rays.

    ``panels`` items: ``{"image_path", "title", "gt": [(box,label)],
    "pred": [(box,label)]}``. Ground truth is drawn solid blue, predictions
    dashed orange — encoded by both colour and line style.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    use_paper_style()
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns,
                             figsize=(DOUBLE_COLUMN, panel_height * rows))
    axes = [axes] if rows * columns == 1 else list(axes.flat)
    for axis, panel in zip(axes, panels):
        with Image.open(panel["image_path"]) as image:
            axis.imshow(image, cmap="gray")
        gt = panel.get("gt") or []
        pred = panel.get("pred") or []
        draw_boxes(axis, [b for b, _ in gt], [l for _, l in gt], PALETTE[0], "-")
        draw_boxes(axis, [b for b, _ in pred], [l for _, l in pred], PALETTE[1], "--")
        axis.set_title(panel.get("title", ""), fontsize=7)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig
