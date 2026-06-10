"""
Phase 3 — Prediction Visualization: Predicted vs Ground-Truth Bounding Boxes

Overlays predicted and ground-truth bounding boxes on panoramic X-rays for
qualitative comparison across all three detection tiers simultaneously.

For each image, produces a three-panel figure:
  Left panel   — Tier 1: quadrant predictions vs GT
  Middle panel — Tier 2: enumeration predictions vs GT
  Right panel  — Tier 3: diagnosis predictions vs GT

Also produces a single-image view for the "best" and "worst" predictions
(by F1 score approximation) for the paper's qualitative figure.

Usage:
    export DATA_ROOT=/path/to/sorted/challenge
    python phase3_visualize_predictions.py \
        --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml \
        --weights-t1 output/tier1/model_final.pth \
        --weights-t2 output/tier2/model_final.pth \
        --weights-t3 output/tier3/model_final.pth \
        --val-json /path/to/validation_triple.json \
        --img-dir  /path/to/val_images \
        --n 8 \
        --out-dir results/visualizations/

Outputs:
    results/visualizations/<imgid>_tripanel.jpg     — 3-panel tier comparison
    results/visualizations/best_detection.jpg        — highest-quality prediction
    results/visualizations/worst_detection.jpg       — hardest failure case
    results/visualizations/paper_figure_candidates/  — top-N images for the paper figure
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file",  required=True)
    p.add_argument("--weights-t1",   required=True, help="Tier-1 (quadrant) checkpoint")
    p.add_argument("--weights-t2",   required=True, help="Tier-2 (enumeration) checkpoint")
    p.add_argument("--weights-t3",   required=True, help="Tier-3 (diagnosis) checkpoint")
    p.add_argument("--val-json",     required=True)
    p.add_argument("--img-dir",      required=True)
    p.add_argument("--n",            type=int, default=8, help="Images to visualize")
    p.add_argument("--image-ids",    nargs="+", type=int, default=None)
    p.add_argument("--score-thresh", type=float, default=0.4)
    p.add_argument("--scale",        type=float, default=0.35,
                   help="Output scale relative to original resolution")
    p.add_argument("--out-dir",      default="results/visualizations")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── Color scheme ───────────────────────────────────────────────────────────────
# GT boxes: solid thick border
# Predicted boxes: dashed thin border + score label

QUADRANT_COLORS = {
    1: (0, 210, 0),     # green  — upper right
    2: (0, 165, 255),   # orange — upper left
    3: (0, 0, 220),     # red    — lower left
    4: (200, 0, 200),   # purple — lower right
    0: (140, 140, 140), # gray   — unknown
}

DIAGNOSIS_COLORS = {
    0: (140, 140, 140),
    1: (0, 220, 220),   # yellow   — impacted
    2: (0, 165, 255),   # orange   — caries
    3: (0, 0, 220),     # red      — periapical
    4: (220, 0, 220),   # magenta  — deep caries
}

DIAG_NAMES  = {0: "None", 1: "Impacted", 2: "Caries", 3: "Periapical", 4: "DeepCaries"}
QUAD_NAMES  = {0: "?", 1: "UprR", 2: "UprL", 3: "LwrL", 4: "LwrR"}
TIER_LABELS = {0: "TIER 1: Quadrant", 1: "TIER 2: Enumeration", 2: "TIER 3: Diagnosis"}


def load_model(config_file, weights):
    import torch
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, \
        EMADetectionCheckpointer

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.freeze()

    model = build_model(cfg)
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, **kwargs).resume_or_load(weights, resume=False)
    else:
        DetectionCheckpointer(model).resume_or_load(weights, resume=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, cfg


def infer(model, cfg, tier, img_path):
    """Return (boxes_xyxy, scores, labels) for one image."""
    import torch
    from detectron2.data import transforms as T

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, None

    img_rgb = img_bgr[:, :, ::-1]
    h, w = img_rgb.shape[:2]
    resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
    img_r, _ = T.apply_transform_gens([resize], img_rgb)
    img_t = torch.as_tensor(np.ascontiguousarray(img_r.transpose(2, 0, 1))).float()
    if torch.cuda.is_available():
        img_t = img_t.cuda()

    inp = [{"image": img_t, "height": h, "width": w, "image_id": 0, "file_name": str(img_path)}]
    with torch.no_grad():
        out = model(inp, k=tier)

    inst = out[0]["instances"].to("cpu")
    return (inst.pred_boxes.tensor.numpy(),
            inst.scores.numpy(),
            inst.pred_classes.numpy() if inst.has("pred_classes") else np.zeros(len(inst)))


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8):
    """Draw a dashed rectangle for predicted boxes."""
    x1, y1, x2, y2 = int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1])
    for side_pts in [
        [(x1, y1, x2, y1), True],   # top
        [(x2, y1, x2, y2), False],  # right
        [(x2, y2, x1, y2), True],   # bottom
        [(x1, y2, x1, y1), False],  # left
    ]:
        (ax, ay, bx, by), horizontal = side_pts
        length = abs(bx - ax) if horizontal else abs(by - ay)
        n_dashes = max(1, length // (2 * dash_len))
        for i in range(n_dashes):
            t0 = i / n_dashes
            t1 = (i + 0.5) / n_dashes
            if horizontal:
                sx = int(ax + t0 * (bx - ax)); ex = int(ax + t1 * (bx - ax))
                cv2.line(img, (sx, ay), (ex, ay), color, thickness)
            else:
                sy = int(ay + t0 * (by - ay)); ey = int(ay + t1 * (by - ay))
                cv2.line(img, (ax, sy), (ax, ey), color, thickness)


def render_tier_panel(img_bgr, gt_anns, pred_boxes, pred_scores, pred_labels,
                      tier, scale, score_thresh, title):
    """Render one tier's GT + predictions onto a scaled copy of the image."""
    h, w = img_bgr.shape[:2]
    sw, sh = int(w * scale), int(h * scale)
    panel = cv2.resize(img_bgr, (sw, sh))

    def s(v):
        return int(v * scale)

    # Draw GT boxes (solid, thick)
    for ann in gt_anns:
        x, y, bw, bh = ann["bbox"]
        c1, c2, c3 = (ann.get("category_id_1", 0),
                      ann.get("category_id_2", 0),
                      ann.get("category_id_3", 0))

        color = QUADRANT_COLORS.get(c1, (140, 140, 140))
        cv2.rectangle(panel, (s(x), s(y)), (s(x+bw), s(y+bh)), color, 2)

        if tier == 0:
            lbl = QUAD_NAMES.get(c1, str(c1))
        elif tier == 1:
            lbl = f"Q{c1}N{c2}"
        else:
            lbl = f"Q{c1}N{c2}D{DIAG_NAMES.get(c3,'?')[:3]}"

        cv2.putText(panel, lbl, (s(x), max(4, s(y)-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32 * scale / 0.35, color, 1, cv2.LINE_AA)

    # Draw predictions (dashed, thinner)
    if pred_boxes is not None:
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box
            if tier == 2:
                color = DIAGNOSIS_COLORS.get(int(label), (200, 200, 200))
            else:
                color = QUADRANT_COLORS.get(int(label) + 1, (200, 200, 200))

            draw_dashed_rect(panel, (s(x1), s(y1)), (s(x2), s(y2)), color, thickness=1)
            cv2.putText(panel, f"{score:.2f}", (s(x1), s(y2) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30 * scale / 0.35, color, 1, cv2.LINE_AA)

    # Title bar
    cv2.rectangle(panel, (0, 0), (sw, 22), (30, 30, 30), -1)
    cv2.putText(panel, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"GT=solid  Pred=dashed", (sw - 160, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


def approximate_f1(gt_anns, pred_boxes, pred_scores, score_thresh=0.4, iou_thresh=0.5):
    """
    Quick approximation of detection F1 for ranking images by quality.
    Not a replacement for COCO mAP — used only for sorting visualizations.
    """
    if not gt_anns or pred_boxes is None or len(pred_boxes) == 0:
        return 0.0

    gt_boxes = [[a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2],
                 a["bbox"][1] + a["bbox"][3]] for a in gt_anns]
    active_preds = [(b, s) for b, s in zip(pred_boxes, pred_scores) if s >= score_thresh]

    tp = 0
    matched_gt = set()
    for pred_box, _ in active_preds:
        best_iou, best_gi = 0.0, -1
        for gi, gt_box in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            x1, y1 = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
            x2, y2 = min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter == 0:
                continue
            union = ((pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1]) +
                     (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) - inter)
            iou_val = inter / union if union > 0 else 0
            if iou_val > best_iou:
                best_iou, best_gi = iou_val, gi
        if best_iou >= iou_thresh and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)

    precision = tp / len(active_preds) if active_preds else 0
    recall    = tp / len(gt_boxes)     if gt_boxes    else 0
    return 2 * precision * recall / (precision + recall + 1e-9)


def add_legend(panel, tier):
    """Add a color-coded legend to a panel."""
    x0, y0 = 5, panel.shape[0] - 80
    cv2.rectangle(panel, (x0-2, y0-2), (x0+200, panel.shape[0]-2), (20, 20, 20), -1)

    if tier in (0, 1):
        items = [(QUADRANT_COLORS[q], QUAD_NAMES[q]) for q in [1, 2, 3, 4]]
    else:
        items = [(DIAGNOSIS_COLORS[d], DIAG_NAMES[d]) for d in [1, 2, 3, 4]]

    for i, (color, name) in enumerate(items):
        y = y0 + i * 17
        cv2.rectangle(panel, (x0, y), (x0+14, y+12), color, -1)
        cv2.putText(panel, name, (x0+18, y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = out_dir / "paper_figure_candidates"
    paper_dir.mkdir(exist_ok=True)

    # Load annotations
    with open(args.val_json) as f:
        val_data = json.load(f)

    ann_by_img = defaultdict(list)
    for ann in val_data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)
    img_map = {img["id"]: img for img in val_data["images"]}

    img_dir = Path(args.img_dir)

    def find_img(meta):
        fname = meta.get("file_name", "")
        for p in [img_dir / fname, img_dir / Path(fname).name]:
            if p.exists():
                return p
        return None

    # Select images
    if args.image_ids:
        selected = [(iid, find_img(img_map[iid]), ann_by_img[iid])
                    for iid in args.image_ids if iid in img_map]
    else:
        import random
        rng = random.Random(args.seed)
        candidates = [(iid, find_img(meta), ann_by_img[iid])
                      for iid, meta in img_map.items()
                      if find_img(meta) is not None and ann_by_img[iid]]
        rng.shuffle(candidates)
        selected = candidates[:args.n]

    selected = [(iid, p, anns) for iid, p, anns in selected if p is not None]
    print(f"Visualizing {len(selected)} images...")

    # Load all three models
    print("Loading tier-1 model...")
    model_t1, cfg_t1 = load_model(args.config_file, args.weights_t1)
    print("Loading tier-2 model...")
    model_t2, cfg_t2 = load_model(args.config_file, args.weights_t2)
    print("Loading tier-3 model...")
    model_t3, cfg_t3 = load_model(args.config_file, args.weights_t3)

    f1_scores = []

    for iid, img_path, gt_anns in selected:
        print(f"  Image {iid}: {img_path.name}  ({len(gt_anns)} GT boxes)")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"    Cannot load image — skipping")
            continue

        # Run inference for each tier
        boxes_t1, scores_t1, labels_t1 = infer(model_t1, cfg_t1, 0, img_path)
        boxes_t2, scores_t2, labels_t2 = infer(model_t2, cfg_t2, 1, img_path)
        boxes_t3, scores_t3, labels_t3 = infer(model_t3, cfg_t3, 2, img_path)

        n_pred = [sum(1 for s in (sc or []) if s >= args.score_thresh)
                  for sc in [scores_t1, scores_t2, scores_t3]]
        print(f"    Predictions (>={args.score_thresh}): T1={n_pred[0]} T2={n_pred[1]} T3={n_pred[2]}")

        # Build three-panel figure
        scale = args.scale
        panel_t1 = render_tier_panel(img_bgr, gt_anns, boxes_t1, scores_t1, labels_t1,
                                     0, scale, args.score_thresh, TIER_LABELS[0])
        panel_t2 = render_tier_panel(img_bgr, gt_anns, boxes_t2, scores_t2, labels_t2,
                                     1, scale, args.score_thresh, TIER_LABELS[1])
        panel_t3 = render_tier_panel(img_bgr, gt_anns, boxes_t3, scores_t3, labels_t3,
                                     2, scale, args.score_thresh, TIER_LABELS[2])

        # Add legends
        panel_t1 = add_legend(panel_t1, 0)
        panel_t2 = add_legend(panel_t2, 1)
        panel_t3 = add_legend(panel_t3, 2)

        # Image ID label on the left panel
        cv2.putText(panel_t1, f"ID={iid}  GT_boxes={len(gt_anns)}",
                    (5, panel_t1.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        tripanel = np.hstack([panel_t1, panel_t2, panel_t3])

        # Add a thin divider line between panels
        h = tripanel.shape[0]
        pw = panel_t1.shape[1]
        cv2.line(tripanel, (pw,   0), (pw,   h), (80, 80, 80), 2)
        cv2.line(tripanel, (pw*2, 0), (pw*2, h), (80, 80, 80), 2)

        out_path = out_dir / f"{iid}_tripanel.jpg"
        cv2.imwrite(str(out_path), tripanel, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Compute F1 for ranking
        f1 = approximate_f1(gt_anns, boxes_t3, scores_t3, args.score_thresh)
        f1_scores.append((f1, iid, img_path, tripanel))
        print(f"    F1≈{f1:.3f}  Saved: {out_path.name}")

    # Save best/worst for the paper figure
    f1_scores.sort(key=lambda x: x[0], reverse=True)

    if f1_scores:
        best_f1, best_id, _, best_panel = f1_scores[0]
        cv2.imwrite(str(paper_dir / f"best_id{best_id}_f1{best_f1:.3f}.jpg"),
                    best_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        worst_f1, worst_id, _, worst_panel = f1_scores[-1]
        cv2.imwrite(str(paper_dir / f"worst_id{worst_id}_f1{worst_f1:.3f}.jpg"),
                    worst_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Top-4 for paper figure
        for rank, (f1, iid, _, panel) in enumerate(f1_scores[:4]):
            cv2.imwrite(str(paper_dir / f"rank{rank+1}_id{iid}_f1{f1:.3f}.jpg"),
                        panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"\n  Best prediction:  ID={best_id}  F1≈{best_f1:.3f}")
        print(f"  Worst prediction: ID={worst_id} F1≈{worst_f1:.3f}")

    print(f"\nAll visualizations saved to {out_dir}/")
    print(f"Paper figure candidates in {paper_dir}/")


if __name__ == "__main__":
    main()
