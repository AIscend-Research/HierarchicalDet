"""
DETR baseline, fine-tuned from the COCO-pretrained `facebook/detr-resnet-50`.

Deviation, stated plainly: DETR is one of the four baselines the paper reports,
but it is not part of this repository's bundled detectron2 snapshot and the
authors did not release a DETR config. Rather than reimplement DETR, this uses
the Hugging Face `transformers` implementation, which is the reference PyTorch
port of the original Facebook code. That is a different codebase from the one
the paper's DETR numbers came from, so this row is a fair *reproduction of the
comparison*, not a reproduction of the authors' exact DETR run, and should be
labelled as such in the report.

Like the other baselines it trains on a single flat tier of DENTEX, writes
COCO-results predictions, and is scored by tools/coco_eval_standalone.py.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from baselines.flat_data import load_flat_dentex  # noqa: E402


class DentexDetrDataset(Dataset):
    def __init__(self, dicts, processor, train=True):
        self.dicts = dicts
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.dicts)

    def __getitem__(self, index):
        record = self.dicts[index]
        image = Image.open(record["file_name"]).convert("RGB")
        annotations = [{
            "bbox": a["bbox"],  # XYWH absolute, which is what DETR's processor expects
            "category_id": a["category_id"],
            "area": a["bbox"][2] * a["bbox"][3],
            "iscrowd": 0,
        } for a in record["annotations"]]
        encoding = self.processor(
            images=image,
            annotations={"image_id": record["image_id"], "annotations": annotations},
            return_tensors="pt",
        )
        return {
            "pixel_values": encoding["pixel_values"][0],
            "labels": encoding["labels"][0],
            "image_id": record["image_id"],
            "orig_size": (record["height"], record["width"]),
        }


def collate(batch, processor):
    encoding = processor.pad([item["pixel_values"] for item in batch], return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": [item["labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "orig_sizes": [item["orig_size"] for item in batch],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-json", required=True)
    p.add_argument("--train-images", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--test-images", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=40244023)
    p.add_argument("--checkpoint", default="facebook/detr-resnet-50")
    p.add_argument("--eval-only", action="store_true",
                   help="load --resume-from and only run inference")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--score-thresh", type=float, default=0.0)
    args = p.parse_args()

    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except ImportError:
        raise SystemExit(
            "the DETR baseline needs `transformers` (pip install transformers). "
            "It is listed in requirements.txt."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dicts, class_names = load_flat_dentex(args.train_json, args.train_images, args.tier)
    test_dicts, _ = load_flat_dentex(args.test_json, args.test_images, args.tier)
    print("tier {}: {} classes {}".format(args.tier, len(class_names), class_names))

    processor = DetrImageProcessor.from_pretrained(args.checkpoint)
    model = DetrForObjectDetection.from_pretrained(
        args.resume_from or args.checkpoint,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True,
    ).to(device)

    if not args.eval_only:
        train_loader = DataLoader(
            DentexDetrDataset(train_dicts, processor, train=True),
            batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate(b, processor))
        # The standard DETR recipe: a lower learning rate on the backbone.
        param_groups = [
            {"params": [p_ for n, p_ in model.named_parameters()
                        if "backbone" not in n and p_.requires_grad], "lr": args.lr},
            {"params": [p_ for n, p_ in model.named_parameters()
                        if "backbone" in n and p_.requires_grad], "lr": args.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        model.train()
        for epoch in range(args.epochs):
            running, batches = 0.0, 0
            for batch in train_loader:
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    pixel_mask=batch["pixel_mask"].to(device),
                    labels=[{k: v.to(device) for k, v in label.items()} for label in batch["labels"]],
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                running += float(loss)
                batches += 1
            print("epoch {}/{}: loss {:.4f}".format(epoch + 1, args.epochs, running / max(batches, 1)),
                  flush=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

    model.eval()
    test_loader = DataLoader(
        DentexDetrDataset(test_dicts, processor, train=False),
        batch_size=1, shuffle=False, collate_fn=lambda b: collate(b, processor))

    results, per_image = [], []
    with torch.no_grad():
        for batch in test_loader:
            start = time.perf_counter()
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            pixel_mask=batch["pixel_mask"].to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            target_sizes = torch.tensor([batch["orig_sizes"][0]], device=device)
            processed = processor.post_process_object_detection(
                outputs, threshold=args.score_thresh, target_sizes=target_sizes)[0]

            image_id = batch["image_ids"][0]
            for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
                x0, y0, x1, y1 = [float(v) for v in box]
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "category_id_{}".format(args.tier + 1): int(label),
                    "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                    "score": float(score),
                })
            per_image.append({"image_id": image_id, "seconds": round(elapsed, 4),
                              "num_detections": int(len(processed["scores"]))})

    predictions_path = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(results, f)
    times = [r["seconds"] for r in per_image][5:]
    with open(os.path.join(args.output_dir, "predictions_runtime.json"), "w") as f:
        json.dump({"summary": {
            "device": device,
            "images": len(per_image),
            "mean_seconds_per_image": round(sum(times) / max(len(times), 1), 4),
            "images_per_minute": round(60.0 * len(times) / max(sum(times), 1e-9), 2) if times else None,
            "implementation": "huggingface transformers DetrForObjectDetection ({})".format(args.checkpoint),
        }, "per_image": per_image}, f, indent=2)

    print("wrote {} ({} predictions)".format(predictions_path, len(results)))
    print("\nScore it with:\n  python tools/coco_eval_standalone.py --gt-json {} "
          "--predictions {} --tier {} --output {}".format(
              args.test_json, predictions_path, args.tier,
              os.path.join(args.output_dir, "metrics.json")))


if __name__ == "__main__":
    main()
