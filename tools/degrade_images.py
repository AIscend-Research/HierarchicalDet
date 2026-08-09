"""
Build degraded copies of an evaluation split, for the image-quality robustness
experiment.

Clinics running older or cheaper panoramic units get softer, more compressed,
lower-resolution films than the DENTEX images. This writes one degraded copy of
the split per condition, together with a COCO annotation file pointing at it,
so each condition can be evaluated with exactly the same commands as the clean
split -- no changes to the model or the evaluation code.

Conditions (all applied to the original image, never stacked):

  blur_{sigma}        Gaussian blur, sigma in pixels
  jpeg_q{quality}     re-encoded as JPEG at the given quality, then reloaded
  scale_{percent}     downscaled to N% and resampled back to the original size
                      (so boxes stay valid and only detail is lost)

Boxes are untouched by design: every condition preserves the pixel grid, so the
ground truth stays exactly comparable to the clean run.
"""
import argparse
import io
import json
import os

from PIL import Image, ImageFilter

DEFAULT_CONDITIONS = [
    "blur_1", "blur_2", "blur_4",
    "jpeg_q50", "jpeg_q25", "jpeg_q10",
    "scale_50", "scale_25",
]


def parse_condition(name):
    kind, _, value = name.partition("_")
    if kind == "blur":
        return kind, float(value)
    if kind == "jpeg":
        return kind, int(value.lstrip("q"))
    if kind == "scale":
        return kind, int(value)
    raise ValueError("unknown condition {!r} (expected blur_N / jpeg_qN / scale_N)".format(name))


def degrade(image, kind, value):
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=value)
        buffer.seek(0)
        return Image.open(buffer).convert(image.mode)
    if kind == "scale":
        width, height = image.size
        small = image.resize(
            (max(1, width * value // 100), max(1, height * value // 100)), Image.BILINEAR)
        return small.resize((width, height), Image.BILINEAR)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True, help="COCO annotations of the split to degrade")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-root", required=True,
                   help="one subdirectory per condition is created here")
    p.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    args = p.parse_args()

    with open(args.json) as f:
        coco = json.load(f)

    os.makedirs(args.output_root, exist_ok=True)
    index = {"source_json": os.path.abspath(args.json), "conditions": {}}

    for condition in args.conditions:
        kind, value = parse_condition(condition)
        image_dir = os.path.join(args.output_root, condition, "xrays")
        os.makedirs(image_dir, exist_ok=True)

        for image_info in coco["images"]:
            source = os.path.join(args.image_dir, image_info["file_name"])
            target = os.path.join(image_dir, image_info["file_name"])
            if os.path.exists(target):
                continue
            with Image.open(source) as im:
                im.load()
                degraded = degrade(im, kind, value)
                # Always PNG: re-saving as JPEG would add a second, uncontrolled
                # round of compression on top of the condition being tested.
                degraded.save(target, format="PNG")

        json_path = os.path.join(args.output_root, condition, os.path.basename(args.json))
        with open(json_path, "w") as f:
            json.dump(coco, f)

        index["conditions"][condition] = {"json": json_path, "image_dir": image_dir,
                                          "images": len(coco["images"])}
        print("{}: {} images -> {}".format(condition, len(coco["images"]), image_dir))

    index_path = os.path.join(args.output_root, "degradation_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print("wrote {}".format(index_path))


if __name__ == "__main__":
    main()
