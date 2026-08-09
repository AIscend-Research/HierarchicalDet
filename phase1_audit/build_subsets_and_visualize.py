import json, os, random
from collections import Counter, defaultdict
from PIL import Image, ImageDraw

random.seed(42)

VAL_JSON = "DENTEX/validation_triple.json"
VAL_IMG_DIR = "extracted/validation_data/quadrant_enumeration_disease/xrays"

d = json.load(open(VAL_JSON))
images = {im["id"]: im for im in d["images"]}
anns_by_image = defaultdict(list)
for a in d["annotations"]:
    anns_by_image[a["image_id"]].append(a)

cat3_names = {c["id"]: c["name"] for c in d["categories_3"]}
cat1_names = {c["id"]: c["name"] for c in d["categories_1"]}
cat2_names = {c["id"]: c["name"] for c in d["categories_2"]}

# Per-image stats: annotation count, unique diagnosis types, unique quadrants
stats = []
for im_id, im in images.items():
    anns = anns_by_image.get(im_id, [])
    n = len(anns)
    diag_types = set(a["category_id_3"] for a in anns)
    quadrants = set(a["category_id_1"] for a in anns)
    stats.append({
        "image_id": im_id,
        "file_name": im["file_name"],
        "n_annotations": n,
        "n_diagnosis_types": len(diag_types),
        "n_quadrants_involved": len(quadrants),
    })

n_annos_list = [s["n_annotations"] for s in stats]
n_annos_list_sorted = sorted(n_annos_list)
median_n = n_annos_list_sorted[len(n_annos_list_sorted)//2]
print(f"validation set: {len(stats)} images, annotation count per image: min={min(n_annos_list)}, median={median_n}, max={max(n_annos_list)}")
print(f"  distribution: {Counter(n_annos_list)}")

# Clean subset: images with a "typical" annotation count (near median, not zero,
# not a pathological outlier) and more than one diagnosis type covered (so the
# hierarchy is exercised).
clean = [s for s in stats if median_n - 2 <= s["n_annotations"] <= median_n + 2 and s["n_annotations"] > 0]
# Stress-test subset: zero-annotation images (edge case: no pathology detected)
# and high-annotation-count outliers (unusually dense pathology / multiple
# diagnosis types / multiple quadrants involved -- edge-case tooth configs).
zero_anno = [s for s in stats if s["n_annotations"] == 0]
high_outliers = sorted(stats, key=lambda s: -s["n_annotations"])[:10]
multi_diag = [s for s in stats if s["n_diagnosis_types"] >= 3]

print()
print(f"CLEAN subset candidates (near-median annotation count, {median_n}+-2): {len(clean)} images")
print(f"STRESS-TEST subset candidates:")
print(f"  zero-annotation images: {len(zero_anno)} -> {[s['file_name'] for s in zero_anno]}")
print(f"  high annotation-count outliers (top 10): {[(s['file_name'], s['n_annotations']) for s in high_outliers]}")
print(f"  images with >=3 distinct diagnosis types: {len(multi_diag)} -> {[(s['file_name'], s['n_diagnosis_types']) for s in multi_diag]}")

with open("clean_stress_subsets.json", "w") as f:
    json.dump({
        "clean_subset": [s["file_name"] for s in clean],
        "stress_zero_annotation": [s["file_name"] for s in zero_anno],
        "stress_high_density": [s["file_name"] for s in high_outliers],
        "stress_multi_diagnosis": [s["file_name"] for s in multi_diag],
    }, f, indent=2)
print()
print("wrote clean_stress_subsets.json")

# --- Visual verification: pick 5 images (mix of clean + stress) and draw boxes ---
os.makedirs("visual_check", exist_ok=True)
sample_files = []
sample_files += [s["file_name"] for s in clean[:3]]
sample_files += [s["file_name"] for s in high_outliers[:2]]

file_name_to_image = {im["file_name"]: im for im in d["images"]}

for fn in sample_files:
    im_record = file_name_to_image[fn]
    im_id = im_record["id"]
    anns = anns_by_image[im_id]
    img_path = os.path.join(VAL_IMG_DIR, fn)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for a in anns:
        x, y, w, h = a["bbox"]
        q = cat1_names.get(a["category_id_1"], "?")
        e = cat2_names.get(a["category_id_2"], "?")
        diag = cat3_names.get(a["category_id_3"], "?")
        draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
        label = f"Q{q}-T{e}-{diag}"
        draw.text((x, max(0, y - 20)), label, fill="yellow")
    out_path = f"visual_check/{fn.replace('.png','')}_annotated.jpg"
    # downscale for easier viewing
    img.thumbnail((1400, 1400))
    img.save(out_path, quality=90)
    print(f"wrote {out_path}  ({len(anns)} boxes)")
