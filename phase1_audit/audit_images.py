import os, glob, json
from PIL import Image
from collections import defaultdict

EXTRACTED = "extracted"

SPLITS = {
    "quadrant (train)": "extracted/training_data/quadrant/xrays",
    "quadrant-enumeration (train)": "extracted/training_data/quadrant_enumeration/xrays",
    "quadrant-enumeration-diagnosis (train)": "extracted/training_data/quadrant-enumeration-disease/xrays",
    "unlabelled (train)": "extracted/training_data/unlabelled/xrays",
    "quadrant-enumeration-diagnosis (validation)": "extracted/validation_data/quadrant_enumeration_disease/xrays",
    "test (disease/input)": "extracted/disease/input",
}

results = {}
for name, path in SPLITS.items():
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    n_ok = 0
    n_corrupt = []
    resolutions = defaultdict(int)
    for f in files:
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                resolutions[im.size] += 1
            n_ok += 1
        except Exception as e:
            n_corrupt.append((f, str(e)))
    results[name] = {
        "n_files": len(files),
        "n_ok": n_ok,
        "n_corrupt": len(n_corrupt),
        "corrupt_list": n_corrupt[:10],
        "resolutions": {f"{w}x{h}": c for (w, h), c in resolutions.items()},
    }
    print(f"=== {name} ===")
    print(f"  files: {len(files)}  loadable: {n_ok}  corrupt: {len(n_corrupt)}")
    if n_corrupt:
        for f, e in n_corrupt[:5]:
            print("   CORRUPT:", f, e)
    top_res = sorted(resolutions.items(), key=lambda x: -x[1])[:8]
    print(f"  top resolutions (w,h -> count):")
    for res, cnt in top_res:
        print(f"    {res} -> {cnt}")
    print(f"  total distinct resolutions: {len(resolutions)}")
    print()

with open("image_audit_results.json", "w") as f:
    json.dump({k: {**v, "corrupt_list": v["corrupt_list"]} for k, v in results.items()}, f, indent=2, default=str)
print("wrote image_audit_results.json")
