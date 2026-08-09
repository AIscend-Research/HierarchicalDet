# Phase 1 Dataset Audit — DENTEX

Audited 2026-07-15 against the real dataset downloaded from
`huggingface.co/datasets/ibrahimhamamci/DENTEX` (full local extraction, not
sampled), using the bundled/official `hierarchialdet` and `detectron2` code
wherever possible. Scripts live in `phase1_audit/` (not committed — see
bottom of this doc for why).

## Summary table

| Split | Images | Annotations | Tier coverage | Images w/ 0 annotations | Loadable by official `register_coco_instances` + `DiffusionDetDatasetMapper`? |
|---|---|---|---|---|---|
| `quadrant` (train) | 693 | 2,772 | tier 1 only (`category_id`, flat) | 0 | **No** — `KeyError: 'categories_1'` |
| `quadrant_enumeration` (train) | 634 | 18,095 | tiers 1+2 (`category_id_1/2`) | 0 | **No** — `KeyError: 'categories_3'` |
| `quadrant-enumeration-disease` (train) | 705 | 3,529 | tiers 1+2+3 (`category_id_1/2/3`) | 27 | **Yes** — 705/705 mapped successfully |
| `quadrant-enumeration-disease` (validation, `validation_triple.json`) | 50 | 182 | tiers 1+2+3 | 4 | **Yes** — 50/50 mapped successfully |
| `unlabelled` (train) | 1,571 | — (no JSON at all) | none | 1,571 | N/A — no annotations to load |
| test (`disease/input` + `disease/label`) | 250 | 250 per-image LabelMe files | Turkish-string encoded, not COCO | n/a | **No** — incompatible format, needs a conversion script that doesn't exist yet |

**Image integrity**: all 3,903 image files across every split load correctly via PIL (`Image.verify()` + full decode) — zero corrupt files.

**Annotation integrity** (train/val COCO-format tiers only): zero invalid `image_id` references, zero missing `file_name`s on disk, zero out-of-range `category_id`s, zero malformed bboxes (non-positive size or out of image bounds), zero null `category_id_N` values in either the train diagnosis tier or the validation tier (the earlier-noted `null` category IDs come from a different, smaller *bundled sample file* in the repo — `pycocotools/val.json` — not the real HuggingFace-hosted `validation_triple.json`, which is fully annotated on all 3 tiers for every non-empty annotation by construction).

## Official dataloader failure — root cause, precisely located

`quadrant` and `quadrant_enumeration` (train) **cannot be loaded by the official pipeline as-is**. Root cause:

```
detectron2/data/datasets/coco.py:76   cat_ids_1 = sorted(coco_api.getCatIds()[0])
pycocotools/coco.py:184               cat_name_list = ['categories_1', 'categories_2', 'categories_3']
                                       cats_list.append(self.dataset[cat_name])   # KeyError, no existence check
```

Both the bundled `detectron2` loader and the bundled `pycocotools.coco.COCO.getCatIds()` unconditionally assume all three `categories_1/2/3` keys exist. `quadrant`'s JSON only has a flat `categories` key; `quadrant_enumeration`'s JSON only has `categories_1`/`categories_2`. Per the roadmap's "use the official pipeline unless it fails" instruction: **it fails**, for these two tiers specifically. Fixing this (so tier-0/tier-1 hierarchical training in Phase 2 can use the official loader) requires a small, targeted patch — either padding the missing `categories_N`/`category_id_N` fields with `null` placeholders to match the 3-tier shape, or making `getCatIds()`/`load_coco_json()` tolerate missing tiers. Not attempted here; flagged for Phase 2.

## Public release vs. paper — matches exactly on stated numbers

The paper (and DENTEX GitHub README) states 693 quadrant-only, 634 quadrant-enumeration, and 1,005 fully-annotated (705 train / 50 val / 250 test) X-rays. The real downloaded release matches **exactly**: 693 / 634 / 705+50+250=1,005. The only undocumented addition is the `unlabelled/` folder (1,571 images, no annotations at all) — not mentioned in either README, not used by the current pipeline, presumably available for future self-supervised pretraining.

## Important finding: substantial cross-tier / cross-split image duplication

3,903 total image files, but only **2,871 unique images by exact content hash (MD5)**. The same physical X-ray appears under different filenames in different tier folders (confirmed directly: `quadrant/train_673.png` and `quadrant-enumeration-disease/train_673.png` are two *different* images — filenames are tier-local, not global IDs — but plenty of *other* filenames across tiers do share identical byte content).

Precise overlap counts:
- **95 of 250 test images (38%)** are byte-identical to an image somewhere in a training tier.
- **20 of 50 validation images (40%)** are byte-identical to an image somewhere in a training tier.
- Within training: `quadrant` (693 unique files) and `quadrant_enumeration` (634 unique files) share 573 images; `quadrant-enumeration-disease` shares ~259–260 images with each of the other two; 237 images are common to all three training tiers.

**Why this matters for reproduction**: the hierarchical training approach trains tier-0 (quadrant) → runs inference → uses that inference as "noisy boxes" seeding tier-1 (quadrant-enumeration) training → repeats for tier-2 (diagnosis). If a diagnosis-tier *test* image was also present in the quadrant or quadrant-enumeration *training* tiers (which the 38%/40% overlap numbers say is common), the model has effectively already seen that image's pixels — under a different annotation tier — before "testing" on it at the diagnosis stage.

**Resolved against the source papers (2026-07-15)**: checked both the HierarchicalDet paper (arXiv 2303.06500) and the companion DENTEX dataset paper (arXiv 2305.19112) directly. Neither addresses this. The DENTEX paper states plainly that Tier 1 (quadrant) and Tier 2 (quadrant-enumeration) have **no official train/val/test split at all** — they're designated broadly "for training and development purposes." Only Tier 3 (quadrant-enumeration-diagnosis) gets an explicit split (705/50/250). Neither paper discusses cross-tier image overlap or data-leakage prevention in any form.

**Conclusion**: since Tiers 1/2 are pooled entirely into "training" with no held-out portion of their own, and this audit found 38–40% of Tier 3's test/validation images are byte-identical to images in Tiers 1/2, the original authors' own hierarchical training pipeline very likely trained tier-0/tier-1 on images that were later "tested" at the diagnosis tier. This is not a mistake introduced by this reproduction — it's a genuine characteristic (arguably a limitation) of the original paper's own experimental design, baked into the public dataset release itself. Any faithful reproduction inherits it. **This should be stated explicitly in the reproduction report's limitations section**, framed as a property of the original work being reproduced, not a reproduction error.

## Clean and stress-test evaluation subsets (from the 50-image validation set)

Per-image annotation-count distribution: min 0, median 3, max 11 (`Counter({2: 12, 4: 8, 3: 7, 5: 4, 6: 4, 1: 4, 0: 4, 7: 3, 8: 2, 9: 1, 11: 1})`).

- **Clean subset** (35 images): annotation count within 2 of the median (1–5 boxes), non-zero.
- **Stress-test subset**:
  - 4 zero-annotation images (`val_17/23/26/29.png`) — no pathology detected at all, an edge case for a model expected to always predict *something*.
  - Top 10 high-annotation-density images (up to 11 boxes on one film, `val_28.png`).
  - 6 images with ≥3 distinct diagnosis types present simultaneously (most hierarchically complex label structure).

Full file lists in `phase1_audit/clean_stress_subsets.json`.

## Visual spot-check (5 images, mix of clean + stress-density)

Rendered ground-truth boxes + quadrant/tooth/diagnosis labels directly on 5 validation images (`phase1_audit/visual_check/*_annotated.jpg`) and visually inspected each:
- `val_38.png` (5 boxes): 4 impacted third molars (all 4 quadrant corners) + 1 periapical lesion, all boxes tightly bound the corresponding tooth.
- `val_30.png` (5 boxes): 5 tight boxes around lower incisor/canine roots, labeled periapical lesion / deep caries, aligned with visible root-tip radiolucency.
- `val_5.png` (2 boxes): 2 caries boxes on visibly restored/decayed molars.
- `val_28.png` (11 boxes, stress case): dense mix of impacted molars, caries, and a periapical lesion — every box still correctly bounds its labeled tooth despite the high density.
- `val_31.png` (9 boxes, stress case): same pattern, correctly aligned throughout.

All 5 checked out: boxes match visible teeth/pathology, and the quadrant→tooth→diagnosis hierarchy is internally consistent in every case (no box with a diagnosis in a quadrant/tooth combination that doesn't make anatomical sense).

## What's NOT done here (explicitly out of scope for Phase 1)

- No LabelMe→COCO conversion for the test set — needed before test-set evaluation is possible at all (Phase 2 prerequisite).
- ~~No patch to make `quadrant`/`quadrant_enumeration` loadable by the official pipeline~~ — **fixed in Phase 2**, see `docs/phase2_dataloader_fix.md`.
- The train/test image-overlap question above is reported, not resolved — needs a decision informed by the paper's actual methodology text.

## Reproducibility

The audit scripts, their text output, the concrete subset file, and the 5
annotated spot-check images are committed under `phase1_audit/`:
`audit_images.py`, `audit_annotations.py`, `test_official_dataloader.py`,
`build_subsets_and_visualize.py`, `image_audit_results.json`,
`clean_stress_subsets.json`, `audit_images_output.txt`,
`audit_annotations_output.txt`, `dataloader_test_output.txt`,
`visual_check/*.jpg`. The raw downloaded zips (`phase1_audit/DENTEX/`) and
the 15GB full extraction (`phase1_audit/extracted/`) are gitignored — re-run
`snapshot_download` + the scripts against a fresh download to reproduce.
Every number in this document comes from a real, complete run against the
full dataset, not a sample.
