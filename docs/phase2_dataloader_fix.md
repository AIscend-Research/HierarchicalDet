# Phase 2 — `quadrant`/`quadrant_enumeration` dataloader fix

Follow-up to the Phase 1 finding (`docs/phase1_dataset_audit.md`) that the
official pipeline can only load `quadrant-enumeration-disease`, not
`quadrant` or `quadrant_enumeration`. This blocks tier-0/tier-1 of the
hierarchical training sequence entirely. Fixed 2026-07-15.

## Why a code patch to `pycocotools`/`detectron2` was rejected

The first candidate fix was to make `pycocotools/coco.py`'s `getCatIds()`
and `detectron2/data/datasets/coco.py`'s `load_coco_json()` tolerate missing
tiers. Reading `load_coco_json()` in full (not just the part that crashes)
showed this doesn't work cleanly: `id_map_1/2/3` are computed
unconditionally, and `min(cat_ids_3)` / `max(cat_ids_3)` (lines 99-119) raise
`ValueError: min() arg is an empty sequence` on an empty category list — so
even "pad the missing tier with an empty list" doesn't survive that check.

More importantly, the annotation-loading loop already contains dead-looking
code that only makes sense for one specific data shape:

```python
if id_map_1:
    ...
    try:
        obj["category_id_1"] = id_map_1[annotation_category_id]
    except KeyError as e:
        if obj["category_id_1"]==None:
            pass
        else:
            raise KeyError(...)
```

`obj["category_id_1"]==None` only fires if annotations can legitimately have
`category_id_1: null`. That's the tell: the pipeline was designed around a
**normalized 3-tier JSON schema** — every annotation always carries
`category_id_1/2/3`, with `null` for tiers that don't apply — not around the
three different raw DENTEX schemas (flat `categories`/`category_id` for
`quadrant`; `categories_1/2` for `quadrant_enumeration`; `categories_1/2/3`
for the full tier). `configs/diffdet.custom.swinbase.nonpretrain.yaml` backs
this up directly: `DiffusionDet.NUM_CLASSES: [4, 8, 4]` is a **fixed**
3-head model shape, unconditional of which tier is currently training. This
is a curriculum: one model, three fixed-size classification heads, trained
in stages (quadrant → +enumeration → +diagnosis) where earlier stages simply
have no supervision for the not-yet-introduced heads.

**Conclusion**: the bug isn't in the loading code, it's that DENTEX's raw
per-tier JSONs don't conform to the schema the loading code (and model
config) already assumes. The fix is a data-normalization step, not a
framework patch — zero changes to `pycocotools/` or `detectron2/`.

## The fix: `tools/normalize_dentex_tiers.py`

Converts `quadrant` and `quadrant_enumeration` raw JSONs into the normalized
schema, using `quadrant-enumeration-disease`'s `categories_1/2/3` as the
canonical category lists (so `thing_classes1/2/3` — and therefore the
model's class-index assignment — stay identical across all three curriculum
stages):

- `quadrant`: `categories` → `categories_1` (canonical), add canonical
  `categories_2`/`categories_3`. Per annotation: `category_id` →
  `category_id_1`, `category_id_2 = null`, `category_id_3 = null`.
- `quadrant_enumeration`: keep `categories_1`/`categories_2`, add canonical
  `categories_3`. Per annotation: keep `category_id_1`/`category_id_2`, add
  `category_id_3 = null`.

### The id-remap trap this caught

`quadrant`'s own `categories` list uses a **different id assignment** than
`categories_1` in the other two tiers for the same 4 quadrant classes:

| | id 0 | id 1 | id 2 | id 3 |
|---|---|---|---|---|
| `quadrant`'s own `categories` | name `"2"` | name `"1"` | name `"3"` | name `"4"` |
| `quadrant_enumeration`/full tier `categories_1` | name `1` | name `2` | name `3` | name `4` |

A raw id→id copy would have silently swapped quadrant classes 1 and 2 for a
large fraction of annotations. Caught by cross-referencing tooth boxes
(`quadrant_enumeration`) against their containing quadrant box (`quadrant`,
same underlying images) across 80 shared images: the dominant
correspondence in every case matched by **category name**, not by raw id
(e.g. `quadrant.category_id=0`, name `"2"`, dominantly contains teeth whose
`quadrant_enumeration.category_id_1=1`, name `2` — a ~90% majority vote per
class, with the minority attributable to the containment heuristic being
imperfect near quadrant boundaries, not to a different true mapping). The
script remaps by name, not by id.

`categories_2` between `quadrant_enumeration` and the full tier was verified
identical by id already (no remap needed there — only `categories_1` in the
`quadrant`-only file has the scrambled ordering).

## Verification (real local data, not a sample)

Ran against the full local `quadrant` (693 images / 2,772 annotations) and
`quadrant_enumeration` (634 images / 18,095 annotations) tiers:

1. **Per-annotation correctness, 100% of annotations, not sampled**: for
   every one of the 2,772 `quadrant` annotations, the category *name*
   resolved from the raw file and the normalized file matched exactly — 0
   mismatches. For all 18,095 `quadrant_enumeration` annotations,
   `category_id_1`/`category_id_2` were preserved exactly and
   `category_id_3` was `null` — 0 mismatches.
2. **Registration through the real, unmodified pipeline**: `register_coco_instances`
   → `load_coco_json` → bundled `pycocotools.COCO` succeeded for all three
   tiers (`quadrant`, `quadrant_enumeration`, `quadrant-enumeration-disease`)
   with zero errors, and `thing_classes1/2/3` were identical across all
   three (`['1','2','3','4']`, `['1'..'8']`,
   `['Impacted','Caries','Periapical Lesion','Deep Caries']`) — confirming
   the fixed-head model assumption holds. Per-tier annotation coverage
   matched expectations exactly: `quadrant` → cat1 only (2,772/2,772);
   `quadrant_enumeration` → cat1+cat2 only (18,095/18,095); full tier → all
   three (3,529/3,529).
3. **Real end-to-end training smoke test** (CPU, `train_net_patched.py`,
   randomly-initialized weights, `SOLVER.MAX_ITER=2`): the training loop
   started, loaded the normalized `quadrant` tier, and completed a full
   forward/backward pass at iteration 1 with a real logged loss line showing
   **only tier-1 losses** (`loss_ce1`, `loss_bbox`, `loss_giou`
   and their auxiliary-head variants) and correctly **no `loss_ce2`/`loss_ce3`**
   — confirming `category_id_2/3 = null` correctly suppresses supervision
   for the not-yet-introduced heads, exactly as the curriculum design
   requires. The run was killed by a 10-minute local timeout during
   iteration 2 (CPU is far too slow for Swin-B — expected and irrelevant;
   real training runs on the Kaggle T4 per the Phase 0 GPU benchmark), with
   zero errors up to that point.

## What's NOT done yet

- Not wired into `kaggle/kaggle_setup.ipynb` — the data-download cell
  currently only extracts images for `quadrant-enumeration-disease` (the one
  tier the official loader could already read); `quadrant`/`quadrant_enumeration`
  images are read for stats from the zip and never extracted. Extracting
  both (~2.4-2.7GB each) plus running this normalization script needs to be
  added before the actual 3-stage curriculum training can run on Kaggle.
- No decision yet on how the 3 sequential ~10.3h training stages (~30.8h
  total, per the Phase 0 GPU benchmark) get split across Kaggle GPU
  sessions/quota.
