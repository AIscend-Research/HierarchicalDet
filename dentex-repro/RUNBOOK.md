# RUNBOOK — HierarchicalDet reproduction on Kaggle

Everything needed to go from a fresh Kaggle account to a populated
`paper_assets/`. Read the whole of §1 before starting: the session limits shape
every decision below.

---

## 1. Kaggle execution model

| Constraint | Value | Consequence for this suite |
|---|---|---|
| Accelerators | P100 16 GB, or **T4 x2** (preferred) | Swin-B at `IMS_PER_BATCH=1` per GPU with AMP fits 16 GB. Batch 2 on one T4 OOMs — measured, not assumed. |
| GPU quota | ~30 GPU-hours/week | `micro` (the default) is engineered to fit in ~13 h, so a full pass plus re-runs fits one week. |
| Session limit | 12 h | Every training run is budgeted to ≤11 h and is resumable; nothing assumes it finishes in one sitting. |
| CPU sessions | quota-free | Notebook 01 and the CPU pass of notebook 03 cost no GPU quota at all. |
| `/kaggle/working` | ~20 GB, persists within a session; saved by "Save Version" | Checkpoints are pruned to `last` + `best` + trajectory snapshots. Archives are deleted as soon as they are extracted. |
| Internet | must be enabled in notebook Settings | Required for the HuggingFace download, the Swin-B weights, and `pip`. |
| GPU access | requires **phone verification** | Do this before you plan a session. |

**Before running any notebook:** Settings → Internet = *On*; Settings →
Persistence = *Variables and files*. Accelerator is per notebook — see §3.

### Multi-GPU

Notebook 02 probes `launch(num_gpus=2)` with a real 20-iteration run before
committing any quota. If DDP fails in the session, it **falls back to one GPU
automatically** and writes the fallback into `paper_assets/deviations.md`. Do
not fight a flaky DDP setup; the fallback is the supported path, and the batch
size it implies is recorded.

### Persistence between sessions

State moves between sessions through **Kaggle Datasets**. Both are produced by
the notebooks themselves — nothing has to be sourced or uploaded by hand.

* `dentex-repro-data` — `/kaggle/working/dentex_converted/`: the converted COCO
  files and verified images, written by notebook 01. Training never re-downloads
  from HuggingFace.
* `dentex-repro-ckpts` — `/kaggle/working/runs/`: checkpoints, prediction dumps,
  calibration and run records, written by notebook 02.

Both also carry `paper_assets/` (the accumulated `results_raw/`, `manifest.json`
and `deviations.md`). That directory lives inside the *cloned repo*, so a fresh
session would otherwise re-clone it empty and lose every prior result;
`setup_env.bootstrap` restores it from any attached dataset at startup, never
overwriting a file the current session has already produced.

Publishing is automated with the `kaggle` CLI using `KAGGLE_USERNAME` /
`KAGGLE_KEY` from **Add-ons → Secrets**. Credentials are never written to disk
or into a notebook.

**Manual fallback** (if the CLI is unavailable or the secrets are not set):
*Save Version → Save & Run All*, then on the next notebook *Add Data → Your
Work →* select that output. `src.train_utils.seed_from_attached_datasets`
searches every mounted `/kaggle/input/*` for the newest checkpoint of a run, so
the layout inside the attached dataset does not have to match exactly.

---

## 2. Run modes

One switch, set in the parameter cell of every notebook.

| mode | schedule | total GPU-h | what runs |
|---|---|---|---|
| `smoke` | 200 iters/run, 20-image evals | ~0.6 | **every experiment class**, including base DiffusionDet and all four variants. Proves the chain end to end. |
| `micro` *(default)* | hour-capped: quadrant 1.75 h, enumeration 1.75 h, each diagnosis variant 2.75 h | ~13 | 5 training runs. Base DiffusionDet and variant `wo_manip_transfer` are skipped. |
| `budget` | 15k / 15k / 20k iterations | ~30 | full matrix at half the paper's schedules. |
| `full` | 30k / 30k / 40k — the repo's own | ~60 | paper-fidelity, all three base-DiffusionDet tiers. |

### How the hour cap holds

`micro` does not guess an iteration count. Notebook 02 runs **500 genuine
training iterations**, discards 100 warm-up iterations, measures it/s, and
converts each hour budget into `MAX_ITER = remaining_seconds × measured_rate`,
rounded down to a multiple of 300 (so the 1/3 and 2/3 trajectory snapshots land
on whole iterations). A `TimeBudgetHook` enforces the wall-clock cap regardless,
saving `model_final.pth` if it fires. On slower hardware you get fewer
iterations, never more hours.

The measurement is cached and shared, so **all three diagnosis variants get the
same `MAX_ITER`** — their budgets are identical by construction, not
approximately equal. `train_utils.assert_matched_budgets` aborts the run if they
ever diverge; an ablation at unmatched budgets is not an ablation.

---

## 3. Order of execution — three notebooks

The split is on the only two boundaries that matter on Kaggle: **what
accelerator a session needs**, and **what fits in one session**. Anything finer
would only add cross-notebook state to lose.

| # | Notebook | Accelerator | Attach | Wall time (`micro`) | Produces |
|---|---|---|---|---|---|
| 1 | `01_setup_and_data` | **None** (CPU, free) | — | ~50 min | pinned env + vendored-import proof, converted COCO, audit table, GT figure, clean/stress subsets → `dentex-repro-data` |
| 2 | `02_train_all` | **T4 x2** | data (+ ckpts on re-runs) | ~12.25 h, **across sessions** | DDP verdict, pre-flight smoke, calibration, quadrant + enumeration, noisy-box dumps, all diagnosis variants + trajectory snapshots, base DiffusionDet → `dentex-repro-ckpts` |
| 3 | `03_evaluate_and_build_assets` | **T4**, then **None** | data, ckpts | ~2.5 h on GPU, ~5 min on CPU | all metrics × 3 seeds, trajectory sweep, step sweep, degradation grid, clean/stress, fault injection, error analysis, qualitative figures, and every file in `paper_assets/` |
| opt | `opt_baselines_and_simmim` | T4 | data | ~1 GPU-h per baseline; 10+ h for SimMIM | baseline rows; closes the pretraining deviation |

### The two things to know

**Notebook 02 will not finish in one session, and that is the design.** `micro`
needs ~12.25 GPU-h against a 12 h limit. Every run checkpoints every 2,500
iterations, is skipped once complete, and resumes from its last checkpoint
otherwise. Re-run the same notebook in a fresh session with `dentex-repro-ckpts`
attached until it reaches the end. Expect 2 sessions.

**Notebook 03 adapts to whatever accelerator it gets.** On a **T4** it runs
every GPU experiment and then builds the assets. On **Accelerator = None** it
skips the GPU sections, measures CPU-only inference (quota-free), and rebuilds
every table, figure and document from `paper_assets/results_raw/`. So run it
once on GPU, then once on CPU — the CPU pass adds the CPU benchmark and the
CPU-vs-GPU agreement check. After that, any CPU run regenerates every asset in
minutes without touching the quota.

### Smoke first

Set `RUN_MODE = "smoke"` and run 01 → 02 → 03. About 0.6 GPU-h total, and it
exercises every experiment class end to end. Only then switch to `micro`.

---

## 4. Scope cuts in `micro`, and what they cost the paper

These are deliberate, and each is written to `paper_assets/deviations.md` by the
notebook that made it.

1. **Variant 4 (`w/o Manipulation & Transfer`) is not trained.**
   Cost: the paper's "neither component" row carries the original's numbers as
   cited-untested context. Variants 1–3 still give both single-switch contrasts,
   which is what the central claim — *manipulation contributes the gain,
   transfer alone does not* — actually rests on.
2. **Base DiffusionDet is not trained.**
   Cost: the "does the hierarchy beat plain DiffusionDet?" comparison is
   untested in `micro`. Run `budget` or `full` to recover it — its budget is
   independent of the variants, so re-running notebook 02 in `budget` will train
   it and skip everything already finished.
3. **Baselines (RetinaNet / Faster R-CNN / DETR) are optional.**
   Cost: three baseline rows stay cited-untested.
4. **SimMIM pretraining is skipped in every mode by default.**
   The authors' checkpoint is not public and the code lives in another
   repository. Our models start from a backbone that has never seen a panoramic
   radiograph, which confounds any accuracy gap against the paper. Notebook
   opt-11 closes it if you have the quota.

The compute constraint is also turned into a *result*: every diagnosis variant
is evaluated at ~1/3, ~2/3 and 1/1 of its budget. If the variant ordering is
stable across those checkpoints, the ablation conclusion is not an artifact of
the shortened schedule; if it is unstable, that is a reportable finding about
how budget-sensitive the paper's claim is. Notebook 03 prints the verdict and
writes it to the summary.

---

## 5. Things that will bite you

* **Never `pip install detectron2`.** The repo vendors a *modified* detectron2
  and pycocotools (multi-label partial annotations, a 3-tier category schema).
  A pip install silently shadows them and every number changes.
  `setup_env.assert_vendored()` raises on this; it runs in every notebook.
* **The Swin backbone must be a `.pth`.** `DetectionCheckpointer` dispatches on
  file extension: a torch checkpoint named `.pkl` is parsed as a Caffe2 blob and
  the backbone stays random, reported only as warnings. The released config's
  `models/swin_base_patch4_window7_224_22k.pkl` is exactly this trap.
* **`configs/diffdet.custom.swinbase.enumeration.yaml` cannot run.** It sets
  `NUM_CLASSES: 32` as a scalar; `hierarchialdet/head.py:81` indexes that value
  as a 3-element list. Use `configs_repro/` instead.
* **Do not run the training loop in the notebook kernel.** Training goes through
  `src/train_entry.py` as a subprocess so a CUDA OOM or a session kill leaves a
  resumable checkpoint instead of a wedged kernel.
* **A stale `NOISY_BOX_TRAIN` turns the "w/o Manipulation" variant back into the
  full model.** `launch_training` explicitly *unsets* every noisy-box variable
  that was not requested for that run.
* **Disk.** `training_data.zip` alone is 10.9 GB. Notebook 01 downloads one
  archive at a time and deletes each one as soon as it is extracted; do not
  "helpfully" pre-download everything with `snapshot_download`.

---

## 6. Verifying a finished run

```python
from src import manifest
manifest.assert_asset_classes()      # every required asset class accounted for
```

Nothing may be cited in the paper that is not in `paper_assets/manifest.json`.
Asset classes an active run mode deliberately skipped are recorded with
`status="not run"` — that counts as accounted for, and the corresponding table
cells read `not run`. **No number anywhere in `paper_assets/` is interpolated,
extrapolated, or carried over from another run mode.**

## 7. Re-verifying without retraining

A reviewer with the two published Kaggle Datasets attached runs one notebook:

```
notebooks/03_evaluate_and_build_assets.ipynb   →   Accelerator: GPU T4   →   Run All
```

That reproduces every value in `tables/main_results.csv` from the released
checkpoints. Re-running the same notebook on a CPU session rebuilds every table
and figure for free. The exact instructions, with the dataset slugs used for the run, are
regenerated into `paper_assets/scope_and_claims.md`.

---

*All outputs of this suite are research artifacts. Nothing here is a clinical
claim, and nothing here is validated for clinical use.*

---

## 8. Known failure modes and their fixes

**`ValueError: numpy.dtype size changed, Expected 96 ... got 88`**, raised from
`pycocotools/_mask.pyx` inside `detectron2.structures.masks`. This reads as a
detectron2 problem and is not one: the vendored `pycocotools` ships Python
sources only, and its compiled `_mask` extension is grafted in from the
pip-installed package. That extension is compiled against numpy's C ABI, so it
breaks if it was built against a different numpy major version than the one
installed. Pinning a `pycocotools` version *causes* this, because pip then
builds from source in an isolated environment that pulls the newest numpy
regardless of what the image has.

`setup_env.ensure_pycocotools_mask` now owns this end to end: it verifies the
extension by importing it in a subprocess, discards a mismatched one, tries each
candidate in site-packages, and only as a last resort rebuilds with
`--no-build-isolation --no-binary :all:` so the build sees the installed numpy.
If it still cannot, it raises with the exact command to run by hand.
