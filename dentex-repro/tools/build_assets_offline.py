#!/usr/bin/env python3
"""
Run notebook 03's asset build on this machine, with no Kaggle and no GPU.

Every table and figure is a pure function of ``paper_assets/results_raw/``, so
rebuilding them needs no model, no data and no accelerator. Until now the only
way to exercise that code was a Kaggle session, which made a one-line figure fix
a multi-minute round trip and meant figure defects were only ever discovered
after the fact.

This executes the **notebook's own cells**, extracted from the generated
``.ipynb`` rather than reimplemented, so there is nothing to drift out of sync.

Usage:

    # copy paper_assets/results_raw/ out of the Kaggle output first
    python tools/build_assets_offline.py [--results DIR] [--out DIR]

Requires matplotlib; it does not require torch, detectron2 or the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

#: The asset build starts at the cell bearing this marker and runs to the end of
#: the notebook, minus the cells that need a live session. Everything before it
#: evaluates checkpoints; everything from here on is a pure function of
#: ``results_raw/``.
START_MARKER = "---- Load every raw result ----"
#: Cells that talk to Kaggle or need results this run did not produce.
SKIP_IF_PRESENT = ("publish_kaggle_dataset", "write_notebook_summary")


def notebook_cells(path: str):
    """
    Yield the asset-build cells, or raise if the marker has gone missing.

    The marker used to be matched against markdown cells only. These notebooks
    are generated and contain no markdown at all, so the scan fell through every
    cell, yielded nothing, and the build reported success having written nothing
    -- which is how the shipped figures came to predate their own fixes. A
    marker that matches no cell is now a hard error.
    """
    with open(path) as handle:
        notebook = json.load(handle)
    started = False
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if not started:
            if START_MARKER in source:
                started = True
            else:
                continue
        if cell["cell_type"] != "code":
            continue
        if any(marker in source for marker in SKIP_IF_PRESENT):
            continue
        yield source
    if not started:
        raise LookupError(
            "no cell in {} contains the start marker {!r}; the asset build would "
            "have run zero cells and exited successfully.".format(path, START_MARKER))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=None,
                        help="directory holding results_raw/<mode>/ "
                             "(default: <project>/paper_assets)")
    parser.add_argument("--out", default=None,
                        help="where to write tables/ and figures/ "
                             "(default: same as --results)")
    parser.add_argument("--mode", default=os.environ.get("RUN_MODE", "micro"))
    arguments = parser.parse_args()

    os.environ["RUN_MODE"] = arguments.mode
    assets = os.path.abspath(arguments.results or os.path.join(PROJECT_ROOT, "paper_assets"))
    os.environ["DENTEX_PAPER_ASSETS"] = os.path.abspath(arguments.out or assets)

    sys.path.insert(0, PROJECT_ROOT)
    import matplotlib

    matplotlib.use("Agg")

    from src import (degradations, eval_utils, figures, manifest,  # noqa: F401
                     setup_env, tables, train_utils)

    raw = os.path.join(assets, "results_raw", arguments.mode)
    if not os.path.isdir(raw) or not os.listdir(raw):
        print("No results in {}.\n"
              "Copy paper_assets/results_raw/ out of the Kaggle output "
              "(Output tab -> repo/dentex-repro/paper_assets/, or "
              "/kaggle/working/paper_assets/ on newer runs) and point --results "
              "at the folder that contains it.".format(raw))
        return 1
    if os.environ["DENTEX_PAPER_ASSETS"] != assets:
        # Results live elsewhere; make them visible under the output root.
        setup_env.hydrate_from_attached_datasets([assets])

    print("results : {} ({} files)".format(raw, len(os.listdir(raw))))
    print("output  : {}\n".format(setup_env.PAPER_ASSETS))

    # The notebook's asset-build cells expect these to already exist.
    from src import data_convert

    namespace = {
        "__name__": "__main__",
        "os": os, "sys": sys, "json": json,
        "run": setup_env.resolve_run_mode(arguments.mode),
        "setup_env": setup_env, "manifest": manifest, "tables": tables,
        "figures": figures, "eval_utils": eval_utils, "degradations": degradations,
        "train_utils": train_utils, "data_convert": data_convert,
        "paths": data_convert.layout(),
        "HAS_GPU": False,
        "FULL_WEIGHTS": None,
        "models": {},
        "training": setup_env.read_notebook_summary("02_train_all") or {},
        "agreement": None,
        "PUBLISH_KAGGLE_DATASET": False,
        "CKPT_DATASET_SLUG": "dentex-repro-ckpts",
        "DATA_DATASET_SLUG": "dentex-repro-data",
    }

    notebook = os.path.join(PROJECT_ROOT, "notebooks",
                            "03_evaluate_and_build_assets.ipynb")
    for index, source in enumerate(notebook_cells(notebook), 1):
        try:
            exec(compile(source, "<asset-cell-{}>".format(index), "exec"), namespace)
        except Exception as error:                        # noqa: BLE001
            print("\ncell {} failed: {}: {}".format(
                index, type(error).__name__, error))
            print("--- cell source ---")
            print(source[:1500])
            return 1

    print("\n" + "=" * 72)
    for directory, _subdirs, names in sorted(os.walk(setup_env.PAPER_ASSETS)):
        relative = os.path.relpath(directory, setup_env.PAPER_ASSETS)
        for name in sorted(names):
            print("  {:<56} {:>8.1f} KB".format(
                os.path.join(relative, name).lstrip("./"),
                os.path.getsize(os.path.join(directory, name)) / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
