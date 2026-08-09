"""
Shared package for the HierarchicalDet reproducibility study.

All logic lives here; the notebooks are thin orchestration. Import order
matters: `setup_env` puts the HierarchicalDet checkout on sys.path first, so
the vendored (multi-label-modified) detectron2 and pycocotools win over
anything pip installed.
"""
