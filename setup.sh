#!/usr/bin/env bash
# Local CPU/MPS environment for dev + smoke-testing on machines without a CUDA GPU
# (e.g. Apple Silicon Macs). For real training, use the Kaggle GPU notebook instead --
# see kaggle_setup.ipynb.
#
# Usage: bash setup.sh
set -euo pipefail

ENV_NAME="hierarchicaldet"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "${ENV_NAME}" python=3.9 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# CPU/MPS torch build -- no CUDA index-url, this machine has no NVIDIA GPU.
pip install torch torchvision torchaudio

pip install -r requirements.txt

# The bundled pycocotools/ (custom categories_1/2/3 3-tier format) ships without
# its compiled _mask Cython extension, and repo-root imports always shadow the
# pip-installed pycocotools package. Copy the compiled extension for this
# interpreter/platform from the pip install into the bundled package so
# `import pycocotools.mask` resolves.
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
MASK_SO=$(find "${SITE_PACKAGES}/pycocotools" -maxdepth 1 -name '_mask*.so' | head -1)
if [ -z "${MASK_SO}" ]; then
    echo "ERROR: could not find compiled pycocotools _mask extension in ${SITE_PACKAGES}/pycocotools" >&2
    exit 1
fi
cp "${MASK_SO}" pycocotools/

echo "Verifying imports..."
python -c "
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from hierarchialdet import add_diffusiondet_config, DiffusionDetDatasetMapper, DiffusionDetWithTTA
from pycocotools.coco import COCO
import evaluator
print('All imports OK.')
"

echo "Setup complete. Activate with: conda activate ${ENV_NAME}"
