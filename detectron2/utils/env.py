# Copyright (c) Facebook, Inc. and its affiliates.
import importlib
import importlib.util
import logging
import numpy as np
import os
import random
import sys
from datetime import datetime
import torch

__all__ = ["seed_all_rng"]


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
"""
PyTorch version as a tuple of 2 ints. Useful for comparison.
"""


DOC_BUILDING = os.getenv("_DOC_BUILDING", False)  # set in docs/conf.py
"""
Whether we're building documentation.
"""


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # This codebase's transform/augmentation code (e.g.
    # detectron2/data/transforms/transform.py) references PIL.Image.LINEAR,
    # .NEAREST, .BILINEAR, .BICUBIC as module-level default-argument values,
    # evaluated at import time. Pillow >= 10 removed these flat aliases in
    # favor of the Image.Resampling enum. Restoring them here (rather than
    # relying on pinning an old Pillow version) works regardless of whatever
    # Pillow version is actually installed -- pip installs/pins have proven
    # unreliable to enforce on some hosted notebook environments (e.g. Kaggle
    # blocks package reinstalls under PEP 668 in ways that fail silently).
    # Set unconditionally rather than guarding with hasattr(): on some Pillow
    # versions (e.g. 11.3.0) accessing the removed name itself raises
    # RuntimeError instead of returning False, so hasattr() is not safe to
    # use as the check here. Overwriting is harmless on Pillow versions that
    # still define these natively.
    import PIL.Image

    PIL.Image.NEAREST = PIL.Image.Resampling.NEAREST
    PIL.Image.BILINEAR = PIL.Image.Resampling.BILINEAR
    PIL.Image.BICUBIC = PIL.Image.Resampling.BICUBIC
    PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR

    # Similarly, some torchvision versions' `torchvision/utils.py` imports
    # PIL.ImageFont, which itself does `from ._util import is_directory,
    # is_path` -- but newer Pillow removed `is_directory` from PIL._util
    # (only `is_path` remains). Restore it so `import torchvision` (triggered
    # transitively by detectron2/layers/deform_conv.py) doesn't ImportError.
    import PIL._util

    if not hasattr(PIL._util, "is_directory"):
        import os as _os

        def _is_directory(f):
            return isinstance(f, str) and _os.path.isdir(f)

        PIL._util.is_directory = _is_directory

    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("DETECTRON2_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ModuleNotFoundError:
            # Other types of ImportError, if happened, should not be ignored.
            # Because a failed opencv import could mess up address space
            # https://github.com/skvark/opencv-python/issues/381
            pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    # fmt: off
    assert get_version(torch) >= (1, 4), "Requires torch>=1.4"
    import fvcore
    assert get_version(fvcore, 3) >= (0, 1, 2), "Requires fvcore>=0.1.2"
    import yaml
    assert get_version(yaml) >= (5, 1), "Requires pyyaml>=5.1"
    # fmt: on


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("DETECTRON2_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("detectron2.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()


def fixup_module_metadata(module_name, namespace, keys=None):
    """
    Fix the __qualname__ of module members to be their exported api name, so
    when they are referenced in docs, sphinx can find them. Reference:
    https://github.com/python-trio/trio/blob/6754c74eacfad9cc5c92d5c24727a2f3b620624e/trio/_util.py#L216-L241
    """
    if not DOC_BUILDING:
        return
    seen_ids = set()

    def fix_one(qualname, name, obj):
        # avoid infinite recursion (relevant when using
        # typing.Generic, for example)
        if id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))

        mod = getattr(obj, "__module__", None)
        if mod is not None and (mod.startswith(module_name) or mod.startswith("fvcore.")):
            obj.__module__ = module_name
            # Modules, unlike everything else in Python, put fully-qualitied
            # names into their __name__ attribute. We check for "." to avoid
            # rewriting these.
            if hasattr(obj, "__name__") and "." not in obj.__name__:
                obj.__name__ = name
                obj.__qualname__ = qualname
            if isinstance(obj, type):
                for attr_name, attr_value in obj.__dict__.items():
                    fix_one(objname + "." + attr_name, attr_name, attr_value)

    if keys is None:
        keys = namespace.keys()
    for objname in keys:
        if not objname.startswith("_"):
            obj = namespace[objname]
            fix_one(objname, objname, obj)
