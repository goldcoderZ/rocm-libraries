# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Load the rocisa C extension and re-export its contents into this package
# namespace. This allows `from rocisa import rocIsa` (and submodule access like
# `rocisa.enum`) to work when the CWD is the project root and Python's
# PathFinder resolves `rocisa` as a namespace package before the editable-
# install finder can intercept it.
import importlib
import importlib.util
import os
import sys
from pathlib import Path

_so_dir = Path(__file__).parent
_candidates = sorted(_so_dir.glob("rocisa.cpython-*.so"))
if not _candidates:
    raise ImportError(
        f"rocisa C extension (.so) not found in {_so_dir}. "
        "Run `uv sync` to build it."
    )
_so_path = _candidates[0]

# Load the .so as the top-level `rocisa` module, replacing this package.
_spec = importlib.util.spec_from_file_location(
    "rocisa", _so_path, submodule_search_locations=[]
)
_ext = importlib.util.module_from_spec(_spec)
sys.modules["rocisa"] = _ext
_spec.loader.exec_module(_ext)

# Re-populate this module's namespace so `from rocisa import X` works.
globals().update({k: v for k, v in vars(_ext).items() if not k.startswith("__")})
