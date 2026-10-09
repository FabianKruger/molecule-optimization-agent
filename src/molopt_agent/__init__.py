"""Deprecated alias for the seismo package."""

import importlib
import importlib.abc
import importlib.util
import os
import sys
import warnings
from pathlib import Path


def _warn_deprecated() -> None:
    message = "molopt_agent is deprecated; use seismo instead"
    prefix = str(Path(__file__).parent)
    if not prefix.endswith(os.sep):
        prefix += os.sep
    try:
        warnings.warn(
            message,
            FutureWarning,
            stacklevel=2,
            skip_file_prefixes=(prefix,),
        )
    except TypeError:
        # Python 3.11 has no skip_file_prefixes, and it does not skip importlib frames.
        stacklevel = 1
        frame = sys._getframe()
        while frame is not None:
            filename = frame.f_code.co_filename
            if not filename.startswith(prefix) and "importlib" not in filename:
                break
            stacklevel += 1
            frame = frame.f_back
        warnings.warn(message, FutureWarning, stacklevel=stacklevel)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, module):
        self._module = module
        self._canonical_spec = module.__spec__

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        module.__spec__ = self._canonical_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        prefix = "molopt_agent."
        if not fullname.startswith(prefix):
            return None
        real_name = "seismo." + fullname[len(prefix) :]
        real = importlib.import_module(real_name)
        spec = importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(real),
            origin=getattr(real, "__file__", None),
            is_package=hasattr(real, "__path__"),
        )
        if hasattr(real, "__path__"):
            spec.submodule_search_locations = list(real.__path__)
        return spec


sys.meta_path.insert(0, _AliasFinder())
_warn_deprecated()

__all__ = [
    "config",
    "state",
    "save_experiment",
    "oracles",
    "objectives",
    "graph",
]
