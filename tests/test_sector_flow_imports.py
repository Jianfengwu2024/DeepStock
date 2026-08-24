from __future__ import annotations

import os
import subprocess
import sys


def test_sector_flow_types_import_without_torch() -> None:
    script = """
import builtins
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('torch intentionally unavailable')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from bnhmm.sector_flow.types import SectorMinutePanel
from bnhmm.sector_flow import SectorMinutePanelBuilder
assert SectorMinutePanel is not None
assert SectorMinutePanelBuilder is not None
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_boundary_index_import_does_not_require_scipy_topology() -> None:
    script = """
import builtins
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'scipy' or name.startswith('scipy.'):
        raise ModuleNotFoundError('scipy intentionally unavailable')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from bnhmm.boundary_fk.synthetic_index import SyntheticSectorIndexBuilder
from bnhmm.boundary_fk import CausalBoundaryNormalizer
assert SyntheticSectorIndexBuilder is not None
assert CausalBoundaryNormalizer is not None
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
