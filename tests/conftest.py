"""Shared pytest helpers for loading repo-local script modules by path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script_module(script_name: str) -> ModuleType:
    """Load a Python file from ``scripts/`` as a module for testing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
