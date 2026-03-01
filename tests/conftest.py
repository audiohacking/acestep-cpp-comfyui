"""pytest configuration: inject minimal stubs for ComfyUI-only modules so that
nodes.py can be imported in a plain Python environment without ComfyUI.

We load nodes.py directly via importlib.util.spec_from_file_location so that
Python does not try to import the parent package's __init__.py (which uses
relative imports that only work inside ComfyUI).
"""

import importlib.util
import os
import sys
import types

# ---------------------------------------------------------------------------
# folder_paths stub  (must be installed before nodes.py is loaded)
# ---------------------------------------------------------------------------
_fp = types.ModuleType("folder_paths")
_fp._paths: dict = {}      # name -> [folder, ...]
_fp._registered: dict = {} # name -> [folder, ...]  (via add_model_folder_path)


def _get_folder_paths(name):
    return _fp._paths.get(name, [])


def _add_model_folder_path(name, path):
    _fp._registered.setdefault(name, []).append(path)


def _get_filename_list(name):
    files = []
    for p in _fp._registered.get(name, []):
        try:
            files.extend(
                e for e in os.listdir(p)
                if os.path.isfile(os.path.join(p, e))
            )
        except (FileNotFoundError, PermissionError):
            pass
    return sorted(set(files))


_fp.get_folder_paths = _get_folder_paths
_fp.add_model_folder_path = _add_model_folder_path
_fp.get_filename_list = _get_filename_list

sys.modules["folder_paths"] = _fp

# ---------------------------------------------------------------------------
# torch stub
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

# ---------------------------------------------------------------------------
# Load nodes.py by absolute path so Python never imports the repo-root
# __init__.py (which uses relative imports incompatible with test context).
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_nodes_spec = importlib.util.spec_from_file_location(
    "nodes",
    os.path.join(_repo_root, "nodes.py"),
)
_nodes_module = importlib.util.module_from_spec(_nodes_spec)
sys.modules["nodes"] = _nodes_module
_nodes_spec.loader.exec_module(_nodes_module)
