#!/usr/bin/env python3
"""ComfyUI custom-node installer for acestep-cpp-comfyui.

ComfyUI Manager automatically runs this script when the node is installed
or updated.  It clones ``https://github.com/audiohacking/acestep.cpp`` into
the node directory and builds the ``ace-qwen3`` and ``dit-vae`` binaries
that the *Acestep.cpp Generate* node needs at runtime.

If ``git`` or ``cmake`` are not available the script prints a helpful
message and exits cleanly so that ComfyUI itself still loads normally.
Users can trigger the build later via the **Acestep.cpp Builder** node
inside ComfyUI.
"""

import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(NODE_DIR, "acestep.cpp")
REPO_URL = "https://github.com/audiohacking/acestep.cpp"
BINARIES = ("ace-qwen3", "dit-vae")


# ---------------------------------------------------------------------------
# Helpers (mirrors AcestepCPPBuilder logic so install.py is self-contained)
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    if shutil.which("nvcc") or shutil.which("nvidia-smi"):
        return "cuda"
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("pkg-config") and subprocess.run(
        ["pkg-config", "--exists", "openblas"], capture_output=True
    ).returncode == 0:
        return "blas"
    openblas_headers = [
        "/usr/include/openblas/cblas.h",
        "/usr/local/include/openblas/cblas.h",
        "/opt/homebrew/include/openblas/cblas.h",
    ]
    if any(os.path.isfile(h) for h in openblas_headers):
        return "blas"
    return "cpu"


def _cmake_flags(backend: str):
    return {
        "cuda":  ["-DGGML_CUDA=ON"],
        "metal": [],
        "blas":  ["-DGGML_BLAS=ON"],
        "cpu":   [],
    }.get(backend, [])


def _run(cmd, cwd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}"
        )


# ---------------------------------------------------------------------------
# Main installation routine
# ---------------------------------------------------------------------------

def install() -> None:
    print("[acestep-cpp] Checking prerequisites for binary build …", flush=True)

    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            print(
                f"[acestep-cpp] WARNING: '{tool}' not found on PATH.\n"
                "  Skipping automatic binary build.  You can build the binaries\n"
                "  later using the 'Acestep.cpp Builder' node inside ComfyUI.",
                flush=True,
            )
            return

    # Skip rebuild if both binaries already exist
    build_dir = os.path.join(REPO_DIR, "build")
    if all(os.path.isfile(os.path.join(build_dir, b)) for b in BINARIES):
        print(
            f"[acestep-cpp] Binaries already present in {build_dir} — skipping build.",
            flush=True,
        )
        return

    # Clone or update submodules
    if not os.path.isdir(REPO_DIR):
        print(f"[acestep-cpp] Cloning {REPO_URL} …", flush=True)
        _run(
            ["git", "clone", "--recurse-submodules", REPO_URL, REPO_DIR],
            cwd=NODE_DIR,
        )
    else:
        print(f"[acestep-cpp] Updating submodules in {REPO_DIR} …", flush=True)
        _run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=REPO_DIR,
        )

    # Detect & configure
    backend = _detect_backend()
    print(f"[acestep-cpp] Detected compute backend: {backend}", flush=True)

    os.makedirs(build_dir, exist_ok=True)
    print("[acestep-cpp] Running CMake configure …", flush=True)
    _run(["cmake", ".."] + _cmake_flags(backend), cwd=build_dir)

    # Build
    jobs = str(multiprocessing.cpu_count())
    print(f"[acestep-cpp] Building with {jobs} parallel jobs …", flush=True)
    _run(
        ["cmake", "--build", ".", "--config", "Release", f"-j{jobs}"],
        cwd=build_dir,
    )

    # Verify
    missing = [b for b in BINARIES if not os.path.isfile(os.path.join(build_dir, b))]
    if missing:
        raise RuntimeError(
            f"Build finished but expected binaries not found: {', '.join(missing)}"
        )

    print(
        f"[acestep-cpp] ✓ Build complete.  Binaries ready in {build_dir}: "
        + ", ".join(BINARIES),
        flush=True,
    )


if __name__ == "__main__":
    try:
        install()
    except Exception as exc:
        print(f"[acestep-cpp] Build failed: {exc}", file=sys.stderr, flush=True)
        print(
            "[acestep-cpp] You can retry the build later using the\n"
            "  'Acestep.cpp Builder' node inside ComfyUI.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
