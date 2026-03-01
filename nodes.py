import os
import json
import platform
import multiprocessing
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import torch
import folder_paths

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def load_config() -> Dict[str, Any]:
    """Load config from config.json; return empty dict if missing or invalid."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_user_model_folders() -> List[str]:
    """Return user-specified model folders from config.json."""
    return load_config().get("model_folders", [])


def get_merged_model_folders() -> List[str]:
    """Merge ComfyUI text_encoders folders with any user-configured folders."""
    try:
        comfy_folders = folder_paths.get_folder_paths("text_encoders")
    except Exception:
        comfy_folders = []
    all_folders = comfy_folders + get_user_model_folders()
    return [f for f in all_folders if os.path.isdir(f)]


def scan_gguf_models() -> List[str]:
    """Scan registered 'acestep_gguf' folder type for .gguf files.

    Tries ComfyUI's folder_paths registry first (so the built-in model
    manager can track the files), then falls back to manual scanning.
    """
    try:
        all_files = folder_paths.get_filename_list("acestep_gguf")
        gguf_files = sorted(f for f in all_files if f.lower().endswith(".gguf"))
        if gguf_files:
            return gguf_files
    except Exception:
        pass

    # Fallback: manual scan of text_encoders + user-configured folders
    seen_models: set = set()
    models: List[str] = []
    for folder in get_merged_model_folders():
        try:
            for name in os.listdir(folder):
                if name.lower().endswith(".gguf") and name not in seen_models:
                    models.append(name)
                    seen_models.add(name)
        except OSError:
            pass
    return sorted(models)


def find_model_path(model_name: str) -> Optional[str]:
    """Return the full path to a model file, or None if not found."""
    for folder in get_merged_model_folders():
        path = os.path.join(folder, model_name)
        if os.path.isfile(path):
            return path
    return None


def _coerce_float(value: Any, default: float) -> float:
    """Return *value* as a float, falling back to *default* for empty strings.

    Older ComfyUI workflows may store optional FLOAT widget values as ``""``
    instead of the numeric default when the field was not explicitly set.
    """
    if isinstance(value, str):
        return float(value) if value.strip() else default
    return float(value)
def get_binary_path(binary_name: str) -> Optional[str]:
    """
    Locate an acestep.cpp binary.

    Search order:
      1. Explicit path from config.json ``binary_paths`` mapping.
      2. System PATH (via shutil.which).
      3. ``<node_dir>/acestep.cpp/build/<binary_name>`` (local build alongside the node).
    """
    config = load_config()
    explicit = config.get("binary_paths", {}).get(binary_name)
    if explicit and os.path.isfile(explicit):
        return explicit

    on_path = shutil.which(binary_name)
    if on_path:
        return on_path

    local = os.path.join(os.path.dirname(__file__), "acestep.cpp", "build", binary_name)
    if os.path.isfile(local):
        return local

    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# HuggingFace repo that hosts all pre-quantized ACE-Step GGUFs
_HF_REPO = "Serveurperso/ACE-Step-1.5-GGUF"

# Quant resolution rules mirroring models.sh
_EMB_QUANTS = ["Q8_0", "BF16"]
_LM_SMALL_QUANTS = ["Q8_0", "BF16"]
_LM_4B_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "BF16"]
_DIT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "BF16"]
_DIT_VARIANTS = ["turbo", "sft", "base", "turbo-shift1", "turbo-shift3", "turbo-continuous"]
_LM_SIZES = ["4B", "1.7B", "0.6B"]


def _resolve_quant(requested: str, model_type: str) -> str:
    """Mirror the quant resolution logic from models.sh."""
    if model_type in ("emb", "lm_small"):
        return requested if requested == "BF16" else "Q8_0"
    if model_type == "lm_4B":
        if requested in ("Q4_K_M", "Q5_K_M"):
            return "Q5_K_M"
        return requested if requested in ("BF16", "Q8_0", "Q6_K") else "Q8_0"
    return requested  # dit: all quants available


class AcestepCPPModelDownloader:
    """
    Download ACE-Step GGUF models from HuggingFace (``Serveurperso/ACE-Step-1.5-GGUF``)
    directly into a local folder so they appear in the Model Loader dropdown.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            default_dir = folder_paths.get_folder_paths("text_encoders")[0]
        except Exception:
            default_dir = os.path.join(os.path.dirname(__file__), "models")

        return {
            "required": {
                "save_dir": (
                    "STRING",
                    {
                        "default": default_dir,
                        "tooltip": "Directory to save downloaded GGUF files into",
                    },
                ),
                "lm_size": (
                    _LM_SIZES,
                    {"default": "4B", "tooltip": "LM model size"},
                ),
                "quant": (
                    _DIT_QUANTS,
                    {
                        "default": "Q8_0",
                        "tooltip": (
                            "Quantisation level. "
                            "Embedding/LM models fall back to the nearest available quant "
                            "automatically (mirroring models.sh logic)."
                        ),
                    },
                ),
                "dit_variant": (
                    _DIT_VARIANTS,
                    {"default": "turbo", "tooltip": "DiT model variant to download"},
                ),
            },
            "optional": {
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "HuggingFace access token (leave empty for public repos)",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Re-download even if the file already exists",
                        "label_on": "Overwrite",
                        "label_off": "Skip existing",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("downloaded_files",)
    FUNCTION = "download"
    CATEGORY = "AcestepCPP"
    OUTPUT_NODE = True

    def download(
        self,
        save_dir: str,
        lm_size: str = "4B",
        quant: str = "Q8_0",
        dit_variant: str = "turbo",
        hf_token: str = "",
        overwrite: bool = False,
    ):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model downloading. "
                "Install it with: pip install huggingface_hub"
            )

        os.makedirs(save_dir, exist_ok=True)
        token = hf_token.strip() or None

        lm_type = "lm_4B" if lm_size == "4B" else "lm_small"
        files_to_download = [
            "vae-BF16.gguf",
            f"Qwen3-Embedding-0.6B-{_resolve_quant(quant, 'emb')}.gguf",
            f"acestep-5Hz-lm-{lm_size}-{_resolve_quant(quant, lm_type)}.gguf",
            f"acestep-v15-{dit_variant}-{_resolve_quant(quant, 'dit')}.gguf",
        ]

        downloaded: List[str] = []
        skipped: List[str] = []

        for filename in files_to_download:
            dest = os.path.join(save_dir, filename)
            if os.path.isfile(dest) and not overwrite:
                print(f"[AcestepCPP] Skip (exists): {filename}")
                skipped.append(filename)
                continue

            print(f"[AcestepCPP] Downloading: {filename} ...")
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=filename,
                local_dir=save_dir,
                token=token,
            )
            print(f"[AcestepCPP] Done: {filename}")
            downloaded.append(filename)

        summary_parts = []
        if downloaded:
            summary_parts.append(f"Downloaded: {', '.join(downloaded)}")
        if skipped:
            summary_parts.append(f"Skipped (already exist): {', '.join(skipped)}")
        summary = "\n".join(summary_parts) if summary_parts else "Nothing to do."
        return (summary,)


# The upstream source repository for the binaries
_ACESTEP_CPP_REPO = "https://github.com/audiohacking/acestep.cpp"


class AcestepCPPBuilder:
    """
    Clone ``acestep.cpp`` from GitHub and build the ``ace-qwen3`` and
    ``dit-vae`` binaries via CMake.

    The default clone directory (``<node_dir>/acestep.cpp``) is the same path
    that ``get_binary_path()`` already searches, so the built binaries will be
    found automatically by the Generate node without any extra configuration.
    """

    BACKENDS = ["auto", "cuda", "metal", "blas", "cpu"]

    @classmethod
    def INPUT_TYPES(cls):
        default_dir = os.path.join(os.path.dirname(__file__), "acestep.cpp")
        return {
            "required": {
                "clone_dir": (
                    "STRING",
                    {
                        "default": default_dir,
                        "tooltip": (
                            "Directory to clone acestep.cpp into. "
                            "Defaults to <node_dir>/acestep.cpp so that the "
                            "Generate node finds the binaries automatically."
                        ),
                    },
                ),
                "backend": (
                    cls.BACKENDS,
                    {
                        "default": "auto",
                        "tooltip": (
                            "GPU/compute backend for CMake. "
                            "'auto' detects CUDA, Metal, or falls back to CPU."
                        ),
                    },
                ),
            },
            "optional": {
                "force_rebuild": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Delete the existing build directory and rebuild from scratch."
                        ),
                        "label_on": "Force rebuild",
                        "label_off": "Incremental",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("build_log",)
    FUNCTION = "build"
    CATEGORY = "AcestepCPP"
    OUTPUT_NODE = True

    # Maximum number of recent log lines included in error messages
    _MAX_ERROR_LOG_LINES = 40

    # Binaries produced by the cmake build
    _BINARIES = ("ace-qwen3", "dit-vae")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        """Return the best available cmake backend flag value."""
        # CUDA: look for nvcc or nvidia-smi on PATH
        if shutil.which("nvcc") or shutil.which("nvidia-smi"):
            return "cuda"
        # Apple Metal: macOS always supports Metal through ggml
        if platform.system() == "Darwin":
            return "metal"
        # OpenBLAS: check pkg-config first (cross-distro), then common header
        # locations, then dpkg as a last resort on Debian-based systems
        if shutil.which("pkg-config") and subprocess.run(
            ["pkg-config", "--exists", "openblas"],
            capture_output=True,
        ).returncode == 0:
            return "blas"
        openblas_headers = [
            "/usr/include/openblas/cblas.h",
            "/usr/local/include/openblas/cblas.h",
            "/opt/homebrew/include/openblas/cblas.h",
        ]
        if any(os.path.isfile(h) for h in openblas_headers):
            return "blas"
        if shutil.which("dpkg") and subprocess.run(
            ["dpkg", "-l", "libopenblas-dev"],
            capture_output=True,
        ).returncode == 0:
            return "blas"
        return "cpu"

    @staticmethod
    def _cmake_flags(backend: str) -> List[str]:
        """Translate backend name to CMake -D flags."""
        mapping = {
            "cuda":  ["-DGGML_CUDA=ON"],
            "metal": [],  # Metal is auto-enabled on macOS by ggml
            "blas":  ["-DGGML_BLAS=ON"],
            "cpu":   [],
        }
        return mapping.get(backend, [])

    @staticmethod
    def _run(cmd: List[str], cwd: str, log_lines: List[str]) -> None:
        """Run a command, stream output to log_lines, raise on failure."""
        label = " ".join(cmd)
        log_lines.append(f"$ {label}")
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True
        )
        if result.stdout:
            log_lines.extend(result.stdout.splitlines())
        if result.stderr:
            log_lines.extend(result.stderr.splitlines())
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {label}\n"
                + "\n".join(log_lines[-self._MAX_ERROR_LOG_LINES:])
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self,
        clone_dir: str,
        backend: str = "auto",
        force_rebuild: bool = False,
    ):
        log: List[str] = []

        # --- Resolve backend --------------------------------------------------
        resolved = self._detect_backend() if backend == "auto" else backend
        log.append(f"[AcestepCPP] Backend: {resolved} (requested: {backend})")

        # --- Clone or update --------------------------------------------------
        repo_dir = clone_dir
        if not os.path.isdir(repo_dir):
            log.append(f"[AcestepCPP] Cloning {_ACESTEP_CPP_REPO} -> {repo_dir}")
            if not shutil.which("git"):
                raise RuntimeError(
                    "git is not available on PATH. Please install git and retry."
                )
            self._run(
                ["git", "clone", "--recurse-submodules", _ACESTEP_CPP_REPO, repo_dir],
                cwd=os.path.dirname(repo_dir),
                log_lines=log,
            )
        else:
            log.append(f"[AcestepCPP] Repo exists: {repo_dir} — updating submodules")
            self._run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=repo_dir,
                log_lines=log,
            )

        # --- Prepare build directory ------------------------------------------
        build_dir = os.path.join(repo_dir, "build")
        if force_rebuild and os.path.isdir(build_dir):
            log.append(f"[AcestepCPP] Removing existing build dir: {build_dir}")
            shutil.rmtree(build_dir)

        os.makedirs(build_dir, exist_ok=True)

        # --- CMake configure --------------------------------------------------
        cmake_cmd = ["cmake", ".."] + self._cmake_flags(resolved)
        log.append(f"[AcestepCPP] Configuring: {' '.join(cmake_cmd)}")
        if not shutil.which("cmake"):
            raise RuntimeError(
                "cmake is not available on PATH. "
                "Install cmake (e.g. apt install cmake) and retry."
            )
        self._run(cmake_cmd, cwd=build_dir, log_lines=log)

        # --- CMake build ------------------------------------------------------
        jobs = str(multiprocessing.cpu_count())
        build_cmd = ["cmake", "--build", ".", "--config", "Release", f"-j{jobs}"]
        log.append(f"[AcestepCPP] Building with {jobs} parallel jobs ...")
        self._run(build_cmd, cwd=build_dir, log_lines=log)

        # --- Verify outputs ---------------------------------------------------
        missing = []
        for binary in self._BINARIES:
            path = os.path.join(build_dir, binary)
            if not os.path.isfile(path):
                missing.append(binary)

        if missing:
            raise RuntimeError(
                f"Build succeeded but expected binaries not found: {', '.join(missing)}. "
                f"Check the build log above."
            )

        log.append(
            f"[AcestepCPP] Build complete. Binaries in {build_dir}: "
            + ", ".join(self._BINARIES)
        )
        return ("\n".join(log),)


class AcestepCPPLoraLoader:
    """
    Specify a LoRA adapter file for use with acestep.cpp.

    Enter the full path to any ``.gguf`` or ``.safetensors`` LoRA file on
    your filesystem. Outputs an ``ACESTEP_LORA`` dict consumed by the
    generator node.
    """

    _ALLOWED_EXTENSIONS = (".gguf", ".safetensors")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Full path to a LoRA adapter file (.gguf or .safetensors). "
                            "You can use any file location on your filesystem."
                        ),
                    },
                ),
                "lora_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "LoRA adapter scale",
                    },
                ),
            }
        }

    RETURN_TYPES = ("ACESTEP_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "AcestepCPP"

    def load_lora(self, lora_path: str, lora_scale: float):
        path = lora_path.strip()
        if not path:
            raise ValueError(
                "lora_path is empty. Enter the full path to your LoRA file."
            )
        if not any(path.lower().endswith(ext) for ext in self._ALLOWED_EXTENSIONS):
            raise ValueError(
                f"Unsupported LoRA file type: {path!r}. "
                f"Expected one of: {', '.join(self._ALLOWED_EXTENSIONS)}"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LoRA file not found: {path}"
            )
        return ({"path": path, "scale": lora_scale},)


class AcestepCPPModelLoader:
    """
    Select the four GGUF model files required by acestep.cpp.

    Outputs an ``ACESTEP_MODELS`` dict consumed by the generator node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_list = scan_gguf_models()
        options = model_list if model_list else ["No GGUF models found"]
        return {
            "required": {
                "lm_model": (
                    options,
                    {
                        "tooltip": (
                            "LM (ace-qwen3) model GGUF, e.g. "
                            "acestep-5Hz-lm-4B-Q8_0.gguf"
                        )
                    },
                ),
                "text_encoder_model": (
                    options,
                    {
                        "tooltip": (
                            "Text-encoder GGUF, e.g. "
                            "Qwen3-Embedding-0.6B-Q8_0.gguf"
                        )
                    },
                ),
                "dit_model": (
                    options,
                    {
                        "tooltip": (
                            "DiT GGUF, e.g. "
                            "acestep-v15-turbo-Q8_0.gguf"
                        )
                    },
                ),
                "vae_model": (
                    options,
                    {"tooltip": "VAE GGUF, e.g. vae-BF16.gguf"},
                ),
            }
        }

    RETURN_TYPES = ("ACESTEP_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_models"
    CATEGORY = "AcestepCPP"

    def load_models(
        self,
        lm_model: str,
        text_encoder_model: str,
        dit_model: str,
        vae_model: str,
    ):
        paths = {
            "lm_model": find_model_path(lm_model),
            "text_encoder": find_model_path(text_encoder_model),
            "dit": find_model_path(dit_model),
            "vae": find_model_path(vae_model),
        }

        missing = [
            label
            for label, path in [
                ("LM model", paths["lm_model"]),
                ("text encoder", paths["text_encoder"]),
                ("DiT model", paths["dit"]),
                ("VAE model", paths["vae"]),
            ]
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Could not locate model file(s): {', '.join(missing)}. "
                "Check your model folder configuration."
            )

        return (paths,)


class AcestepCPPGenerate:
    """
    Generate music with acestep.cpp.

    Runs ``ace-qwen3`` (LM) followed by ``dit-vae`` (DiT + VAE) and returns
    the result as a ComfyUI AUDIO tensor.
    """

    VOCAL_LANGUAGES = [
        "unknown", "en", "zh", "fr", "de", "es", "ja", "ko", "pt", "ru", "it",
    ]
    TASK_TYPES = ["text2music", "cover", "repaint"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": (
                    "ACESTEP_MODELS",
                    {"tooltip": "Model paths from the AcestepCPP Model Loader node"},
                ),
                "caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Upbeat pop rock with driving guitars and catchy hooks",
                        "tooltip": "Music style/description passed to the LM",
                    },
                ),
            },
            "optional": {
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Song lyrics (leave empty for the LM to generate)",
                    },
                ),
                "task_type": (
                    cls.TASK_TYPES,
                    {"default": "text2music", "tooltip": "Generation mode"},
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Generate an instrumental track (no vocals)",
                        "label_on": "Instrumental",
                        "label_off": "Vocal",
                    },
                ),
                "vocal_language": (
                    cls.VOCAL_LANGUAGES,
                    {"default": "unknown", "tooltip": "Language of the vocals"},
                ),
                "duration": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 300,
                        "tooltip": "Duration in seconds (-1 lets the LM decide)",
                    },
                ),
                "bpm": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 300,
                        "tooltip": "Beats per minute (0 lets the LM decide)",
                    },
                ),
                "keyscale": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Key and scale, e.g. 'C major' (leave empty for the LM to decide)",
                    },
                ),
                "timesignature": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Time signature, e.g. '4/4' (leave empty for the LM to decide)",
                    },
                ),
                "inference_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 200,
                        "tooltip": "DiT diffusion steps (8 = turbo preset, 50 = SFT preset)",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale (ignored by turbo models)",
                    },
                ),
                "shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Flow-matching shift (3.0 = turbo preset, 6.0 = SFT preset)",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "tooltip": "Random seed (-1 for a random seed)",
                    },
                ),
                "lm_temperature": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "LM sampling temperature",
                    },
                ),
                "lm_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "LM classifier-free guidance scale",
                    },
                ),
                "lm_top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "LM nucleus (top-p) sampling probability",
                    },
                ),
                "lm_negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt for the LM",
                    },
                ),
                "reference_audio": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to a WAV/MP3 reference file for timbre transfer",
                    },
                ),
                "src_audio": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to a WAV/MP3 source file for cover mode",
                    },
                ),
                "audio_cover_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Cover influence strength (0 = silence, 1 = full cover)",
                    },
                ),
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional path to a DiT LoRA adapter file",
                    },
                ),
                "lora_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "LoRA adapter scale",
                    },
                ),
                "reference_audio_input": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Reference audio for timbre transfer, connected from a "
                            "Load Audio node. Overrides the 'reference_audio' path "
                            "string when connected."
                        ),
                    },
                ),
                "src_audio_input": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Source audio for cover/repaint mode, connected from a "
                            "Load Audio node. Overrides the 'src_audio' path string "
                            "when connected."
                        ),
                    },
                ),
                "lora": (
                    "ACESTEP_LORA",
                    {
                        "tooltip": (
                            "LoRA adapter from the Acestep.cpp LoRA Loader node. "
                            "Overrides 'lora_path' / 'lora_scale' when connected."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AcestepCPP"

    def generate(
        self,
        models: Dict[str, Any],
        caption: str,
        lyrics: str = "",
        task_type: str = "text2music",
        instrumental: bool = False,
        vocal_language: str = "unknown",
        duration: int = -1,
        bpm: int = 0,
        keyscale: str = "",
        timesignature: str = "",
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        shift: float = 3.0,
        seed: int = -1,
        lm_temperature: float = 0.85,
        lm_cfg_scale: float = 2.0,
        lm_top_p: float = 0.9,
        lm_negative_prompt: str = "",
        reference_audio: str = "",
        src_audio: str = "",
        audio_cover_strength: float = 1.0,
        lora_path: str = "",
        lora_scale: float = 1.0,
        reference_audio_input: Optional[Dict[str, Any]] = None,
        src_audio_input: Optional[Dict[str, Any]] = None,
        lora: Optional[Dict[str, Any]] = None,
    ):
        import torchaudio

        # Coerce optional FLOAT inputs that may arrive as empty strings from
        # workflows saved with an older version of the node schema.
        lm_top_p = _coerce_float(lm_top_p, 0.9)
        audio_cover_strength = _coerce_float(audio_cover_strength, 1.0)

        ace_qwen3 = get_binary_path("ace-qwen3")
        dit_vae = get_binary_path("dit-vae")

        if not ace_qwen3:
            raise FileNotFoundError(
                "ace-qwen3 binary not found. "
                "Build acestep.cpp and set binary_paths.ace-qwen3 in config.json, "
                "or add the binary to your PATH."
            )
        if not dit_vae:
            raise FileNotFoundError(
                "dit-vae binary not found. "
                "Build acestep.cpp and set binary_paths.dit-vae in config.json, "
                "or add the binary to your PATH."
            )

        # If a LoRA was supplied via the LoRA Loader node, it takes priority
        # over the freeform lora_path / lora_scale widget values.
        if lora is not None:
            lora_path = lora["path"]
            lora_scale = lora["scale"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Materialise AUDIO tensor inputs as WAV files so the binary can
            # read them.  These override the freeform string-path widgets.
            if reference_audio_input is not None:
                ref_wav = os.path.join(tmpdir, "reference_audio.wav")
                torchaudio.save(
                    ref_wav,
                    reference_audio_input["waveform"].squeeze(0),
                    reference_audio_input["sample_rate"],
                )
                reference_audio = ref_wav

            if src_audio_input is not None:
                src_wav = os.path.join(tmpdir, "src_audio.wav")
                torchaudio.save(
                    src_wav,
                    src_audio_input["waveform"].squeeze(0),
                    src_audio_input["sample_rate"],
                )
                src_audio = src_wav

            request_path = os.path.join(tmpdir, "request.json")

            request: Dict[str, Any] = {
                "task_type": task_type,
                "caption": caption,
                "lyrics": lyrics,
                "instrumental": instrumental,
                "vocal_language": vocal_language,
                "duration": duration,
                "bpm": bpm,
                "seed": seed,
                "lm_temperature": lm_temperature,
                "lm_cfg_scale": lm_cfg_scale,
                "lm_top_p": lm_top_p,
                "lm_negative_prompt": lm_negative_prompt,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "audio_cover_strength": audio_cover_strength,
            }

            if keyscale.strip():
                request["keyscale"] = keyscale.strip()
            if timesignature.strip():
                request["timesignature"] = timesignature.strip()
            if reference_audio.strip():
                request["reference_audio"] = reference_audio.strip()
            if src_audio.strip():
                request["src_audio"] = src_audio.strip()

            with open(request_path, "w") as f:
                json.dump(request, f)

            # Step 1 -- LM: ace-qwen3 -> request0.json
            lm_cmd = [
                ace_qwen3,
                "--request", request_path,
                "--model", models["lm_model"],
            ]
            lm_result = subprocess.run(
                lm_cmd, capture_output=True, text=True, cwd=tmpdir
            )
            if lm_result.returncode != 0:
                raise RuntimeError(
                    f"ace-qwen3 failed (exit {lm_result.returncode}):\n"
                    f"{lm_result.stderr}"
                )

            lm_output = os.path.join(tmpdir, "request0.json")
            if not os.path.isfile(lm_output):
                raise RuntimeError(
                    "ace-qwen3 did not produce request0.json.\n"
                    f"stdout: {lm_result.stdout}\nstderr: {lm_result.stderr}"
                )

            # Step 2 -- DiT+VAE: dit-vae -> request00.wav
            dit_cmd = [
                dit_vae,
                "--request", lm_output,
                "--text-encoder", models["text_encoder"],
                "--dit", models["dit"],
                "--vae", models["vae"],
            ]
            if lora_path.strip():
                dit_cmd += ["--lora", lora_path.strip(), "--lora-scale", str(lora_scale)]

            dit_result = subprocess.run(
                dit_cmd, capture_output=True, text=True, cwd=tmpdir
            )
            if dit_result.returncode != 0:
                raise RuntimeError(
                    f"dit-vae failed (exit {dit_result.returncode}):\n"
                    f"{dit_result.stderr}"
                )

            wav_path = os.path.join(tmpdir, "request00.wav")
            if not os.path.isfile(wav_path):
                raise RuntimeError(
                    "dit-vae did not produce request00.wav.\n"
                    f"stdout: {dit_result.stdout}\nstderr: {dit_result.stderr}"
                )

            waveform, sample_rate = torchaudio.load(wav_path)
            # ComfyUI AUDIO format: waveform shape (batch, channels, samples)
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

        return (audio,)
