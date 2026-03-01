import os
import json
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
    """Scan all model folders for .gguf files, deduplicated and sorted."""
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
    ):
        import torchaudio

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

        with tempfile.TemporaryDirectory() as tmpdir:
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
