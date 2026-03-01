# acestep-cpp-comfyui

ComfyUI custom nodes that wrap [acestep.cpp](https://github.com/audiohacking/acestep.cpp) — a portable C++17 implementation of ACE-Step 1.5 music generation using GGML. Text + lyrics in, stereo 48 kHz WAV out. Runs on CPU, CUDA, Metal, and Vulkan.

## Features

- **Download** the required GGUF models directly from HuggingFace without leaving ComfyUI
- Load the four GGUF model files required by acestep.cpp (LM, text encoder, DiT, VAE)
- Generate music from a caption and optional lyrics/metadata
- Full control over generation parameters (turbo and SFT presets)
- Cover mode, reference-audio timbre transfer, and LoRA adapter support
- Returns a ComfyUI **AUDIO** tensor, compatible with any audio preview or save node

## Prerequisites

### 1 – Build acestep.cpp

```bash
git clone https://github.com/audiohacking/acestep.cpp
cd acestep.cpp
git submodule update --init
mkdir build && cd build

# Linux (NVIDIA GPU)
cmake .. -DGGML_CUDA=ON
# macOS (Metal auto-enabled)
cmake ..
# CPU with OpenBLAS
cmake .. -DGGML_BLAS=ON

cmake --build . --config Release -j$(nproc)
```

This produces two binaries: `ace-qwen3` (LM) and `dit-vae` (DiT + VAE).

### 2 – Download GGUF models

**Option A – Acestep.cpp Model Downloader node (recommended)**

After installing this custom node package, use the **Acestep.cpp Model Downloader** node inside ComfyUI. It downloads the required GGUFs from [`Serveurperso/ACE-Step-1.5-GGUF`](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF) straight into your model folder. `huggingface_hub` must be available:

```bash
pip install huggingface_hub
```

**Option B – Command line**

```bash
pip install huggingface_hub[cli]   # installs the 'hf' CLI tool
./models.sh          # Q8_0 turbo essentials (~7.7 GB)
```

Pre-quantized GGUFs are available on [Hugging Face](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF).

Default models (placed in `models/`):

| GGUF | Role | Size |
|------|------|------|
| `acestep-5Hz-lm-4B-Q8_0.gguf` | LM (ace-qwen3) | 4.2 GB |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | Text encoder | 748 MB |
| `acestep-v15-turbo-Q8_0.gguf` | DiT | 2.4 GB |
| `vae-BF16.gguf` | VAE | 322 MB |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/audiohacking/acestep-cpp-comfyui
```

Restart ComfyUI.

## Configuration

Copy `config.example.json` to `config.json` in the node directory and edit it:

```json
{
  "model_folders": [
    "/path/to/acestep.cpp/models"
  ],
  "binary_paths": {
    "ace-qwen3": "/path/to/acestep.cpp/build/ace-qwen3",
    "dit-vae": "/path/to/acestep.cpp/build/dit-vae"
  }
}
```

**`model_folders`** – list of directories to scan for `.gguf` files. These are merged with ComfyUI's built-in `text_encoders` folder. Non-existent paths are silently ignored.

**`binary_paths`** – explicit paths to the `ace-qwen3` and `dit-vae` binaries. If omitted, the node also searches your system `PATH` and `<node_dir>/acestep.cpp/build/`.

`config.json` is optional; if both binaries are on `PATH` and the GGUF files are in a standard ComfyUI model folder, no configuration file is needed.

## Node Reference

### Acestep.cpp Model Downloader

Downloads the required ACE-Step GGUF files from [`Serveurperso/ACE-Step-1.5-GGUF`](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF) on HuggingFace into a local directory. Quant availability per model type mirrors the logic in `models.sh`.

**Inputs (required)**

| Name | Default | Description |
|------|---------|-------------|
| `save_dir` | ComfyUI `text_encoders` folder | Directory to save downloaded GGUF files |
| `lm_size` | `4B` | LM model size: `4B`, `1.7B`, or `0.6B` |
| `quant` | `Q8_0` | Quantisation level (falls back to nearest valid quant for each model type) |
| `dit_variant` | `turbo` | DiT variant: `turbo`, `sft`, `base`, `turbo-shift1`, `turbo-shift3`, `turbo-continuous` |

**Inputs (optional)**

| Name | Default | Description |
|------|---------|-------------|
| `hf_token` | *(empty)* | HuggingFace access token (not needed for public repos) |
| `overwrite` | `false` | Re-download even if the file already exists |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `downloaded_files` | `STRING` | Summary of files downloaded / skipped |

> **Tip**: Run this node once to populate your model folder, then bypass/disable it and connect the **Model Loader** node to the same `save_dir`.

---

### Acestep.cpp Model Loader

Selects the four GGUF model files and validates that they exist on disk.

**Inputs (required)**

| Name | Description |
|------|-------------|
| `lm_model` | LM GGUF (e.g. `acestep-5Hz-lm-4B-Q8_0.gguf`) |
| `text_encoder_model` | Text-encoder GGUF (e.g. `Qwen3-Embedding-0.6B-Q8_0.gguf`) |
| `dit_model` | DiT GGUF (e.g. `acestep-v15-turbo-Q8_0.gguf`) |
| `vae_model` | VAE GGUF (e.g. `vae-BF16.gguf`) |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `models` | `ACESTEP_MODELS` | Model path bundle passed to the generator |

---

### Acestep.cpp Generate

Runs `ace-qwen3` (LM) then `dit-vae` (DiT + VAE) and returns the generated audio.

**Inputs (required)**

| Name | Description |
|------|-------------|
| `models` | Output of the Model Loader |
| `caption` | Music style/description (required by the LM) |

**Inputs (optional)**

| Name | Default | Description |
|------|---------|-------------|
| `lyrics` | *(empty)* | Song lyrics; leave empty for the LM to generate |
| `task_type` | `text2music` | Generation mode: `text2music`, `cover`, or `repaint` |
| `instrumental` | `false` | Generate an instrumental track (no vocals) |
| `vocal_language` | `unknown` | Language of the vocals (`en`, `fr`, `zh`, …) |
| `duration` | `-1` | Duration in seconds; `-1` lets the LM decide |
| `bpm` | `0` | Beats per minute; `0` lets the LM decide |
| `keyscale` | *(empty)* | Key and scale, e.g. `C major`; leave empty for the LM to decide |
| `timesignature` | *(empty)* | Time signature, e.g. `4/4`; leave empty for the LM to decide |
| `inference_steps` | `8` | DiT diffusion steps (8 = turbo preset, 50 = SFT preset) |
| `guidance_scale` | `7.0` | CFG scale (ignored by turbo models) |
| `shift` | `3.0` | Flow-matching shift (3.0 = turbo, 6.0 = SFT) |
| `seed` | `-1` | Random seed; `-1` picks one at random |
| `lm_temperature` | `0.85` | LM sampling temperature |
| `lm_cfg_scale` | `2.0` | LM classifier-free guidance scale |
| `lm_top_p` | `0.9` | LM nucleus sampling probability |
| `lm_negative_prompt` | *(empty)* | Negative prompt for the LM |
| `reference_audio` | *(empty)* | Path to a WAV/MP3 for timbre transfer |
| `src_audio` | *(empty)* | Path to a WAV/MP3 source for cover mode |
| `audio_cover_strength` | `1.0` | Cover influence strength (0 = silence, 1 = full) |
| `lora_path` | *(empty)* | Path to a DiT LoRA adapter file |
| `lora_scale` | `1.0` | LoRA adapter scale |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `audio` | `AUDIO` | Generated stereo 48 kHz audio |

## Quick Start Presets

**Turbo (fast, 8 steps)**
```
inference_steps = 8
shift           = 3.0
```

**SFT (higher quality, 50 steps)**
```
inference_steps = 50
guidance_scale  = 4.0
shift           = 6.0
```

## License

MIT — see [LICENSE](LICENSE).
