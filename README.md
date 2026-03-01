# acestep-cpp-comfyui

ComfyUI custom nodes that wrap [acestep.cpp](https://github.com/audiohacking/acestep.cpp) — a portable C++17 implementation of ACE-Step 1.5 music generation using GGML. Text + lyrics in, stereo 48 kHz WAV out. Runs on CPU, CUDA, Metal, and Vulkan.

## Features

- **Build** the `ace-qwen3` and `dit-vae` binaries from source via the **Acestep.cpp Builder** node (no terminal required)
- **Download** the required GGUF models directly from HuggingFace without leaving ComfyUI
- Load the four GGUF model files required by acestep.cpp (LM, text encoder, DiT, VAE)
- Load LoRA adapters from a dedicated **Acestep.cpp LoRA Loader** node (scans `loras/` subdirectories)
- Generate music from a caption and optional lyrics/metadata
- Full control over generation parameters (turbo and SFT presets)
- Cover mode, reference-audio timbre transfer, and LoRA adapter support
- Connect **AUDIO tensors** from any `LoadAudio` node directly to the generator for reference/source audio
- Returns a ComfyUI **AUDIO** tensor, compatible with any audio preview or save node
- Ready-to-use **example workflows** in `workflow-examples/`

## Prerequisites

`git` and `cmake` must be on your system `PATH` before using the Builder node (or the manual build below). Everything else is handled inside ComfyUI.

```bash
# Debian/Ubuntu
apt install git cmake build-essential

# macOS (Homebrew)
brew install cmake
```

### 1 – Build acestep.cpp

**Option A – Acestep.cpp Builder node (recommended)**

After installing this custom node package, drop the **Acestep.cpp Builder** node onto your canvas and click *Queue*. It will:

1. Clone `https://github.com/audiohacking/acestep.cpp` into `<node_dir>/acestep.cpp`
2. Run `git submodule update --init --recursive`
3. Configure with CMake (auto-detecting CUDA → Metal → CPU)
4. Build `ace-qwen3` and `dit-vae` using all available CPU cores

The binaries land in `<node_dir>/acestep.cpp/build/`, which is where the **Generate** node looks first — no extra config needed.

**Option B – Command line**

Clone `acestep.cpp` **inside the node directory** so the Generate node finds the binaries automatically — no configuration required:

```bash
cd ComfyUI/custom_nodes/acestep-cpp-comfyui
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

This produces `ace-qwen3` and `dit-vae` in `<node_dir>/acestep.cpp/build/`, which is where the Generate node already looks — no extra config needed.

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

Restart ComfyUI. On startup the node will attempt to build the `ace-qwen3` and `dit-vae` binaries automatically if `git` and `cmake` are available. If the automatic build does not complete, use the **Acestep.cpp Builder** node inside ComfyUI — no manual file editing required.

## Advanced Configuration

`config.json` is **optional** and only needed if you store binaries or models in non-standard locations.

Copy `config.example.json` to `config.json` in the node directory and set only the keys you need:

```json
{
  "model_folders": [
    "/custom/path/to/models"
  ],
  "binary_paths": {
    "ace-qwen3": "/custom/path/to/build/ace-qwen3",
    "dit-vae": "/custom/path/to/build/dit-vae"
  }
}
```

**`model_folders`** – additional directories to scan for `.gguf` files, merged with ComfyUI's built-in `text_encoders` folder.

**`binary_paths`** – override the automatic binary search. The node already looks in your system `PATH` and `<node_dir>/acestep.cpp/build/`, so this is only needed for custom build locations.

## Example Workflows

Ready-to-use workflow JSON files are in the `workflow-examples/` directory. Drag one onto the ComfyUI canvas or load it via *Load workflow*.

| File | Description |
|------|-------------|
| `acestep-cpp-text2music.json` | Basic text-to-music generation |
| `acestep-cpp-lora.json` | Text-to-music with a LoRA adapter |
| `acestep-cpp-reference-audio.json` | Timbre transfer using a reference audio file |
| `acestep-cpp-cover.json` | Cover/remix mode using a source audio file |

> **Prerequisites**: download the GGUF models (use the **Model Downloader** node) and build the binaries (use the **Builder** node) before running a generation workflow.

## Node Reference

### Acestep.cpp Builder

Clones [`audiohacking/acestep.cpp`](https://github.com/audiohacking/acestep.cpp) from GitHub and builds the `ace-qwen3` and `dit-vae` binaries using CMake. Requires `git` and `cmake` on the system PATH.

**Inputs (required)**

| Name | Default | Description |
|------|---------|-------------|
| `clone_dir` | `<node_dir>/acestep.cpp` | Directory to clone the repo into |
| `backend` | `auto` | CMake backend: `auto` (detects CUDA → Metal → CPU), `cuda`, `metal`, `blas`, `cpu` |

**Inputs (optional)**

| Name | Default | Description |
|------|---------|-------------|
| `force_rebuild` | `false` | Remove the existing `build/` directory and rebuild from scratch |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `build_log` | `STRING` | Full cmake configure + build output |

> **Tip**: Run this node once to compile the binaries. The default `clone_dir` places them where the **Generate** node already searches, so no further configuration is needed.

---

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

### Acestep.cpp LoRA Loader

Specify a LoRA adapter file and scale, ready to connect to the **Generate** node.
Enter the full path to any `.gguf` or `.safetensors` LoRA file anywhere on your filesystem.

**Inputs (required)**

| Name | Description |
|------|-------------|
| `lora_path` | Full filesystem path to the LoRA adapter file (`.gguf` or `.safetensors`) |
| `lora_scale` | Adapter scale (default `1.0`) |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `lora` | `ACESTEP_LORA` | LoRA bundle passed to the generator |

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
| `reference_audio` | *(empty)* | Path to a WAV/MP3 for timbre transfer (use `reference_audio_input` instead when possible) |
| `src_audio` | *(empty)* | Path to a WAV/MP3 source for cover mode (use `src_audio_input` instead when possible) |
| `audio_cover_strength` | `1.0` | Cover influence strength (0 = silence, 1 = full) |
| `lora_path` | *(empty)* | Path to a DiT LoRA adapter file (use the LoRA Loader node instead when possible) |
| `lora_scale` | `1.0` | LoRA adapter scale |
| `reference_audio_input` | *(not connected)* | **AUDIO** tensor for timbre transfer — connect from a `Load Audio` node; overrides `reference_audio` |
| `src_audio_input` | *(not connected)* | **AUDIO** tensor for cover/repaint — connect from a `Load Audio` node; overrides `src_audio` |
| `lora` | *(not connected)* | **ACESTEP_LORA** from the LoRA Loader node; overrides `lora_path` / `lora_scale` |

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
