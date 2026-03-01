"""Unit tests for nodes.py — covers node helpers, INPUT_TYPES, and validation
logic.  All ComfyUI / torch imports are stubbed via tests/conftest.py so the
suite runs in any plain Python environment."""

import os
import sys

import pytest

import nodes  # pre-loaded into sys.modules by tests/conftest.py


# ===========================================================================
# _coerce_float
# ===========================================================================

class TestCoerceFloat:
    def test_float_passthrough(self):
        assert nodes._coerce_float(0.9, 0.5) == pytest.approx(0.9)

    def test_int_converts(self):
        assert nodes._coerce_float(1, 0.0) == pytest.approx(1.0)

    def test_valid_string_converts(self):
        assert nodes._coerce_float("0.75", 0.0) == pytest.approx(0.75)

    def test_empty_string_uses_default(self):
        assert nodes._coerce_float("", 0.9) == pytest.approx(0.9)

    def test_whitespace_string_uses_default(self):
        assert nodes._coerce_float("   ", 1.0) == pytest.approx(1.0)

    def test_different_defaults(self):
        assert nodes._coerce_float("", 0.0) == pytest.approx(0.0)
        assert nodes._coerce_float("", 2.5) == pytest.approx(2.5)


# ===========================================================================
# scan_gguf_models / get_merged_model_folders / find_model_path
# ===========================================================================

class TestScanGgufModels:
    def test_returns_list(self):
        result = nodes.scan_gguf_models()
        assert isinstance(result, list)

    def test_finds_gguf_files_via_folder_paths(self, tmp_path, monkeypatch):
        (tmp_path / "model-a.gguf").write_text("mock")
        (tmp_path / "model-b.gguf").write_text("mock")
        (tmp_path / "readme.txt").write_text("not a model")

        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [str(tmp_path)])

        result = nodes.scan_gguf_models()
        assert "model-a.gguf" in result
        assert "model-b.gguf" in result
        assert "readme.txt" not in result

    def test_result_is_sorted(self, tmp_path, monkeypatch):
        for name in ("z-model.gguf", "a-model.gguf", "m-model.gguf"):
            (tmp_path / name).write_text("mock")

        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [str(tmp_path)])

        result = nodes.scan_gguf_models()
        assert result == sorted(result)

    def test_falls_back_to_manual_scan(self, tmp_path, monkeypatch):
        """When folder_paths returns no files the manual scan path is used."""
        (tmp_path / "fallback.gguf").write_text("mock")

        # Ensure the registered list is empty (no acestep_gguf entry)
        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [])

        # Point manual scan at tmp_path via get_merged_model_folders
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [str(tmp_path)])

        result = nodes.scan_gguf_models()
        assert "fallback.gguf" in result

    def test_no_models_returns_empty_list(self, monkeypatch):
        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [])
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [])

        result = nodes.scan_gguf_models()
        assert result == []


class TestFindModelPath:
    def test_finds_existing_model(self, tmp_path, monkeypatch):
        model = tmp_path / "test-model.gguf"
        model.write_text("mock")
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [str(tmp_path)])

        result = nodes.find_model_path("test-model.gguf")
        assert result == str(model)

    def test_returns_none_for_missing_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [str(tmp_path)])

        assert nodes.find_model_path("nonexistent.gguf") is None

    def test_searches_multiple_folders(self, tmp_path, monkeypatch):
        folder_a = tmp_path / "a"
        folder_b = tmp_path / "b"
        folder_a.mkdir()
        folder_b.mkdir()
        (folder_b / "model.gguf").write_text("mock")

        monkeypatch.setattr(
            nodes, "get_merged_model_folders",
            lambda: [str(folder_a), str(folder_b)],
        )

        result = nodes.find_model_path("model.gguf")
        assert result == str(folder_b / "model.gguf")


# ===========================================================================
# AcestepCPPModelLoader
# ===========================================================================

class TestAcestepCPPModelLoaderInputTypes:
    def test_has_required_section(self):
        result = nodes.AcestepCPPModelLoader.INPUT_TYPES()
        assert "required" in result

    def test_has_four_model_fields(self):
        req = nodes.AcestepCPPModelLoader.INPUT_TYPES()["required"]
        assert set(req) == {"lm_model", "text_encoder_model", "dit_model", "vae_model"}

    def test_placeholder_when_no_models(self, monkeypatch):
        monkeypatch.setattr(nodes, "scan_gguf_models", lambda: [])
        req = nodes.AcestepCPPModelLoader.INPUT_TYPES()["required"]
        assert req["lm_model"][0] == ["No GGUF models found"]
        assert req["dit_model"][0] == ["No GGUF models found"]

    def test_model_names_appear_in_dropdown(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "scan_gguf_models",
            lambda: ["acestep-v15-turbo-Q8_0.gguf", "vae-BF16.gguf"],
        )
        req = nodes.AcestepCPPModelLoader.INPUT_TYPES()["required"]
        assert "acestep-v15-turbo-Q8_0.gguf" in req["dit_model"][0]
        assert "vae-BF16.gguf" in req["vae_model"][0]

    def test_return_types(self):
        assert nodes.AcestepCPPModelLoader.RETURN_TYPES == ("ACESTEP_MODELS",)

    def test_category(self):
        assert nodes.AcestepCPPModelLoader.CATEGORY == "AcestepCPP"


# ===========================================================================
# AcestepCPPLoraLoader
# ===========================================================================

class TestAcestepCPPLoraLoader:
    @pytest.fixture
    def loader(self):
        return nodes.AcestepCPPLoraLoader()

    def test_empty_path_raises_value_error(self, loader):
        with pytest.raises(ValueError, match="lora_path is empty"):
            loader.load_lora("", 1.0)

    def test_whitespace_path_raises_value_error(self, loader):
        with pytest.raises(ValueError, match="lora_path is empty"):
            loader.load_lora("   ", 1.0)

    def test_unsupported_extension_raises(self, loader, tmp_path):
        f = tmp_path / "lora.bin"
        f.write_text("mock")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_lora(str(f), 1.0)

    def test_missing_file_raises_file_not_found(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load_lora(str(tmp_path / "missing.gguf"), 1.0)

    def test_valid_gguf_returns_correct_dict(self, loader, tmp_path):
        f = tmp_path / "lora.gguf"
        f.write_text("mock")
        result = loader.load_lora(str(f), 0.8)
        assert result == ({"path": str(f), "scale": 0.8},)

    def test_valid_safetensors_accepted(self, loader, tmp_path):
        f = tmp_path / "lora.safetensors"
        f.write_text("mock")
        result = loader.load_lora(str(f), 1.0)
        assert result[0]["path"] == str(f)

    def test_return_types(self):
        assert nodes.AcestepCPPLoraLoader.RETURN_TYPES == ("ACESTEP_LORA",)


# ===========================================================================
# AcestepCPPGenerate — INPUT_TYPES and coercion
# ===========================================================================

class TestAcestepCPPGenerateInputTypes:
    def test_required_has_models_and_caption(self):
        req = nodes.AcestepCPPGenerate.INPUT_TYPES()["required"]
        assert "models" in req
        assert "caption" in req

    def test_optional_contains_float_fields(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "lm_top_p" in opt
        assert "audio_cover_strength" in opt

    def test_lm_top_p_default_in_range(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        spec = opt["lm_top_p"][1]
        assert 0.0 <= spec["default"] <= 1.0

    def test_audio_cover_strength_default_in_range(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        spec = opt["audio_cover_strength"][1]
        assert 0.0 <= spec["default"] <= 1.0

    def test_optional_connections_present(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "reference_audio_input" in opt
        assert "src_audio_input" in opt
        assert "lora" in opt

    def test_return_types(self):
        assert nodes.AcestepCPPGenerate.RETURN_TYPES == ("AUDIO",)

    def test_task_types_list(self):
        assert "text2music" in nodes.AcestepCPPGenerate.TASK_TYPES
        assert "cover" in nodes.AcestepCPPGenerate.TASK_TYPES


# ===========================================================================
# AcestepCPPModelDownloader
# ===========================================================================

class TestAcestepCPPModelDownloader:
    def test_input_types_has_required_fields(self):
        req = nodes.AcestepCPPModelDownloader.INPUT_TYPES()["required"]
        assert "save_dir" in req
        assert "lm_size" in req
        assert "quant" in req
        assert "dit_variant" in req

    def test_is_output_node(self):
        assert nodes.AcestepCPPModelDownloader.OUTPUT_NODE is True


# ===========================================================================
# AcestepCPPBuilder
# ===========================================================================

class TestAcestepCPPBuilder:
    def test_is_output_node(self):
        assert nodes.AcestepCPPBuilder.OUTPUT_NODE is True

    def test_backends_list(self):
        assert "auto" in nodes.AcestepCPPBuilder.BACKENDS
        assert "cpu" in nodes.AcestepCPPBuilder.BACKENDS

    def test_detect_backend_returns_string(self):
        backend = nodes.AcestepCPPBuilder._detect_backend()
        assert isinstance(backend, str)
        assert backend in nodes.AcestepCPPBuilder.BACKENDS

    def test_cmake_flags_cpu(self):
        assert nodes.AcestepCPPBuilder._cmake_flags("cpu") == []

    def test_cmake_flags_cuda(self):
        assert "-DGGML_CUDA=ON" in nodes.AcestepCPPBuilder._cmake_flags("cuda")

    def test_cmake_flags_blas(self):
        assert "-DGGML_BLAS=ON" in nodes.AcestepCPPBuilder._cmake_flags("blas")
