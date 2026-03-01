"""Tests for workflow example JSON files.

Validates structure, model metadata, output nodes, and widget value types
without requiring ComfyUI to be installed.
"""

import glob
import json
import os

import pytest

WORKFLOW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workflow-examples",
)
WORKFLOW_FILES = sorted(glob.glob(os.path.join(WORKFLOW_DIR, "*.json")))

EXPECTED_MODEL_NAMES = {
    "acestep-5Hz-lm-4B-Q8_0.gguf",
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "acestep-v15-turbo-Q8_0.gguf",
    "vae-BF16.gguf",
}

HF_BASE = "https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _nodes_by_type(wf, node_type):
    return [n for n in wf["nodes"] if n["type"] == node_type]


# ---------------------------------------------------------------------------
# Parametrised: every workflow file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "workflow_path",
    WORKFLOW_FILES,
    ids=lambda p: os.path.basename(p),
)
class TestWorkflowStructure:

    # --- Basic structure ---------------------------------------------------

    def test_valid_json(self, workflow_path):
        wf = _load(workflow_path)
        assert isinstance(wf, dict)

    def test_has_nodes_list(self, workflow_path):
        wf = _load(workflow_path)
        assert isinstance(wf.get("nodes"), list)
        assert len(wf["nodes"]) >= 2

    def test_has_links_list(self, workflow_path):
        wf = _load(workflow_path)
        assert isinstance(wf.get("links"), list)

    def test_has_version(self, workflow_path):
        wf = _load(workflow_path)
        assert "version" in wf

    # --- Required custom nodes present ------------------------------------

    def test_has_model_loader_node(self, workflow_path):
        wf = _load(workflow_path)
        assert _nodes_by_type(wf, "AcestepCPPModelLoader"), \
            "Workflow must include AcestepCPPModelLoader"

    def test_has_generate_node(self, workflow_path):
        wf = _load(workflow_path)
        assert _nodes_by_type(wf, "AcestepCPPGenerate"), \
            "Workflow must include AcestepCPPGenerate"

    # --- Output node: save-then-play --------------------------------------

    def test_uses_save_audio_not_preview_audio(self, workflow_path):
        wf = _load(workflow_path)
        node_types = [n["type"] for n in wf["nodes"]]
        assert "PreviewAudio" not in node_types, \
            "PreviewAudio (play-only) should be replaced by SaveAudio (save-then-play)"
        assert "SaveAudio" in node_types, \
            "SaveAudio must be present so generated audio is persisted to disk"

    def test_save_audio_has_filename_prefix(self, workflow_path):
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "SaveAudio"):
            wv = node.get("widgets_values", [])
            assert len(wv) >= 1 and isinstance(wv[0], str) and wv[0], \
                "SaveAudio must have a non-empty filename_prefix widget value"

    # --- Model loader widget values ---------------------------------------

    def test_model_loader_has_four_widget_values(self, workflow_path):
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            assert len(node.get("widgets_values", [])) == 4, \
                "AcestepCPPModelLoader needs exactly 4 widget values"

    def test_model_loader_widget_values_are_known_models(self, workflow_path):
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for v in node.get("widgets_values", []):
                assert v in EXPECTED_MODEL_NAMES, \
                    f"Unexpected model name in ModelLoader: {v!r}"

    # --- Generate node widget value types --------------------------------

    def test_generate_lm_top_p_is_numeric(self, workflow_path):
        """Widget index 15 (lm_top_p) must be a number, never an empty string."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 15:
                assert isinstance(wv[15], (int, float)), \
                    f"lm_top_p (index 15) should be numeric, got {type(wv[15])}"

    def test_generate_audio_cover_strength_is_numeric(self, workflow_path):
        """Widget index 19 (audio_cover_strength) must be a number."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 19:
                assert isinstance(wv[19], (int, float)), \
                    f"audio_cover_strength (index 19) should be numeric, got {type(wv[19])}"

    def test_generate_task_type_is_valid(self, workflow_path):
        valid = {"text2music", "cover", "repaint"}
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 2:
                assert wv[2] in valid, \
                    f"task_type (index 2) must be one of {valid}, got {wv[2]!r}"

    # --- extra.models download metadata ----------------------------------

    def test_extra_models_present(self, workflow_path):
        wf = _load(workflow_path)
        models = wf.get("extra", {}).get("models", [])
        assert len(models) == 4, \
            f"Expected 4 entries in extra.models, got {len(models)}"

    def test_extra_models_names_match_expected(self, workflow_path):
        wf = _load(workflow_path)
        names = {m["name"] for m in wf.get("extra", {}).get("models", [])}
        assert names == EXPECTED_MODEL_NAMES

    def test_extra_models_have_huggingface_urls(self, workflow_path):
        wf = _load(workflow_path)
        for m in wf.get("extra", {}).get("models", []):
            assert m.get("url", "").startswith(HF_BASE), \
                f"{m.get('name')} has unexpected URL: {m.get('url')}"

    def test_extra_models_save_path_in_text_encoders(self, workflow_path):
        wf = _load(workflow_path)
        for m in wf.get("extra", {}).get("models", []):
            assert m.get("save_path", "").startswith("text_encoders/"), \
                f"{m.get('name')} save_path should be under text_encoders/"

    def test_extra_models_save_path_matches_name(self, workflow_path):
        wf = _load(workflow_path)
        for m in wf.get("extra", {}).get("models", []):
            expected = f"text_encoders/{m['name']}"
            assert m.get("save_path") == expected, \
                f"save_path mismatch for {m['name']}: {m.get('save_path')}"

    # --- Node properties models (ComfyUI auto-download) ------------------

    def test_model_loader_properties_has_models(self, workflow_path):
        """AcestepCPPModelLoader.properties must include a 'models' list."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            models = node.get("properties", {}).get("models")
            assert isinstance(models, list) and len(models) == 4, \
                "AcestepCPPModelLoader properties must have exactly 4 model entries"

    def test_model_loader_properties_models_names(self, workflow_path):
        """Models in node properties must match expected GGUF filenames."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            names = {m["name"] for m in node["properties"].get("models", [])}
            assert names == EXPECTED_MODEL_NAMES

    def test_model_loader_properties_models_urls(self, workflow_path):
        """Models in node properties must have HuggingFace download URLs."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for m in node["properties"].get("models", []):
                assert m.get("url", "").startswith(HF_BASE), \
                    f"{m.get('name')} has unexpected URL: {m.get('url')}"

    def test_model_loader_properties_models_directory(self, workflow_path):
        """Models in node properties must specify 'text_encoders' directory."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for m in node["properties"].get("models", []):
                assert m.get("directory") == "text_encoders", \
                    f"{m.get('name')} directory should be 'text_encoders', got {m.get('directory')!r}"

    def test_model_loader_properties_models_url_matches_name(self, workflow_path):
        """Each model's URL must end with its filename."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for m in node["properties"].get("models", []):
                assert m.get("url", "").endswith(m["name"]), \
                    f"URL does not end with filename for {m['name']}: {m.get('url')}"

    # --- Graph connectivity -----------------------------------------------

    def test_model_loader_output_connected(self, workflow_path):
        """The ACESTEP_MODELS output of ModelLoader must be connected."""
        wf = _load(workflow_path)
        link_src_nodes = {lnk[1] for lnk in wf.get("links", [])}
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            assert node["id"] in link_src_nodes, \
                "AcestepCPPModelLoader output is not connected to any node"

    def test_generate_output_connected_to_save_audio(self, workflow_path):
        """AcestepCPPGenerate audio output must feed into SaveAudio."""
        wf = _load(workflow_path)
        node_ids = {n["id"]: n for n in wf["nodes"]}
        save_audio_ids = {n["id"] for n in _nodes_by_type(wf, "SaveAudio")}
        # Walk links: find any link whose destination is a SaveAudio node
        connected = any(lnk[3] in save_audio_ids for lnk in wf.get("links", []))
        assert connected, "No link from any node to SaveAudio found"
