try:
    import folder_paths as _fp

    # Register a dedicated model type for ACE-Step GGUF files.
    # This makes the files appear in ComfyUI's model manager, enables the
    # built-in download prompt, and lets get_filename_list() find them.
    for _p in _fp.get_folder_paths("text_encoders"):
        _fp.add_model_folder_path("acestep_gguf", _p)
except Exception:
    pass

# Run the same auto-build that ComfyUI Manager triggers via install.py so
# that manual git-clone users also get the binaries built automatically the
# first time ComfyUI loads this node.
try:
    import threading as _threading
    from .install import install as _install

    def _background_install():
        try:
            _install()
        except Exception as _exc:
            print(
                f"[acestep-cpp] Background build failed: {_exc}\n"
                "[acestep-cpp] Use the 'Acestep.cpp Builder' node or run "
                "install.py from the node directory for detailed output.",
                flush=True,
            )

    _threading.Thread(target=_background_install, daemon=True).start()
except Exception:
    pass

try:
    from .nodes import AcestepCPPModelLoader, AcestepCPPLoraLoader, AcestepCPPModelDownloader, AcestepCPPBuilder, AcestepCPPGenerate

    NODE_CLASS_MAPPINGS = {
        "AcestepCPPModelLoader": AcestepCPPModelLoader,
        "AcestepCPPLoraLoader": AcestepCPPLoraLoader,
        "AcestepCPPModelDownloader": AcestepCPPModelDownloader,
        "AcestepCPPBuilder": AcestepCPPBuilder,
        "AcestepCPPGenerate": AcestepCPPGenerate,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "AcestepCPPModelLoader": "Acestep.cpp Model Loader",
        "AcestepCPPLoraLoader": "Acestep.cpp LoRA Loader",
        "AcestepCPPModelDownloader": "Acestep.cpp Model Downloader",
        "AcestepCPPBuilder": "Acestep.cpp Builder",
        "AcestepCPPGenerate": "Acestep.cpp Generate",
    }

except ImportError:
    # Imported as a standalone module (e.g., by pytest during conftest
    # discovery) rather than as a ComfyUI package — skip node registration.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
