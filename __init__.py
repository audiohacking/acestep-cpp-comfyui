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
