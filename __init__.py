from .nodes import AcestepCPPModelLoader, AcestepCPPModelDownloader, AcestepCPPGenerate

NODE_CLASS_MAPPINGS = {
    "AcestepCPPModelLoader": AcestepCPPModelLoader,
    "AcestepCPPModelDownloader": AcestepCPPModelDownloader,
    "AcestepCPPGenerate": AcestepCPPGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AcestepCPPModelLoader": "Acestep.cpp Model Loader",
    "AcestepCPPModelDownloader": "Acestep.cpp Model Downloader",
    "AcestepCPPGenerate": "Acestep.cpp Generate",
}
