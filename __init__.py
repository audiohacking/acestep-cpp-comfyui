from .nodes import AcestepCPPModelLoader, AcestepCPPModelDownloader, AcestepCPPBuilder, AcestepCPPGenerate

NODE_CLASS_MAPPINGS = {
    "AcestepCPPModelLoader": AcestepCPPModelLoader,
    "AcestepCPPModelDownloader": AcestepCPPModelDownloader,
    "AcestepCPPBuilder": AcestepCPPBuilder,
    "AcestepCPPGenerate": AcestepCPPGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AcestepCPPModelLoader": "Acestep.cpp Model Loader",
    "AcestepCPPModelDownloader": "Acestep.cpp Model Downloader",
    "AcestepCPPBuilder": "Acestep.cpp Builder",
    "AcestepCPPGenerate": "Acestep.cpp Generate",
}
