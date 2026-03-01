from .nodes import AcestepCPPModelLoader, AcestepCPPGenerate

NODE_CLASS_MAPPINGS = {
    "AcestepCPPModelLoader": AcestepCPPModelLoader,
    "AcestepCPPGenerate": AcestepCPPGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AcestepCPPModelLoader": "Acestep.cpp Model Loader",
    "AcestepCPPGenerate": "Acestep.cpp Generate",
}
