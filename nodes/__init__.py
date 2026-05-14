"""ComfyUI-MoGe2 — node registry."""

from .load_model import DownloadAndLoadMoGe2Model
from .nodes_inference import MoGe2Inference
from .nodes_mesh import MoGe2SaveMesh


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMoGe2Model": DownloadAndLoadMoGe2Model,
    "MoGe2Inference": MoGe2Inference,
    "MoGe2SaveMesh": MoGe2SaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMoGe2Model": "(Down)Load MoGe-2 Model",
    "MoGe2Inference": "MoGe-2 Geometry",
    "MoGe2SaveMesh": "MoGe-2 Save Mesh (GLB/PLY)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
