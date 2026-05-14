"""Mesh + point cloud export for MoGe-2.

Lifts the mesh-from-point-map logic from upstream `moge/scripts/infer.py` lines 124-156
and writes .glb / .ply files into ComfyUI's output directory.

Inputs come straight from the inference node:
- `points_raw`     (B, H, W, 3) camera-space, OpenCV (x right, y down, z forward)
- `valid_mask`     (B, H, W)
- `normal`         (B, H, W, 3) — colorized; we use the *raw* normals from the same
                                  inference pass if available, otherwise None. We
                                  reconstruct from depth in trimesh if needed.
- `images`         (B, H, W, 3) source RGB images for textures / vertex colors

We export the *first* image in the batch (ComfyUI mesh export idiom — multi-frame
mesh export bloats outputs and Sharp / SAM3D pipelines all chain per-frame).
"""

import os
from pathlib import Path

import numpy as np
import torch

import folder_paths
from comfy_api.latest import io

from .utils import logger


class MoGe2SaveMesh(io.ComfyNode):
    """Build a textured GLB and/or vertex-colored PLY from MoGe-2 output."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoGe2SaveMesh",
            display_name="MoGe-2 Save Mesh (GLB/PLY)",
            category="MoGe2",
            is_output_node=True,
            description=(
                "Build a triangle mesh from MoGe-2's point map and write it to ComfyUI/output/. "
                "Writes both .glb (textured) and .ply (vertex-colored) by default. "
                "Coordinates are converted from OpenCV (camera-space, y-down, z-forward) to "
                "OpenGL world conventions (y-up, z-backward) on export."
            ),
            inputs=[
                io.Image.Input("points_raw", tooltip="MoGe-2 'points_raw' output (B,H,W,3)."),
                io.Image.Input("image", tooltip="Original RGB image used to texture the mesh."),
                io.Mask.Input("valid_mask"),
                io.String.Input(
                    "filename_prefix", default="moge2_mesh",
                    tooltip="Base filename. Written under ComfyUI/output/.",
                ),
                io.Boolean.Input("save_glb", default=True, optional=True),
                io.Boolean.Input("save_ply", default=True, optional=True),
                io.Float.Input(
                    "edge_threshold", default=0.04, min=0.0, max=1.0, step=0.001,
                    tooltip="Relative depth threshold for cleaning up depth-discontinuity edges. "
                            "Smaller = more aggressive trimming. 0 disables.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="glb_path"),
                io.String.Output(display_name="ply_path"),
            ],
        )

    @classmethod
    def execute(cls, points_raw, image, valid_mask,
                filename_prefix="moge2_mesh",
                save_glb=True, save_ply=True, edge_threshold=0.04):
        import utils3d
        from .moge_pkg.utils.io import save_glb as _save_glb, save_ply as _save_ply

        # Take first frame.
        points = points_raw[0].numpy()                  # (H, W, 3)
        rgb = image[0].clamp(0, 1).numpy()              # (H, W, 3), [0, 1]
        mask = (valid_mask[0] > 0.5).numpy()            # (H, W) bool
        H, W = points.shape[:2]
        depth = points[..., 2]

        # Edge cleanup — drop pixels on steep depth discontinuities.
        if edge_threshold > 0:
            edges = utils3d.np.depth_map_edge(depth, rtol=edge_threshold)
            mask = mask & ~edges

        if not mask.any():
            raise RuntimeError("No valid pixels remain after masking — try a smaller edge_threshold.")

        # Build mesh from the camera-space point map.
        uv = utils3d.np.uv_map(H, W)
        mesh_args = (points, rgb.astype(np.float32), uv)
        try:
            faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                *mesh_args, mask=mask, tri=True,
            )
            vertex_normals = None
        except Exception:
            # Older / newer utils3d versions accept a normal-map argument; we don't have raw normals here,
            # so let trimesh recompute them on save if needed.
            raise

        # Convert OpenCV camera space -> OpenGL world space (y up, z backward, flipped V).
        vertices = vertices * np.array([1, -1, -1], dtype=vertices.dtype)
        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=vertex_uvs.dtype) + np.array([0, 1], dtype=vertex_uvs.dtype)

        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(parents=True, exist_ok=True)

        glb_path_out = ""
        ply_path_out = ""

        if save_glb:
            glb_path = output_dir / f"{filename_prefix}.glb"
            texture = (rgb * 255).clip(0, 255).astype(np.uint8)
            _save_glb(glb_path, vertices, faces, vertex_uvs, texture, vertex_normals)
            glb_path_out = str(glb_path)
            logger.info(f"Saved GLB: {glb_path}")

        if save_ply:
            ply_path = output_dir / f"{filename_prefix}.ply"
            _save_ply(ply_path, vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)
            ply_path_out = str(ply_path)
            logger.info(f"Saved PLY: {ply_path}")

        return io.NodeOutput(glb_path_out, ply_path_out)


NODE_CLASS_MAPPINGS = {
    "MoGe2SaveMesh": MoGe2SaveMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoGe2SaveMesh": "MoGe-2 Save Mesh (GLB/PLY)",
}
