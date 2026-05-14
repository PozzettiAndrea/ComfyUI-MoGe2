"""MoGe-2 inference node.

ComfyUI IMAGE layout is (B, H, W, 3) in [0, 1]; MoGe expects (B, 3, H, W).
We loop one image at a time because MoGe.infer() does focal/shift recovery per-sample
and the upstream API takes a single image tensor (3, H, W) or a batched (B, 3, H, W).
"""

import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar
from comfy_api.latest import io

from .utils import logger


def _mm():
    import comfy.model_management
    return comfy.model_management


def _colorize_depth_tensor(depth_np, valid_mask_np):
    """Colorize a (H, W) numpy depth array to a (H, W, 3) RGB float32 image in [0, 1]."""
    from .moge_pkg.utils.vis import colorize_depth
    rgb = colorize_depth(depth_np, mask=valid_mask_np)
    return torch.from_numpy(rgb).float()


def _colorize_normal_tensor(normal_np):
    from .moge_pkg.utils.vis import colorize_normal
    rgb = colorize_normal(normal_np)
    return torch.from_numpy(rgb).float()


def _to_image_batch(tensor_list):
    """List of (H, W, 3) float tensors -> (B, H, W, 3) IMAGE batch."""
    return torch.stack(tensor_list, dim=0).cpu().float().clamp(0, 1)


def _to_mask_batch(tensor_list):
    """List of (H, W) bool/float tensors -> (B, H, W) MASK batch in [0, 1]."""
    return torch.stack([m.float() for m in tensor_list], dim=0).cpu()


class MoGe2Inference(io.ComfyNode):
    """Single inference node — all outputs in one pass. Connect what you need."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoGe2Inference",
            display_name="MoGe-2 Geometry",
            category="MoGe2",
            description=(
                "MoGe-2 monocular geometry estimation.\n\n"
                "Outputs depth, point map, normal map, valid mask, and camera intrinsics in a single forward pass. "
                "Set fov_x > 0 to constrain inference with a known horizontal FoV; leave at 0 to estimate it."
            ),
            inputs=[
                io.Custom("MOGE2_MODEL").Input("moge_model"),
                io.Image.Input("images"),
                io.Int.Input(
                    "resolution_level", default=9, min=0, max=9,
                    tooltip="0-9; higher = finer detail, slower. Ignored if num_tokens > 0.",
                    optional=True,
                ),
                io.Int.Input(
                    "num_tokens", default=0, min=0, max=4096,
                    tooltip="Override token count (suggested 1200-2500). 0 = use resolution_level.",
                    optional=True,
                ),
                io.Float.Input(
                    "fov_x_deg", default=0.0, min=0.0, max=170.0,
                    tooltip="Known horizontal FoV in degrees. 0 = estimate from image.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "force_projection", default=True,
                    tooltip="Recompute point map from depth+intrinsics for consistency.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "apply_mask", default=True,
                    tooltip="Mask out invalid (sky / off-surface) pixels in points/depth.",
                    optional=True,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="depth_visualization"),
                io.Image.Output(display_name="depth_raw"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="points_raw"),
                io.Mask.Output(display_name="valid_mask"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Float.Output(display_name="fov_x"),
                io.Float.Output(display_name="fov_y"),
            ],
        )

    @classmethod
    def execute(cls, moge_model, images,
                resolution_level=9, num_tokens=0, fov_x_deg=0.0,
                force_projection=True, apply_mask=True):
        from .load_model import _get_or_build_moge_model

        device = _mm().get_torch_device()
        patcher = _get_or_build_moge_model(moge_model)
        dtype = patcher.model_options["moge_dtype"]
        capabilities = patcher.model_options["moge_capabilities"]

        B, H, W, _ = images.shape
        memory_required = H * W * 3 * B * _mm().dtype_size(dtype) * 8  # generous estimate
        _mm().load_models_gpu([patcher], memory_required=memory_required)
        model = patcher.model

        # (B, H, W, 3) -> (B, 3, H, W), [0, 1]
        images_pt = images.permute(0, 3, 1, 2).clamp(0, 1)

        use_fp16 = dtype == torch.float16
        fov_x = float(fov_x_deg) if fov_x_deg > 0 else None
        num_tokens_arg = int(num_tokens) if num_tokens > 0 else None

        pbar = ProgressBar(B)
        depth_vis_out, normal_out, points_out = [], [], []
        mask_out, depth_raw_out = [], []
        intrinsics_list = []
        fov_x_list, fov_y_list = [], []

        import utils3d

        for i in range(B):
            _mm().throw_exception_if_processing_interrupted()
            img = images_pt[i].to(device=device, dtype=dtype)  # (3, H, W)

            with torch.inference_mode():
                output = model.infer(
                    img,
                    num_tokens=num_tokens_arg,
                    resolution_level=resolution_level,
                    fov_x=fov_x,
                    force_projection=force_projection,
                    apply_mask=apply_mask,
                    use_fp16=use_fp16,
                )

            depth = output.get("depth")           # (H, W)
            normal = output.get("normal")         # (H, W, 3) or None
            points = output.get("points")         # (H, W, 3) or None
            mask = output.get("mask")             # (H, W) bool or None
            intrinsics = output.get("intrinsics") # (3, 3) normalized

            if depth is None:
                raise RuntimeError("MoGe model did not return a depth map for this checkpoint.")

            depth_np = depth.float().cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None

            # Colorized depth for previewing in ComfyUI's image lane.
            depth_vis_out.append(_colorize_depth_tensor(depth_np, mask_np))

            # Normal: model returns (H, W, 3) in camera space.
            if normal is not None and capabilities["has_normal"]:
                normal_np = normal.float().cpu().numpy()
                normal_out.append(_colorize_normal_tensor(normal_np))
            else:
                normal_out.append(torch.zeros(H, W, 3))

            # Raw point map kept as (H, W, 3) — downstream geometry consumers can read it back.
            # NaN/inf cleanup: replace masked-out infs with 0 so the IMAGE lane stays well-defined.
            if points is not None:
                pts = points.float().cpu()
                pts = torch.where(torch.isfinite(pts), pts, torch.zeros_like(pts))
                points_out.append(pts)
            else:
                points_out.append(torch.zeros(H, W, 3))

            # Mask (replace inf-depth pixels too).
            valid = mask if mask is not None else torch.isfinite(depth)
            mask_out.append(valid.float().cpu())

            # Raw depth, NaN-cleaned, stored as a 1-channel image so workflows can sample it.
            depth_raw = depth.float().cpu()
            depth_raw = torch.where(torch.isfinite(depth_raw), depth_raw, torch.zeros_like(depth_raw))
            depth_raw_out.append(depth_raw)

            if intrinsics is not None:
                k = intrinsics.float().cpu()  # (3, 3) normalized
                k4 = torch.eye(4, dtype=k.dtype)
                k4[:3, :3] = k
                intrinsics_list.append(k4)
                fov_x_rad, fov_y_rad = utils3d.np.intrinsics_to_fov(k[:3, :3].numpy())
                fov_x_list.append(float(torch.rad2deg(torch.tensor(float(fov_x_rad)))))
                fov_y_list.append(float(torch.rad2deg(torch.tensor(float(fov_y_rad)))))
            else:
                intrinsics_list.append(torch.eye(4))
                fov_x_list.append(0.0)
                fov_y_list.append(0.0)

            pbar.update(1)

        depth_vis = _to_image_batch(depth_vis_out)
        normal_img = _to_image_batch(normal_out)
        points_img = torch.stack(points_out, dim=0).cpu().float()      # (B, H, W, 3), raw metric, not clamped
        valid_mask = _to_mask_batch(mask_out)
        # Raw metric depth, broadcast to a 3-channel IMAGE so it can flow on the IMAGE lane.
        depth_raw_batch = torch.stack(depth_raw_out, dim=0).cpu().float()             # (B, H, W)
        depth_raw_img = depth_raw_batch.unsqueeze(-1).repeat(1, 1, 1, 3)              # (B, H, W, 3)
        intrinsics_batch = torch.stack(intrinsics_list, dim=0).cpu().float()          # (B, 4, 4)

        # FOV outputs follow batch[0] (ComfyUI Float port is scalar).
        fov_x_out = fov_x_list[0]
        fov_y_out = fov_y_list[0]

        return io.NodeOutput(
            depth_vis,
            depth_raw_img,
            normal_img,
            points_img,
            valid_mask,
            intrinsics_batch,
            fov_x_out,
            fov_y_out,
        )


NODE_CLASS_MAPPINGS = {
    "MoGe2Inference": MoGe2Inference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoGe2Inference": "MoGe-2 Geometry",
}
