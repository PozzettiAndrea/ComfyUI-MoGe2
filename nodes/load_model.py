"""MoGe-2 model loader node.

Builds a JSON-safe config in the loader; the actual nn.Module is constructed lazily
on first use in the inference node and wrapped in a ComfyUI ModelPatcher so VRAM
management routes through comfy.model_management.
"""

import os

import torch
import folder_paths
from comfy_api.latest import io

from .utils import MODEL_REPOS, V1_MODELS, logger, patch_sage_attention


def _mm():
    import comfy.model_management
    return comfy.model_management


_moge2_model_dir = os.path.join(folder_paths.models_dir, "moge2")
os.makedirs(_moge2_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("moge2", _moge2_model_dir)


def _comfy_tqdm():
    """tqdm subclass that drives a ComfyUI ProgressBar (so HF downloads show in the UI)."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None

    holder = {"pbar": None, "total": 0, "done": 0}

    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret

    return _T


def _get_moge_model_list():
    local = folder_paths.get_filename_list("moge2")
    known = list(MODEL_REPOS.keys())
    return list(dict.fromkeys(known + local))


_MOGE_MODEL_CACHE = {}


class MoGeWrapper(torch.nn.Module):
    """Thin nn.Module wrapper around MoGeModel.

    MoGeModel exposes ``device`` and ``dtype`` as read-only @property values, but
    ComfyUI's ModelPatcher assigns ``model.device = ...`` during partial loads.
    Wrapping moves the property behind ``self.moge.device`` (still readable as
    a getter) and lets ComfyUI freely set ``self.device`` on the wrapper.
    """

    def __init__(self, model):
        super().__init__()
        self.moge = model

    def forward(self, *args, **kwargs):
        return self.moge(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self.moge.infer(*args, **kwargs)


def _build_moge_model(model_path, version, dtype, attention):
    """Construct the MoGe model from a .pt checkpoint on disk."""
    if attention == "sage":
        patch_sage_attention()
    elif attention == "auto":
        # Best-effort: prefer sage when available, otherwise leave SDPA alone.
        patch_sage_attention()

    if version == "v1":
        from .moge_pkg.model.v1 import MoGeModel
    else:
        from .moge_pkg.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(model_path)
    model.to(dtype=dtype)
    model.eval()
    return MoGeWrapper(model)


def _get_or_build_moge_model(config):
    import comfy.model_patcher
    from .utils import check_model_capabilities

    key = (config["model_path"], config["dtype"], config.get("attention", "auto"))
    if key in _MOGE_MODEL_CACHE:
        return _MOGE_MODEL_CACHE[key]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["dtype"]]
    model = _build_moge_model(config["model_path"], config["version"], dtype, config.get("attention", "auto"))

    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=_mm().get_torch_device(),
        offload_device=_mm().unet_offload_device(),
    )
    # Capabilities are read off the underlying MoGeModel (the wrapper has no heads).
    inner = model.moge if hasattr(model, "moge") else model
    patcher.model_options["moge_capabilities"] = check_model_capabilities(inner)
    patcher.model_options["moge_dtype"] = dtype
    patcher.model_options["moge_version"] = config["version"]

    _MOGE_MODEL_CACHE[key] = patcher
    return patcher


def _resolve_dtype(precision):
    device = _mm().get_torch_device()
    if precision == "auto":
        if _mm().should_use_fp16(device):
            return torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


class DownloadAndLoadMoGe2Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DownloadAndLoadMoGe2Model",
            display_name="(Down)Load MoGe-2 Model",
            category="MoGe2",
            description=(
                "Load a MoGe-2 (or MoGe-1) model from ComfyUI/models/moge2/. "
                "Auto-downloads from HuggingFace if not found locally. "
                "Variants: vits/vitb/vitl (with optional normal head, metric scale). "
                "vitl-normal is the full-capability checkpoint (~1.2 GB)."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=_get_moge_model_list(),
                    default="moge-2-vitl-normal.pt",
                    tooltip="Auto-downloads from HuggingFace if missing.",
                ),
                io.Combo.Input(
                    "precision",
                    options=["auto", "fp16", "fp32", "bf16"],
                    default="auto",
                    optional=True,
                ),
                io.Combo.Input(
                    "attention",
                    options=["auto", "sdpa", "sage"],
                    default="auto",
                    optional=True,
                    tooltip=(
                        "auto: sage if available else SDPA. "
                        "sage: monkey-patch F.scaled_dot_product_attention with sageattention (20-40%% faster on Ampere/Ada)."
                    ),
                ),
            ],
            outputs=[
                io.Custom("MOGE2_MODEL").Output(display_name="moge_model"),
            ],
        )

    @classmethod
    def execute(cls, model, precision="auto", attention="auto"):
        # Resolve full path; download from HuggingFace if needed.
        model_path = folder_paths.get_full_path("moge2", model)
        if model_path is None:
            if model not in MODEL_REPOS:
                raise FileNotFoundError(
                    f"Model '{model}' not in ComfyUI/models/moge2/ and not a known HuggingFace repo."
                )
            from huggingface_hub import hf_hub_download

            target_path = os.path.join(_moge2_model_dir, model)
            logger.info(f"Auto-downloading {model} from {MODEL_REPOS[model]} ...")
            downloaded = hf_hub_download(
                repo_id=MODEL_REPOS[model],
                filename="model.pt",
                local_dir=_moge2_model_dir,
                tqdm_class=_comfy_tqdm(),
            )
            # HF saves as model.pt; rename to the user-friendly name.
            if downloaded != target_path and not os.path.exists(target_path):
                os.rename(downloaded, target_path)
            model_path = target_path

        dtype = _resolve_dtype(precision)
        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
        version = "v1" if model in V1_MODELS else "v2"

        config = {
            "model_path": str(model_path),
            "version": version,
            "dtype": dtype_str,
            "attention": attention,
        }
        return io.NodeOutput(config)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMoGe2Model": DownloadAndLoadMoGe2Model,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMoGe2Model": "(Down)Load MoGe-2 Model",
}
