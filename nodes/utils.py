"""Shared utilities for ComfyUI-MoGe2 nodes."""

import logging

logger = logging.getLogger("MoGe2")

# MoGe-2 HuggingFace model repos
MODEL_REPOS = {
    "moge-2-vits-normal.pt": "Ruicheng/moge-2-vits-normal",
    "moge-2-vitb-normal.pt": "Ruicheng/moge-2-vitb-normal",
    "moge-2-vitl.pt": "Ruicheng/moge-2-vitl",
    "moge-2-vitl-normal.pt": "Ruicheng/moge-2-vitl-normal",
    "moge-vitl.pt": "Ruicheng/moge-vitl",   # MoGe v1
}

V1_MODELS = {"moge-vitl.pt"}


def check_model_capabilities(model):
    """Return which output heads are populated on this checkpoint."""
    return {
        "has_normal": hasattr(model, "normal_head"),
        "has_metric_scale": hasattr(model, "scale_head"),
        "has_mask": hasattr(model, "mask_head"),
    }


def patch_sage_attention():
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention with sageattention.

    MoGe-2's DINOv2 backbone calls F.scaled_dot_product_attention directly, so this is a
    zero-code-change speedup. Falls back silently if sageattention is not installed.
    """
    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("sageattention not installed — falling back to PyTorch SDPA.")
        return False

    import torch.nn.functional as F

    if getattr(F.scaled_dot_product_attention, "_moge_sage_patched", False):
        return True

    original = F.scaled_dot_product_attention

    def _sdpa_sage(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        # sageattn currently has no attn_mask / dropout / causal-with-mask support; fall back when needed.
        if attn_mask is not None or dropout_p != 0.0 or scale is not None or enable_gqa:
            return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        try:
            return sageattn(q, k, v, is_causal=is_causal)
        except Exception:
            return original(q, k, v, is_causal=is_causal)

    _sdpa_sage._moge_sage_patched = True
    _sdpa_sage._original = original
    F.scaled_dot_product_attention = _sdpa_sage
    logger.info("sageattention patched onto F.scaled_dot_product_attention.")
    return True


def unpatch_sage_attention():
    import torch.nn.functional as F
    fn = F.scaled_dot_product_attention
    if getattr(fn, "_moge_sage_patched", False):
        F.scaled_dot_product_attention = fn._original
