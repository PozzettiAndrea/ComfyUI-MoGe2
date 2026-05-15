# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Callable, Optional
import warnings

import comfy.ops
from torch import Tensor, nn
import torch.nn.functional as F

ops = comfy.ops.disable_weight_init


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = operations.Linear(in_features, 2 * hidden_features, bias=bias, dtype=dtype, device=device)
        self.w3 = operations.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (SwiGLU)")
    else:
        # warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

    # warnings.warn("xFormers is not available (SwiGLU)")


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        # xformers SwiGLU (the alternate base class) doesn't accept dtype/device/operations.
        # When XFORMERS_AVAILABLE is False, SwiGLU IS our SwiGLUFFN above which does.
        super_kwargs = dict(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
        if not XFORMERS_AVAILABLE:
            super_kwargs.update(dtype=dtype, device=device, operations=operations)
        super().__init__(**super_kwargs)
