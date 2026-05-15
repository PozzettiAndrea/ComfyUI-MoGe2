# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import comfy.ops
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm

ops = comfy.ops.disable_weight_init


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers, in_dim, bottleneck_dim,
            hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias,
            dtype=dtype, device=device, operations=operations,
        )
        self.apply(self._init_weights)
        # last_layer uses weight_norm which needs an nn.Linear instance with .weight; we
        # use operations.Linear (still an nn.Linear subclass via comfy.ops) so weight_norm
        # works the same. dtype/device threaded for consistency.
        self.last_layer = weight_norm(operations.Linear(bottleneck_dim, out_dim, bias=False, dtype=dtype, device=device))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True, dtype=None, device=None, operations=ops):
    if nlayers == 1:
        return operations.Linear(in_dim, bottleneck_dim, bias=bias, dtype=dtype, device=device)
    else:
        layers = [operations.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(operations.Linear(hidden_dim, hidden_dim, bias=bias, dtype=dtype, device=device))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(operations.Linear(hidden_dim, bottleneck_dim, bias=bias, dtype=dtype, device=device))
        return nn.Sequential(*layers)
