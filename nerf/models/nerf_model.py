from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class NeRFModel(nn.Module):

    def __init__(self, D: int = 8, W: int = 256, input_ch: int = 3, input_ch_views: int = 3, output_ch: int = 4,
                 skips: Tuple[int, ...] = (4,), use_view_dirs: bool = False):
        """
            D: number of layers for density (sigma) encoder
            W: number of hidden units in each layer
            input_ch: number of input channels for xyz (3+3*10*2=63 by default)
            in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
            skips: layer index to add skip connection in the Dth layer
        """
        super(NeRFModel, self).__init__()

        self._D = D
        self._W = W
        self._input_ch = input_ch
        self._input_ch_views = input_ch_views
        self._output_ch = output_ch
        self._skips = skips
        self._use_view_dirs = use_view_dirs

        # Positional encoder
        self._pts_linears = nn.ModuleList([nn.Linear(self._input_ch, self._W)] +
                                          [nn.Linear(self._W, self._W) if i not in self._skips else
                                           nn.Linear(self._W + self._input_ch, self._W) for i in range(self._D - 1)])

        self._views_linears = nn.ModuleList([nn.Linear(self._input_ch_views + self._W, self._W // 2)])

        if self._use_view_dirs:
            self._feature_linear = nn.Linear(self._W, self._W)
            self._alpha_linear = nn.Linear(self._W, 1)
            self._rgb_linear = nn.Linear(self._W // 2, 3)
        else:
            self._output_linear = nn.Linear(self._W, self._output_ch)

    def forward(self, x, show_endpoint=False):
        """
        Encodes input (xyz+dir) to rgb+sigma raw output
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of 3D xyz position and viewing direction
        """

        input_pts, input_views = torch.split(x, [self._input_ch, self._input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self._pts_linears):
            h = self._pts_linears[i](h)
            h = F.relu(h)
            if i in self._skips:
                h = torch.cat([input_pts, h], -1)

        if self._use_view_dirs:
            # if using view-dirs, output occupancy alpha as well as features for concatenation
            alpha = self._alpha_linear(h)
            feature = self._feature_linear(h)

            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self._views_linears):
                h = self._views_linears[i](h)
                h = F.relu(h)

            if show_endpoint:
                endpoint_feat = h
            rgb = self._rgb_linear(h)

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self._output_linear(h)

        if show_endpoint:
            return torch.cat([outputs, endpoint_feat], -1)
        else:
            return outputs
