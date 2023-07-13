import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class NeRF(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        
        """
            D: number of layers for density (sigma) encoder
            W: number of hidden units in each layer
            input_ch: number of input channels for xyz (3+3*10*2=63 by default)
            in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
            skips: layer index to add skip connection in the Dth layer
        """

        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # positional encoder
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
            

    def forward(self, x, show_endpoint=False):

        """
        Encodes input (xyz+dir) to rgb+sigma raw output
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of 3D xyz position and viewing direction
        """

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # if using view-dirs, output occupancy alpha as well as features for concatenation
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
                
            if show_endpoint:
                endpoint_feat = h
            rgb = self.rgb_linear(h)

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if show_endpoint is False:
            return outputs
        else:
            return torch.cat([outputs, endpoint_feat], -1)
