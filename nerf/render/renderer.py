import os
from typing import Dict

import imageio
import numpy as np
import torch
from tqdm import tqdm

from nerf.configs.config_parser import ConfigParser
from nerf.models.model_utils import run_network, raw2outputs, to8b_np
from nerf.render.rays import sample_pdf
from nerf.training.training_utils import batchify_rays


class Renderer:

    def __init__(self, config: Dict) -> None:

        self._config = config
        self._config_parser = ConfigParser(self._config)

        # Setting experiment options
        self._endpoint_feat = self._config_parser.get_param(("experiment", "endpoint_feat"), bool, default=False)

        # Setting model options
        self._net_chunk = eval(self._config_parser.get_param(("model", "net_chunk"), str))
        self._chunk = eval(self._config_parser.get_param(("model", "chunk"), str))

        # Setting rendering options
        self._n_samples = self._config_parser.get_param(("rendering", "n_samples"), int)
        self._n_importance = self._config_parser.get_param(("rendering", "n_importance"), int)
        self._num_freqs_3d = self._config_parser.get_param(("rendering", "num_freqs_3d"), int)
        self._num_freqs_2d = self._config_parser.get_param(("rendering", "num_freqs_2d"), int)
        self._use_view_dirs = self._config_parser.get_param(("rendering", "use_view_dirs"), bool)
        self._raw_noise_std = self._config_parser.get_param(("rendering", "raw_noise_std"), float)
        self._white_bkgd = self._config_parser.get_param(("rendering", "white_background"), bool)
        self._perturb = self._config_parser.get_param(("rendering", "perturb"), float)

    def render_rays(self, flat_rays):
        """
        Render rays, run in optimization loop.

        Returns:
            List of:
                rgb_map: [batch_size, 3]. Predicted RGB values for rays.
                disp_map: [batch_size]. Disparity map. Inverse of depth.
                acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.

        Dict of extras: dict with everything returned by render_rays().
        """

        # Render and reshape
        ray_shape = flat_rays.shape  # num_rays, 11

        # assert ray_shape[0] == self.n_rays  # this is not satisfied in test model
        fn = self.volumetric_rendering
        all_ret = batchify_rays(fn, flat_rays.cuda(), self._chunk)

        for k in all_ret:
            k_sh = list(ray_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        return all_ret

    def volumetric_rendering(self, ray_batch):

        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self._n_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals)  # use linear sampling in depth space

        z_vals = z_vals.expand([N_rays, self._n_samples])

        if self._perturb > 0. and self._train_mode:  # perturb sampling depths (z_vals)
            if self._train_mode is True:  # only add perturbation during train_mode intead of testing
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).cuda()

                z_vals = lower + (upper - lower) * t_rand

        pts_coarse_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                           None]  # [N_rays, N_samples, 3]

        raw_noise_std = self._raw_noise_std if self._train_mode else 0
        raw_coarse = run_network(pts_coarse_sampled, viewdirs, self._nerf_net_coarse,
                                 self._embed_fcn, self._embed_dirs_fcn, netchunk=self._net_chunk)

        rgb_coarse, disp_coarse, acc_coarse, \
            weights_coarse, depth_coarse, feat_map_coarse = raw2outputs(raw_coarse, z_vals, rays_d,
                                                                        raw_noise_std, self._white_bkgd,
                                                                        endpoint_feat=False)

        if self._n_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (N_rays, N_samples-1) interval mid points
            z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], self._n_importance,
                                   det=(self._perturb == 0.) or (not self._train_mode))
            z_samples = z_samples.detach()
            # detach so that grad doesn't propogate to weights_coarse from here
            # values are interleaved actually, so maybe can do better than sort?

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            # [N_rays, N_samples + N_importance, 3]
            pts_fine_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            raw_fine = run_network(pts_fine_sampled, viewdirs, lambda x: self._nerf_net_fine(x, self._endpoint_feat),
                                   self._embed_fcn, self._embed_dirs_fcn, netchunk=self._net_chunk)

            rgb_fine, disp_fine, acc_fine, \
                weights_fine, depth_fine, feat_map_fine = raw2outputs(raw_fine, z_vals, rays_d,
                                                                      raw_noise_std, self._white_bkgd,
                                                                      endpoint_feat=self._endpoint_feat)

        ret = {}
        ret["raw_coarse"] = raw_coarse
        ret["rgb_coarse"] = rgb_coarse
        ret["disp_coarse"] = disp_coarse
        ret["acc_coarse"] = acc_coarse
        ret["depth_coarse"] = depth_coarse

        ret["rgb_fine"] = rgb_fine
        ret["disp_fine"] = disp_fine
        ret["acc_fine"] = acc_fine
        ret["depth_fine"] = depth_fine
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret["raw_fine"] = raw_fine  # model's raw, unprocessed predictions.

        if self._endpoint_feat:
            ret["feat_map_fine"] = feat_map_fine

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def render_path(self, rays, save_dir=None):

        if save_dir is None or not os.path.exists(save_dir):
            raise RuntimeError(f"Cannot store rendered images. Path {save_dir} does not exist.")

        rgbs = []

        for i, ray in enumerate(tqdm(rays)):
            output_dict = self.render_rays(ray)

            rgb = output_dict["rgb_fine"]
            rgb = rgb.cpu().numpy().reshape((self._img_h_scaled, self._img_w_scaled, 3))
            rgbs.append(rgb)

            imageio.imwrite(os.path.join(save_dir, f"rgb_{i:03d}.png"), to8b_np(rgbs[-1]))

        rgbs = np.stack(rgbs, 0)

        return rgbs
