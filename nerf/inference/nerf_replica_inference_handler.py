import copy
import math
import os
from typing import Dict, Any

import numpy as np
import torch
import yaml

from nerf.configs.config_parser import ConfigParser
from nerf.models.embedding import Embedding
from nerf.models.model_utils import run_network, raw2outputs, to8b_np
from nerf.models.nerf_model import NeRFModel
from nerf.rays.rays import create_rays, sample_pdf
from utils.batch_utils import batchify_rays
from utils.camera_poses import get_camera_poses_from_list_of_coordinates
from utils.data_descriptors import COORD

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")


class NeRFReplicaInferenceHandler:

    def __init__(self, office_name: str, ckpt_path: str) -> None:

        self._office_name = office_name
        self._ckpt_path = ckpt_path

        self._config_path = os.path.join(CONFIGS_DIR, f"{office_name}_config.yaml")

        with open(self._config_path, "r") as f:
            self._config = yaml.safe_load(f)

        self._config_parser = ConfigParser()
        self._config_parser.config = self._config

        # Setting experiment options
        self._endpoint_feat = self._config_parser.get_param(("experiment", "endpoint_feat"), bool, default=False)

        # Setting model options
        self._net_depth_coarse = eval(self._config_parser.get_param(("model", "net_depth"), str))
        self._net_width_coarse = eval(self._config_parser.get_param(("model", "net_width"), str))
        self._net_depth_fine = eval(self._config_parser.get_param(("model", "net_depth_fine"), str))
        self._net_width_fine = eval(self._config_parser.get_param(("model", "net_width_fine"), str))
        self._net_chunk = eval(self._config_parser.get_param(("model", "net_chunk"), str))
        self._chunk = eval(self._config_parser.get_param(("inference", "chunk"), str))

        # Setting rendering options
        self._n_rays = eval(self._config_parser.get_param(("rendering", "n_rays"), str))
        self._n_samples = self._config_parser.get_param(("rendering", "n_samples"), int)
        self._n_importance = self._config_parser.get_param(("rendering", "n_importance"), int)
        self._num_freqs_3d = self._config_parser.get_param(("rendering", "num_freqs_3d"), int)
        self._num_freqs_2d = self._config_parser.get_param(("rendering", "num_freqs_2d"), int)
        self._use_view_dirs = self._config_parser.get_param(("rendering", "use_view_dirs"), bool)
        self._raw_noise_std = self._config_parser.get_param(("rendering", "raw_noise_std"), float)
        self._white_bkgd = self._config_parser.get_param(("rendering", "white_background"), bool)
        self._perturb = self._config_parser.get_param(("rendering", "perturb"), float)

        # Image height and width
        self._img_h = self._config_parser.get_param(("experiment", "image_height"), int)
        self._img_w = self._config_parser.get_param(("experiment", "image_width"), int)

        # Number of pixels, aspect ratio and horizontal field of view
        self._n_pix = self._img_h * self._img_w
        self._aspect_ratio = self._img_w / self._img_h
        self._hfov = 90

        # Camera Matrix components
        # The pin-hole camera has the same value for fx and fy
        self._fx = self._img_w / 2.0 / math.tan(math.radians(self._hfov / 2.0))
        self._fy = self._fx
        self._cx = (self._img_w - 1.0) / 2.0
        self._cy = (self._img_h - 1.0) / 2.0

        # Depth bounds
        self._depth_close_bound, self._depth_far_bound = self._config_parser.get_param(("rendering", "depth_range"),
                                                                                       list)
        # Models
        self._nerf_net_coarse = None
        self._nerf_net_fine = None

        self._embed_fcn = None
        self._embed_dirs_fcn = None

        torch.cuda.empty_cache()

    def initialize_models(self):
        """
        Creating coarse and fine NeRF models with previously "embedded" inputs.
        """

        embedding_3d_location = Embedding(num_freqs=self._num_freqs_3d, scalar_factor=10)
        embed_fcn = embedding_3d_location.embed
        input_ch = embedding_3d_location.output_dim

        input_ch_views = 0
        embed_dirs_fcn = None

        if self._use_view_dirs:
            embedding_2d_direction = Embedding(num_freqs=self._num_freqs_2d, scalar_factor=1)
            embed_dirs_fcn = embedding_2d_direction.embed
            input_ch_views = embedding_2d_direction.output_dim

        # Creating NeRF model - coarse
        model = NeRFModel(D=self._net_depth_coarse,
                          W=self._net_width_coarse,
                          input_ch=input_ch,
                          output_ch=5,
                          input_ch_views=input_ch_views,
                          use_view_dirs=self._use_view_dirs).cuda()

        # Creating NeRF model - fine
        model_fine = NeRFModel(D=self._net_depth_fine,
                               W=self._net_width_fine,
                               input_ch=input_ch,
                               output_ch=5,
                               input_ch_views=input_ch_views,
                               use_view_dirs=self._use_view_dirs).cuda()

        self._nerf_net_coarse = model
        self._nerf_net_fine = model_fine

        self._embed_fcn = embed_fcn
        self._embed_dirs_fcn = embed_dirs_fcn

        self._nerf_net_coarse.eval()
        self._nerf_net_fine.eval()

        try:
            # Loading models saved state dictionary
            checkpoint = torch.load(self._ckpt_path, map_location=torch.device("cuda"))

            checkpoint["network_coarse_state_dict"] = self.transform_state_dict(
                checkpoint["network_coarse_state_dict"])

            checkpoint["network_fine_state_dict"] = self.transform_state_dict(
                checkpoint["network_fine_state_dict"])

            self._nerf_net_coarse.load_state_dict(checkpoint["network_coarse_state_dict"])
            self._nerf_net_fine.load_state_dict(checkpoint["network_fine_state_dict"])

            # Setting evaluation mode for models
            self._nerf_net_coarse.eval()
            self._nerf_net_fine.eval()

        except FileNotFoundError as exc:
            raise RuntimeError(f"Checkpoint path: {self._ckpt_path} for model cannot be found!") from exc

    @staticmethod
    def transform_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Due to mismatch of key names while training and evaluating,
        some keys changed expected names inside state dict.
        This function handles this.
        """
        new_dict = {}
        for key in state_dict:
            if key.endswith("weight") or key.endswith("bias"):
                new_dict[f"_{key}"] = state_dict[key]
            else:
                new_dict[key] = state_dict[key]

        return copy.deepcopy(new_dict)

    def render_coordinates(self, init_coordinates: COORD, coordinates: COORD) -> np.ndarray:

        with torch.no_grad():
            # Getting camera pose matrix -> shape = (1, 4, 4)
            camera_pose = get_camera_poses_from_list_of_coordinates(init_coordinates, [coordinates])

            rays = create_rays(camera_pose.shape[0], camera_pose, self._img_h, self._img_w, self._fx, self._fy,
                               self._cx, self._cy, self._depth_close_bound, self._depth_far_bound, self._use_view_dirs)
            rays = rays.cuda()

            # Rendering rays
            output_dict = self._render_rays(rays[0])

            # Gathering output data
            rgb_output = output_dict["rgb_fine"]
            rgb_output = rgb_output.detach().cpu().numpy().reshape((self._img_h, self._img_w, 3))  # H, W, C

            rgb_image = to8b_np(rgb_output)

        return rgb_image

    def _render_rays(self, flat_rays: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference rendering of all rays created for one image with requested coordinates
        """

        ray_shape = flat_rays.shape  # num_rays, 11

        all_outputs = batchify_rays(self._volumetric_rendering, flat_rays, self._chunk)

        for key in all_outputs:
            # each key should contain data with shape [num_rays, *data_shape]
            key_shape = list(ray_shape[:-1]) + list(all_outputs[key].shape[1:])
            all_outputs[key] = torch.reshape(all_outputs[key], key_shape)

        return all_outputs

    def _volumetric_rendering(self, ray_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Volumetric rendering for inference over whole image
        """

        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self._n_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals)  # use linear sampling in depth space

        z_vals = z_vals.expand([N_rays, self._n_samples])

        # [N_rays, N_samples, 3]
        pts_coarse_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        raw_noise_std = 0
        raw_coarse = run_network(pts_coarse_sampled, viewdirs, self._nerf_net_coarse,
                                 self._embed_fcn, self._embed_dirs_fcn, netchunk=self._net_chunk)

        rgb_coarse, disp_coarse, acc_coarse, \
        weights_coarse, depth_coarse, feat_map_coarse = raw2outputs(raw_coarse, z_vals, rays_d,
                                                                    raw_noise_std, self._white_bkgd,
                                                                    endpoint_feat=False)

        if self._n_importance > 0:
            # (N_rays, N_samples-1) interval mid points
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], self._n_importance,
                                   det=(self._perturb == 0.) or True)
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

        all_outputs = {}
        all_outputs["rgb_coarse"] = rgb_coarse
        all_outputs["disp_coarse"] = disp_coarse
        all_outputs["acc_coarse"] = acc_coarse
        all_outputs["depth_coarse"] = depth_coarse
        all_outputs["raw_coarse"] = raw_coarse

        all_outputs["rgb_fine"] = rgb_fine
        all_outputs["disp_fine"] = disp_fine
        all_outputs["acc_fine"] = acc_fine
        all_outputs["depth_fine"] = depth_fine
        all_outputs["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        all_outputs["raw_fine"] = raw_fine  # model's raw, unprocessed predictions.

        if self._endpoint_feat:
            all_outputs["feat_map_fine"] = feat_map_fine

        for key in all_outputs:
            if (torch.isnan(all_outputs[key]).any() or torch.isinf(all_outputs[key]).any()):
                print(f"[Numerical Error] {key} contains NaN or inf.")

        return all_outputs
