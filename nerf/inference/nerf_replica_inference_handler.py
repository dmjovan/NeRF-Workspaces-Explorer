import math
import os

import numpy as np
import torch
import yaml

from nerf.configs.config_parser import ConfigParser
from nerf.models.embedding import Embedding
from nerf.models.nerf_model import NeRFModel
from nerf.render.rays import create_rays
from utils.camera_poses import get_camera_pose_from_coordinates
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

        # Setting model options
        self._net_depth_coarse = eval(self._config_parser.get_param(("model", "net_depth"), str))
        self._net_width_coarse = eval(self._config_parser.get_param(("model", "net_width"), str))
        self._net_depth_fine = eval(self._config_parser.get_param(("model", "net_depth_fine"), str))
        self._net_width_fine = eval(self._config_parser.get_param(("model", "net_width_fine"), str))
        self._net_chunk = eval(self._config_parser.get_param(("model", "net_chunk"), str))
        self._chunk = eval(self._config_parser.get_param(("model", "chunk"), str))

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


        self._nerf_net_coarse = None
        self._nerf_net_fine = None

        self._embed_fcn = None
        self._embed_dirs_fcn = None


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
                          use_view_dirs=self._use_view_dirs).cpu()

        # Creating NeRF model - fine
        model_fine = NeRFModel(D=self._net_depth_fine,
                               W=self._net_width_fine,
                               input_ch=input_ch,
                               output_ch=5,
                               input_ch_views=input_ch_views,
                               use_view_dirs=self._use_view_dirs).cpu()

        self._nerf_net_coarse = model
        self._nerf_net_fine = model_fine

        self._nerf_net_coarse.eval()
        self._nerf_net_fine.eval()

        self._embed_fcn = embed_fcn
        self._embed_dirs_fcn = embed_dirs_fcn


    def render_coordinates(self, coordinates: COORD) -> np.ndarray:

        if not os.path.exists(self._ckpt_path):
            raise RuntimeError(f"Cannot load models from following path: {self._ckpt_path}. Path doesn't exist.")

        else:
            # Loading models saved state dictionary
            checkpoint = torch.load(self._ckpt_path)
            self._nerf_net_coarse.load_state_dict(checkpoint["network_coarse_state_dict"])
            self._nerf_net_fine.load_state_dict(checkpoint["network_fine_state_dict"])

            # Setting evaluation mode for models
            self._nerf_net_coarse.eval()
            self._nerf_net_fine.eval()

        with torch.no_grad():
            # Getting camera pose matrix -> shape = (1, 4, 4)
            camera_pose = get_camera_pose_from_coordinates(coordinates)

            rays = create_rays(camera_pose.shape[0], camera_pose, self._img_h, self._img_w, self._fx, self._fy,
                               self._cx, self._cy, self._depth_close_bound, self._depth_far_bound, self._use_view_dirs)

            rgb_image = self.render_path(rays, save_dir=video_save_dir, save_img=False)

        return rgb_image

