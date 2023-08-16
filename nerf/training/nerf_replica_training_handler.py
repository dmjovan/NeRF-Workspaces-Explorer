import math
import os
from typing import Dict, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from imgviz import depth2rgb
from tqdm import tqdm

from nerf.configs.config_parser import ConfigParser
from nerf.datasets.replica_dataset import ReplicaDataset
from nerf.models.embedding import Embedding
from nerf.models.model_utils import run_network, raw2outputs, img2mse, mse2psnr, to8b_np
from nerf.models.nerf_model import NeRFModel
from nerf.rays.rays import create_rays, sample_pdf
from nerf.visualisation.tensorboard_writer import TensorboardWriter
from utils.batch_utils import batchify_rays

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")


class NeRFReplicaTrainingHandler:

    def __init__(self, office_name: str, config: Dict) -> None:

        self._office_name = office_name
        self._config = config

        self._train_mode = True

        training_dir_number = 1
        if os.path.exists(os.path.join(EXPERIMENTS_DIR, self._office_name)):
            training_dir_number = len(os.listdir(os.path.join(EXPERIMENTS_DIR, self._office_name))) + 1

        self._save_dir = os.path.join(EXPERIMENTS_DIR, self._office_name, str(training_dir_number))

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        self._config_parser = ConfigParser(self._config)

        # Setting experiment options
        self._endpoint_feat = self._config_parser.get_param(("experiment", "endpoint_feat"), bool, default=False)

        # Setting training options
        self._learning_rate = self._config_parser.get_param(("training", "learning_rate"), float)
        self._learning_rate_decay_rate = self._config_parser.get_param(("training", "learning_rate_decay_rate"), float)
        self._learning_rate_decay_steps = self._config_parser.get_param(("training", "learning_rate_decay_steps"),
                                                                        float)

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

        # Tensorboard Writer initialization
        self._tensorboard_writer = TensorboardWriter(self._save_dir, self._config)

        # Replica Dataset initialization
        self._dataset = ReplicaDataset(self._office_name, self._config)
        print(self._dataset)

        # Dataset parameters

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

        # Scaled images for test and visualisation (scaling usually ends up the same)
        # It influences the new camera matrix for test/visualisation purposes
        self._test_viz_factor = self._config_parser.get_param(("rendering", "test_viz_factor"), int)
        self._img_h_scaled = self._img_h // self._test_viz_factor
        self._img_w_scaled = self._img_w // self._test_viz_factor

        self._fx_scaled = self._img_w_scaled / 2.0 / math.tan(math.radians(self._hfov / 2.0))
        self._fy_scaled = self._fx_scaled
        self._cx_scaled = (self._img_w_scaled - 1.0) / 2.0
        self._cy_scaled = (self._img_h_scaled - 1.0) / 2.0

        # Logging parameters
        self._step_log_print = self._config_parser.get_param(("logging", "step_log_print"), int)
        self._step_save_ckpt = self._config_parser.get_param(("logging", "step_save_ckpt"), int)
        self._step_render_train = self._config_parser.get_param(("logging", "step_render_train"), int)
        self._step_render_test = self._config_parser.get_param(("logging", "step_render_test"), int)

    def prepare_data(self):
        """
        Preparing the data from loaded datasets
        """

        ############################# Training Data #############################

        # RGB images
        # regular rgb training images - (N, H, W, C); C = 3
        self._train_rgbs = torch.from_numpy(self._dataset.train_dataset["rgb"])

        # scaled rgb training images -> (N, H, W, C) -> (N, C, H, W) -> bilinear transformation -> (N, H, W, C)
        self._train_rgbs_scaled = F.interpolate(self._train_rgbs.permute(0, 3, 1, 2),
                                                scale_factor=1 / self._test_viz_factor, mode="bilinear").permute(0, 2,
                                                                                                                 3, 1)

        # Depth maps
        # regular depth training maps - (N, H, W)
        self._train_depths = torch.from_numpy(self._dataset.train_dataset["depth"])

        # converting depth maps into rgb images using depth bounds - (N, H, W, C); C = 3
        self._viz_train_depths = np.stack([depth2rgb(dep, min_value=self._depth_close_bound,
                                                     max_value=self._depth_far_bound) for dep in
                                           self._dataset.train_dataset["depth"]], axis=0)

        # (N, H, W) -> unsqueeze(dim = 1) -> C = 1 -> (N, 1, H, W) -> bilinear -> squeeze(1) -> (N, H, W)
        self._train_depths_scaled = F.interpolate(torch.unsqueeze(self._train_depths, dim=1).float(),
                                                  scale_factor=1 / self._test_viz_factor,
                                                  mode="bilinear").squeeze(1).cpu().numpy()
        # training camera poses (N, 4, 4)
        self._train_camera_poses = torch.from_numpy(self._dataset.train_dataset["camera_pose"]).float()

        ############################# Test Data #############################

        # RGB images
        # regular rgb test images - (N, H, W, C); C = 3
        self._test_rgbs = torch.from_numpy(self._dataset.test_dataset["rgb"])

        # scaled rgb test images -> (N, H, W, C) -> (N, C, H, W) -> bilinear transformation -> (N, H, W, C)
        self._test_rgbs_scaled = F.interpolate(self._test_rgbs.permute(0, 3, 1, 2),
                                               scale_factor=1 / self._test_viz_factor, mode="bilinear").permute(0, 2, 3,
                                                                                                                1)

        # Depth maps
        # regular depth test maps - (N, H, W)
        self._test_depths = torch.from_numpy(self._dataset.test_dataset["depth"])

        # converting depth maps into rgb images using depth bounds - (N, H, W, C); C = 3
        self._viz_test_depths = np.stack([depth2rgb(dep, min_value=self._depth_close_bound,
                                                    max_value=self._depth_far_bound) for dep in
                                          self._dataset.test_dataset["depth"]], axis=0)

        # (N, H, W) -> unsqueeze(dim = 1) -> C = 1 -> (N, 1, H, W) -> bilinear -> squeeze(1) -> (N, H, W)
        self._test_depths_scaled = F.interpolate(torch.unsqueeze(self._test_depths, dim=1).float(),
                                                 scale_factor=1 / self._test_viz_factor,
                                                 mode="bilinear").squeeze(1).cpu().numpy()

        # test camera poses (N, 4, 4)
        self._test_camera_poses = torch.from_numpy(self._dataset.test_dataset["camera_pose"]).float()

        # After creating all torch.Tensor(s) comes sending all data to the CUDA cores
        self._train_rgbs = self._train_rgbs.cuda()
        self._train_rgbs_scaled = self._train_rgbs_scaled.cuda()
        self._train_depths = self._train_depths.cuda()

        self._test_rgbs = self._test_rgbs.cuda()
        self._test_rgbs_scaled = self._test_rgbs_scaled.cuda()
        self._test_depths = self._test_depths.cuda()

        # Adding all datasets to Tensorboard for comparison with rendered images
        self._tensorboard_writer.summary_writer.add_image("Train/rgb_ground_truth",
                                                          self._dataset.train_dataset["rgb"], 0,
                                                          dataformats="NHWC")

        self._tensorboard_writer.summary_writer.add_image("Test/rgb_ground_truth",
                                                          self._dataset.test_dataset["rgb"], 0,
                                                          dataformats="NHWC")

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

        learnable_params = list(model.parameters())

        # Creating NeRF model - fine
        model_fine = NeRFModel(D=self._net_depth_fine,
                               W=self._net_width_fine,
                               input_ch=input_ch,
                               output_ch=5,
                               input_ch_views=input_ch_views,
                               use_view_dirs=self._use_view_dirs).cuda()

        learnable_params += list(model_fine.parameters())

        # Adam optimizer
        optimizer = torch.optim.Adam(params=learnable_params, lr=self._learning_rate)

        self._nerf_net_coarse = model
        self._nerf_net_fine = model_fine
        self._optimizer = optimizer

        self._embed_fcn = embed_fcn
        self._embed_dirs_fcn = embed_dirs_fcn

    def initialize_rays(self):
        """
        Creating rays for all datasets
        """

        # Creating rays
        rays_train = create_rays(self._dataset.train_dataset_len, self._train_camera_poses, self._img_h, self._img_w,
                                 self._fx, self._fy, self._cx, self._cy, self._depth_close_bound, self._depth_far_bound,
                                 self._use_view_dirs)

        rays_vis = create_rays(self._dataset.train_dataset_len, self._train_camera_poses, self._img_h_scaled,
                               self._img_w_scaled, self._fx_scaled, self._fy_scaled, self._cx_scaled, self._cy_scaled,
                               self._depth_close_bound, self._depth_far_bound, self._use_view_dirs)

        rays_test = create_rays(self._dataset.test_dataset_len, self._test_camera_poses, self._img_h_scaled,
                                self._img_w_scaled, self._fx_scaled, self._fy_scaled, self._cx_scaled, self._cy_scaled,
                                self._depth_close_bound, self._depth_far_bound, self._use_view_dirs)

        self.rays_train = rays_train.cuda()  # [num_images, H*W, 11]
        self.rays_vis = rays_vis.cuda()
        self.rays_test = rays_test.cuda()

    def step(self, global_step):

        """
        One step in model optimization loop.
        Contains:
            - Randomly sampling rays and ground truth for them
            - Rendering of sampled rays
            - Loss and metrics calculations
            - Logging and visualization
            - Model saving
        """

        # Sampling rays to query and optimize
        sampled_rays, sampled_gt_rgb = self._sample_training_data()

        # Rendering rays using models
        output_dict = self._render_rays(sampled_rays)

        rgb_coarse = output_dict["rgb_coarse"]  # N_rays x 3
        rgb_fine = output_dict["rgb_fine"]

        # Calculation of loss and gradients for coarse and fine networks

        self._optimizer.zero_grad()

        # MSE on rgb pixel values for coarse-net RGB output
        rgb_loss_coarse = img2mse(rgb_coarse, sampled_gt_rgb)

        # PSNR value for coarse-net RGB output
        with torch.no_grad():
            psnr_coarse = mse2psnr(rgb_loss_coarse)

        # MSE on rgb pixel values for fine-net RGB output
        rgb_loss_fine = img2mse(rgb_fine, sampled_gt_rgb)

        # PSNR value for fine-net RGB output
        with torch.no_grad():
            psnr_fine = mse2psnr(rgb_loss_fine)

        # Total loss is the sum of coarse-net loss and fine-net loss
        total_loss = rgb_loss_coarse + rgb_loss_fine

        # Backpropagation of loss using total loss
        total_loss.backward()
        self._optimizer.step()

        # Updating learning rate
        new_learning_rate = self._learning_rate * (self._learning_rate_decay_rate **
                                                   (global_step / self._learning_rate_decay_steps))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        # Printing progress
        if global_step % self._step_log_print == 0:
            tqdm.write(f"[TRAIN] Iter: {global_step} "
                       f"Loss: {total_loss.item()}, rgb_coarse: {rgb_loss_coarse.item()}, "
                       f"rgb_fine: {rgb_loss_fine.item()}, "
                       f"PSNR_coarse: {psnr_coarse.item()}, PSNR_fine: {psnr_fine.item()}")

        # Logging losses, metrics and histograms to Tensorboard
        if global_step % float(self._tensorboard_writer.log_interval) == 0:
            self._log_to_tensorboard(global_step, output_dict, rgb_loss_coarse, rgb_loss_fine,
                                     total_loss, psnr_coarse, psnr_fine)

        # Rendering training images
        if global_step % self._step_render_train == 0 and global_step > 0:
            self._render_train_images(global_step)

        # Rendering test images
        if global_step % self._step_render_test == 0 and global_step > 0:
            self._render_test_images(global_step)

        # Saving models checkpoint
        if global_step % float(self._step_save_ckpt) == 0:
            self._save_models_checkpoint(global_step)

    def _sample_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling rays and ground truth data
        """

        def sample_indices(n_rays: int, num_images: int, h: int, w: int) -> Tuple[np.ndarray, torch.Tensor]:
            # Randomly sampling one image from the full training set
            image_index = np.random.choice(np.arange(num_images)).reshape((1, 1))

            # Randomly sampling n_rays pixels from image (actually pixel indices)
            hw_pixel_inds = torch.randint(0, h * w, (1, n_rays))

            return image_index, hw_pixel_inds

        # generate sampling index
        num_img, num_ray, ray_dim = self.rays_train.shape

        if num_ray != self._img_h * self._img_w:
            raise RuntimeError("Bad configuration of training rays")

        # Sample random pixels from one random image from training dataset
        image_idx, hw_pixel_indices = sample_indices(self._n_rays, num_img, self._img_h, self._img_w)

        sampled_rays = self.rays_train[image_idx, hw_pixel_indices, :]
        sampled_rays = sampled_rays.reshape([-1, ray_dim]).float()

        sampled_gt_rgbs = self._train_rgbs.reshape(self._dataset.train_dataset_len,
                                                   -1, 3)[image_idx, hw_pixel_indices, :].reshape(-1, 3)

        return sampled_rays, sampled_gt_rgbs

    def _log_to_tensorboard(self, global_step: int, network_output_dict: Dict, rgb_loss_coarse: float,
                            rgb_loss_fine: float, total_loss: float, psnr_coarse: float, psnr_fine: float) -> None:
        """
        Logging data on Tensorboard
        """

        # Adding losses
        self._tensorboard_writer.write_scalars(global_step, [rgb_loss_coarse, rgb_loss_fine, total_loss],
                                               ["Train/Loss/rgb_loss_coarse", "Train/Loss/rgb_loss_fine",
                                                "Train/Loss/total_loss"])

        # Adding raw transparency value into Tensorboard histogram
        trans_coarse = network_output_dict["raw_coarse"][..., 3]
        self._tensorboard_writer.write_histogram(global_step, trans_coarse, "trans_coarse")

        trans_fine = network_output_dict["raw_fine"][..., 3]
        self._tensorboard_writer.write_histogram(global_step, trans_fine, "trans_fine")

        # Adding PSNR metrics
        self._tensorboard_writer.write_scalars(global_step, [psnr_coarse, psnr_fine],
                                               ["Train/Metric/psnr_coarse", "Train/Metric/psnr_fine"])

    def _save_models_checkpoint(self, global_step: int) -> None:
        """
        Saving models into .ckpt file
        """

        ckpt_dir = os.path.join(self._save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_file = os.path.join(ckpt_dir, f"{global_step:06d}.ckpt")

        torch.save({"global_step": global_step,
                    "network_coarse_state_dict": self._nerf_net_coarse.state_dict(),
                    "network_fine_state_dict": self._nerf_net_fine.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict()}, ckpt_file)

        print(f"Saved checkpoints at {ckpt_file}")

    def _render_train_images(self, global_step: int) -> None:
        """
        Evaluation of models over training images - rendering and calculating mean squared error and PSNR
        """

        train_save_dir = os.path.join(self._save_dir, "train_render", f"step_{global_step:06d}")
        os.makedirs(train_save_dir, exist_ok=True)

        # Setting evaluation mode
        self._train_mode = False
        self._nerf_net_coarse.eval()
        self._nerf_net_fine.eval()

        with torch.no_grad():
            rgbs = self._render_images_for_camera_path(self.rays_vis, save_dir=train_save_dir)

        print("Saved rendered images from training dataset")

        with torch.no_grad():
            batch_train_img_mse = img2mse(torch.from_numpy(rgbs), self._train_rgbs_scaled.cpu())
            batch_train_img_psnr = mse2psnr(batch_train_img_mse)

            self._tensorboard_writer.write_scalars(global_step, [batch_train_img_psnr, batch_train_img_mse],
                                                   ["Train/Metric/batch_PSNR", "Train/Metric/batch_MSE"])

        # Creating a video from rendered images
        imageio.mimwrite(os.path.join(train_save_dir, "rgb.mp4"), to8b_np(rgbs), fps=30, quality=8)

        # Adding rendered images into Tensorboard
        self._tensorboard_writer.summary_writer.add_image("Train/rgb", rgbs, global_step, dataformats="NHWC")

        # Back to train mode
        self._train_mode = True
        self._nerf_net_coarse.train()
        self._nerf_net_fine.train()

    def _render_test_images(self, global_step: int) -> None:
        """
        Evaluation of models over test images - rendering and calculating mean squared error and PSNR
        """

        test_save_dir = os.path.join(self._save_dir, "test_render", f"step_{global_step:06d}")
        os.makedirs(test_save_dir, exist_ok=True)

        # Setting evaluation mode
        self._train_mode = False
        self._nerf_net_coarse.eval()
        self._nerf_net_fine.eval()

        with torch.no_grad():
            rgbs = self._render_images_for_camera_path(self.rays_test, save_dir=test_save_dir)

        print("Saved rendered images from test dataset")

        with torch.no_grad():
            batch_test_img_mse = img2mse(torch.from_numpy(rgbs), self._test_rgbs_scaled.cpu())
            batch_test_img_psnr = mse2psnr(batch_test_img_mse)
            self._tensorboard_writer.write_scalars(global_step, [batch_test_img_psnr, batch_test_img_mse],
                                                   ["Test/Metric/batch_PSNR", "Test/Metric/batch_MSE"])

        # Creating a video from rendered images
        imageio.mimwrite(os.path.join(test_save_dir, "rgb.mp4"), to8b_np(rgbs), fps=30, quality=8)

        # Adding rendered images into Tensorboard
        self._tensorboard_writer.summary_writer.add_image("Test/rgb", rgbs, global_step, dataformats="NHWC")

        # Back to train mode
        self._train_mode = True
        self._nerf_net_coarse.train()
        self._nerf_net_fine.train()

    def _render_images_for_camera_path(self, rays: torch.Tensor, save_dir: str) -> np.ndarray:
        """
        Rendering images for one "path" of camera. Path is already embedded into rays,
        since they are created earlier before.
        """

        if not os.path.exists(save_dir):
            raise RuntimeError(f"Cannot store rendered images. Path {save_dir} does not exist.")

        rgb_images = []

        for i, ray in enumerate(tqdm(rays)):
            # Rendering rays
            output_dict = self._render_rays(ray)

            # Gathering output data
            rgb = output_dict["rgb_fine"]
            rgb = rgb.cpu().numpy().reshape((self._img_h_scaled, self._img_w_scaled, 3))  # H, W, C
            rgb_images.append(rgb)

            # Writing it into a file
            imageio.imwrite(os.path.join(save_dir, f"rgb_{i:03d}.png"), to8b_np(rgb_images[-1]))

        # Stacking images over batch dimension
        rgb_images = np.stack(rgb_images, 0)

        return rgb_images

    def _render_rays(self, flat_rays: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Render rays, run in optimization loop.

        Returns:
            List of:
                rgb_map: [batch_size, 3]. Predicted RGB values for rays.
                disp_map: [batch_size]. Disparity map. Inverse of depth.
                acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.

        Dict of extras: dict with everything returned by render_rays().
        """

        ray_shape = flat_rays.shape  # num_rays, 11

        all_outputs = batchify_rays(self._volumetric_rendering, flat_rays.cuda(), self._chunk)

        for key in all_outputs:
            # each key should contain data with shape [num_rays, *data_shape]
            key_shape = list(ray_shape[:-1]) + list(all_outputs[key].shape[1:])
            all_outputs[key] = torch.reshape(all_outputs[key], key_shape)

        return all_outputs

    def _volumetric_rendering(self, ray_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Volumetric rendering over batch of rays (chunk).
        """

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
