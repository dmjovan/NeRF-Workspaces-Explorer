import math
import os
import time
from typing import Dict

import imageio
from imgviz import depth2rgb
from tqdm import tqdm

from nerf.configs.config_parser import ConfigParser
from nerf.datasets.replica_dataset import ReplicaDataset
from nerf.models.embedder import get_embedder
from nerf.models.model_utils import *
from nerf.models.nerf_model import NeRF
from nerf.models.rays import sampling_index, sample_pdf, create_rays
from nerf.training.training_utils import *
from nerf.visualisation.tensorboard_writer import TensorboardWriter
from nerf.visualisation.video_utils import *

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


class NeRFTrainer:

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
        self._convention = self._config_parser.get_param(("experiment", "convention"), str)
        self._endpoint_feat = self._config_parser.get_param(("experiment", "endpoint_feat"), bool, default=False)

        # Setting training options
        self._learning_rate = self._config_parser.get_param(("training", "learning_rate"), float)
        self._learning_rate_decay_rate = self._config_parser.get_param(("training", "learning_rate_decay_rate"), float)
        self._learning_rate_decay_steps = self._config_parser.get_param(("training", "learning_rate_decay_steps"),
                                                                        float)

        # Setting model options
        self._net_chunk = eval(self._config_parser.get_param(("model", "net_chunk"), str))
        self._chunk = eval(self._config_parser.get_param(("model", "chunk"), str))

        # Setting rendering options
        self._n_rays = eval(self._config_parser.get_param(("rendering", "n_rays"), str))
        self._n_samples = self._config_parser.get_param(("rendering", "n_samples"), int)
        self._n_importance = self._config_parser.get_param(("rendering", "n_importance"), int)
        self._use_view_dirs = self._config_parser.get_param(("rendering", "use_view_dirs"), bool)
        self._raw_noise_std = self._config_parser.get_param(("rendering", "raw_noise_std"), float)
        self._white_bkgd = self._config_parser.get_param(("rendering", "white_background"), bool)
        self._perturb = self._config_parser.get_param(("rendering", "perturb"), float)
        self._no_batching = self._config_parser.get_param(("rendering", "no_batching"), bool)

        self._dataset = ReplicaDataset(self._office_name, self._config)

        print(self._dataset)

        self._tensorboard_visualizer = TensorboardWriter(self._save_dir, self._config)

        # setting params for data from dataset
        self.set_data_params()

        # preparing data from dataset
        self.prepare_data()

        # create nerf model, initialize optimizer
        self.create_nerf()

        # create rays in world coordinates
        self.init_rays()

    def set_data_params(self):

        self.H = self._config["experiment"]["image_height"]
        self.W = self._config["experiment"]["image_width"]

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W / self.H

        self.hfov = 90

        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.fy = self.fx

        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0

        self.depth_close_bound, self.depth_far_bound = self._config["render"]["depth_range"]
        self.c2w_staticcam = None

        # use scaled size for test and visualisation purpose
        self.test_viz_factor = int(self._config["render"]["test_viz_factor"])
        self.H_scaled = self.H // self.test_viz_factor
        self.W_scaled = self.W // self.test_viz_factor

        self.fx_scaled = self.W_scaled / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.fy_scaled = self.fx_scaled

        self.cx_scaled = (self.W_scaled - 1.0) / 2.0
        self.cy_scaled = (self.H_scaled - 1.0) / 2.0

    def prepare_data(self):

        # shift numpy data to torch
        train_samples = self._dataset.train_samples
        test_samples = self._dataset.test_samples

        self.train_ids = self._dataset._train_ids
        self.test_ids = self._dataset.test_ids

        self.num_train = self._dataset.train_num
        self.num_test = self._dataset.test_num

        ##### Train Data #####

        # RGB
        self.train_image = torch.from_numpy(train_samples["image"])
        self.train_image_scaled = F.interpolate(self.train_image.permute(0, 3, 1, 2, ),
                                                scale_factor=1 / self._config["render"]["test_viz_factor"],
                                                mode='bilinear').permute(0, 2, 3, 1)
        # Depth
        self.train_depth = torch.from_numpy(train_samples["depth"])
        self.viz_train_depth = np.stack(
            [depth2rgb(dep, min_value=self.depth_close_bound, max_value=self.depth_far_bound) for dep in
             train_samples["depth"]], axis=0)  # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.train_depth_scaled = F.interpolate(torch.unsqueeze(self.train_depth, dim=1).float(),
                                                scale_factor=1 / self._config["render"]["test_viz_factor"],
                                                mode='bilinear').squeeze(1).cpu().numpy()
        # pose 
        self.train_pose = torch.from_numpy(train_samples["pose"]).float()

        ##### Test Data #####

        # RGB
        self.test_image = torch.from_numpy(test_samples["image"])  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(self.test_image.permute(0, 3, 1, 2, ),
                                               scale_factor=1 / self._config["render"]["test_viz_factor"],
                                               mode='bilinear').permute(0, 2, 3, 1)
        # Depth
        self.test_depth = torch.from_numpy(test_samples["depth"])  # [num_test, H, W]
        self.viz_test_depth = np.stack(
            [depth2rgb(dep, min_value=self.depth_close_bound, max_value=self.depth_far_bound) for dep in
             test_samples["depth"]], axis=0)  # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.test_depth_scaled = F.interpolate(torch.unsqueeze(self.test_depth, dim=1).float(),
                                               scale_factor=1 / self._config["render"]["test_viz_factor"],
                                               mode='bilinear').squeeze(1).cpu().numpy()
        # pose 
        self.test_pose = torch.from_numpy(test_samples["pose"]).float()  # [num_test, 4, 4]

        self.train_image = self.train_image.cuda()
        self.train_image_scaled = self.train_image_scaled.cuda()
        self.train_depth = self.train_depth.cuda()

        self.test_image = self.test_image.cuda()
        self.test_image_scaled = self.test_image_scaled.cuda()
        self.test_depth = self.test_depth.cuda()

        # set the data sampling paras which need the number of training images
        if self.no_batching is False:  # False means we need to sample from all rays instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train * self.H * self.W)

        # add datasets to tfboard for comparison to rendered images
        self._tensorboard_visualizer._summary_writer.add_image('Train/rgb_ground_truth', train_samples["image"], 0,
                                                               dataformats='NHWC')

        self._tensorboard_visualizer._summary_writer.add_image('Test/rgb_ground_truth', test_samples["image"], 0,
                                                               dataformats='NHWC')

    def init_rays(self):

        # create rays
        rays = create_rays(self.num_train,
                           self.train_pose,
                           self.H,
                           self.W,
                           self.fx,
                           self.fy,
                           self.cx,
                           self.cy,
                           self.depth_close_bound,
                           self.depth_far_bound,
                           use_viewdirs=self.use_view_dirs,
                           convention=self._convention)

        rays_vis = create_rays(self.num_train,
                               self.train_pose,
                               self.H_scaled,
                               self.W_scaled,
                               self.fx_scaled,
                               self.fy_scaled,
                               self.cx_scaled,
                               self.cy_scaled,
                               self.depth_close_bound,
                               self.depth_far_bound,
                               use_viewdirs=self.use_view_dirs,
                               convention=self._convention)

        rays_test = create_rays(self.num_test,
                                self.test_pose,
                                self.H_scaled,
                                self.W_scaled,
                                self.fx_scaled,
                                self.fy_scaled,
                                self.cx_scaled,
                                self.cy_scaled,
                                self.depth_close_bound,
                                self.depth_far_bound,
                                use_viewdirs=self.use_view_dirs,
                                convention=self._convention)

        # init rays
        self.rays = rays.cuda()  # [num_images, H*W, 11]
        self.rays_vis = rays_vis.cuda()
        self.rays_test = rays_test.cuda()

    def sample_data(self, step, rays, h, w, no_batching=True, mode="train"):

        # generate sampling index
        num_img, num_ray, ray_dim = rays.shape

        assert num_ray == h * w
        total_ray_num = num_img * h * w

        if mode == "train":
            image = self.train_image
            sample_num = self.num_train

        elif mode == "test":
            image = self.test_image
            sample_num = self.num_test

        # sample rays and ground truth data

        if no_batching:  # sample random pixels from one random images

            index_batch, index_hw = sampling_index(self._n_rays, num_img, h, w)
            sampled_rays = rays[index_batch, index_hw, :]

            flat_sampled_rays = sampled_rays.reshape([-1, ray_dim]).float()
            gt_image = image.reshape(sample_num, -1, 3)[index_batch, index_hw, :].reshape(-1, 3)

        else:  # sample from all random pixels

            index_hw = self.rand_idx[self.i_batch: self.i_batch + self._n_rays]

            flat_rays = rays.reshape([-1, ray_dim]).float()
            flat_sampled_rays = flat_rays[index_hw, :]
            gt_image = image.reshape(-1, 3)[index_hw, :]

            self.i_batch += self._n_rays
            if self.i_batch >= total_ray_num:
                print("Shuffle data after an epoch!")
                self.rand_idx = torch.randperm(total_ray_num)
                self.i_batch = 0

        sampled_rays = flat_sampled_rays
        sampled_gt_rgb = gt_image

        return sampled_rays, sampled_gt_rgb

    def render_rays(self, flat_rays):

        """
            Render rays, run in optimisation loop.

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
        all_ret = batchify_rays(fn, flat_rays.cuda(), self.chunk)

        for k in all_ret:
            k_sh = list(ray_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        return all_ret

    def volumetric_rendering(self, ray_batch):

        """
            Volumetric Rendering
        """
        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals)  # use linear sampling in depth space

        z_vals = z_vals.expand([N_rays, self.N_samples])

        if self.perturb > 0. and self._train_mode:  # perturb sampling depths (z_vals)
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

        raw_noise_std = self.raw_noise_std if self._train_mode else 0
        raw_coarse = run_network(pts_coarse_sampled, viewdirs, self.nerf_net_coarse,
                                 self.embed_fn, self.embeddirs_fn, netchunk=self._net_chunk)

        rgb_coarse, disp_coarse, acc_coarse, weights_coarse, depth_coarse, feat_map_coarse = raw2outputs(raw_coarse,
                                                                                                         z_vals, rays_d,
                                                                                                         raw_noise_std,
                                                                                                         self.white_bkgd,
                                                                                                         endpoint_feat=False)

        if self.N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (N_rays, N_samples-1) interval mid points
            z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], self.N_importance,
                                   det=(self.perturb == 0.) or (not self._train_mode))
            z_samples = z_samples.detach()
            # detach so that grad doesn't propogate to weights_coarse from here
            # values are interleaved actually, so maybe can do better than sort?

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts_fine_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                             None]  # [N_rays, N_samples + N_importance, 3]

            raw_fine = run_network(pts_fine_sampled, viewdirs, lambda x: self.nerf_net_fine(x, self._endpoint_feat),
                                   self.embed_fn, self.embeddirs_fn, netchunk=self._net_chunk)

            rgb_fine, disp_fine, acc_fine, weights_fine, depth_fine, feat_map_fine = raw2outputs(raw_fine, z_vals,
                                                                                                 rays_d, raw_noise_std,
                                                                                                 self.white_bkgd,
                                                                                                 endpoint_feat=self._endpoint_feat)

        ret = {}
        ret['raw_coarse'] = raw_coarse
        ret['rgb_coarse'] = rgb_coarse
        ret['disp_coarse'] = disp_coarse
        ret['acc_coarse'] = acc_coarse
        ret['depth_coarse'] = depth_coarse

        ret['rgb_fine'] = rgb_fine
        ret['disp_fine'] = disp_fine
        ret['acc_fine'] = acc_fine
        ret['depth_fine'] = depth_fine
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['raw_fine'] = raw_fine  # model's raw, unprocessed predictions.

        if self._endpoint_feat:
            ret['feat_map_fine'] = feat_map_fine

        for k in ret:
            # if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and self.config["experiment"]["debug"]:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def create_nerf(self):

        """ 
            Instantiate NeRF's MLP model 
        """

        embed_fn, input_ch = get_embedder(self._config["render"]["multires"], self._config["render"]["i_embed"],
                                          scalar_factor=10)

        input_ch_views = 0
        embeddirs_fn = None
        if self._config["render"]["use_viewdirs"]:
            embeddirs_fn, input_ch_views = get_embedder(self._config["render"]["multires_views"],
                                                        self._config["render"]["i_embed"], scalar_factor=1)

        # creating NeRF model - coarse
        model = NeRF(D=self._config["model"]["netdepth"],
                     W=self._config["model"]["netwidth"],
                     input_ch=input_ch,
                     output_ch=5,
                     skips=[4],
                     input_ch_views=input_ch_views,
                     use_viewdirs=self._config["render"]["use_viewdirs"]).cuda()

        grad_vars = list(model.parameters())

        # creating NeRF model - fine
        model_fine = NeRF(D=self._config["model"]["netdepth_fine"],
                          W=self._config["model"]["netwidth_fine"],
                          input_ch=input_ch,
                          output_ch=5,
                          skips=[4],
                          input_ch_views=input_ch_views,
                          use_viewdirs=self._config["render"]["use_viewdirs"]).cuda()

        grad_vars += list(model_fine.parameters())

        # create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=self.lrate)

        self.nerf_net_coarse = model
        self.nerf_net_fine = model_fine
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.optimizer = optimizer

    def step(self, global_step):

        """
            Running one optimization step
        """

        # sample rays to query and optimise
        sampled_rays, sampled_gt_rgb = self.sample_data(global_step, self.rays, self.H, self.W, no_batching=True,
                                                        mode="train")

        output_dict = self.render_rays(sampled_rays)

        rgb_coarse = output_dict["rgb_coarse"]  # N_rays x 3
        rgb_fine = output_dict["rgb_fine"]

        # calculation of loss and gradients for coarse and fine networks 

        self.optimizer.zero_grad()

        img_loss_coarse = img2mse(rgb_coarse, sampled_gt_rgb)

        with torch.no_grad():
            psnr_coarse = mse2psnr(img_loss_coarse)

        img_loss_fine = img2mse(rgb_fine, sampled_gt_rgb)

        with torch.no_grad():
            psnr_fine = mse2psnr(img_loss_fine)

        total_loss = img_loss_coarse + img_loss_fine

        total_loss.backward()
        self.optimizer.step()

        # updating learning rate
        new_lrate = self.lrate * (self.lrate_decay_rate ** (global_step / self.lrate_decay_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Visualization
        if global_step % float(self._config["logging"]["step_log_tensorboard"]) == 0:
            self._tensorboard_visualizer.visualize_scalars(global_step,
                                                           [img_loss_coarse, img_loss_fine, total_loss],
                                                           ['Train/Loss/rgb_loss_coarse', 'Train/Loss/rgb_loss_fine',
                                                            'Train/Loss/total_loss'])

            # add raw transparancy value into tfb histogram
            trans_coarse = output_dict["raw_coarse"][..., 3]
            self._tensorboard_visualizer.visualize_histogram(global_step, trans_coarse, 'trans_coarse')

            trans_fine = output_dict['raw_fine'][..., 3]
            self._tensorboard_visualizer.visualize_histogram(global_step, trans_fine, 'trans_fine')

            self._tensorboard_visualizer.visualize_scalars(global_step, [psnr_coarse, psnr_fine],
                                                           ['Train/Metric/psnr_coarse', 'Train/Metric/psnr_fine'])

        # Saving checkpoint
        if global_step % float(self._config["logging"]["step_save_ckpt"]) == 0:

            ckpt_dir = os.path.join(self.save_dir, "checkpoints")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            ckpt_file = os.path.join(ckpt_dir, '{:06d}.ckpt'.format(global_step))

            torch.save({'global_step': global_step,
                        'network_coarse_state_dict': self.nerf_net_coarse.state_dict(),
                        'network_fine_state_dict': self.nerf_net_fine.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(), }, ckpt_file)

            print(f"Saved checkpoints at {ckpt_file}")

        # Rendering training images
        if global_step % self._config["logging"]["step_render_train"] == 0 and global_step > 0:
            self._train_mode = False
            self.nerf_net_coarse.eval()
            self.nerf_net_fine.eval()

            train_save_dir = os.path.join(self._config["experiment"]["save_dir"], "train_render",
                                          'step_{:06d}'.format(global_step))
            os.makedirs(train_save_dir, exist_ok=True)

            with torch.no_grad():
                rgbs = self.render_path(self.rays_vis, save_dir=train_save_dir)

            print('Saved rendered images from training set')

            self._train_mode = True
            self.nerf_net_coarse.train()
            self.nerf_net_fine.train()

            with torch.no_grad():
                batch_train_img_mse = img2mse(torch.from_numpy(rgbs), self.train_image_scaled.cpu())
                batch_train_img_psnr = mse2psnr(batch_train_img_mse)

                self._tensorboard_visualizer.visualize_scalars(global_step, [batch_train_img_psnr, batch_train_img_mse],
                                                               ['Train/Metric/batch_PSNR', 'Train/Metric/batch_MSE'])

            imageio.mimwrite(os.path.join(train_save_dir, 'rgb.mp4'), to8b_np(rgbs), fps=30, quality=8)

            # add rendered image into tf-board
            self._tensorboard_visualizer._summary_writer.add_image('Train/rgb', rgbs, global_step, dataformats='NHWC')

        # Rendering test images
        if global_step % self._config["logging"]["step_render_test"] == 0 and global_step > 0:
            self._train_mode = False
            self.nerf_net_coarse.eval()
            self.nerf_net_fine.eval()

            test_save_dir = os.path.join(self._config["experiment"]["save_dir"], "test_render",
                                         'step_{:06d}'.format(global_step))
            os.makedirs(test_save_dir, exist_ok=True)

            with torch.no_grad():
                rgbs = self.render_path(self.rays_test, save_dir=test_save_dir)

            print('Saved rendered images from test set')

            self._train_mode = True
            self.nerf_net_coarse.train()
            self.nerf_net_fine.train()

            with torch.no_grad():
                batch_test_img_mse = img2mse(torch.from_numpy(rgbs), self.test_image_scaled.cpu())
                batch_test_img_psnr = mse2psnr(batch_test_img_mse)
                self._tensorboard_visualizer.visualize_scalars(global_step, [batch_test_img_psnr, batch_test_img_mse],
                                                               ['Test/Metric/batch_PSNR', 'Test/Metric/batch_MSE'])

            imageio.mimwrite(os.path.join(test_save_dir, 'rgb.mp4'), to8b_np(rgbs), fps=30, quality=8)

            # add rendered image into tensor board
            self._tensorboard_visualizer._summary_writer.add_image('Test/rgb', rgbs, global_step, dataformats='NHWC')

        # Printing progress
        if global_step % self._config["logging"]["step_log_print"] == 0:
            tqdm.write(f"[TRAIN] Iter: {global_step} "
                       f"Loss: {total_loss.item()}, rgb_coarse: {img_loss_coarse.item()}, rgb_fine: {img_loss_fine.item()}, "
                       f"PSNR_coarse: {psnr_coarse.item()}, PSNR_fine: {psnr_fine.item()}")

    def render_path(self, rays, save_dir=None, save_img=True):

        rgbs = []

        t = time.time()
        for i, c2w in enumerate(tqdm(rays)):

            print(i, time.time() - t)
            t = time.time()
            output_dict = self.render_rays(rays[i])

            rgb = output_dict["rgb_fine"]

            rgb = rgb.cpu().numpy().reshape((self.H_scaled, self.W_scaled, 3))

            rgbs.append(rgb)

            if save_dir is not None:
                assert os.path.exists(save_dir)
                rgb8 = to8b_np(rgbs[-1])

                if save_img:
                    rgb_filename = os.path.join(save_dir, 'rgb_{:03d}.png'.format(i))
                    imageio.imwrite(rgb_filename, rgb8)

        rgbs = np.stack(rgbs, 0)

        return rgbs

    def create_video(self, video_name, use_current_models=False):

        ckpt_path = self._config["video"]["prev_ckpt_path"]

        print("here", os.path.exists(ckpt_path))

        if os.path.exists(ckpt_path) and not use_current_models:

            self._train_mode = False

            checkpoint = torch.load(ckpt_path)
            self.nerf_net_coarse.load_state_dict(checkpoint['network_coarse_state_dict'])
            self.nerf_net_fine.load_state_dict(checkpoint['network_fine_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.nerf_net_coarse.eval()
            self.nerf_net_fine.eval()

        elif not os.path.exists(ckpt_path) and not use_current_models:
            print("Specified checkpoint doesn't exist!")
            return

        else:

            self._train_mode = False

            self.nerf_net_coarse.eval()
            self.nerf_net_fine.eval()

        video_save_dir = os.path.join(self._config["experiment"]["save_dir"], "videos")

        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)

        with torch.no_grad():

            poses = generate_new_poses(x=0.0)

            rays = create_rays(poses.shape[0],
                               poses,
                               self.H_scaled,
                               self.W_scaled,
                               self.fx_scaled,
                               self.fy_scaled,
                               self.cx_scaled,
                               self.cy_scaled,
                               self.depth_close_bound,
                               self.depth_far_bound,
                               use_viewdirs=self.use_view_dirs,
                               convention=self._convention)

            rgbs = self.render_path(rays, save_dir=video_save_dir, save_img=False)

            imageio.mimwrite(os.path.join(video_save_dir, video_name), to8b_np(rgbs), fps=30, quality=8)

            self._train_mode = True
            self.nerf_net_coarse.train()
            self.nerf_net_fine.train()
