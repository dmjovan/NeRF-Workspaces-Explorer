from typing import Tuple

import torch


def create_rays(num_images: int, Ts_c2w: torch.Tensor, height: int, width: int, fx: float, fy: float, cx: float,
                cy: float, near: float, far: float, use_view_dirs: bool = True):
    """
    Convention details: "opencv" or "opengl".
    It defines the coordinates convention of rays from cameras.
    OpenCV defines x,y,z as right, down, forward while OpenGL defines x,y,z as right, up, backward
    (camera looking towards forward direction still, -z!).

    Note: Use either convention is fine, but the corresponding pose should follow the same convention.
    """

    print("Preparing rays")

    rays_cam = _get_rays_camera(num_images, height, width, fx, fy, cx, cy)  # [N, H, W, 3]

    dirs_C = rays_cam.view(num_images, -1, 3)  # [N, HW, 3]
    rays_o, rays_d = _get_rays_world(Ts_c2w, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

    if use_view_dirs:
        # Providing ray directions as input
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_view_dirs:
        rays = torch.cat([rays, view_dirs], -1)

    return rays


def _get_rays_camera(B: int, H: int, W: int, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """
    Getting rays from camera perspective
    """

    # Pytorch's meshgrid has indexing "ij", by transposing it we get "xy"
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H))
    i = i.t().float()
    j = j.t().float()

    size = [B, H, W]

    i_batch = torch.empty(size)
    j_batch = torch.empty(size)
    i_batch[:, :, :] = i[None, :, :]
    j_batch[:, :, :] = j[None, :, :]

    x = (i_batch - cx) / fx
    y = (j_batch - cy) / fy
    z = torch.ones(size)

    dirs = torch.stack((x, y, z), dim=3)  # shape of [B, H, W, 3], 3 comes from x, y, z not channels,
    # channels contain redundant info for spa
    return dirs


def _get_rays_world(T_WC: torch.Tensor, dirs_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Getting rays in world coordinates
    """

    R_WC = T_WC[:, :3, :3]  # Bx3x3
    dirs_W = torch.matmul(R_WC[:, None, ...], dirs_C[..., None]).squeeze(-1)
    origins = T_WC[:, :3, -1]  # Bx3
    origins = torch.broadcast_tensors(origins[:, None, :], dirs_W)[0]

    return origins, dirs_W


def sample_pdf(bins, weights, N_samples, det=False):
    """
    Hierarchical sampling using inverse CDF transformations.
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: N_rays x (N_samples_coarse - 1)
        weights: N_rays x (N_samples_coarse - 2)
        N_samples: N_samples_fine
        det: deterministic or not
    """

    # Get pdf
    weights = weights + 1e-5  # prevent nans, prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)  # N_rays x (N_samples - 2)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # N_rays x (N_samples_coarse - 1)
    # padded to 0~1 inclusive, (N_rays, N_samples-1)

    # Take uniform samples
    if det:  # generate deterministic samples
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)
        # (N_rays, N_samples_fine)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # N_rays x N_samples_fine
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples_fine, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # (N_rays, N_samples_fine, N_samples_coarse - 1)

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # N_rays, N_samples_fine, 2
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)  # N_rays, N_samples_fine, 2

    denom = (cdf_g[..., 1] - cdf_g[..., 0])  # # N_rays, N_samples_fine
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
