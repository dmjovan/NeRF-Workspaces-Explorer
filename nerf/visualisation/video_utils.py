import torch
import numpy as np


trans_xyz = lambda x, y, z: np.array([[1, 0, 0, x],
                                      [0, 1, 0, y],
                                      [0, 0, 1, z],
                                      [0, 0, 0, 1]], dtype=np.float32)

rot_phi = lambda phi: np.array([[1, 0, 0, 0],
                                [0, np.cos(phi), -np.sin(phi), 0],
                                [0, np.sin(phi), np.cos(phi), 0],
                                [0, 0, 0, 1]], dtype=np.float32)

rot_theta = lambda th: np.array([[np.cos(th), 0, -np.sin(th), 0],
                                 [0, 1, 0, 0],
                                 [np.sin(th), 0, np.cos(th), 0],
                                 [0, 0, 0, 1]], dtype=np.float32)

def pose_spherical(x, y, z, theta, phi):
    c2w = trans_xyz(x, y, z)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w

    return c2w

def generate_new_poses(num_samples = 120, x = 0.0, y = 0.0, z = 0.0, phi = 0.0):

    print("Generating new poses for video")

    Ts_c2w = []
    
    for th in np.linspace(0., 360., num_samples, endpoint=False):
        c2w = pose_spherical(x, y, z, th, phi).reshape((4, 4))
        Ts_c2w.append(c2w)

    Ts_c2w = np.asarray(Ts_c2w, dtype=np.float32).reshape((num_samples, 4, 4))
    
    print("Generated new poses for video finished")

    return torch.tensor(Ts_c2w).cpu()