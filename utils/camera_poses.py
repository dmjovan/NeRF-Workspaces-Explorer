import numpy as np
import torch

from utils.data_descriptors import COORD

trans_xyz = lambda x, y, z: np.array([[1, 0, 0, x],
                                      [0, 1, 0, y],
                                      [0, 0, 1, z],
                                      [0, 0, 0, 1]], dtype=np.float32)

pitch_rotation = lambda phi: np.array([[1, 0, 0, 0],
                                       [0, np.cos(phi), -np.sin(phi), 0],
                                       [0, np.sin(phi), np.cos(phi), 0],
                                       [0, 0, 0, 1]], dtype=np.float32)

yaw_rotation = lambda th: np.array([[np.cos(th), 0, -np.sin(th), 0],
                                    [0, 1, 0, 0],
                                    [np.sin(th), 0, np.cos(th), 0],
                                    [0, 0, 0, 1]], dtype=np.float32)


def _get_camera_to_world_matrix(coordinates: COORD):
    c2w = trans_xyz(coordinates.x, coordinates.y, coordinates.z)
    c2w = pitch_rotation(coordinates.pitch / 180.0 * np.pi) @ c2w
    c2w = yaw_rotation(coordinates.yaw / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w

    return c2w


def get_camera_pose_from_coordinates(coordinates: COORD):
    c2w = _get_camera_to_world_matrix(coordinates).reshape((4, 4))

    Ts_c2w = np.asarray(c2w, dtype=np.float32).reshape((-1, 4, 4))

    return torch.tensor(Ts_c2w).cpu()
