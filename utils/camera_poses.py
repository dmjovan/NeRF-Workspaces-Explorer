from typing import List

import cv2
import numpy as np
import torch

from utils.data_descriptors import COORD

trans_xyz = lambda x, y, z: np.array([[1, 0, 0, x],
                                      [0, 1, 0, y],
                                      [0, 0, 1, z],
                                      [0, 0, 0, 1]], dtype=np.float32)

yaw_rotation = lambda th: np.array([[np.cos(th), 0, np.sin(th), 0],
                                    [0, 1, 0, 0],
                                    [-np.sin(th), 0, np.cos(th), 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

pitch_rotation = lambda th: np.array([[1, 0, 0, 0],
                                      [0, np.cos(th), -np.sin(th), 0],
                                      [0, np.sin(th), np.cos(th), 0],
                                      [0, 0, 0, 1]], dtype=np.float32)

roll_rotation = lambda th: np.array([[np.cos(th), -np.sin(th), 0, 0],
                                     [np.sin(th), np.cos(th), 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)


def _get_camera_to_world_matrix(coordinates: COORD) -> np.ndarray:
    """
    Converting camera poses into transformation matrix.
    """

    # Rotation matrices
    R_roll = roll_rotation(coordinates.roll / 180.0 * np.pi)
    R_pitch = pitch_rotation(coordinates.pitch / 180.0 * np.pi)
    R_yaw = yaw_rotation(coordinates.yaw / 180.0 * np.pi)

    # Combined rotation matrix
    R = R_roll @ R_pitch @ R_yaw

    # Translation matrix
    T = trans_xyz(coordinates.x, coordinates.y, coordinates.z)

    # Final camera-to-world transformation matrix
    c2w = R @ T

    return c2w


def get_camera_poses_from_list_of_coordinates(init_coordinates: COORD, coordinates: List[COORD]) -> torch.Tensor:
    """
    Getting camera-to-world poses from list of coordinates.
    """

    Ts_c2w = []
    for coord in coordinates:
        extrinsic_matrix = _get_camera_to_world_matrix(init_coordinates).reshape((4, 4))

        # Creating rotation matrices using OpenCV's Rodrigues function
        horizontal_rotation_matrix, _ = cv2.Rodrigues(np.array([0, 0, coord.yaw / 180.0 * np.pi]))
        vertical_rotation_matrix, _ = cv2.Rodrigues(np.array([coord.pitch / 180.0 * np.pi, 0, 0]))

        # Applying the rotations to the existing rotation matrix in the extrinsic matrix
        new_rotation_matrix = horizontal_rotation_matrix @ vertical_rotation_matrix @ extrinsic_matrix[:3, :3]

        # Updating the rotation component in the extrinsic matrix
        extrinsic_matrix[:3, :3] = new_rotation_matrix

        Ts_c2w.append(extrinsic_matrix)

    Ts_c2w = np.asarray(Ts_c2w, dtype=np.float32).reshape((-1, 4, 4))

    return torch.tensor(Ts_c2w)
