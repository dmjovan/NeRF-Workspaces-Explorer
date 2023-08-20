import os
from abc import ABCMeta, abstractmethod

import numpy as np

from nerf.inference.nerf_replica_inference_handler import NeRFReplicaInferenceHandler
from utils.data_descriptors import HW, COORD

PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..")


class Workspace(metaclass=ABCMeta):

    def __init__(self, name: str, folder_path: str, floor_plan_scale: HW, model_path: str) -> None:
        super().__init__()

        self._name: str = name
        self._folder_path: str = folder_path

        self._floor_plan_scale: HW = floor_plan_scale

        self._model_path: str = os.path.join(PROJECT_PATH, model_path)

        self._nerf_inference = NeRFReplicaInferenceHandler(office_name=self._name.replace(" ", "_").lower(),
                                                           ckpt_path=os.path.normpath(self._model_path))

    def __repr__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name

    @property
    def folder_path(self) -> str:
        return self._folder_path

    @property
    def floor_plan_scale(self) -> HW:
        return self._floor_plan_scale

    @abstractmethod
    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> COORD:
        pass

    def initialize_models(self):
        self._nerf_inference.initialize_models()

    def render_image(self, rel_x: float, rel_y: float, horizontal_angle: int, vertical_angle: int) -> np.ndarray:
        coordinates: COORD = self._transform_relative_coordinates(rel_x, rel_y, horizontal_angle, vertical_angle)

        print(f"Transformed coordinates are: \n{coordinates}\n"
              f"-------------------------------------------------------------")

        image_array = self._nerf_inference.render_coordinates(coordinates)

        return image_array  # H, W, C


class OfficeTokyoWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Tokyo",
                         folder_path="application/workspaces/office_tokyo/",
                         floor_plan_scale=HW(600, 600),
                         model_path="nerf/experiments/office_tokyo/1/checkpoints/100000.ckpt")

        # X, Y, Z coordinates
        self._x_prim_max = 1.5
        self._x_prim_min = -1.2

        self._fixed_y = -0.5

        self._z_prim_max = 1.2
        self._z_prim_min = -0.6

        # Coordinate systems difference angle
        self._angle_diff = -10.0

        # Euler angles
        self._fixed_roll = 180.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> COORD:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_y + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_x + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z,
                     yaw=(-1.0) * hor_angle, pitch=(-1.0) * ver_angle, roll=self._fixed_roll)


class OfficeNewYorkWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office New York",
                         folder_path="application/workspaces/office_new_york/",
                         floor_plan_scale=HW(600, 800),
                         model_path="nerf/experiments/office_new_york/1/checkpoints/200000.ckpt")

        # X, Y, Z coordinates
        self._x_prim_max = 1.8
        self._x_prim_min = -1.2

        self._fixed_y = -0.5

        self._z_prim_max = 1.2
        self._z_prim_min = -1.0

        # Coordinate systems difference angle
        self._angle_diff = 45.0

        # Euler angles
        self._fixed_roll = 180.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> COORD:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_x + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_y + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z,
                     yaw=(-1.0) * hor_angle, pitch=(-1.0) * ver_angle, roll=self._fixed_roll)


class OfficeGeneveWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Geneve",
                         folder_path="application/workspaces/office_geneve/",
                         floor_plan_scale=HW(600, 1000),
                         model_path="nerf/experiments/office_geneve/1/checkpoints/200000.ckpt")

        # X, Y, Z coordinates
        self._x_prim_max = 1.0
        self._x_prim_min = -1.8

        self._fixed_y = -0.5

        self._z_prim_max = 3.3
        self._z_prim_min = -1.4

        # Coordinate systems difference angle
        self._angle_diff = 20.0

        # Euler angles
        self._fixed_roll = 180.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> COORD:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_y + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_x + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z,
                     yaw=(-1.0) * hor_angle, pitch=(-1.0) * ver_angle, roll=self._fixed_roll)


class OfficeBelgradeWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Belgrade",
                         folder_path="application/workspaces/office_belgrade/",
                         floor_plan_scale=HW(600, 750),
                         model_path="nerf/experiments/office_belgrade/1/checkpoints/200000.ckpt")

        # X, Y, Z coordinates
        self._x_prim_max = 3.0
        self._x_prim_min = -0.5

        self._fixed_y = -0.5

        self._z_prim_max = 3.0
        self._z_prim_min = -1.0

        # Coordinate systems difference angle
        self._angle_diff = -10.0

        # Euler angles
        self._fixed_roll = 180.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> COORD:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_y + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_x + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z,
                     yaw=(-1.0) * hor_angle, pitch=(-1.0) * ver_angle, roll=self._fixed_roll)
