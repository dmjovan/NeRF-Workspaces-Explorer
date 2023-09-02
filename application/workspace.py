import os
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from nerf.inference.nerf_replica_inference_handler import NeRFReplicaInferenceHandler
from utils.data_descriptors import HW, COORD

PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..")


class Workspace(metaclass=ABCMeta):

    def __init__(self, name: str, floor_plan_scale: HW) -> None:
        super().__init__()

        self._name: str = name
        self._floor_plan_scale: HW = floor_plan_scale

        self._office_name = self._name.replace(" ", "_").lower()
        self._folder_path: str = os.path.normpath(
            os.path.join(PROJECT_PATH, "application", "workspaces", f"{self._office_name}"))

        self._model_path: str = os.path.normpath(
            os.path.join(PROJECT_PATH, "nerf", "final_models", self._office_name, "model.ckpt"))

        self._nerf_inference = NeRFReplicaInferenceHandler(office_name=self._office_name,
                                                           ckpt_path=self._model_path)

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
    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> Tuple[
        COORD, COORD]:
        pass

    def initialize_models(self):
        self._nerf_inference.initialize_models()

    def render_image(self, rel_x: float, rel_y: float, horizontal_angle: int, vertical_angle: int) -> np.ndarray:
        init_coordinates, coordinates = self._transform_relative_coordinates(rel_x, rel_y, horizontal_angle,
                                                                             vertical_angle)

        print(f"Virtual camera coordinates and orientation: \n{init_coordinates}\n"
              f"-------------------------------------\n"
              f"Virtual camera local orientation: \n"
              f"yaw (left-right): {coordinates.yaw:.3f}\n"
              f"pitch (up-down): {coordinates.pitch:.3f}\n"
              f"roll (twist): {coordinates.roll:.3f}\n"
              f"-------------------------------------------------------------")

        image_array = self._nerf_inference.render_coordinates(init_coordinates, coordinates)

        return image_array  # H, W, C


class OfficeTokyoWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Tokyo", floor_plan_scale=HW(600, 600))

        # X, Y, Z coordinates
        self._x_prim_max = 2.0
        self._x_prim_min = -2.0

        self._fixed_y = -0.5

        self._z_prim_max = 1.5
        self._z_prim_min = -3.0

        # Coordinate systems difference angle
        self._angle_diff = -10.0

        # Euler angles
        self._init_pitch = -90.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> Tuple[
        COORD, COORD]:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_y + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_x + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z, yaw=0.0, pitch=self._init_pitch, roll=0.0), \
               COORD(x=0.0, y=0.0, z=0.0, yaw=-float(hor_angle), pitch=float(ver_angle), roll=0.0)


class OfficeNewYorkWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office New York", floor_plan_scale=HW(600, 800))

        # X, Y, Z coordinates
        self._x_prim_max = 1.8
        self._x_prim_min = -1.2

        self._fixed_y = -0.5

        self._z_prim_max = 2.0
        self._z_prim_min = -1.6

        # Coordinate systems difference angle
        self._angle_diff = 45.0

        # Euler angles
        self._init_pitch = -90.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> Tuple[
        COORD, COORD]:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_x + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_y + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z, yaw=0.0, pitch=self._init_pitch, roll=0.0), \
               COORD(x=0.0, y=0.0, z=0.0, yaw=-float(hor_angle), pitch=float(ver_angle), roll=0.0)


class OfficeGeneveWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Geneve", floor_plan_scale=HW(600, 1000))

        # X, Y, Z coordinates
        self._x_prim_max = 1.7
        self._x_prim_min = -2.5

        self._fixed_y = -0.5

        self._z_prim_max = 4.2
        self._z_prim_min = -2.8

        # Coordinate systems difference angle
        self._angle_diff = 35.0

        # Euler angles
        self._init_pitch = -90.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> Tuple[
        COORD, COORD]:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_y + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_x + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z, yaw=0.0, pitch=self._init_pitch, roll=0.0), \
               COORD(x=0.0, y=0.0, z=0.0, yaw=-float(hor_angle), pitch=float(ver_angle), roll=0.0)


class OfficeBelgradeWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Belgrade", floor_plan_scale=HW(600, 750))

        # X, Y, Z coordinates
        self._x_prim_max = 4.7
        self._x_prim_min = -0.7

        self._fixed_y = -0.5

        self._z_prim_max = 3.5
        self._z_prim_min = -2.3

        # Coordinate systems difference angle
        self._angle_diff = -10.0

        # Euler angles
        self._init_pitch = -90.0

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, hor_angle: int, ver_angle: int) -> Tuple[
        COORD, COORD]:
        x_prim = (self._x_prim_min - self._x_prim_max) * rel_y + self._x_prim_max
        z_prim = (self._z_prim_min - self._z_prim_max) * rel_x + self._z_prim_max

        x = x_prim / np.cos(self._angle_diff / 180.0 * np.pi)
        z = z_prim / np.cos(self._angle_diff / 180.0 * np.pi)

        return COORD(x=x, y=self._fixed_y, z=z, yaw=0.0, pitch=self._init_pitch, roll=0.0), \
               COORD(x=0.0, y=0.0, z=0.0, yaw=-float(hor_angle), pitch=float(ver_angle), roll=0.0)
