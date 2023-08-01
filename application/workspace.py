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
    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> COORD:
        pass

    def render_image(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> np.ndarray:

        coords: COORD = self._transform_relative_coordinates(rel_x, rel_y, yaw, pitch)

        image_array = self._nerf_inference.render_coordinates(coords)

        return image_array  # H, W, C


class OfficeTokyoWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Tokyo",
                         folder_path="application/workspaces/office_tokyo/",
                         floor_plan_scale=HW(600, 600),
                         model_path="nerf/experiments/office_tokyo/1/checkpoints/100000.ckpt")

        self._x_max = 1.5
        self._x_min = -1.2

        self._fixed_y = -0.5

        self._z_max = 1.2
        self._z_min = -0.6

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> COORD:
        x = (self._x_min - self._x_max) * rel_y + self._x_max
        z = (self._z_min - self._z_max) * rel_x + self._z_max

        return COORD(x=x, y=self._fixed_y, z=z, yaw=float(yaw), pitch=float(pitch))


class OfficeNewYorkWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office New York",
                         folder_path="application/workspaces/office_new_york/",
                         floor_plan_scale=HW(600, 800),
                         model_path="nerf/experiments/office_new_york/1/checkpoints/200000.ckpt")

        self._fixed_y = -0.5

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> COORD:
        return COORD(x=rel_x, y=self._fixed_y, z=rel_y, yaw=float(yaw), pitch=float(pitch))


class OfficeGeneveWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Geneve",
                         folder_path="application/workspaces/office_geneve/",
                         floor_plan_scale=HW(600, 1000),
                         model_path="")

        self._fixed_y = -0.5

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> COORD:
        return COORD(x=rel_x, y=self._fixed_y, z=rel_y, yaw=float(yaw), pitch=float(pitch))


class OfficeBelgradeWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Belgrade",
                         folder_path="application/workspaces/office_belgrade/",
                         floor_plan_scale=HW(600, 750),
                         model_path="")

        self._fixed_y = -0.5

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> COORD:
        return COORD(x=rel_x, y=self._fixed_y, z=rel_y, yaw=float(yaw), pitch=float(pitch))
