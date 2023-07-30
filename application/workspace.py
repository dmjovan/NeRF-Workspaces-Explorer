from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

from utils.data_descriptors import HW, XYZ


class Workspace(metaclass=ABCMeta):

    def __init__(self, name: str, folder_path: str, floor_plan_scale: HW) -> None:
        super().__init__()

        self._name: str = name
        self._folder_path: str = folder_path

        self._floor_plan_scale: HW = floor_plan_scale

        self._model = None

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
    def _transform_relative_coordinates(self, rel_x: float, rel_y: float) -> XYZ:
        pass

    def render_image(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> np.ndarray:

        coords: XYZ = self._transform_relative_coordinates(rel_x, rel_y)
        print(coords)

        image_array = np.random.randint(low=0, high=255, size=(600, 800, 3), dtype=np.uint8)

        return image_array  # H, W, C


class OfficeTokyoWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Tokyo",
                         folder_path="application/workspaces/office_tokyo/",
                         floor_plan_scale=HW(600, 600))

        self._x_max = 1.5
        self._x_min = -1.2

        self._fixed_y = -0.5

        self._z_max = 1.2
        self._z_min = -0.6

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float) -> XYZ:
        x = (self._x_min - self._x_max) * rel_y + self._x_max
        z = (self._z_min - self._z_max) * rel_x + self._z_max

        return XYZ(x, self._fixed_y, z)


class OfficeNewYorkWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office New York",
                         folder_path="application/workspaces/office_new_york/",
                         floor_plan_scale=HW(600, 800))

        self._fixed_y = -0.5

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float) -> XYZ:
        return XYZ(rel_x, self._fixed_y, rel_y)


class OfficeGeneveWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Geneve",
                         folder_path="application/workspaces/office_geneve/",
                         floor_plan_scale=HW(600, 1000))

        self._fixed_y = -0.5

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float) -> XYZ:
        return XYZ(rel_x, self._fixed_y, rel_y)


class OfficeBelgradeWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Belgrade",
                         folder_path="application/workspaces/office_belgrade/",
                         floor_plan_scale=HW(600, 750))

        self._fixed_y = -0.5

    def _transform_relative_coordinates(self, rel_x: float, rel_y: float) -> XYZ:
        return XYZ(rel_x, self._fixed_y, rel_y)
