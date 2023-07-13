from abc import ABCMeta
from typing import Tuple

import numpy as np


class Workspace(metaclass=ABCMeta):

    def __init__(self, name: str, folder_path: str, thumbnail_size: Tuple[int, int],
                 floor_plan_size: Tuple[int, int], floor_plan_scale: Tuple[int, int]) -> None:
        self._name: str = name
        self._folder_path: str = folder_path

        self._thumbnail_size: Tuple[int, int] = thumbnail_size  # H, W
        self._floor_plan_size: Tuple[int, int] = floor_plan_size  # H, W
        self._floor_plan_scale: Tuple[int, int] = floor_plan_scale  # H, W

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
    def thumbnail_size(self) -> Tuple[int, int]:
        return self._thumbnail_size

    @property
    def floor_plan_size(self) -> Tuple[int, int]:
        return self._floor_plan_size

    @property
    def floor_plan_scale(self) -> Tuple[int, int]:
        return self._floor_plan_scale

    def render_image(self, rel_x: float, rel_y: float, yaw: int, pitch: int) -> np.ndarray:
        image_array = np.random.randint(low=0, high=255, size=(600, 800, 3), dtype=np.uint8)

        return image_array  # H, W, C


class OfficeTokyoWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Tokyo",
                         folder_path="application/workspaces/office_tokyo/",
                         thumbnail_size=(239, 308),
                         floor_plan_size=(794, 786),
                         floor_plan_scale=(600, 600))

        self.scale = 800


class OfficeNewYorkWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office New York",
                         folder_path="application/workspaces/office_new_york/",
                         thumbnail_size=(243, 344),
                         floor_plan_size=(802, 1072),
                         floor_plan_scale=(600, 800))

        self.scale = 1000


class OfficeGeneveWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Geneve",
                         folder_path="application/workspaces/office_geneve/",
                         thumbnail_size=(257, 349),
                         floor_plan_size=(807, 1380),
                         floor_plan_scale=(600, 1000))

        self.scale = 1000


class OfficeBelgradeWorkspace(Workspace):

    def __init__(self) -> None:
        super().__init__(name="Office Belgrade",
                         folder_path="application/workspaces/office_belgrade/",
                         thumbnail_size=(262, 348),
                         floor_plan_size=(890, 1102),
                         floor_plan_scale=(600, 750))

        self.scale = 1000
