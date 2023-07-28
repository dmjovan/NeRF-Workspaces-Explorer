import glob
import os
from enum import Enum
from typing import Dict, Tuple, Union, List

import cv2
import numpy as np
from torch.utils.data import Dataset

from nerf.configs.config_parser import ConfigParser

DATASETS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "replica_dataset")


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"


class ReplicaDataset(Dataset):

    def __init__(self, office_name: str, config: Dict) -> None:

        # Path to the dataset folder
        self._dataset_dir = os.path.join(DATASETS_PATH, office_name, "Sequence_1")

        # Gathering information about image height and width
        self._config_parser = ConfigParser(config)
        self._img_h = self._config_parser.get_param(("experiment", "image_height"), int)
        self._img_w = self._config_parser.get_param(("experiment", "image_width"), int)

        # Camera poses during trajectory execution
        self._traj_file = os.path.join(self._dataset_dir, "traj_w_c.txt")

        # RGB images for each camera pose
        self._rgb_dir = os.path.join(self._dataset_dir, "rgb")

        # Depth for each camera pose
        self._depth_dir = os.path.join(self._dataset_dir, "depth")

        # Gathering indices for train and test datasets
        self._train_ids = list(range(0, len(os.listdir(self._rgb_dir)), 5))
        self._test_ids = [x + 2 for x in self._train_ids]

        # Loading camera poses and reshaping them into matrix Nx4x4
        self._camera_poses = np.loadtxt(self._traj_file, delimiter=" ").reshape(-1, 4, 4)

        # Sorted RGB and depth images
        self._rgb_images = sorted(glob.glob(self._rgb_dir + '/rgb*.png'),
                                  key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self._depth_images = sorted(glob.glob(self._depth_dir + '/depth*.png'),
                                    key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        # Populating TRAIN dataset
        self._train_dataset = {"rgb": [],
                               "depth": [],
                               "camera_pose": []}
        self._populate_dataset_samples(DatasetType.TRAIN)

        # Populating TEST dataset
        self._test_dataset = {"rgb": [],
                              "depth": [],
                              "camera_pose": []}
        self._populate_dataset_samples(DatasetType.TRAIN)

    @property
    def train_dataset(self) -> Dict[str, Union[List, np.ndarray]]:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dict[str, Union[List, np.ndarray]]:
        return self._test_dataset

    @property
    def train_dataset_len(self) -> int:
        return self._train_dataset["rgb"].shape[0] if \
            isinstance(self._train_dataset["rgb"], np.ndarray) else len(self._train_dataset["rgb"])

    @property
    def test_dataset_len(self) -> int:
        return self._test_dataset["rgb"].shape[0] if \
            isinstance(self._test_dataset["rgb"], np.ndarray) else len(self._test_dataset["rgb"])

    def __str__(self) -> str:

        train_dataset_string = f"Dataset length: {self.train_dataset_len}\n"

        for key in self._train_dataset.keys():
            train_dataset_string += f"{key} has shape of {self._train_dataset[key].shape}, " \
                                    f"type {self._train_dataset[key].dtype}\n"

        test_dataset_string = f"Dataset length: {self.test_dataset_len}\n"

        for key in self._test_dataset.keys():
            test_dataset_string += f"{key} has shape of {self._test_dataset[key].shape}, " \
                                   f"type {self._test_dataset[key].dtype}\n"

        dataset_string = f"################################################################################\n" \
                         f"------------------------------- Datasets summary  ------------------------------\n" \
                         f"################################################################################\n" \
                         f"----->---->---->---->---->----> Training dataset <----<----<----<----<----<-----\n" \
                         f"{train_dataset_string}" \
                         f"----->---->---->---->---->----> Testing dataset <-----<----<----<----<----<-----\n" \
                         f"{test_dataset_string}" \
                         f"################################################################################"

        return dataset_string

    def _populate_dataset_samples(self, dataset_type: DatasetType) -> None:

        def get_rgb_and_depth_image_with_index(idx: int) -> Tuple[np.ndarray, np.ndarray]:

            # Changing loaded image from BGR uint8 to RGB float
            rgb_image = cv2.imread(self._rgb_images[idx])[:, :, ::-1] / 255.0

            # Using uint16 mm for depth - scaling with 1/1000 for meters
            depth = cv2.imread(self._depth_images[idx], cv2.IMREAD_UNCHANGED) / 1000.0

            # Running interpolation for image resizing if needed
            if (self._img_h is not None and self._img_h != rgb_image.shape[0]) or (
                    self._img_w is not None and self._img_w != rgb_image.shape[1]):
                rgb_image = cv2.resize(rgb_image, (self._img_w, self._img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self._img_w, self._img_h), interpolation=cv2.INTER_LINEAR)

            return rgb_image, depth

        # training samples
        if dataset_type == DatasetType.TRAIN:
            for idx in self._train_ids:
                # Getting adequate RGB and depth images
                image, depth = get_rgb_and_depth_image_with_index(idx)

                # Getting adequate camera pose
                camera_pose = self._camera_poses[idx]

                self._train_dataset["rgb"].append(image)
                self._train_dataset["depth"].append(depth)
                self._train_dataset["camera_pose"].append(camera_pose)

            # Transforming list of np.ndarrays to array with batch dimension
            for key in self._train_dataset.keys():
                self._train_dataset[key] = np.asarray(self._train_dataset[key])


        elif dataset_type == DatasetType.TEST:

            # test samples
            for idx in self._test_ids:
                # Getting adequate RGB and depth images
                image, depth = get_rgb_and_depth_image_with_index(idx)

                # Getting adequate camera pose
                camera_pose = self._camera_poses[idx]

                self._test_dataset["rgb"].append(image)
                self._test_dataset["depth"].append(depth)
                self._test_dataset["camera_pose"].append(camera_pose)

            # Transforming list of np.ndarrays to array with batch dimension
            for key in self._train_dataset.keys():
                self._train_dataset[key] = np.asarray(self._train_dataset[key])
