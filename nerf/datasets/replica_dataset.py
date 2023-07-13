import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

class ReplicaDataset(Dataset):

    def __init__(self, config):

        self.traj_file = os.path.join(config["experiment"]["dataset_dir"], "traj_w_c.txt")
        self.rgb_dir = os.path.join(config["experiment"]["dataset_dir"], "rgb")
        self.depth_dir = os.path.join(config["experiment"]["dataset_dir"], "depth")

        self.train_ids = list(range(0, len(os.listdir(self.rgb_dir)), 5))
        self.train_num = len(self.train_ids)

        self.test_ids = [x + 2 for x in self.train_ids]
        self.test_num = len(self.test_ids)

        self.img_h=config["experiment"]["height"]
        self.img_w=config["experiment"]["width"]

        self.poses = np.loadtxt(self.traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.train_samples = {'image': [], 'depth': [], 'pose': []}
        self.test_samples = {'image': [], 'depth': [], 'pose': []}

       # training samples
        for idx in self.train_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter

            if (self.img_h is not None and self.img_h != image.shape[0]) or (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

            pose = self.poses[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["pose"].append(pose)

        # test samples
        for idx in self.test_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter

            if (self.img_h is not None and self.img_h != image.shape[0]) or (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                
            pose = self.poses[idx]

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["pose"].append(pose)

        for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        print("#####################################################################")
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print(f"{key} has shape of {self.train_samples[key].shape}, type {self.train_samples[key].dtype} ")
        
        print("#####################################################################")
        print("Testing Sample Summary:")
        for key in self.test_samples.keys(): 
            print(f"{key} has shape of {self.test_samples[key].shape}, type {self.test_samples[key].dtype}")