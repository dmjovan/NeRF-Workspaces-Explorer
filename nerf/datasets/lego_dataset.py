import os
import numpy as np
from torch.utils.data import Dataset

class LegoDataset(Dataset):

    def __init__(self, config):
        
        self.img_h=config["experiment"]["height"]
        self.img_w=config["experiment"]["width"]

        self.rgb_file = os.path.join(config["experiment"]["dataset_dir"], 'images.npy')
        self.poses_file = os.path.join(config["experiment"]["dataset_dir"], 'poses.npy')
        self.focal_file = os.path.join(config["experiment"]["dataset_dir"], 'focal.npy')

        self.rgb_list = np.load(self.rgb_file).reshape(-1, 100, 100, 3)
        self.poses =  np.load(self.poses_file).reshape(-1, 4, 4)
        self.focal =  np.load(self.focal_file)

        self.train_ids = np.random.randint(low=0, high=self.rgb_list.shape[0], size=self.rgb_list.shape[0]-20)
        self.train_num = len(self.train_ids)

        self.test_ids = list(set(range(0, self.rgb_list.shape[0])).difference(self.train_ids))
        self.test_num = len(self.test_ids)

        self.train_samples = {'image': [], 'depth': [], 'pose': []}
        self.test_samples = {'image': [], 'depth': [], 'pose': []}

       # training samples
        for idx in self.train_ids:
            image = self.rgb_list[idx]
            depth = np.random.uniform(low=config["render"]["depth_range"][0]/config["render"]["depth_range"][1], high=1.0, size=image[:,:,0].shape)
            pose = self.poses[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["pose"].append(pose)

        # test samples
        for idx in self.test_ids:
            image = self.rgb_list[idx]
            depth = np.random.uniform(low=config["render"]["depth_range"][0]/config["render"]["depth_range"][1], size=image[:,:,0].shape)
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