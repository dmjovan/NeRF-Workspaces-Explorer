import argparse
import os
import time

import yaml

from nerf.training.nerf_replica_training_handler import NeRFReplicaTrainingHandler

AVAILABLE_OFFICES = ("tokyo", "new_york", "geneve", "belgrade")

if __name__ == "__main__":

    # Reading office name to run training on its data
    parser = argparse.ArgumentParser()
    parser.add_argument("--office", type=str, default="tokyo", dest="office")
    args = parser.parse_args()

    office_name = (str(args.office).lower().strip()).replace(" ", "_")

    if office_name not in AVAILABLE_OFFICES:
        raise RuntimeError(f"Office {office_name} not available for training.")

    # Reading YAML config file
    with open(f"nerf/configs/office_{office_name}_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Creating trainer for NeRF algorithm for Replica dataset
    nerf_trainer = NeRFReplicaTrainingHandler(f"office_{office_name}", config)

    # Preparing the data from datasets
    nerf_trainer.prepare_data()

    # Creating and initializing the NeRF models
    nerf_trainer.initialize_models()

    # Initializing rays for training, testing and visualization for NeRF models (in world coordinates)
    nerf_trainer.initialize_rays()

    # Gathering number of training iterations
    num_iteration = int(config["training"]["n_iterations"])

    print("################################################################################")
    print("------------------------------- Training loop ----------------------------------")
    print("################################################################################")

    for i in range(0, num_iteration):
        step_start_time = time.time()

        # Running one step of model training
        nerf_trainer.step(i)

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        print(f"Finished step: {i + 1}/{num_iteration} --> Step duration: {step_duration} sec")
