import os
import time

import yaml

from nerf.training.trainer import NeRFTrainer

if __name__ == "__main__":

    # read YAML file
    with open("configs/office0_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Creating trainer for NeRF algorithm
    nerf_trainer = NeRFTrainer(config)

    N_iters = int(config["train"]["N_iters"]) + 1

    print("###############################################################################")
    print("-------------------------- Begining of training loop -------------------------")

    for i in range(0, N_iters):
        step_start_time = time.time()
        nerf_trainer.step(i)
        step_end_time = time.time()

        step_duration = step_end_time - step_start_time
        print(f"Finished step: {i}/{N_iters} --> Step duration: {step_duration}")
