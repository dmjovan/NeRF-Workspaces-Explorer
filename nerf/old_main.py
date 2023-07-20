import yaml
import argparse
import time
import os

from nerf.training.trainer import NeRFTrainer, DATASET_TO_CONFIG_PATH


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", 
                        type=str, 
                        default="replica", 
                        choices= ["replica", "lego", "custom"], 
                        help="the dataset to be used")

    parser.add_argument("--video", 
                        type=str, 
                        default="true",
                        help="create video initially from previous models")

    args = parser.parse_args()

    # read YAML file
    with open(DATASET_TO_CONFIG_PATH[args.dataset], "r") as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Creating trainer for NeRF algorithm
    nerf_trainer = NeRFTrainer(args.dataset, config)

    print("########################################################################################")
    print(f"------------------------------- Using {args.dataset.capitalize()} Dataset ------------------------------")

    if args.video.lower() == "true":

        print("------------------------------- Creating initial video --------------------------------")
        nerf_trainer.create_video("video_init.mp4")

        print("Video created")

    N_iters = int(config["train"]["N_iters"]) + 1

    print("###############################################################################")
    print("-------------------------- Begining of training loop -------------------------")

    try:
        for i in range(0, N_iters):

            step_start_time = time.time()
            nerf_trainer.step(i)
            step_end_time = time.time()

            step_duration = step_end_time - step_start_time
            print(f"Finished step: {i}/{N_iters} --> Step duration: {step_duration}")

        print("Training finished")

        print("------------------------------- Creating video --------------------------------")
        nerf_trainer.create_video("video_final.mp4", use_current_models=True)

        print("Video created")

    except KeyboardInterrupt:
        print("###############################################################################")
        print("---------------------------- Training Interupted ------------------------------")

        print("------------------------------- Creating video --------------------------------")
        nerf_trainer.create_video("video_final_training_interupted.mp4", use_current_models=True)

        print("Video created")
