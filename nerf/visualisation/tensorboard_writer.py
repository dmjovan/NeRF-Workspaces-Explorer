import os
from typing import Dict

import yaml
from torch.utils.tensorboard import SummaryWriter

from nerf.configs.config_parser import ConfigParser


class TensorboardWriter:
    """
    Class for handling data subscription on to the Tensorboard.
    """

    def __init__(self, experiment_dir: str, config: Dict) -> None:

        # Setting the logging path
        self._log_dir = os.path.join(experiment_dir, "tensorboard_logs")
        os.makedirs(self._log_dir, exist_ok=True)

        # Creating SummaryWriter object
        self.summary_writer = SummaryWriter(log_dir=self._log_dir)

        # Getting logging interval
        self._config_parser = ConfigParser(config)
        self.log_interval = self._config_parser.get_param(("logging", "step_log_tensorboard"), int)

        # Adding experiment arguments into Tensorboard
        self.summary_writer.add_text("Experiment arguments", str(yaml.dump(config, sort_keys=False, indent=4)), 0)

    def write_scalars(self, i_iter: int, losses, names):
        for i, loss in enumerate(losses):
            self.summary_writer.add_scalar(names[i], loss, i_iter)

    def write_histogram(self, i_iter: int, value, names):
        self.summary_writer.add_histogram(tag=names, values=value, global_step=i_iter)
