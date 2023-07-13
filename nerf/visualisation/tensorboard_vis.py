import os
import yaml
from torch.utils.tensorboard import SummaryWriter

class TensorboardVisualizer:

    def __init__(self, config):

        log_dir = os.path.join(config["experiment"]["save_dir"], "tensorboard_logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir))
        self.vis_interval = config["logging"]["step_log_tensorboard"]

        # dump args to tensorboard
        args_str = '{}'.format(yaml.dump(config, sort_keys=False, indent=4))
        self.tb_writer.add_text('Exp_args', args_str, 0)

    def visualize_scalars(self, i_iter, losses, names):
        for i, loss in enumerate(losses):
            self.tb_writer.add_scalar(names[i], loss, i_iter)
  
    def visualize_histogram(self, i_iter, value, names):
            self.tb_writer.add_histogram(tag=names, values=value, global_step=i_iter)
