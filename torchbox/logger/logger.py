import glog as log
import torchbox.utils.distributed as du

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self):
        if du.is_master_proc():
            self.writer = SummaryWriter()

    def info(self, s):
        if du.is_master_proc():
            log.info(s)

    def add_scalar(self, tag, value, step):
        if du.is_master_proc():
            self.writer.add_scalar(tag, value, step)


if __name__ == "__main__":
    logger = Logger()
    log.info("aaa")
