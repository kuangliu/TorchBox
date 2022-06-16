import glog as log
import torchbox.utils.distributed as du

from torch.utils.tensorboard import SummaryWriter


def master_proc_check(func):
    def wrapper(ref, *args, **kwargs):
        if du.is_master_proc():
            if ref.writer is None:
                ref.writer = SummaryWriter()
            func(ref, *args, **kwargs)
    return wrapper


class Logger:
    def __init__(self):
        self.writer = None

    @master_proc_check
    def info(self, s):
        log.info(s)

    @master_proc_check
    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)


if __name__ == "__main__":
    logger = Logger()
    logger.info("aaa")
