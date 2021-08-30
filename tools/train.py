import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchbox.utils.checkpoint as cu
import torchbox.utils.distributed as du
import torchbox.utils.multiprocessing as mp
import torchbox.models.optimizer as optim

from test.test_model import ResNet18
from test.test_dataset import ExampleDataset

from config.defaults import get_cfg
from torchbox.logger import Logger
from torchbox.models import build_model
from torchbox.utils.metrics import topks_correct
from torchbox.datasets import construct_loader, shuffle_dataset


log = Logger()


def train_epoch(train_loader, model, optimizer, epoch, cfg):
    """Epoch training.

    Args:
      train_loader (DataLoader): training data loader.
      model (model): the video model to train.
      optimizer (optim): the optimizer to perform optimization on the model's parameters.
      epoch (int): current epoch of training.
      cfg (CfgNode): configs. Details can be found in config/defaults.py
    """
    if du.is_master_proc():
       log.info('Epoch: %d' % epoch)

    model.train()
    num_batches = len(train_loader)
    train_loss = 0.0
    correct = total = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda()

        # Update lr.
        lr = optim.get_epoch_lr(epoch + float(batch_idx) / num_batches, cfg)
        optim.set_lr(optimizer, lr)

        # Forward.
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, reduction="mean")

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gather all predictions across all devices.
        if cfg.NUM_GPUS > 1:
           loss = du.all_reduce([loss])[0]
           outputs, labels = du.all_gather([outputs, labels])

        # Accuracy.
        batch_correct = topks_correct(outputs, labels, (1,))[0]
        correct += batch_correct.item()
        total += labels.size(0)

        if du.is_master_proc():
            train_loss += loss.item()
            train_acc = correct / total
            log.info("Loss: %.3f | Acc: %.3f | LR: %.3f" %
                     (train_loss/(batch_idx+1), train_acc, lr))
            log.add_scalar("train_loss", train_loss/(batch_idx+1), batch_idx)
            log.add_scalar("train_acc", train_acc, batch_idx)


@torch.no_grad()
def eval_epoch(val_loader, model, epoch, cfg):
    '''Evaluate the model on the val set.
    Args:
      val_loader (loader): data loader to provide validation data.
      model (model): model to evaluate the performance.
      epoch (int): number of the current epoch of training.
      cfg (CfgNode): configs. Details can be found in config/defaults.py
    '''
    if du.is_master_proc():
        log.info('Testing..')

    model.eval()
    test_loss = 0.0
    correct = total = 0.0
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, reduction="mean")

        # Gather all predictions across all devices.
        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]
            outputs, labels = du.all_gather([outputs, labels])

        # Accuracy.
        batch_correct = topks_correct(outputs, labels, (1,))[0]
        correct += batch_correct.item()
        total += labels.size(0)

        if du.is_master_proc():
            test_loss += loss.item()
            test_acc = correct / total
            log.info("Loss: %.3f | Acc: %.3f" %
                     (test_loss/(batch_idx+1), test_acc))
            log.add_scalar("test_loss", test_loss/(batch_idx+1), batch_idx)
            log.add_scalar("test_acc", test_acc, batch_idx)


def train(cfg):
    train_loader = construct_loader(cfg, mode="train")
    val_loader = construct_loader(cfg, mode="val")
    model = build_model(cfg)
    optimizer = optim.construct_optimizer(model, cfg)

    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        shuffle_dataset(train_loader, epoch)
        train_epoch(train_loader, model, optimizer, epoch, cfg)
        eval_epoch(val_loader, model, epoch, cfg)
        cu.save_checkpoint(model, optimizer, epoch, cfg)


if __name__ == "__main__":
    cfg = get_cfg()
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mp.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                train,
                'tcp://localhost:9999',
                0,  # shard_id
                1,  # num_shards
                'nccl',
                cfg,
            ),
            daemon=False,
        )
    else:
        train(cfg)
