import torch

from torchbox.datasets.build import build_dataset
from torch.utils.data.distributed import DistributedSampler


def construct_loader(cfg, mode):
    """Construct data loader.

    Args:
      cfg (CfgNode): configs.
      mode: (str) train/val/test.
    """
    train = (mode == "train")

    # Perform shuffle & drop_last only during training.
    shuffle = drop_last = train

    # Construct dataset.
    dataset = build_dataset(cfg.DATA.DATASET, cfg, mode)

    # Create a sampler for multi-process training.
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    # Create a loader.
    batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None)
    return loader


def shuffle_dataset(loader, epoch):
    """Shuffle the dataset.

    Args:
      loader (DataLoader): data loader to perform shuffle.
      epoch (int): current epoch.
    """
    # RandomSampler handles shuffling automatically.
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch.
        loader.sampler.set_epoch(epoch)


if __name__ == "__main__":
    from config.defaults import get_cfg
    from test.test_dataset import ExampleDataset
    cfg = get_cfg()
    dataloader = construct_loader(cfg, "train")
    for inputs, labels in dataloader:
       print(inputs.shape)
       print(labels)
       break
