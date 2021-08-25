import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

from torchbox.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ExampleDataset(data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.train = (mode == "train")

        if self.train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.dataset = torchvision.datasets.CIFAR10(
            root=cfg.DATA.ROOT, train=self.train, download=True, transform=transform)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def test_dataset():
    from config.defaults import get_cfg
    cfg = get_cfg()
    dataset = ExampleDataset(cfg, mode="train")
    x, y = dataset[4]
    print(x.shape)
    print(y)


def test_dataset_build():
    from config.defaults import get_cfg
    from torchbox.datasets import build_dataset, DATASET_REGISTRY
    print(DATASET_REGISTRY)
    cfg = get_cfg()
    dataset = build_dataset("ExampleDataset", cfg, "train")
    x, y = dataset[4]
    print(x.shape)
    print(y)


if __name__ == '__main__':
    test_dataset()
    test_dataset_build()
