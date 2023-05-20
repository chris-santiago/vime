import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, transforms

from vime import Constants


class MnistData(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = Constants.DATA, batch_size: int = 128, n_workers: int = 1
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.flatten = Compose(
            [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
        )

    def prepare_data(self) -> None:
        MNIST(Constants.DATA, train=True, download=True, transform=self.flatten)
        MNIST(
            Constants.DATA, train=False, download=True, transform=transforms.ToTensor()
        )

    def setup(self, stage: str) -> None:
        rng = torch.Generator().manual_seed(Constants.SEED)
        if stage == "fit":
            data = MNIST(Constants.DATA, train=True, transform=self.flatten)
            self.train, self.valid = random_split(
                dataset=data, lengths=[55_000, 5_000], generator=rng
            )
        if stage == "test":
            self.test = MNIST(
                Constants.DATA, train=False, transform=transforms.ToTensor()
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.n_workers
        )
