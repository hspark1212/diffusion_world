from typing import Tuple

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms as T


class MinimalDiffusionDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "mnist",
        data_dir: str = "./data",
        train: bool = True,
        discrete: bool = False,
        num_bins: int = 10,
    ):
        """MNIST dataset.

        Args:
            dataset_name (str, optional): Name of the dataset. Either "mnist" or "cifar10".
            data_dir (str, optional): Directory to store the dataset.
            discrete (bool, optional): Whether to use discrete or continuous data.
            num_bins (int, optional): Number of bins for discretization.
        """
        if discrete:
            transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.Lambda(
                        lambda x: (x * (num_bins - 1))
                        .round()
                        .long()
                        .clamp(0, num_bins - 1)
                    ),
                ]
            )
        else:
            transforms = T.ToTensor()
        if dataset_name == "mnist":
            self.dataset = torchvision.datasets.MNIST(
                root=data_dir,
                train=train,
                transform=transforms,
                download=True,
            )
        elif dataset_name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_dir,
                train=train,
                transform=transforms,
                download=True,
            )
        else:
            raise ValueError("Dataset name should be either 'mnist' or 'cifar10'.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        return img, label
