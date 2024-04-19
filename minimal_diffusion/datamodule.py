from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from minimal_diffusion.dataset import MinimalDiffusionDataset


class Datamodule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "mnist",
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 8,
        discrete: bool = False,
        num_bins: int = 10,
    ):
        """DataModule for minimal diffusion.

        Args:
            dataset_name (str, optional): Name of the dataset. Either "mnist" or "cifar10".
            data_dir (str, optional): Directory to store the dataset.
            batch_size (int, optional): Batch size for the dataloaders.
            num_workers (int, optional): Number of workers for the dataloaders.
            discrete (bool, optional): Whether to use discrete or continuous data.
            num_bins (int, optional): Number of bins for discretization.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.discrete = discrete
        self.num_bins = num_bins

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = MinimalDiffusionDataset(
                self.dataset_name,
                self.data_dir,
                train=True,
                discrete=self.discrete,
                num_bins=self.num_bins,
            )

            self.val_dataset = MinimalDiffusionDataset(
                self.dataset_name,
                self.data_dir,
                train=False,
                discrete=self.discrete,
                num_bins=self.num_bins,
            )

            self.test_dataset = MinimalDiffusionDataset(
                self.dataset_name,
                self.data_dir,
                train=False,
                discrete=self.discrete,
                num_bins=self.num_bins,
            )
        elif stage == "test" or stage == "validate":
            self.val_dataset = MinimalDiffusionDataset(
                self.dataset_name,
                self.data_dir,
                train=False,
                discrete=self.discrete,
                num_bins=self.num_bins,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
