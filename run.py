from pathlib import Path
from fire import Fire

import pytorch_lightning as pl

from minimal_diffusion.datamodule import Datamodule
from minimal_diffusion.models import DDPM, D3PM


def main(
    model_name: str = "ddpm",  # "ddpm", "dsm", "d3pm"
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 8,
    discrete: bool = False,
    num_bins: int = 10,
    num_timesteps: int = 1000,
    scheduler_mode: str = "linear",  # "linear", "quadratic", "sigmoid", "jsd"
    num_samples: int = 10,
    save_dir: str = "./results",
    max_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda",
):
    pl.seed_everything(42)
    save_dir = Path(save_dir)

    # set Datamodule
    dm = Datamodule(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        discrete=discrete,
        num_bins=num_bins,
    )

    # set model
    if dataset_name == "mnist":
        img_channels = 1
        img_size = 28
    elif dataset_name == "cifar10":
        img_channels = 3
        img_size = 32
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")
    model_name = model_name.lower()
    if model_name == "ddpm":
        model = DDPM(
            img_channels=img_channels,
            img_size=img_size,
            learning_rate=learning_rate,
            num_timesteps=num_timesteps,
            scheduler_mode=scheduler_mode,
            num_samples=num_samples,
            save_dir=save_dir,
        )
    elif model_name == "dsm":
        model = None

    elif model_name == "d3pm":
        model = D3PM(
            img_channels=img_channels,
            img_size=img_size,
            learning_rate=learning_rate,
            num_timesteps=num_timesteps,
            scheduler_mode=scheduler_mode,
            num_samples=num_samples,
            save_dir=save_dir,
            discrete=discrete,
            num_bins=num_bins,
        )
    else:
        raise ValueError(f"Model name {model_name} is not supported.")

    # set trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1 if device == "cuda" else 0,
        default_root_dir=save_dir / model_name,
    )

    # train
    trainer.fit(model, dm)


if __name__ == "__main__":
    Fire(main)
