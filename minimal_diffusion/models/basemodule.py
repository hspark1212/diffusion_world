import torch
from pytorch_lightning import LightningModule

from minimal_diffusion.models.unet import UNet
from minimal_diffusion.utils import save2img, make_gif


class BaseModule(LightningModule):
    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 28,
        learning_rate: float = 1e-3,
        num_samples: int = 10,
        discrete: bool = False,
        num_bins: int = 10,
    ):
        """Base module for minimal diffusion.

        Args:
            img_channels (int, optional): Number of image channels.
                Defaults to 1.
            img_size (int, optional): Size of the image.
                Defaults to 28.
            learning_rate (float, optional): Learning rate for the optimizer.
                Defaults to 1e-3.
            num_samples (int, optional): Number of samples to generate.
                Defaults to 10.
            discrete (bool, optional): Whether the data is discrete.
                Defaults to False.
            num_bins (int, optional): Number of bins for the data.
                Defaults to 10.
        """
        super().__init__()
        self.in_channels = self.out_channels = img_channels
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.discrete = discrete
        self.decoder = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_dims=[64, 128, 256],
            image_size=self.img_size,
            discrete=self.discrete,
            num_bins=num_bins,
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, _):
        imgs, _ = batch
        loss = self.forward(imgs)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        trajectory = self.sample(self.num_samples)
        for t, imgs in enumerate(trajectory):
            if t % 100 == 0:
                save2img(
                    imgs,
                    self.save_dir / f"sample_{self.model_name}_{t}.png",
                )
            elif t == self.num_timesteps - 1:
                save2img(
                    imgs,
                    self.save_dir / f"sample_{self.model_name}_{t}.png",
                )

        make_gif(
            [
                str(self.save_dir / f"sample_{self.model_name}_{t}.png")
                for t in range(0, self.num_timesteps + 1, 100)
            ],
            self.save_dir / f"sample_{self.model_name}_epoch_{self.current_epoch}.gif",
        )

    def validation_step(self, batch, _):
        imgs, _ = batch
        loss = self.forward(imgs)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        imgs, _ = batch
        loss = self.forward(imgs)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.learning_rate)
        return optimizer
