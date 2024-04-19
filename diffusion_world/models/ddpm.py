"""Denoising Diffusion Probabilistic Models (DDPMs)
https://arxiv.org/abs/2006.11239
"""

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from diffusion_world.models.basemodule import BaseModule
from diffusion_world.diff_utils import BetaScheduler


class DDPM(BaseModule):
    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 28,
        learning_rate: float = 1e-3,
        num_timesteps: int = 1000,
        scheduler_mode: str = "linear",
        num_samples: int = 10,
        save_dir: str = "./results",
        discrete: bool = False,
    ):
        """Denoising Diffusion Probabilistic Models (DDPMs)

        Args:
            img_channels (int, optional): Number of image channels.
                Defaults to 1.
            img_size (int, optional): Size of the image.
                Defaults to 28.
            learning_rate (float, optional): Learning rate for the optimizer.
                Defaults to 1e-3.
            num_timesteps (int, optional): Number of timesteps.
                Defaults to 1000.
            scheduler_mode (str, optional): Mode of the scheduler.
                Defaults to "linear".
            num_samples (int, optional): Number of samples to generate.
                Defaults to 10.
            save_dir (str, optional): Directory to save the results.
                Defaults to "./results".
            discrete (bool, optional): Whether the data is discrete.
                Defaults to False.
        """
        super().__init__(img_channels, img_size, learning_rate, num_samples, discrete)
        self.num_timesteps = num_timesteps
        self.scheduler = BetaScheduler(
            timesteps=num_timesteps, scheduler_mode=scheduler_mode
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.discrete = discrete
        assert not self.discrete, "Discrete data is not supported yet."

    @property
    def model_name(self):
        return "ddpm"

    def q_sample(
        self, x_0: torch.Tensor, batched_t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample from q(x_t | x_start) (i.e. add noise to the data).
        q(x_t | x_0) = N(x_t | sqrt(alpha_cumprod) * x_0, sqrt(1 - alpha_cumprod) * I)

        Args:
            x_0 (torch.Tensor): [B, C, H, W]
            batched_t (torch.Tensor): [B]
            noise (torch.Tensor): [B, C, H, W]

        Returns:
            torch.Tensor: [B, C, H, W]
        """
        # coefficients
        alpha_cumprod = self.scheduler.alphas_cumprod[batched_t][
            :, None, None, None
        ]  # [B, 1, 1, 1]

        return (
            torch.sqrt(alpha_cumprod) * x_0 + torch.sqrt((1 - alpha_cumprod)) * noise
        )  # [B, C, H, W]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Calculate P(x_t | x_0) loss.

        The forward process of DDPM is defined as:
        x_t ~ q(x_t | x_0) = N(x_t | sqrt(alpha_cumprod) * x_0, sqrt(1 - alpha_cumprod) * I)
        where alpha_cumprod is the cumulative product of the schedule alphas.

        Args:
            imgs (torch.Tensor): [B, C, H, W]
        """
        batch_size = imgs.shape[0]
        # Sample a random timestep for each image in the batch
        batched_t = self.scheduler.uniform_sample_t(batch_size, self.device)
        # pure noise image at t=T
        noise = torch.randn_like(imgs)  # [B, C, H, W]
        # q_sample
        x_t = self.q_sample(imgs, batched_t, noise)  # [B, C, H, W]
        # predict noise at t
        pred_noise = self.decoder(x_t, batched_t)  # [B, C, H, W]
        # loss
        loss = F.mse_loss(pred_noise, noise)  # [B, C, H, W]
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int):
        """Sample from the model.

        Args:
            num_samples: Number of samples to generate.
        """
        # pure noise image at t=T
        rand_imgs = torch.randn(
            num_samples, self.in_channels, self.img_size, self.img_size
        ).to(self.device)
        x_t = rand_imgs.clone()  # [B, C, H, W]

        # reverse sampling
        trajectory = [x_t]
        for t in tqdm(range(self.num_timesteps, 0, -1), desc="Sampling"):
            batched_t = torch.full(
                (num_samples,), t, dtype=torch.long, device=self.device
            )
            # coefficients
            alpha = self.scheduler.alphas[batched_t][:, None, None, None]
            alpha_cumprod = self.scheduler.alphas_cumprod[batched_t][
                :, None, None, None
            ]
            sigma = self.scheduler.sigmas[batched_t][:, None, None, None]

            # predict noise at t
            pred_noise = self.decoder(x_t, batched_t)  # [B, C, H, W]
            model_mean = (1 / torch.sqrt(alpha)) * (
                x_t - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * pred_noise
            )
            # x_{t-1} = model_mean + model_variance * noise
            x_t = model_mean + sigma * torch.randn_like(rand_imgs)
            trajectory.append(x_t)

        return trajectory
