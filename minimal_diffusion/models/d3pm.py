"""Discrete Denoising Diffusion Probabilistic Models (D3PMs)
https://arxiv.org/abs/2107.03006
"""

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from minimal_diffusion.models.basemodule import BaseModule
from minimal_diffusion.diff_utils import BetaScheduler


class D3PM(BaseModule):
    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 28,
        learning_rate: float = 1e-3,
        num_timesteps: int = 1000,
        scheduler_mode: str = "jsd",
        num_samples: int = 10,
        save_dir: str = "./results",
        discrete: bool = True,
        num_bins: int = 10,
        hybrid_coeff: float = 0.01,
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
                Defaults to "jsd".
            num_samples (int, optional): Number of samples to generate.
                Defaults to 10.
            save_dir (str, optional): Directory to save the results.
                Defaults to "./results".
            discrete (bool, optional): Whether the data is discrete.
                Defaults to True.
            num_bins (int, optional): Number of pixel values.
                Defaults to 256.
            hybrid_coeff (float, optional): Hybrid coefficient for loss.
                Defaults to 0.001.
        """
        super().__init__(img_channels, img_size, learning_rate, num_samples, discrete)
        self.num_timesteps = num_timesteps
        self.scheduler = BetaScheduler(
            timesteps=num_timesteps, scheduler_mode=scheduler_mode
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.discrete = discrete
        assert self.discrete, "Continuous data is not supported yet."

        self.eps = 1.0e-6
        self.num_bins = num_bins
        self.hybrid_coeff = hybrid_coeff

        # transition matrix for absorbing
        q_one_step_mats = torch.stack(
            [
                self._get_absorbing_transition_mat(t)
                for t in range(0, num_timesteps + 1)
            ],
            dim=0,
        )
        self.register_buffer("q_one_step_mats", q_one_step_mats)

        # construct transition matrices for q(x_t | x_0)
        q_mat_t = self.q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, num_timesteps + 1):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = q_mat_t @ self.q_one_step_mats[t]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            num_timesteps + 1,
            self.num_bins,
            self.num_bins,
        )

        self.q_one_step_transposed = self.q_one_step_mats.transpose(1, 2)

    @property
    def model_name(self):
        return "d3pm"

    def _get_absorbing_transition_mat(self, t: int):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_bins//2.

        Args:
          t (int): timestep.

        Returns:
          Q_t: transition matrix. shape = (num_bins, num_bins).
        """
        # get beta at timestep t
        beta_t = self.scheduler.betas[t]

        diag = torch.full((self.num_bins,), 1 - beta_t)
        mat = torch.diag(diag, 0)
        # Add beta_t to the num_bins//2-th diagonal entry.
        mat[:, self.num_bins // 2] += beta_t
        # ADD beta_t at final column
        # mat[:, 0] += beta_t
        return mat

    def _at(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
            a (torch.Tensor): matrix of coefficients. [num_timesteps, num_bins, num_bins]
            t (torch.Tensor): timesteps.[B]
            x (torch.Tensor): image data. [B, C, H, W]

        Returns:
            a[t, x] (torch.Tensor): coefficients at timesteps t and data x. [B, C, H, W, num_bins]
        """
        a = a.to(x.device)
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_sample(self, x_0: torch.Tensor, batched_t: torch.Tensor, noise: torch.Tensor):
        """Sample from q(x_t | x_0) (i.e. add noise to the data).
        q(x_t | x_0) = Categorical(x_t ; p = x_0 Q_{1...t})

        Args:
            x_0 (torch.Tensor): Image data at t=0. [B, C, H, W]
            batched_t (torch.Tensor): Timesteps. [B]
            noise (torch.Tensor): Noise. [B, C, H, W]

        Returns:
            torch.Tensor: [B, C, H, W]
        """
        logits = torch.log(self._at(self.q_mats, batched_t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        batched_t: torch.Tensor,
    ):
        """Compute logits for q(x_{t-1} | x_t, x_0)."""
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_bins) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_bins,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        fact1 = self._at(self.q_one_step_transposed, batched_t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)
        qmats2 = self.q_mats[batched_t - 2]  # t - 1
        fact2 = torch.einsum("b...c, bcd -> b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = batched_t.reshape((batched_t.shape[0], *[1] * (x_t.dim())))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def forward(self, imgs: torch.Tensor):
        """Calculate P(x_t | x_0) loss.

        The forward process of D3PM is defined as:
        x_t ~ q(x_t | x_0) = Categorical(x_t ; p = x_0 Q_{1...t})

        Args:
            imgs (torch.Tensor): x_start images. [B, C, H, W]

        Returns:
            loss: loss value. [B]
        """
        imgs = imgs.long()
        batch_size = imgs.shape[0]
        # Sample a random timestep for each image in the batch
        batched_t = self.scheduler.uniform_sample_t(batch_size, self.device)
        # pure uniform noise image at t=T
        noise = torch.rand(imgs.shape + (self.num_bins,)).to(self.device)
        # q_sample
        x_t = self.q_sample(imgs, batched_t, noise)  # [B, C, H, W]
        # predict x0
        pred_x_start_logits = self.decoder(x_t, batched_t)  # [B, C, H, W, num_bins]
        # 1. variational bound (VL) loss
        true_q_posterior_logits = self.q_posterior_logits(imgs, x_t, batched_t)
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t, batched_t
        )
        vb_loss = categorical_kl_logits(
            true_q_posterior_logits, pred_q_posterior_logits
        )
        # 2. cross entropy loss for x_0
        ce_loss = F.cross_entropy(pred_x_start_logits.flatten(0, -2), imgs.flatten())
        self.log("loss/vb_loss", vb_loss, prog_bar=True)
        self.log("loss/ce_loss", ce_loss, prog_bar=True)
        return vb_loss + self.hybrid_coeff * ce_loss

    # === Sampling ===
    def p_sample(self, x: torch.Tensor, batched_t: torch.Tensor, noise: torch.Tensor):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        pred_x_start_logits = self.decoder(x, batched_t)
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x, batched_t
        )

        noise = torch.clamp(noise, min=self.eps, max=1.0)

        nonzero_mask = (batched_t != 1).to(x.dtype).view(-1, *([1] * (x.ndim)))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * nonzero_mask, dim=-1
        )
        return sample

    @torch.no_grad()
    def sample(self, num_samples: int):
        """Sample from the model.

        Args:
            num_samples: Number of samples to generate.
        """
        # pure noise image at t=T
        imgs_shape = (num_samples, self.in_channels, self.img_size, self.img_size)

        x_init = torch.full(imgs_shape, self.num_bins // 2, dtype=torch.long).to(
            self.device
        )  # [B, C, H, W]

        x_t = x_init.clone()  # [B, C, H, W]
        trajectory = [x_t]
        for t in tqdm(range(self.num_timesteps, 0, -1), desc="Sampling"):
            batched_t = torch.full(
                (num_samples,), t, dtype=torch.long, device=self.device
            )
            noise = torch.rand(imgs_shape + (self.num_bins,)).to(self.device)
            # p sampling
            x_t = self.p_sample(x_t, batched_t, noise)
            trajectory.append(x_t)

        return trajectory


# utils for loss


def categorical_kl_logits(logits1, logits2, eps=1.0e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
        logits1: logits of the first distribution. Last dim is class dim.
        logits2: logits of the second distribution. Last dim is class dim.
        eps: float small number to avoid numerical issues.

    Returns:
        KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = torch.softmax(logits1 + eps, dim=-1) * (
        torch.log_softmax(logits1 + eps, dim=-1)
        - torch.log_softmax(logits2 + eps, dim=-1)
    )
    return out.sum(dim=-1).mean()
