"""
Living AI System — Generative Models Module
All generative architectures: GAN, DCGAN, WGAN, Conditional GAN, CycleGAN,
VAE, Beta-VAE, VQ-VAE, Diffusion Model, Normalizing Flow, Autoregressive Model,
Masked Autoencoder.
"""

import asyncio
import math
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base import BaseModule, ModuleOutput

log = structlog.get_logger(__name__)


# ─── GAN Family ───────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Standard GAN generator."""

    def __init__(self, latent_dim: int = 128, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Standard GAN discriminator."""

    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalGenerator(nn.Module):
    """Conditional GAN generator — conditioned on class label."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 10, output_dim: int = 784):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embed(labels)
        return self.net(torch.cat([z, label_emb], dim=1))


class WassersteinCritic(nn.Module):
    """Wasserstein GAN critic — no sigmoid, unbounded output."""

    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
        """Gradient penalty for WGAN-GP."""
        alpha = torch.rand(real.size(0), 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        d_out = self.forward(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_out, inputs=interpolated,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True, retain_graph=True,
        )[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ─── VAE Family ───────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    Variational Autoencoder.
    The latent space is the compressed path space.
    Sampling from the posterior is the filter operation.
    The reconstruction is the revealed answer.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        recon_loss = F.binary_cross_entropy(recon, x.view(recon.size(0), -1), reduction="sum")
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return recon_loss + beta * kld


class VectorQuantizedVAE(nn.Module):
    """
    VQ-VAE — discrete latent space using learned codebook.
    The codebook is the discrete filter structure.
    Commitment loss prevents codebook collapse.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 num_embeddings: int = 512, embedding_dim: int = 64,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x.view(x.size(0), -1))
        # Find nearest codebook entries
        dists = (z.pow(2).sum(1, keepdim=True)
                 - 2 * z @ self.codebook.weight.T
                 + self.codebook.weight.pow(2).sum(1))
        indices = dists.argmin(1)
        z_q = self.codebook(indices)
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()
        recon = self.decoder(z_q_st)
        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z, z_q.detach())
        return recon, loss, indices


# ─── Diffusion Models ─────────────────────────────────────────────────────────

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DenoisingNetwork(nn.Module):
    """Simple denoising network for DDPM."""

    def __init__(self, data_dim: int = 784, hidden_dim: int = 512, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t.float())
        return self.net(torch.cat([x, t_emb], dim=-1))


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    The forward process adds noise — expanding the path space.
    The reverse process is the filter — revealing the clean signal.
    """

    def __init__(self, data_dim: int = 784, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.denoiser = DenoisingNetwork(data_dim)

        # Precompute noise schedule
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def forward_diffuse(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        x_noisy, noise = self.forward_diffuse(x0.view(B, -1), t)
        predicted_noise = self.denoiser(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)


# ─── Normalizing Flows ────────────────────────────────────────────────────────

class RealNVPCouplingLayer(nn.Module):
    """Real-valued Non-Volume Preserving coupling layer."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 256):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim), nn.Tanh(),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor):
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.shift_net(x_masked) * (1 - self.mask)
        y = x_masked + (1 - self.mask) * (x * s.exp() + t)
        log_det = (s * (1 - self.mask)).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        y_masked = y * self.mask
        s = self.scale_net(y_masked) * (1 - self.mask)
        t = self.shift_net(y_masked) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * ((y - t) * (-s).exp())
        return x


class NormalizingFlow(nn.Module):
    """Normalizing flow with Real-NVP coupling layers."""

    def __init__(self, dim: int = 64, num_layers: int = 8):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask = torch.zeros(dim)
            mask[::2] = 1 if i % 2 == 0 else 0
            mask[1::2] = 0 if i % 2 == 0 else 1
            self.layers.append(RealNVPCouplingLayer(dim, mask))

    def forward(self, x: torch.Tensor):
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        log_pz = -0.5 * (x ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return -(log_pz + log_det_total).mean()

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.layers[0].scale_net[0].in_features, device=device)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


class GenerativeModule(BaseModule):
    """
    Generative models module.
    Activates for generation, synthesis, and creation tasks.
    Houses all generative architectures.
    """

    def __init__(self):
        self._vae: VAE | None = None
        self._vqvae: VectorQuantizedVAE | None = None
        self._generator: Generator | None = None
        self._discriminator: Discriminator | None = None
        self._cond_generator: ConditionalGenerator | None = None
        self._wgan_critic: WassersteinCritic | None = None
        self._ddpm: DDPM | None = None
        self._flow: NormalizingFlow | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "generative"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._vae = VAE().to(self._device)
        self._vqvae = VectorQuantizedVAE().to(self._device)
        self._generator = Generator().to(self._device)
        self._discriminator = Discriminator().to(self._device)
        self._cond_generator = ConditionalGenerator().to(self._device)
        self._wgan_critic = WassersteinCritic().to(self._device)
        self._ddpm = DDPM().to(self._device)
        self._flow = NormalizingFlow().to(self._device)
        for model in [self._vae, self._vqvae, self._generator, self._discriminator,
                      self._cond_generator, self._wgan_critic, self._ddpm, self._flow]:
            model.eval()
        log.info("generative_module.initialised", variants=8)

    async def execute(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> ModuleOutput:
        return ModuleOutput(
            content="",
            confidence=0.0,
            output_type=self.output_type,
            source=self.name,
        )
