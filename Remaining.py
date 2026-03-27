"""
Living AI System — Remaining Neural Network Architectures
Spiking Neural Networks, Capsule Networks, Autoencoders,
Hopfield Networks, Memory-Augmented Networks, Siamese Networks,
Mixture of Experts, Radial Basis Function Networks,
Self-Organising Maps, Deep Belief Networks.
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


# ─── Spiking Neural Networks ──────────────────────────────────────────────────

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron.
    Models the hyperpolarized membrane potential mechanism.
    τ · dV/dt = -(V - V_rest) + R · I(t)
    Spike emitted when V(t) ≥ Θ; then V(t) ← V_reset
    Computation occurs only in response to meaningful input changes —
    near-zero power when below threshold.
    """

    def __init__(self, threshold: float = 1.0, tau: float = 20.0,
                 v_rest: float = 0.0, v_reset: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset

    def forward(self, current: torch.Tensor, v: torch.Tensor, dt: float = 1.0):
        """Single timestep update."""
        dv = (-(v - self.v_rest) + current) / self.tau * dt
        v_new = v + dv
        spike = (v_new >= self.threshold).float()
        v_new = torch.where(spike.bool(), torch.full_like(v_new, self.v_reset), v_new)
        return spike, v_new


class SpikingLayer(nn.Module):
    """Fully connected spiking layer with LIF neurons."""

    def __init__(self, in_features: int, out_features: int, num_timesteps: int = 20):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.lif = LIFNeuron()
        self.num_timesteps = num_timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in = x.shape
        v = torch.zeros(B, self.fc.out_features, device=x.device)
        spike_train = []
        current = self.fc(x)
        for t in range(self.num_timesteps):
            spike, v = self.lif(current, v)
            spike_train.append(spike)
        return torch.stack(spike_train, dim=1).mean(dim=1)  # Rate coding


class SpikingNetwork(nn.Module):
    """Multi-layer spiking neural network."""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super().__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([
            SpikingLayer(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ─── Capsule Networks ─────────────────────────────────────────────────────────

class CapsuleLayer(nn.Module):
    """
    Capsule layer with dynamic routing.
    Capsules represent entities and their properties as vectors.
    Agreement between capsule outputs determines routing weights.
    """

    def __init__(self, num_capsules: int, in_channels: int, in_dim: int,
                 out_dim: int, num_routing: int = 3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(num_capsules, in_channels, in_dim, out_dim))

    @staticmethod
    def squash(x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        return norm ** 2 / (1 + norm ** 2) * x / (norm + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # x: (B, in_channels, in_dim)
        u_hat = torch.einsum("bci,cijo->bcjo", x, self.W)
        b = torch.zeros(B, self.num_capsules, x.size(1), device=x.device)
        for _ in range(self.num_routing):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)
            v = self.squash(s)
            b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)
        return v


# ─── Autoencoder Family ───────────────────────────────────────────────────────

class DenoisingAutoencoder(nn.Module):
    """Autoencoder trained to reconstruct clean inputs from noisy ones."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, noise_factor: float = 0.3):
        super().__init__()
        self.noise_factor = noise_factor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noisy = x + self.noise_factor * torch.randn_like(x)
            noisy = noisy.clamp(0.0, 1.0)
        else:
            noisy = x
        return self.decoder(self.encoder(noisy.view(x.size(0), -1)))


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with L1 penalty on latent activations."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 1024, sparsity: float = 0.05):
        super().__init__()
        self.sparsity = sparsity
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        h = self.encoder(x.view(x.size(0), -1))
        recon = self.decoder(h)
        l1_penalty = self.sparsity * h.abs().sum(dim=-1).mean()
        return recon, l1_penalty


# ─── Hopfield Networks ────────────────────────────────────────────────────────

class ModernHopfieldNetwork(nn.Module):
    """
    Modern Hopfield Network with exponential storage capacity.
    Can store exponentially many patterns.
    The stored patterns are the path space.
    Retrieval is the filter operation.
    The recalled pattern is the revealed answer.
    """

    def __init__(self, hidden_dim: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor, stored: torch.Tensor) -> torch.Tensor:
        """Retrieve pattern closest to x from stored patterns."""
        scores = self.beta * (x @ stored.T)
        attn = F.softmax(scores, dim=-1)
        return attn @ stored


# ─── Memory-Augmented Networks ────────────────────────────────────────────────

class NeuralTuringMachineMemory(nn.Module):
    """
    Neural Turing Machine — differentiable external memory.
    The memory matrix is an external knowledge store.
    Attention-based reading and writing are the filter operations.
    """

    def __init__(self, memory_size: int = 128, memory_dim: int = 64,
                 controller_dim: int = 256, output_size: int = 128):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller = nn.LSTMCell(output_size + memory_dim, controller_dim)
        self.read_head = nn.Linear(controller_dim, memory_size)
        self.write_head = nn.Linear(controller_dim, memory_size)
        self.erase_head = nn.Linear(controller_dim, memory_dim)
        self.add_head = nn.Linear(controller_dim, memory_dim)
        self.output_layer = nn.Linear(controller_dim + memory_dim, output_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, state: tuple):
        B = x.size(0)
        read_w = F.softmax(self.read_head(state[0]), dim=-1)
        read_vector = (read_w.unsqueeze(-1) * memory).sum(dim=1)
        controller_input = torch.cat([x, read_vector], dim=-1)
        h, c = self.controller(controller_input, state)
        write_w = F.softmax(self.write_head(h), dim=-1)
        erase = torch.sigmoid(self.erase_head(h))
        add = torch.tanh(self.add_head(h))
        memory = memory * (1 - write_w.unsqueeze(-1) * erase.unsqueeze(1))
        memory = memory + write_w.unsqueeze(-1) * add.unsqueeze(1)
        output = self.output_layer(torch.cat([h, read_vector], dim=-1))
        return output, memory, (h, c)


# ─── Siamese Networks ─────────────────────────────────────────────────────────

class SiameseNetwork(nn.Module):
    """
    Siamese network for similarity learning.
    Both branches share weights — the filter is identical for both inputs.
    The distance between outputs measures similarity.
    """

    def __init__(self, input_dim: int = 784, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x.view(x.size(0), -1)), p=2, dim=-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.forward_one(x1), self.forward_one(x2)
        return F.pairwise_distance(z1, z2)

    def contrastive_loss(self, dist: torch.Tensor, label: torch.Tensor,
                         margin: float = 1.0) -> torch.Tensor:
        return (label * dist.pow(2) +
                (1 - label) * F.relu(margin - dist).pow(2)).mean()


# ─── Mixture of Experts ───────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts — only top-k experts activate per input.
    Models threshold-gated computation — the metabolic engineering principle.
    Only a small subset of parameters fire for any given input.
    Output = Σₖ G(x)ₖ · Eₖ(x), where ||G(x)||₀ ≤ k_active
    """

    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 8,
                 k_active: int = 2):
        super().__init__()
        self.k_active = k_active
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = gate_logits.topk(self.k_active, dim=-1)
        gate_weights = F.softmax(topk_vals, dim=-1)
        out = torch.zeros(x.size(0), self.experts[0].net[-1].out_features, device=x.device)
        for k in range(self.k_active):
            expert_idx = topk_idx[:, k]
            weight = gate_weights[:, k].unsqueeze(-1)
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i)
                if mask.any():
                    out[mask] += weight[mask] * expert(x[mask])
        return out, gate_logits


# ─── Radial Basis Function Network ───────────────────────────────────────────

class RBFNetwork(nn.Module):
    """Radial Basis Function Network."""

    def __init__(self, in_features: int, num_centers: int, out_features: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.log_sigmas = nn.Parameter(torch.zeros(num_centers))
        self.linear = nn.Linear(num_centers, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.centers)
        phi = torch.exp(-dists ** 2 / (2 * self.log_sigmas.exp() ** 2))
        return self.linear(phi)


# ─── Self-Organising Map ─────────────────────────────────────────────────────

class SelfOrganisingMap(nn.Module):
    """Self-Organising Map — unsupervised topological mapping."""

    def __init__(self, map_size: int = 10, input_dim: int = 64):
        super().__init__()
        self.map_size = map_size
        self.weights = nn.Parameter(torch.randn(map_size * map_size, input_dim))
        grid = torch.stack(torch.meshgrid(
            torch.arange(map_size), torch.arange(map_size), indexing="ij"
        ), dim=-1).float().view(-1, 2)
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.weights)
        bmu_indices = dists.argmin(dim=-1)
        return bmu_indices

    def neighbourhood(self, bmu_idx: torch.Tensor, sigma: float) -> torch.Tensor:
        bmu_pos = self.grid[bmu_idx]
        dists = torch.cdist(self.grid, bmu_pos)
        return torch.exp(-dists ** 2 / (2 * sigma ** 2))


# ─── Module wrappers ──────────────────────────────────────────────────────────

class SpikingModule(BaseModule):
    @property
    def name(self) -> str:
        return "snn"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._snn = SpikingNetwork(784, [512, 256], 128)
        log.info("snn_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)


class CapsuleModule(BaseModule):
    @property
    def name(self) -> str:
        return "capsule"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._capsule = CapsuleLayer(10, 32, 8, 16)
        log.info("capsule_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)


class AutoencoderModule(BaseModule):
    @property
    def name(self) -> str:
        return "autoencoder"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._dae = DenoisingAutoencoder()
        self._sae = SparseAutoencoder()
        log.info("autoencoder_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)


class HopfieldModule(BaseModule):
    @property
    def name(self) -> str:
        return "hopfield"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._hopfield = ModernHopfieldNetwork(256)
        log.info("hopfield_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)


class MemoryAugmentedModule(BaseModule):
    @property
    def name(self) -> str:
        return "memory_augmented"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._ntm = NeuralTuringMachineMemory()
        log.info("memory_augmented_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)


class SiameseModule(BaseModule):
    @property
    def name(self) -> str:
        return "siamese"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._siamese = SiameseNetwork()
        log.info("siamese_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)


class MixtureOfExpertsModule(BaseModule):
    @property
    def name(self) -> str:
        return "moe"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._moe = SparseMixtureOfExperts(512, 512, num_experts=8, k_active=2)
        log.info("moe_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context, working_memory):
        return ModuleOutput(content="", confidence=0.0, output_type=self.output_type, source=self.name)
