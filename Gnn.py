"""
Living AI System — Graph Neural Network Module
All GNN architectures: GCN, GAT, GraphSAGE, MPNN, Graph Autoencoder,
Variational Graph Autoencoder, Temporal Graph Network, Graph Isomorphism Network,
Jumping Knowledge Network, Gated Graph Neural Network, and more.
"""

import asyncio
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base import BaseModule, ModuleOutput

log = structlog.get_logger(__name__)


def degree_norm(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Compute symmetric degree normalisation for GCN."""
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    return deg_inv_sqrt


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer — spectral convolution."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        norm = degree_norm(edge_index, num_nodes, x.device)
        # Aggregate neighbour features
        agg = torch.zeros_like(x)
        weighted = x[col] * norm[col].unsqueeze(-1)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(weighted), weighted)
        agg = agg * norm.unsqueeze(-1)
        return F.relu(self.linear(agg))


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network."""

    def __init__(self, in_features: int, hidden: int, out_features: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden, hidden))
        self.layers.append(GCNLayer(hidden, out_features))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GATLayer(nn.Module):
    """Graph Attention Network layer."""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        h = self.W(x).view(num_nodes, self.num_heads, self.head_dim)
        # Attention coefficients
        edge_h = torch.cat([h[row], h[col]], dim=-1)
        attn = self.leaky_relu((edge_h * self.a).sum(dim=-1))
        # Softmax per node
        attn = attn - attn.max()
        attn_exp = attn.exp()
        attn_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attn_sum.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.num_heads), attn_exp)
        attn_norm = attn_exp / (attn_sum[row] + 1e-9)
        attn_norm = self.dropout(attn_norm)
        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        weighted = h[col] * attn_norm.unsqueeze(-1)
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)
        return F.elu(out.view(num_nodes, self.out_features))


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer with mean aggregation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        # Mean aggregation
        neighbour_sum = torch.zeros_like(x)
        neighbour_count = torch.zeros(num_nodes, 1, device=x.device)
        neighbour_sum.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(-1)), x[col])
        neighbour_count.scatter_add_(0, row.unsqueeze(-1), torch.ones(row.size(0), 1, device=x.device))
        neighbour_mean = neighbour_sum / (neighbour_count + 1e-9)
        combined = torch.cat([x, neighbour_mean], dim=-1)
        return F.relu(self.linear(combined))


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer — maximally powerful GNN."""

    def __init__(self, in_features: int, out_features: int, epsilon: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(epsilon))
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(-1)), x[col])
        return self.mlp((1 + self.eps) * x + agg)


class GatedGraphNeuralNetwork(nn.Module):
    """Gated Graph Neural Network using GRU propagation."""

    def __init__(self, hidden_size: int, num_steps: int = 5):
        super().__init__()
        self.num_steps = num_steps
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.message_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        h = x
        for _ in range(self.num_steps):
            messages = self.message_linear(h[col])
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
            h = self.gru_cell(agg, h)
        return h


class GraphAutoencoder(nn.Module):
    """Graph Autoencoder for link prediction and node embedding."""

    def __init__(self, in_features: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.ModuleList([
            GCNLayer(in_features, hidden),
            GCNLayer(hidden, latent),
        ])

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x, edge_index)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z), z


class VariationalGraphAutoencoder(nn.Module):
    """Variational Graph Autoencoder."""

    def __init__(self, in_features: int, hidden: int, latent: int):
        super().__init__()
        self.gcn_shared = GCNLayer(in_features, hidden)
        self.gcn_mu = GCNLayer(hidden, latent)
        self.gcn_logvar = GCNLayer(hidden, latent)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.gcn_shared(x, edge_index)
        return self.gcn_mu(h, edge_index), self.gcn_logvar(h, edge_index)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterise(mu, logvar)
        recon = torch.sigmoid(z @ z.T)
        return recon, mu, logvar


class JumpingKnowledgeNetwork(nn.Module):
    """
    Jumping Knowledge Network — aggregates representations from all layers.
    Addresses the over-smoothing problem in deep GNNs.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int, num_layers: int = 4):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCNLayer(in_features, hidden)] +
                                        [GCNLayer(hidden, hidden) for _ in range(num_layers - 1)])
        # Concatenate all layer outputs
        self.output_layer = nn.Linear(hidden * num_layers, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            layer_outputs.append(x)
        return self.output_layer(torch.cat(layer_outputs, dim=-1))


class TemporalGraphNetwork(nn.Module):
    """Temporal Graph Network with memory modules for dynamic graphs."""

    def __init__(self, node_features: int, edge_features: int, memory_dim: int, hidden: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.message_fn = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.memory_updater = nn.GRUCell(hidden, memory_dim)
        self.embedding = nn.Linear(memory_dim + node_features, hidden)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor, memory: torch.Tensor):
        row, col = edge_index
        messages = self.message_fn(
            torch.cat([memory[row], memory[col], edge_features], dim=-1)
        )
        new_memory = self.memory_updater(messages, memory[row])
        updated_memory = memory.clone()
        updated_memory[row] = new_memory
        embeddings = self.embedding(torch.cat([updated_memory, node_features], dim=-1))
        return embeddings, updated_memory


class GraphModule(BaseModule):
    """
    Graph Neural Network module.
    Activates for relational reasoning, graph-structured data,
    knowledge graph traversal, and network analysis tasks.
    All GNN variants are available and selectable by the router.
    """

    def __init__(self):
        self._gcn: GCN | None = None
        self._gat: GATLayer | None = None
        self._sage: GraphSAGELayer | None = None
        self._gin: GINLayer | None = None
        self._ggnn: GatedGraphNeuralNetwork | None = None
        self._gae: GraphAutoencoder | None = None
        self._vgae: VariationalGraphAutoencoder | None = None
        self._jkn: JumpingKnowledgeNetwork | None = None
        self._tgn: TemporalGraphNetwork | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "graph"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._gcn = GCN(64, 128, 64).to(self._device)
        self._gat = GATLayer(64, 64).to(self._device)
        self._sage = GraphSAGELayer(64, 64).to(self._device)
        self._gin = GINLayer(64, 64).to(self._device)
        self._ggnn = GatedGraphNeuralNetwork(64).to(self._device)
        self._gae = GraphAutoencoder(64, 128, 32).to(self._device)
        self._vgae = VariationalGraphAutoencoder(64, 128, 32).to(self._device)
        self._jkn = JumpingKnowledgeNetwork(64, 128, 64).to(self._device)
        self._tgn = TemporalGraphNetwork(64, 16, 64, 128).to(self._device)
        for model in [self._gcn, self._gat, self._sage, self._gin,
                      self._ggnn, self._gae, self._vgae, self._jkn, self._tgn]:
            model.eval()
        log.info("graph_module.initialised", variants=9)

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
