"""
Living AI System — Recurrent Neural Network Module
All RNN architectures: vanilla RNN, LSTM, GRU, Bidirectional variants,
Deep RNN, Echo State Network, Clockwork RNN.
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


class VanillaRNN(nn.Module):
    """Standard recurrent neural network."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        out, hn = self.rnn(x, h0)
        return self.fc(out), hn


class LSTMNetwork(nn.Module):
    """Long Short-Term Memory network."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, state: tuple | None = None):
        out, (hn, cn) = self.lstm(x, state)
        return self.fc(out), (hn, cn)


class GRUNetwork(nn.Module):
    """Gated Recurrent Unit network."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        out, hn = self.gru(x, h0)
        return self.fc(out), hn


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM — processes sequence in both directions."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor, state: tuple | None = None):
        out, (hn, cn) = self.lstm(x, state)
        return self.fc(out), (hn, cn)


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        out, hn = self.gru(x, h0)
        return self.fc(out), hn


class EchoStateNetwork(nn.Module):
    """
    Echo State Network — reservoir computing.
    A fixed random reservoir with a trained readout layer.
    The reservoir is the infinite path space of recurrent dynamics.
    Only the readout weights are trained.
    """

    def __init__(self, input_size: int, reservoir_size: int, output_size: int,
                 spectral_radius: float = 0.9, sparsity: float = 0.1, leak_rate: float = 0.3):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate

        # Fixed random reservoir weights
        W_res = torch.randn(reservoir_size, reservoir_size) * (1.0 / math.sqrt(reservoir_size))
        # Apply sparsity
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        W_res[mask] = 0.0
        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        max_eig = eigenvalues.max().item()
        if max_eig > 0:
            W_res = W_res * (spectral_radius / max_eig)

        self.register_buffer("W_res", W_res)
        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)
        self.W_out = nn.Linear(reservoir_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = torch.zeros(B, self.reservoir_size, device=x.device)
        states = []
        for t in range(T):
            h_new = torch.tanh(self.W_in(x[:, t]) + h @ self.W_res.T)
            h = (1 - self.leak_rate) * h + self.leak_rate * h_new
            states.append(h)
        states = torch.stack(states, dim=1)
        return self.W_out(states)


class ClockworkRNN(nn.Module):
    """
    Clockwork RNN — different neuron groups update at different frequencies.
    Models biological rhythmic activity at multiple timescales.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_modules: int = 4):
        super().__init__()
        self.num_modules = num_modules
        self.module_size = hidden_size // num_modules
        self.periods = [2 ** i for i in range(num_modules)]

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = torch.zeros(B, self.num_modules * self.module_size, device=x.device)
        outputs = []
        for t in range(T):
            h_new = self.tanh(self.W_in(x[:, t]) + self.W_h(h))
            # Only update modules whose period divides current timestep
            for m, period in enumerate(self.periods):
                if t % period == 0:
                    start = m * self.module_size
                    end = start + self.module_size
                    h[:, start:end] = h_new[:, start:end]
            outputs.append(self.W_out(h))
        return torch.stack(outputs, dim=1)


class RecurrentModule(BaseModule):
    """RNN module — activates for sequential and temporal processing tasks."""

    def __init__(self):
        self._lstm: LSTMNetwork | None = None
        self._gru: GRUNetwork | None = None
        self._bi_lstm: BidirectionalLSTM | None = None
        self._esn: EchoStateNetwork | None = None
        self._clockwork: ClockworkRNN | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "rnn"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._lstm = LSTMNetwork(128, 256, 128).to(self._device)
        self._gru = GRUNetwork(128, 256, 128).to(self._device)
        self._bi_lstm = BidirectionalLSTM(128, 256, 128).to(self._device)
        self._esn = EchoStateNetwork(128, 512, 128).to(self._device)
        self._clockwork = ClockworkRNN(128, 256, 128).to(self._device)
        for model in [self._lstm, self._gru, self._bi_lstm, self._esn, self._clockwork]:
            model.eval()
        log.info("rnn_module.initialised")

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
