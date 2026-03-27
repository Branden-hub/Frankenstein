"""
Living AI System — Metacognitive Monitor
Tracks performance drift using KL divergence.
Performance_drift(t) = KL[ p(y|x, θ_t) || p(y|x, θ_{t-τ}) ]
"""

import structlog
import torch
import torch.nn.functional as F

log = structlog.get_logger(__name__)


class MetacognitiveMonitor:
    """
    Monitors the system's own cognitive performance.
    Detects drift from baseline and triggers corrective consolidation.
    The system's self-awareness of its own computational state.
    """

    def __init__(self):
        self._baseline_distribution: torch.Tensor | None = None
        self._current_distribution: torch.Tensor | None = None

    async def compute_drift(self) -> float:
        """
        Compute KL divergence between current and baseline distributions.
        Returns drift score — higher means more drift from baseline.
        """
        if self._baseline_distribution is None or self._current_distribution is None:
            return 0.0

        try:
            kl = F.kl_div(
                self._current_distribution.log(),
                self._baseline_distribution,
                reduction="sum",
            ).item()
            return abs(kl)
        except Exception as exc:
            log.error("metacognitive_monitor.compute_drift_error", error=str(exc))
            return 0.0

    def update_baseline(self, distribution: torch.Tensor) -> None:
        """Set a new performance baseline."""
        self._baseline_distribution = F.softmax(distribution, dim=-1).detach()

    def update_current(self, distribution: torch.Tensor) -> None:
        """Update current performance distribution."""
        self._current_distribution = F.softmax(distribution, dim=-1).detach()
