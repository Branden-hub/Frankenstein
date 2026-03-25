"""
Living AI System — Ouroboros Pruner
The adversarial critic network for autonomous node pruning.
Models PROTAC-mediated protein degradation — the system's
mechanism for preventing parameter explosion (HeLa Risk).

Governing differential equation:
dP_param/dt = k_alloc - (k_prune × [Utility] × [Compute_Available])

When degradation rate far exceeds synthesis rate,
uncontrolled growth is mathematically impossible.
The system is self-limiting by design.
"""

from collections import defaultdict
from pathlib import Path
import json

import structlog

log = structlog.get_logger(__name__)

ALPHA = 0.4  # Gradient magnitude weight
BETA = 0.4   # Activation frequency weight
GAMMA = 0.2  # Weight magnitude weight
WINDOW = 1000


class OuroborosPruner:
    """
    Secondary critic network that evaluates utility of every parameter.
    The critic has no output responsibility — its sole function is
    evaluation and deletion.

    Utility score:
    U(θᵢ) = α · |∂L/∂θᵢ| + β · f_activation(θᵢ) + γ · |θᵢ|

    Parameters scoring below the dynamic threshold are zeroed out.
    """

    def __init__(self, threshold: float = ALPHA):
        self.threshold = threshold
        self.activation_counts: dict[str, float] = defaultdict(float)
        self.window = WINDOW
        self._state_path = Path("data/pruner_state.json")
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted activation counts from disk."""
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    state = json.load(f)
                    self.activation_counts = defaultdict(float, state.get("activation_counts", {}))
            except Exception as exc:
                log.error("pruner.load_state_error", error=str(exc))

    def _save_state(self) -> None:
        """Persist activation counts to disk."""
        try:
            with open(self._state_path, "w") as f:
                json.dump({"activation_counts": dict(self.activation_counts)}, f)
        except Exception as exc:
            log.error("pruner.save_state_error", error=str(exc))

    def record_activation(self, parameter_name: str, count: float = 1.0) -> None:
        """Record an activation event for a parameter."""
        self.activation_counts[parameter_name] = (
            self.activation_counts[parameter_name] * 0.99 + count * 0.01
        )

    def score_parameter(
        self,
        name: str,
        weight_magnitude: float,
        grad_magnitude: float | None,
    ) -> float:
        """
        Compute composite utility score for a single parameter.
        U(θᵢ) = α · |∂L/∂θᵢ| + β · f_activation(θᵢ) + γ · |θᵢ|
        """
        grad_mag = grad_magnitude if grad_magnitude is not None else 0.0
        act_freq = self.activation_counts[name] / self.window
        return ALPHA * grad_mag + BETA * act_freq + GAMMA * weight_magnitude

    async def prune(self, memory_pressure: float = 0.0) -> int:
        """
        Run a pruning pass across all registered modules.
        Dynamic threshold scales with memory pressure —
        pruning accelerates when resources are constrained.
        Returns the count of pruned parameters.
        """
        dynamic_threshold = self.threshold * (1.0 + memory_pressure)
        pruned_count = 0

        try:
            # Load module registry to get all active parameter sets
            from modules.homeostasis.parameter_registry import ParameterRegistry
            registry = ParameterRegistry()
            parameter_sets = await registry.get_all_parameters()

            for param_set in parameter_sets:
                for param_name, param_info in param_set.items():
                    score = self.score_parameter(
                        name=param_name,
                        weight_magnitude=param_info.get("weight_magnitude", 0.0),
                        grad_magnitude=param_info.get("grad_magnitude"),
                    )
                    if score < dynamic_threshold:
                        await registry.zero_parameter(param_name)
                        pruned_count += 1

            if pruned_count > 0:
                self._save_state()
                log.info(
                    "pruner.pruned",
                    count=pruned_count,
                    threshold=dynamic_threshold,
                    memory_pressure=memory_pressure,
                )

        except Exception as exc:
            log.error("pruner.prune_error", error=str(exc))

        return pruned_count

    async def evaluate_structural_unit(
        self,
        unit_name: str,
        unit_parameters: dict,
    ) -> float:
        """
        Evaluate utility of an entire structural unit
        (attention head, layer, filter) rather than individual weights.
        Returns aggregate utility score for the unit.
        """
        if not unit_parameters:
            return 0.0

        scores = [
            self.score_parameter(
                name=f"{unit_name}.{p}",
                weight_magnitude=info.get("weight_magnitude", 0.0),
                grad_magnitude=info.get("grad_magnitude"),
            )
            for p, info in unit_parameters.items()
        ]
        return sum(scores) / len(scores)
