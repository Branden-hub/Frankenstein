"""
Living AI System — Parameter Registry
Tracks all learnable parameters across all modules.
Used by the Ouroboros pruner to evaluate and zero out low-utility parameters.
"""

import structlog

log = structlog.get_logger(__name__)


class ParameterRegistry:
    """
    Central registry of all model parameters across all modules.
    The pruner queries this to score and prune low-utility parameters.
    """

    _registry: dict = {}

    @classmethod
    def register(cls, module_name: str, parameters: dict) -> None:
        cls._registry[module_name] = parameters

    async def get_all_parameters(self) -> list[dict]:
        """Return all registered parameter sets."""
        return list(self._registry.values())

    async def zero_parameter(self, param_name: str) -> None:
        """Zero out a parameter identified for pruning."""
        for module_params in self._registry.values():
            if param_name in module_params:
                module_params[param_name]["weight_magnitude"] = 0.0
                log.debug("parameter_registry.zeroed", param=param_name)
                return
