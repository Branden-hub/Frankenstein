"""
Living AI System — Base Module
Every AI architecture, every learning paradigm, every graph network,
every generative model — all are modules that inherit from this base class.
The base class enforces the interface that the controller requires
to route inputs, execute filters, and aggregate outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ModuleOutput:
    content: str
    confidence: float
    output_type: str
    source: str
    is_primary: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModule(ABC):
    """
    Base class for all AI modules in the Living AI System.
    Every neural network, every learning paradigm, every graph network,
    every generative model implements this interface.
    The controller uses this interface to route, execute, and aggregate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this module."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> str:
        """Type of output this module produces: text, code, vision, audio, plan, retrieval."""
        ...

    @property
    def required_capabilities(self) -> list[str]:
        """Capability gate names required before this module can execute."""
        return []

    async def initialise(self) -> None:
        """Initialise the module — load weights, connect to resources, etc."""
        pass

    @abstractmethod
    async def execute(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> ModuleOutput:
        """
        Apply this module's filter to the input.
        The input is the current message plus all relevant context.
        The output is the revealed answer after this filter is applied.
        """
        ...

    async def stream(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> AsyncIterator[str]:
        """
        Streaming version of execute.
        Yields tokens as they are generated.
        Default implementation runs execute() and yields the full content.
        Override for true streaming.
        """
        output = await self.execute(
            message=message,
            episodic_context=episodic_context,
            knowledge_context=knowledge_context,
            working_memory=working_memory,
        )
        yield output.content

    async def find_matching_skills(self, query: str) -> list:
        """
        Check whether this module has skills matching the query.
        Used by the router for skill-based routing.
        Override in skill-capable modules.
        """
        return []

    def get_status(self) -> dict:
        """Return module status for health and monitoring."""
        return {"name": self.name, "initialised": True}
