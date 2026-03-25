"""
Living AI System — Router
The router is the first filter in the Game of Infinite Paths.
It takes the infinite space of all possible module combinations
and narrows it to the set of modules whose filters are relevant
to the current input.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from controller.main import InputMessage

log = structlog.get_logger(__name__)


@dataclass
class RoutingDecision:
    modules: list
    primary_module: object
    required_capabilities: list[str]
    reasoning: str
    confidence: float
    metadata: dict = field(default_factory=dict)


class Router:
    """
    Routes each input to the correct set of modules.
    Uses a combination of pattern matching, keyword detection,
    modality detection, and learned routing weights to determine
    which modules are relevant for each input.
    """

    def __init__(self):
        self._modules: dict = {}
        self._routing_weights: dict = {}
        self._initialised = False

    async def initialise(self) -> None:
        """Load and register all available modules."""
        from modules.neural_networks.language import LanguageModule
        from modules.neural_networks.transformer import TransformerModule
        from modules.neural_networks.cnn import ConvolutionalModule
        from modules.neural_networks.rnn import RecurrentModule
        from modules.neural_networks.gnn import GraphModule
        from modules.neural_networks.generative import GenerativeModule
        from modules.neural_networks.snn import SpikingModule
        from modules.neural_networks.capsule import CapsuleModule
        from modules.neural_networks.autoencoder import AutoencoderModule
        from modules.neural_networks.hopfield import HopfieldModule
        from modules.neural_networks.memory_augmented import MemoryAugmentedModule
        from modules.neural_networks.siamese import SiameseModule
        from modules.neural_networks.moe import MixtureOfExpertsModule
        from modules.learning_paradigms.reinforcement import ReinforcementModule
        from modules.learning_paradigms.meta import MetaLearningModule
        from modules.learning_paradigms.continual import ContinualLearningModule
        from modules.learning_paradigms.bayesian import BayesianModule
        from modules.learning_paradigms.evolutionary import EvolutionaryModule
        from modules.reasoning.planner import PlannerModule
        from modules.reasoning.chain_of_thought import ChainOfThoughtModule
        from modules.reasoning.multi_agent import MultiAgentModule
        from modules.hands.code_executor import CodeExecutorModule
        from modules.hands.browser import BrowserModule
        from modules.hands.api_orchestrator import APIOrchestatorModule
        from modules.knowledge.retrieval import RetrievalModule
        from modules.skills.registry import SkillRegistryModule
        from modules.sensory.vision import VisionModule
        from modules.sensory.audio import AudioModule
        from modules.voice.tts import VoiceModule

        # Register all modules
        all_modules = [
            LanguageModule(),
            TransformerModule(),
            ConvolutionalModule(),
            RecurrentModule(),
            GraphModule(),
            GenerativeModule(),
            SpikingModule(),
            CapsuleModule(),
            AutoencoderModule(),
            HopfieldModule(),
            MemoryAugmentedModule(),
            SiameseModule(),
            MixtureOfExpertsModule(),
            ReinforcementModule(),
            MetaLearningModule(),
            ContinualLearningModule(),
            BayesianModule(),
            EvolutionaryModule(),
            PlannerModule(),
            ChainOfThoughtModule(),
            MultiAgentModule(),
            CodeExecutorModule(),
            BrowserModule(),
            APIOrchestatorModule(),
            RetrievalModule(),
            SkillRegistryModule(),
            VisionModule(),
            AudioModule(),
            VoiceModule(),
        ]

        for module in all_modules:
            await module.initialise()
            self._modules[module.name] = module

        self._initialised = True
        log.info("router.initialised", module_count=len(self._modules))

    async def route(
        self,
        message: "InputMessage",
        episodic_context: list,
        knowledge_context: list,
    ) -> RoutingDecision:
        """
        Determine which modules to activate for this input.
        The routing decision narrows the infinite space of possible
        module combinations to the relevant subset.
        """
        if not self._initialised:
            raise RuntimeError("Router not initialised — call initialise() first")

        activated_modules = []
        required_capabilities = []
        routing_reasoning = []

        content_lower = message.content.lower()

        # Language module is always primary for text inputs
        language_module = self._modules["language"]
        activated_modules.append(language_module)
        routing_reasoning.append("language: always active for text input")

        # Code detection — activate code executor
        code_indicators = [
            "write code", "write a", "create a script", "function",
            "implement", "debug", "error in", "syntax", "python",
            "javascript", "typescript", "rust", "go ", "java ",
            "c++", "sql ", "bash ", "powershell", "dockerfile",
            "```", "class ", "def ", "async ", "import ",
        ]
        if any(ind in content_lower for ind in code_indicators) or message.modality == "code":
            activated_modules.append(self._modules["code_executor"])
            required_capabilities.append("code_execution")
            routing_reasoning.append("code_executor: code indicators detected")

        # Vision — activate for image modality
        if message.modality == "image":
            activated_modules.append(self._modules["vision"])
            required_capabilities.append("vision")
            routing_reasoning.append("vision: image modality")

        # Audio — activate for audio modality
        if message.modality == "audio":
            activated_modules.append(self._modules["audio"])
            required_capabilities.append("audio_input")
            routing_reasoning.append("audio: audio modality")

        # Complex reasoning — activate chain of thought and planner
        complexity_indicators = [
            "how do i", "step by step", "plan", "strategy", "analyse",
            "compare", "explain why", "what is the best", "design",
            "architecture", "solve", "reason", "think through",
        ]
        if any(ind in content_lower for ind in complexity_indicators):
            activated_modules.append(self._modules["chain_of_thought"])
            routing_reasoning.append("chain_of_thought: complex reasoning detected")

        # Multi-step tasks — activate planner
        task_indicators = [
            "build", "create", "make me", "generate", "produce",
            "write a report", "research", "find all", "complete",
        ]
        if any(ind in content_lower for ind in task_indicators):
            activated_modules.append(self._modules["planner"])
            routing_reasoning.append("planner: multi-step task detected")

        # Knowledge retrieval — always activate for factual queries
        factual_indicators = [
            "what is", "who is", "when did", "where is", "how does",
            "tell me about", "explain", "describe", "define",
        ]
        if any(ind in content_lower for ind in factual_indicators):
            activated_modules.append(self._modules["retrieval"])
            routing_reasoning.append("retrieval: factual query detected")

        # Graph reasoning — activate for relational queries
        graph_indicators = [
            "relationship", "connected", "network", "graph",
            "linked", "related to", "dependencies", "path between",
        ]
        if any(ind in content_lower for ind in graph_indicators):
            activated_modules.append(self._modules["graph"])
            routing_reasoning.append("graph: relational query detected")

        # Generative tasks — activate generative module
        generative_indicators = [
            "generate", "create an image", "draw", "synthesise",
            "produce", "imagine", "design a",
        ]
        if any(ind in content_lower for ind in generative_indicators):
            activated_modules.append(self._modules["generative"])
            routing_reasoning.append("generative: generation task detected")

        # Skill lookup — check skill registry for matching patterns
        skill_module = self._modules["skill_registry"]
        matching_skills = await skill_module.find_matching_skills(message.content)
        if matching_skills:
            activated_modules.append(skill_module)
            routing_reasoning.append(
                f"skill_registry: {len(matching_skills)} matching skills found"
            )

        # Deduplicate modules
        seen = set()
        unique_modules = []
        for m in activated_modules:
            if m.name not in seen:
                seen.add(m.name)
                unique_modules.append(m)

        log.info(
            "router.decision",
            trace_id=message.trace_id,
            modules=[m.name for m in unique_modules],
            reasoning=routing_reasoning,
        )

        return RoutingDecision(
            modules=unique_modules,
            primary_module=language_module,
            required_capabilities=required_capabilities,
            reasoning="; ".join(routing_reasoning),
            confidence=0.9,
        )
