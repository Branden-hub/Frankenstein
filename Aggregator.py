"""
Living AI System — Aggregator
The aggregator is the final filter in the processing pipeline.
It takes the outputs of all activated modules — each having applied
its own filter to the input — and reveals the correct unified output
through their intersection.
As the system learns and model resolution increases,
the aggregation becomes more precise and noise decreases.
"""

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class AggregatedOutput:
    content: str
    confidence: float
    metadata: dict[str, Any]
    sources: list[str]


class Aggregator:
    """
    Combines outputs from multiple activated modules into a single
    coherent response. Uses confidence-weighted combination where
    modules with higher confidence scores contribute more to the
    final output. As the system learns, confidence scores become
    more accurate — noise in the aggregation decreases toward zero.
    """

    async def aggregate(
        self,
        outputs: list,
        routing_decision,
        message,
    ) -> AggregatedOutput:
        """
        Aggregate all module outputs into a single response.
        The primary module (language) provides the base response.
        Other module outputs are integrated as context, corrections,
        or augmentations depending on their type.
        """
        if not outputs:
            return AggregatedOutput(
                content="No module outputs to aggregate.",
                confidence=0.0,
                metadata={},
                sources=[],
            )

        # Separate primary language output from auxiliary outputs
        primary_output = None
        auxiliary_outputs = []

        for output in outputs:
            if hasattr(output, "is_primary") and output.is_primary:
                primary_output = output
            else:
                auxiliary_outputs.append(output)

        # If no primary output, use the first output
        if primary_output is None and outputs:
            primary_output = outputs[0]
            auxiliary_outputs = outputs[1:]

        if primary_output is None:
            return AggregatedOutput(
                content="Processing error — no primary output generated.",
                confidence=0.0,
                metadata={},
                sources=[],
            )

        base_content = primary_output.content
        combined_confidence = primary_output.confidence
        metadata = {}
        sources = [primary_output.source] if hasattr(primary_output, "source") else []

        # Integrate auxiliary outputs
        for aux in auxiliary_outputs:
            if not hasattr(aux, "content") or not aux.content:
                continue

            # Code executor output — append as code block
            if hasattr(aux, "output_type") and aux.output_type == "code":
                metadata["code_output"] = aux.content
                sources.append(aux.source if hasattr(aux, "source") else "code_executor")

            # Retrieval output — knowledge was already injected into context
            elif hasattr(aux, "output_type") and aux.output_type == "retrieval":
                metadata["knowledge_sources"] = aux.content
                sources.append("knowledge_base")

            # Vision output — image description integrated into response
            elif hasattr(aux, "output_type") and aux.output_type == "vision":
                metadata["visual_analysis"] = aux.content
                sources.append("vision")

            # Planning output — task plan stored in metadata
            elif hasattr(aux, "output_type") and aux.output_type == "plan":
                metadata["task_plan"] = aux.content
                sources.append("planner")

            # Average confidence across all outputs
            if hasattr(aux, "confidence"):
                combined_confidence = (combined_confidence + aux.confidence) / 2

        log.info(
            "aggregator.complete",
            trace_id=message.trace_id,
            output_count=len(outputs),
            auxiliary_count=len(auxiliary_outputs),
            confidence=combined_confidence,
            sources=sources,
        )

        return AggregatedOutput(
            content=base_content,
            confidence=combined_confidence,
            metadata=metadata,
            sources=sources,
        )
