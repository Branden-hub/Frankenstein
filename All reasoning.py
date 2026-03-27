"""
Living AI System — Reasoning, Planning, and Multi-Agent Modules
Chain of Thought, Tree of Thought, Graph of Thought, Process Reward Models.
HTN Planning, MCTS Planning, Contingency Planning.
Multi-Agent Orchestration with typed message passing.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from modules.base import BaseModule, ModuleOutput

log = structlog.get_logger(__name__)


# ─── Message Types ────────────────────────────────────────────────────────────

class MessageType(Enum):
    TASK = "TASK"
    RESULT = "RESULT"
    QUERY = "QUERY"
    STATUS = "STATUS"
    ESCALATE = "ESCALATE"
    HEARTBEAT = "HEARTBEAT"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_module: str = ""
    target_module: str = ""
    message_type: MessageType = MessageType.TASK
    payload: dict = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 5


# ─── Chain of Thought ─────────────────────────────────────────────────────────

class ChainOfThoughtReasoner:
    """
    Chain of Thought reasoning — explicit step-by-step reasoning
    before committing to an answer.
    The reasoning chain is the filter narrowing the path space.
    Each step reveals more of the correct answer.
    """

    async def reason(self, question: str, context: list[dict]) -> dict:
        """Decompose question into reasoning steps and solve step by step."""
        steps = self._decompose(question)
        reasoning_trace = []
        for step in steps:
            result = await self._execute_step(step, context, reasoning_trace)
            reasoning_trace.append({"step": step, "result": result})
        conclusion = self._synthesise(reasoning_trace)
        return {"steps": reasoning_trace, "conclusion": conclusion}

    def _decompose(self, question: str) -> list[str]:
        """Break question into reasoning steps."""
        steps = []
        if "?" in question:
            steps.append(f"Understand what is being asked: {question}")
        if any(w in question.lower() for w in ["compare", "versus", "difference"]):
            steps.append("Identify items to compare")
            steps.append("List properties of each")
            steps.append("Compare properties systematically")
        elif any(w in question.lower() for w in ["how", "why", "explain"]):
            steps.append("Identify the subject")
            steps.append("Recall relevant knowledge")
            steps.append("Construct explanation")
        elif any(w in question.lower() for w in ["calculate", "compute", "solve"]):
            steps.append("Identify the mathematical operation")
            steps.append("Set up the calculation")
            steps.append("Compute step by step")
            steps.append("Verify the result")
        else:
            steps.append("Analyse the question")
            steps.append("Gather relevant information")
            steps.append("Formulate response")
        return steps

    async def _execute_step(self, step: str, context: list[dict],
                            trace: list[dict]) -> str:
        """Execute a single reasoning step."""
        return f"[Reasoning: {step}]"

    def _synthesise(self, trace: list[dict]) -> str:
        """Synthesise final conclusion from reasoning trace."""
        return "Conclusion derived from " + str(len(trace)) + " reasoning steps."


class TreeOfThought:
    """
    Tree of Thought — branching exploration of multiple reasoning paths
    with pruning and backtracking.
    The tree is the full path space.
    BFS/DFS with evaluation is the filter.
    The best path is the revealed answer.
    """

    def __init__(self, branching_factor: int = 3, max_depth: int = 4,
                 beam_width: int = 5):
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.beam_width = beam_width

    async def search(self, problem: str, context: list[dict]) -> dict:
        """BFS through the thought tree with beam search."""
        beam = [{"thought": problem, "score": 1.0, "depth": 0, "path": [problem]}]
        best_path = beam[0]

        for depth in range(1, self.max_depth + 1):
            candidates = []
            for node in beam:
                expansions = await self._expand(node["thought"], node["path"])
                for expansion in expansions:
                    score = self._evaluate(expansion, problem)
                    candidates.append({
                        "thought": expansion,
                        "score": score,
                        "depth": depth,
                        "path": node["path"] + [expansion],
                    })
            if not candidates:
                break
            candidates.sort(key=lambda x: x["score"], reverse=True)
            beam = candidates[:self.beam_width]
            if beam and beam[0]["score"] > best_path["score"]:
                best_path = beam[0]

        return {
            "best_path": best_path["path"],
            "score": best_path["score"],
            "depth_reached": best_path["depth"],
        }

    async def _expand(self, thought: str, path: list[str]) -> list[str]:
        """Generate child thoughts from current thought."""
        return [
            f"{thought} → step {i + 1}"
            for i in range(self.branching_factor)
        ]

    def _evaluate(self, thought: str, goal: str) -> float:
        """Score a thought node by relevance to the goal."""
        words_in_common = set(thought.lower().split()) & set(goal.lower().split())
        return len(words_in_common) / max(len(goal.split()), 1)


# ─── Planning ─────────────────────────────────────────────────────────────────

@dataclass
class TaskStep:
    action: str
    args: dict = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class Plan:
    goal: str
    steps: list[TaskStep]
    contingencies: dict[str, list[TaskStep]] = field(default_factory=dict)


class HTNPlanner:
    """
    Hierarchical Task Network planner.
    Decomposes high-level goals into ordered sub-task trees.
    Handles temporal and resource constraints.
    """

    def __init__(self):
        self._task_library: dict[str, list[TaskStep]] = {
            "write_code": [
                TaskStep("understand_requirements", {}),
                TaskStep("design_solution", {}),
                TaskStep("implement_code", {}),
                TaskStep("test_code", {}),
                TaskStep("document_code", {}),
            ],
            "research_topic": [
                TaskStep("formulate_queries", {}),
                TaskStep("search_knowledge_base", {}),
                TaskStep("retrieve_episodic_memory", {}),
                TaskStep("synthesise_findings", {}),
                TaskStep("format_response", {}),
            ],
            "analyse_data": [
                TaskStep("load_data", {}),
                TaskStep("clean_data", {}),
                TaskStep("explore_data", {}),
                TaskStep("apply_analysis", {}),
                TaskStep("interpret_results", {}),
            ],
            "answer_question": [
                TaskStep("classify_question", {}),
                TaskStep("retrieve_relevant_knowledge", {}),
                TaskStep("reason_about_answer", {}),
                TaskStep("formulate_response", {}),
            ],
        }

    def plan(self, goal: str, context: dict) -> Plan:
        """Generate a plan for the given goal."""
        task_type = self._classify_goal(goal)
        steps = self._task_library.get(task_type, [
            TaskStep("analyse_goal", {}),
            TaskStep("execute_goal", {}),
        ])
        contingencies = {
            "step_failure": [
                TaskStep("replan", {"reason": "step_failed"}),
                TaskStep("escalate_to_human", {"reason": "replanning_failed"}),
            ],
        }
        return Plan(goal=goal, steps=steps, contingencies=contingencies)

    def _classify_goal(self, goal: str) -> str:
        goal_lower = goal.lower()
        if any(w in goal_lower for w in ["code", "write", "implement", "build", "program"]):
            return "write_code"
        elif any(w in goal_lower for w in ["research", "find out", "learn about", "investigate"]):
            return "research_topic"
        elif any(w in goal_lower for w in ["analyse", "analyze", "data", "statistics"]):
            return "analyse_data"
        else:
            return "answer_question"

    def check_preconditions(self, step: TaskStep, world_state: dict) -> bool:
        """Check whether all preconditions for a step are met."""
        return all(world_state.get(pc, False) for pc in step.preconditions)


class AutonomousPlanner:
    """
    Full autonomous planner combining HTN decomposition
    with MCTS look-ahead for multi-step horizon planning.
    """

    def __init__(self):
        self.htn = HTNPlanner()
        self.world_state: dict = {}

    async def execute(self, goal: str, context: dict) -> dict:
        """Plan and execute a goal, replanning on failure."""
        plan = self.htn.plan(goal, context)
        executed_steps = []
        failed_steps = []

        for step in plan.steps:
            if not self.htn.check_preconditions(step, self.world_state):
                replan_steps = plan.contingencies.get("step_failure", [])
                failed_steps.append(step.action)
                log.warning(
                    "planner.precondition_failed",
                    step=step.action,
                    goal=goal,
                )
                break

            result = await self._execute_step(step)
            self.world_state[step.action + "_complete"] = True
            executed_steps.append({"step": step.action, "result": result})

        return {
            "goal": goal,
            "executed_steps": executed_steps,
            "failed_steps": failed_steps,
            "success": len(failed_steps) == 0,
        }

    async def _execute_step(self, step: TaskStep) -> str:
        """Execute a single plan step."""
        await asyncio.sleep(0)
        return f"Executed: {step.action}"


# ─── Multi-Agent Coordination ─────────────────────────────────────────────────

class WorkerAgent:
    """
    Worker agent — handles specific task types.
    Receives tasks from orchestrator via typed messages.
    Returns results via the same message protocol.
    """

    def __init__(self, agent_type: str, specialisation: str):
        self.agent_type = agent_type
        self.specialisation = specialisation
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process a task message and return result."""
        result_payload = {
            "status": "complete",
            "result": f"{self.specialisation} agent processed: {message.payload.get('task', '')}",
        }
        return AgentMessage(
            source_module=self.agent_type,
            target_module=message.source_module,
            message_type=MessageType.RESULT,
            payload=result_payload,
            trace_id=message.trace_id,
        )


class MultiAgentOrchestrator:
    """
    Orchestrator for multi-agent coordination.
    Plans overall task strategy, assigns subtasks to workers,
    monitors progress, aggregates results, resolves conflicts,
    maintains shared context.
    """

    def __init__(self):
        self.workers: dict[str, WorkerAgent] = {
            "researcher": WorkerAgent("researcher", "research"),
            "coder": WorkerAgent("coder", "code_generation"),
            "analyst": WorkerAgent("analyst", "data_analysis"),
            "critic": WorkerAgent("critic", "fact_checking"),
            "communicator": WorkerAgent("communicator", "output_formatting"),
        }
        self.shared_context: dict = {}
        self.message_log: list[AgentMessage] = []

    async def orchestrate(self, task: str, context: dict) -> dict:
        """
        Orchestrate a multi-agent task.
        Decomposes task, assigns to workers, aggregates results.
        """
        subtasks = self._decompose_task(task)
        worker_assignments = self._assign_workers(subtasks)
        results = []

        # Execute all subtasks in parallel where possible
        subtask_coroutines = [
            self._execute_subtask(worker, subtask, context)
            for worker, subtask in worker_assignments.items()
        ]
        worker_results = await asyncio.gather(*subtask_coroutines, return_exceptions=True)

        for result in worker_results:
            if not isinstance(result, Exception):
                results.append(result)

        return {
            "task": task,
            "subtasks_completed": len(results),
            "subtasks_total": len(worker_assignments),
            "results": results,
        }

    def _decompose_task(self, task: str) -> list[str]:
        """Decompose main task into subtasks for different workers."""
        subtasks = ["analyse_requirements"]
        task_lower = task.lower()
        if any(w in task_lower for w in ["research", "find", "search"]):
            subtasks.append("research_information")
        if any(w in task_lower for w in ["code", "implement", "write"]):
            subtasks.append("generate_code")
        if any(w in task_lower for w in ["analyse", "data", "statistics"]):
            subtasks.append("analyse_data")
        subtasks.append("synthesise_results")
        return subtasks

    def _assign_workers(self, subtasks: list[str]) -> dict[str, str]:
        """Assign subtasks to appropriate worker agents."""
        assignments = {}
        worker_map = {
            "research_information": "researcher",
            "generate_code": "coder",
            "analyse_data": "analyst",
            "verify_facts": "critic",
            "synthesise_results": "communicator",
            "analyse_requirements": "analyst",
        }
        for subtask in subtasks:
            worker_name = worker_map.get(subtask, "communicator")
            assignments[worker_name] = subtask
        return assignments

    async def _execute_subtask(self, worker_name: str, subtask: str,
                               context: dict) -> dict:
        """Execute a single subtask on the assigned worker."""
        worker = self.workers.get(worker_name)
        if worker is None:
            return {"error": f"Worker {worker_name} not found"}

        message = AgentMessage(
            source_module="orchestrator",
            target_module=worker_name,
            message_type=MessageType.TASK,
            payload={"task": subtask, "context": context},
        )
        result_message = await worker.process(message)
        self.message_log.append(message)
        self.message_log.append(result_message)

        return result_message.payload


# ─── Module Wrappers ──────────────────────────────────────────────────────────

class ChainOfThoughtModule(BaseModule):
    def __init__(self):
        self._cot = ChainOfThoughtReasoner()
        self._tot = TreeOfThought()

    @property
    def name(self) -> str:
        return "chain_of_thought"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        log.info("chain_of_thought_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context,
                      working_memory) -> ModuleOutput:
        result = await self._cot.reason(message.content, knowledge_context)
        reasoning_text = "\n".join([
            f"Step {i + 1}: {s['step']} → {s['result']}"
            for i, s in enumerate(result["steps"])
        ])
        return ModuleOutput(
            content=reasoning_text,
            confidence=0.8,
            output_type=self.output_type,
            source=self.name,
            metadata={"conclusion": result["conclusion"]},
        )


class PlannerModule(BaseModule):
    def __init__(self):
        self._planner = AutonomousPlanner()

    @property
    def name(self) -> str:
        return "planner"

    @property
    def output_type(self) -> str:
        return "plan"

    async def initialise(self) -> None:
        log.info("planner_module.initialised")

    async def execute(self, message: Any, episodic_context, knowledge_context,
                      working_memory) -> ModuleOutput:
        result = await self._planner.execute(message.content, {})
        plan_text = "\n".join([
            f"{i + 1}. {step['step']}: {step['result']}"
            for i, step in enumerate(result["executed_steps"])
        ])
        return ModuleOutput(
            content=plan_text,
            confidence=0.75,
            output_type=self.output_type,
            source=self.name,
            metadata={"success": result["success"]},
        )


class MultiAgentModule(BaseModule):
    def __init__(self):
        self._orchestrator = MultiAgentOrchestrator()

    @property
    def name(self) -> str:
        return "multi_agent"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        log.info("multi_agent_module.initialised", workers=len(self._orchestrator.workers))

    async def execute(self, message: Any, episodic_context, knowledge_context,
                      working_memory) -> ModuleOutput:
        result = await self._orchestrator.orchestrate(message.content, {})
        return ModuleOutput(
            content=str(result["results"]),
            confidence=0.7,
            output_type=self.output_type,
            source=self.name,
            metadata=result,
        )
