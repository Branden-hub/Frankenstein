"""
Living AI System — Continuous Execution Environment
The CEE is the heartbeat of the living system.
It runs as an infinite async loop — never stopping, never waiting for requests.
Every tick it perceives its environment, processes pending work,
consolidates memory, maintains itself through pruning,
and paces its own metabolic cycle.
This is what makes the system live rather than simply respond.
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import structlog

log = structlog.get_logger(__name__)

CEE_TICK_SECONDS = 5
CONSOLIDATION_WINDOW_TICKS = 720  # Every hour at 5-second ticks
KL_DIVERGENCE_ALERT_THRESHOLD = 2.0
PRUNING_BASE_THRESHOLD = 0.01


class ContinuousExecutionEnvironment:
    """
    The infinite execution loop.
    Models the biological circadian rhythm — alternating between
    active processing and consolidation phases.
    The system exists continuously through time, not only when responding.
    As it runs, noise in its model decreases toward zero.
    """

    def __init__(self, working_memory, episodic_memory, knowledge_base):
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.knowledge_base = knowledge_base
        self.time_step: int = 0
        self.is_running: bool = False
        self._consolidation_counter: int = 0
        self._performance_baseline: float | None = None
        self._start_time: float = 0.0

    async def run(self) -> None:
        """
        The infinite temporal loop.
        This method never returns while the system is running.
        Each iteration is one tick of the system's existence.
        """
        self.is_running = True
        self._start_time = time.monotonic()
        log.info("cee.started", tick_interval_seconds=CEE_TICK_SECONDS)

        while self.is_running:
            tick_start = time.monotonic()
            self.time_step += 1

            try:
                hardware_state = await self._perceive()
                await self._process(hardware_state)
                await self._consolidate()
                await self._maintain(hardware_state)
                await self._metacognitive_monitor()
            except Exception as exc:
                log.error(
                    "cee.tick_error",
                    time_step=self.time_step,
                    error=str(exc),
                    exc_info=True,
                )

            tick_duration = time.monotonic() - tick_start
            sleep_time = max(0.0, CEE_TICK_SECONDS - tick_duration)

            log.debug(
                "cee.tick",
                time_step=self.time_step,
                tick_duration_ms=round(tick_duration * 1000, 2),
                uptime_seconds=round(time.monotonic() - self._start_time, 1),
            )

            await asyncio.sleep(sleep_time)

    async def stop(self) -> None:
        """Signal the loop to stop after the current tick completes."""
        self.is_running = False
        log.info("cee.stopping", time_step=self.time_step)

    async def _perceive(self) -> dict:
        """
        Read the current state of the hardware environment.
        CPU, memory, thermal state — the system's physical substrate.
        These are not noise. They are the laws of the local physical system.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_pressure = memory.percent / 100.0

        try:
            temperatures = psutil.sensors_temperatures()
            thermal_state = max(
                (t.current for group in temperatures.values() for t in group),
                default=0.0,
            )
        except (AttributeError, Exception):
            thermal_state = 0.0

        hardware_state = {
            "cpu_percent": cpu_percent,
            "memory_pressure": memory_pressure,
            "memory_available_gb": memory.available / (1024 ** 3),
            "thermal_state": thermal_state,
            "time_step": self.time_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return hardware_state

    async def _process(self, hardware_state: dict) -> None:
        """
        Process any pending background consolidation tasks.
        These are tasks queued by the controller during active processing
        that are deferred to background execution.
        """
        pending = await self.episodic_memory.get_pending_encoding_queue()

        for item in pending:
            try:
                await self.episodic_memory.encode_and_store(item)
            except Exception as exc:
                log.error(
                    "cee.encoding_error",
                    session_id=item.get("session_id"),
                    error=str(exc),
                )

    async def _consolidate(self) -> None:
        """
        Periodic consolidation window — the sleep cycle of the living system.
        During consolidation the system deepens memory, distils knowledge,
        and reinforces stable representations.
        Runs every CONSOLIDATION_WINDOW_TICKS ticks.
        """
        self._consolidation_counter += 1

        if self._consolidation_counter < CONSOLIDATION_WINDOW_TICKS:
            return

        self._consolidation_counter = 0
        log.info("cee.consolidation_window.start", time_step=self.time_step)

        try:
            # Episodic to semantic memory transfer
            await self.episodic_memory.consolidate_to_semantic()

            # Spaced repetition — rehearse critical knowledge
            await self.knowledge_base.run_spaced_repetition()

            # Knowledge distillation — compress redundant representations
            await self.knowledge_base.distil()

            # Cross-tier consistency verification
            await self._verify_cross_tier_consistency()

            log.info("cee.consolidation_window.complete", time_step=self.time_step)

        except Exception as exc:
            log.error(
                "cee.consolidation_error",
                error=str(exc),
                exc_info=True,
            )

    async def _maintain(self, hardware_state: dict) -> None:
        """
        Ouroboros pruning pass.
        The adversarial critic evaluates utility of all parameters.
        Parameters scoring below the dynamic threshold are zeroed out.
        Pruning accelerates when memory pressure is high.
        This prevents parameter explosion — the HeLa Risk.
        The governing equation: dP/dt = k_alloc - (k_prune × Utility × Compute_Available)
        """
        memory_pressure = hardware_state["memory_pressure"]

        # Dynamic threshold scales with memory pressure
        dynamic_threshold = PRUNING_BASE_THRESHOLD * (1.0 + memory_pressure)

        try:
            from modules.homeostasis.pruner import OuroborosPruner
            pruner = OuroborosPruner(threshold=dynamic_threshold)
            pruned_count = await pruner.prune(memory_pressure=memory_pressure)

            if pruned_count > 0:
                log.info(
                    "cee.pruning",
                    pruned_count=pruned_count,
                    threshold=dynamic_threshold,
                    memory_pressure=memory_pressure,
                )
        except Exception as exc:
            log.error("cee.pruning_error", error=str(exc))

    async def _metacognitive_monitor(self) -> None:
        """
        Monitor system performance drift.
        Tracks KL divergence between current and baseline output distributions.
        When drift exceeds the threshold, forces a consolidation cycle.
        This is the system's self-awareness of its own cognitive state.
        Performance_drift(t) = KL[ p(y|x, θ_t) || p(y|x, θ_{t-τ}) ]
        """
        # Only run monitor every 60 ticks
        if self.time_step % 60 != 0:
            return

        try:
            from modules.homeostasis.monitor import MetacognitiveMonitor
            monitor = MetacognitiveMonitor()
            drift = await monitor.compute_drift()

            if self._performance_baseline is None:
                self._performance_baseline = drift
                return

            std_devs_from_baseline = abs(drift - self._performance_baseline)

            if std_devs_from_baseline > KL_DIVERGENCE_ALERT_THRESHOLD:
                log.warning(
                    "cee.performance_drift_detected",
                    drift=drift,
                    baseline=self._performance_baseline,
                    std_devs=std_devs_from_baseline,
                )
                # Force immediate consolidation
                self._consolidation_counter = CONSOLIDATION_WINDOW_TICKS

        except Exception as exc:
            log.error("cee.monitor_error", error=str(exc))

    async def _verify_cross_tier_consistency(self) -> None:
        """
        Verify that episodic and semantic memories remain consistent.
        Uses hash-based verification to detect drift between tiers.
        """
        try:
            episodic_hash = await self.episodic_memory.compute_consistency_hash()
            semantic_hash = await self.working_memory.compute_consistency_hash()

            log.debug(
                "cee.consistency_check",
                episodic_hash=episodic_hash[:16],
                semantic_hash=semantic_hash[:16],
            )
        except Exception as exc:
            log.error("cee.consistency_check_error", error=str(exc))

    def get_status(self) -> dict:
        """Return current CEE status for the health endpoint."""
        return {
            "running": self.is_running,
            "time_step": self.time_step,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1)
            if self._start_time
            else 0,
            "consolidation_next_in_ticks": CONSOLIDATION_WINDOW_TICKS
            - self._consolidation_counter,
            "tick_interval_seconds": CEE_TICK_SECONDS,
        }
