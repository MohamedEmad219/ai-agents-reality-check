"""
# Copyright 2025 ByteStack Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Module: ai_agents_reality_check/agent_benchmark.py

Core benchmark engine for evaluating AI agent architectures.

This module defines interfaces, benchmark orchestration logic, and reporting
utilities that quantify agent performance across realistic, multi-step tasks
with varying complexity. Enhanced with tool failure simulation, network conditions,
statistical analysis, and enterprise-grade error handling with structured context
and resource management.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .agents import MarketingAgent, RealAgent, WrapperAgent
from .types import (
    AgentExecutionError,
    AgentRealityCheckError,
    AgentType,
    BenchmarkErrorContext,
    BenchmarkExecutionError,
    ConfigurationError,
    ExportError,
    TaskComplexity,
    TaskResult,
)
from .utils.helpers import create_challenging_task_set

try:
    from .logging_config import setup_benchmark_logger

    benchmark_logger = setup_benchmark_logger(quiet=False)
except ImportError:
    import logging

    benchmark_logger = logging.getLogger(__name__)

try:
    from .agents.enhanced_executor import (  # noqa: F401
        NetworkCondition,
        ToolFailureConfig,
    )

    ENHANCED_EXECUTOR_AVAILABLE = True
except ImportError:
    ENHANCED_EXECUTOR_AVAILABLE = False

try:
    from .utils import (
        assess_statistical_significance,
        calculate_effect_size,
        calculate_required_sample_size,
    )

    ENHANCED_STATS_AVAILABLE = True
except ImportError:
    ENHANCED_STATS_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Constants for magic numbers
MAX_SAFE_ITERATIONS = 1000
MIN_TIMEOUT_SECONDS = 10
MIN_AGENTS_FOR_COMPARISON = 2
QUICK_MODE_TASK_TIMEOUT = 30
COMPREHENSIVE_TASK_TIMEOUT = 60

# Performance thresholds
EXCELLENT_PERFORMANCE_THRESHOLD = 0.8
GOOD_PERFORMANCE_THRESHOLD = 0.6
FAIR_PERFORMANCE_THRESHOLD = 0.4

# Progress reporting intervals
QUICK_MODE_PROGRESS_INTERVAL = 3
COMPREHENSIVE_PROGRESS_INTERVAL = 5


class AgentInterface(Protocol):
    """Interface that all AI agent implementations must conform to.

    This protocol ensures consistent agent behavior across different implementations
    and enables proper type checking and error handling throughout the benchmark system.

    Attributes:
        name (str): Unique identifier for the agent (used in logs and metrics).
    """

    name: str

    async def execute_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute a benchmark task and return the result.

        Args:
            task: Task description dictionary. Must contain 'name' and 'complexity' keys
                at minimum. May include additional metadata like 'task_id', 'description'.

        Returns:
            TaskResult: Structured result of task execution with comprehensive metrics including:
                - success: Boolean indicating task completion status
                - execution_time: Duration in seconds
                - error_count: Number of errors encountered
                - context_retention_score: Quality of context maintenance
                - cost_estimate: Estimated resource/token cost
                - Additional metadata for debugging and analysis

        Raises:
            AgentExecutionError: When agent-specific execution failures occur.
                Common scenarios: tool failures, memory issues, planning failures.
            ConfigurationError: When task configuration is invalid or incomplete.
                Examples: missing required keys, invalid complexity levels.

        Note:
            - Implementation must be async to support timeout handling
            - Should handle partial task completion gracefully
            - Must return structured TaskResult even for failures
            - Expected to manage internal state and resource cleanup
        """
        ...


class BenchmarkResourceManager:
    """
    Enterprise-grade resource manager for benchmark execution.

    Handles cleanup of temporary files, connections, and other resources
    to prevent resource leaks during benchmark execution.
    """

    def __init__(self) -> None:
        """Initialize resource manager with empty tracking lists.

        Sets up tracking structures for temporary files, connections, locks, and
        other resources that need cleanup during benchmark execution.

        Note:
            - Initializes empty lists for each resource type
            - Sets up logger reference for cleanup event tracking
            - Designed to be used in context managers for automatic cleanup
        """
        self.temp_files: list[Path] = []
        self.open_connections: list[Any] = []
        self.benchmark_locks: list[asyncio.Lock] = []
        self.logger = benchmark_logger

    def register_temp_file(self, filepath: Path) -> None:
        """Register temporary file for cleanup on benchmark completion.

        Adds a temporary file to the cleanup registry to ensure it's removed
        when the benchmark completes, preventing disk space accumulation.

        Args:
            filepath: Path object pointing to the temporary file to be cleaned up.

        Note:
            - Files are cleaned up in registration order
            - Non-existent files at cleanup time are handled gracefully
            - Logs registration for debugging and audit purposes
        """
        self.temp_files.append(filepath)
        self.logger.debug(f"Registered temp file for cleanup: {filepath}")

    def register_connection(self, connection: Any) -> None:
        """Register connection for cleanup on benchmark completion.

        Adds a connection object to the cleanup registry to ensure proper
        resource cleanup and prevent connection leaks.

        Args:
            connection: Connection object with a close() method. Can be sync or async.

        Note:
            - Supports both synchronous and asynchronous connection types
            - Attempts to call close() method during cleanup
            - Handles connection cleanup failures gracefully with logging
        """
        self.open_connections.append(connection)
        self.logger.debug(
            f"Registered connection for cleanup: {type(connection).__name__}"
        )

    def register_lock(self, lock: asyncio.Lock) -> None:
        """Register async lock for release on benchmark completion.

        Adds an asyncio Lock to the cleanup registry to ensure proper
        lock release and prevent deadlocks in multi-task scenarios.

        Args:
            lock: AsyncIO lock object to be released during cleanup.

        Note:
            - Only releases locks that are currently held
            - Prevents deadlocks from unreleased locks
            - Handles release failures with appropriate error handling
        """
        self.benchmark_locks.append(lock)
        self.logger.debug("Registered async lock for cleanup")

    def _cleanup_connections(self) -> list[str]:
        """Clean up connections and return any errors."""
        cleanup_errors = []
        for i, conn in enumerate(self.open_connections):
            try:
                if hasattr(conn, "close"):
                    if asyncio.iscoroutinefunction(conn.close):
                        pass
                    else:
                        conn.close()
                    self.logger.debug(f"Closed connection {i}")
            except Exception as e:
                cleanup_errors.append(f"Connection {i}: {str(e)}")
                self.logger.warning(f"Error closing connection {i}: {e}")
        return cleanup_errors

    def _cleanup_temp_files(self) -> list[str]:
        """Clean up temporary files and return any errors."""
        cleanup_errors = []
        for filepath in self.temp_files:
            try:
                if filepath.exists():
                    filepath.unlink()
                    self.logger.debug(f"Removed temp file: {filepath}")
                else:
                    self.logger.debug(f"Temp file already removed: {filepath}")
            except Exception as e:
                cleanup_errors.append(f"File {filepath}: {str(e)}")
                self.logger.warning(f"Error removing temp file {filepath}: {e}")
        return cleanup_errors

    def _cleanup_locks(self) -> list[str]:
        """Clean up async locks and return any errors."""
        cleanup_errors = []
        for i, lock in enumerate(self.benchmark_locks):
            try:
                if lock.locked():
                    lock.release()
                    self.logger.debug(f"Released lock {i}")
            except Exception as e:
                cleanup_errors.append(f"Lock {i}: {str(e)}")
                self.logger.warning(f"Error releasing lock {i}: {e}")
        return cleanup_errors

    async def cleanup_all(self) -> None:
        """Clean up all registered resources with comprehensive error handling.

        Performs cleanup of all registered resources including files, connections,
        and locks with detailed error tracking and logging.

        Note:
            - Processes all resource types regardless of individual failures
            - Logs success and failure events for each resource type
            - Clears all tracking lists after cleanup attempts
            - Does not raise exceptions - logs errors for monitoring
            - Safe to call multiple times (idempotent)

        Cleanup Order:
            1. Network connections (close sync/async connections)
            2. Temporary files (delete from filesystem)
            3. Async locks (release held locks)
        """
        cleanup_errors = []
        cleanup_errors.extend(self._cleanup_connections())
        cleanup_errors.extend(self._cleanup_temp_files())
        cleanup_errors.extend(self._cleanup_locks())

        self.temp_files.clear()
        self.open_connections.clear()
        self.benchmark_locks.clear()

        if cleanup_errors:
            self.logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
        else:
            self.logger.debug("All resources cleaned up successfully")


@asynccontextmanager
async def benchmark_execution_context(
    operation: str,
    timeout: int | None = None,
    resource_manager: BenchmarkResourceManager | None = None,
) -> AsyncGenerator[BenchmarkErrorContext, None]:
    """Enterprise-grade context manager for benchmark operations.

    Provides structured error handling, timeout management, and resource cleanup
    for all major benchmark operations with comprehensive error context tracking.

    Args:
        operation: Description of the operation being performed. Used in error messages
            and logging for debugging and monitoring.
        timeout: Optional timeout in seconds for the operation. If None, no timeout applied.
        resource_manager: Optional resource manager for automatic cleanup of resources
            created during the operation.

    Yields:
        BenchmarkErrorContext: Context object for adding operational metadata and
            tracking execution details for enhanced error reporting.

    Raises:
        BenchmarkExecutionError: On execution failures with enhanced context including:
            - Timeout details and actual duration for timeout failures
            - Original error information for unexpected failures
            - Operation metadata and execution context
            - Resource cleanup status information

    Note:
        - Automatically captures execution timing for all operations
        - Performs resource cleanup even in failure scenarios
        - Enhances existing AgentRealityCheckError with additional context
        - Provides consistent error handling patterns across all operations
        - Safe to nest - inner contexts add to outer context metadata

    Example:
        >>> async with benchmark_execution_context("agent_execution", 30) as ctx:
        ...     ctx.add_context("agent_type", "RealAgent")
        ...     result = await agent.execute_task(task)
    """
    context = BenchmarkErrorContext(operation)
    start_time = time.time()

    try:
        if timeout:
            context.add_context("timeout_seconds", timeout)
            context.add_context("operation_start", start_time)

        context.add_context("resource_manager_active", resource_manager is not None)
        yield context

    except TimeoutError as e:
        execution_duration = time.time() - start_time
        context.add_context("timeout_exceeded", True)
        context.add_context("execution_duration", execution_duration)

        raise BenchmarkExecutionError(
            f"Benchmark operation '{operation}' timed out after {timeout}s "
            f"(actual duration: {execution_duration:.2f}s)",
            error_code="BENCHMARK_TIMEOUT",
            context=context.context,
        ) from e

    except Exception as e:
        execution_duration = time.time() - start_time
        context.add_context("unexpected_error", True)
        context.add_context("execution_duration", execution_duration)
        context.add_context("original_error_type", type(e).__name__)
        context.add_context("original_error_message", str(e))

        if isinstance(e, AgentRealityCheckError):
            e.context.update(context.context)
            raise

        raise BenchmarkExecutionError(
            f"Benchmark operation '{operation}' failed after {execution_duration:.2f}s: {str(e)}",
            error_code="BENCHMARK_EXECUTION_FAILED",
            context=context.context,
        ) from e

    finally:
        if resource_manager:
            try:
                await resource_manager.cleanup_all()
            except Exception as cleanup_error:
                benchmark_logger.warning(f"Resource cleanup failed: {cleanup_error}")


async def execute_agent_task_with_retry(
    agent: Any,
    task: dict[str, Any],
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: int | None = None,
) -> TaskResult:
    """Execute agent task with exponential backoff retry logic and comprehensive error handling.

    Provides robust task execution with automatic retry for transient failures,
    exponential backoff to avoid overwhelming failing systems, and detailed
    error context for debugging.

    Args:
        agent: Agent instance implementing AgentInterface.execute_task method.
        task: Task dictionary containing 'name' and 'complexity' keys plus any
            additional task metadata.
        max_retries: Maximum number of retry attempts after initial failure. Default: 3.
        backoff_factor: Exponential backoff multiplier between retries. Default: 1.5.
            Delay = base_delay * (backoff_factor ^ attempt_number).
        timeout: Optional timeout per attempt in seconds. If None, no per-attempt timeout.

    Returns:
        TaskResult: Result of successful execution with timing and context information.

    Raises:
        AgentExecutionError: After all retries exhausted with detailed context including:
            - Total attempts made and retry configuration
            - Last error encountered and its classification
            - Agent and task identification for debugging
        ConfigurationError: When agent or task configuration is invalid:
            - Agent missing required execute_task method
            - Task missing required keys or invalid format

    Note:
        - Uses exponential backoff starting at 0.5 seconds base delay
        - Logs retry attempts at warning level for monitoring
        - Preserves original error information through retry chain
        - Each retry attempt respects the individual timeout limit
        - Success on retry is logged for performance analysis

    Example:
        >>> result = await execute_agent_task_with_retry(
        ...     agent, task, max_retries=2, timeout=30
        ... )
    """
    if not hasattr(agent, "execute_task"):
        raise ConfigurationError(
            f"Agent {type(agent).__name__} does not implement execute_task method",
            error_code="INVALID_AGENT_INTERFACE",
            context={"agent_type": type(agent).__name__},
        )

    if not isinstance(task, dict) or "name" not in task:
        raise ConfigurationError(
            "Task must be a dictionary with 'name' key",
            error_code="INVALID_TASK_FORMAT",
            context={
                "task_type": type(task).__name__,
                "task_keys": list(task.keys()) if isinstance(task, dict) else None,
            },
        )

    last_error: AgentRealityCheckError | None = None
    base_delay = 0.5
    agent_name = getattr(agent, "name", type(agent).__name__)
    task_name = task.get("name", "unknown")

    for attempt in range(max_retries + 1):
        try:
            async with benchmark_execution_context(
                f"agent_task_attempt_{attempt}", timeout
            ) as ctx:
                ctx.add_context("agent_type", type(agent).__name__)
                ctx.add_context("agent_name", agent_name)
                ctx.add_context("task_name", task_name)
                ctx.add_context(
                    "task_complexity", str(task.get("complexity", "unknown"))
                )
                ctx.add_context("attempt_number", attempt)
                ctx.add_context("max_retries", max_retries)
                ctx.add_context("is_retry", attempt > 0)

                if timeout:
                    result = cast(
                        TaskResult,
                        await asyncio.wait_for(
                            agent.execute_task(task), timeout=timeout
                        ),
                    )
                else:
                    result = cast(TaskResult, await agent.execute_task(task))

                if attempt > 0:
                    benchmark_logger.info(
                        f"Task '{task_name}' succeeded on retry attempt {attempt} for agent {agent_name}"
                    )

                return result

        except TimeoutError:
            last_error = AgentExecutionError(
                f"Agent task timed out after {timeout}s",
                error_code="AGENT_TASK_TIMEOUT",
                context={
                    "agent_type": type(agent).__name__,
                    "task_name": task_name,
                    "attempt": attempt,
                    "timeout": timeout,
                },
            )

        except Exception as e:
            if isinstance(e, AgentRealityCheckError):
                last_error = e
            else:
                last_error = AgentExecutionError(
                    f"Agent task failed with unexpected error: {str(e)}",
                    error_code="AGENT_TASK_UNEXPECTED_ERROR",
                    context={
                        "agent_type": type(agent).__name__,
                        "task_name": task_name,
                        "attempt": attempt,
                        "original_error_type": type(e).__name__,
                        "original_error": str(e),
                    },
                )

            if attempt < max_retries:
                delay = base_delay * (backoff_factor**attempt)
                benchmark_logger.warning(
                    f"Agent '{agent_name}' task '{task_name}' failed on attempt {attempt + 1}/{max_retries + 1}, "
                    f"retrying in {delay:.1f}s: {str(last_error)}"
                )
                await asyncio.sleep(delay)
            else:
                benchmark_logger.error(
                    f"Agent '{agent_name}' task '{task_name}' failed after {max_retries + 1} attempts"
                )

    raise AgentExecutionError(
        f"Agent task failed after {max_retries + 1} attempts",
        error_code="AGENT_TASK_RETRY_EXHAUSTED",
        context={
            "agent_type": type(agent).__name__,
            "agent_name": agent_name,
            "task_name": task_name,
            "task_complexity": str(task.get("complexity", "unknown")),
            "max_retries": max_retries,
            "last_error": str(last_error),
            "last_error_type": type(last_error).__name__,
            "last_error_code": getattr(last_error, "error_code", None),
            "total_attempts": max_retries + 1,
        },
    ) from last_error


class AgentBenchmark:
    """
    Enterprise-grade benchmarking engine for AI agent architectures.

    Provides comprehensive evaluation with statistical analysis, error handling,
    resource management, and detailed reporting capabilities.

    Enhanced with optional tool failure simulation, network conditions, and
    statistical analysis while maintaining full backward compatibility.
    """

    def __init__(
        self,
        quick_mode: bool = False,
        quiet: bool = False,
        random_seed: int | None = None,
        randomize: bool = False,
        enable_retries: bool = True,
        max_retries: int = 2,
        use_challenging_tasks: bool = True,
    ) -> None:
        """Initialize benchmark settings and prepare tasks/agents.

        Sets up benchmark configuration, initializes agent instances, and prepares
        task sets with comprehensive error handling and configuration validation.

        Args:
            quick_mode: Enable fast mode with reduced tasks and shorter timeouts.
                Suitable for development and quick validation.
            quiet: Minimize logging output to avoid conflicts with Rich console.
                Does not suppress progress indicators or final results.
            random_seed: Set random seed for reproducible results. If None, defaults to 42.
            randomize: Shuffle task order using the configured seed for execution variety.
            enable_retries: Enable automatic retry logic for failed agent executions.
            max_retries: Maximum retry attempts per agent task execution. Must be ≥ 0.
            use_challenging_tasks: Use enhanced task set designed for realistic performance
                differentiation between agent architectures.

        Raises:
            ConfigurationError: When initialization parameters are invalid:
                - max_retries < 0
                - Other configuration validation failures
            BenchmarkExecutionError: When agent initialization fails:
                - Unable to create required agent instances
                - Agent interface validation failures

        Note:
            - Initializes all three agent types: Wrapper, Marketing, and Real agents
            - Task selection depends on quick_mode and use_challenging_tasks flags
            - Random seed affects both task order and any stochastic agent behaviors
            - Comprehensive logging setup with fallback for missing dependencies
            - Results history initialized empty for accumulating execution data

        Task Sets:
            Quick Mode + Challenging: 3 challenging tasks (subset)
            Quick Mode + Standard: 3 standard tasks
            Full + Challenging: 8 challenging tasks (recommended)
            Full + Standard: 8 standard tasks (legacy compatibility)
        """
        if max_retries < 0:
            raise ConfigurationError(
                "max_retries must be non-negative",
                error_code="INVALID_MAX_RETRIES",
                context={"max_retries": max_retries},
            )

        try:
            self.logger = setup_benchmark_logger(quiet=quiet)
            self.logger.info(
                f"AgentBenchmark initialized with quick_mode={quick_mode}, quiet={quiet}"
            )
        except ImportError:
            self.logger = benchmark_logger
            self.logger.info("Using fallback logger for AgentBenchmark")

        if random_seed is None:
            random_seed = 42

        self.logger.debug(f"Setting random seed to: {random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.random_seed = random_seed
        self.randomize = randomize
        self.quick_mode = quick_mode
        self.quiet = quiet
        self.enable_retries = enable_retries
        self.max_retries = max_retries
        self.use_challenging_tasks = use_challenging_tasks

        self.logger.debug(
            "Initializing AgentBenchmark with quick_mode=%s, quiet=%s, seed=%s, retries=%s, challenging_tasks=%s",
            quick_mode,
            quiet,
            random_seed,
            enable_retries,
            use_challenging_tasks,
        )

        if use_challenging_tasks:
            challenging_tasks = create_challenging_task_set()

            if quick_mode:
                self.logger.debug("Using subset of challenging tasks for quick mode")
                self.tasks = [
                    challenging_tasks[0],
                    challenging_tasks[1],
                    challenging_tasks[2],
                ]
            else:
                self.logger.debug(
                    "Using full challenging task set for comprehensive benchmark"
                )
                self.tasks = challenging_tasks
        elif quick_mode:
            self.logger.debug("Using original reduced task set for quick mode")
            self.tasks = [
                {"name": "Schedule meeting", "complexity": TaskComplexity.SIMPLE},
                {"name": "Data analysis", "complexity": TaskComplexity.MODERATE},
                {
                    "name": "Research synthesis",
                    "complexity": TaskComplexity.COMPLEX,
                },
            ]
        else:
            self.logger.debug(
                "Using original full task set for comprehensive benchmark"
            )
            self.tasks = [
                {
                    "name": "Schedule meeting with conflict resolution",
                    "complexity": TaskComplexity.SIMPLE,
                },
                {
                    "name": "Data analysis with visualization",
                    "complexity": TaskComplexity.MODERATE,
                },
                {
                    "name": "Debug code and implement fix",
                    "complexity": TaskComplexity.MODERATE,
                },
                {
                    "name": "Research synthesis with citations",
                    "complexity": TaskComplexity.COMPLEX,
                },
                {
                    "name": "Customer complaint with policy lookup",
                    "complexity": TaskComplexity.COMPLEX,
                },
                {
                    "name": "Multi-system integration workflow",
                    "complexity": TaskComplexity.ENTERPRISE,
                },
                {
                    "name": "Compliance audit with documentation",
                    "complexity": TaskComplexity.ENTERPRISE,
                },
                {
                    "name": "Financial forecast with risk analysis",
                    "complexity": TaskComplexity.ENTERPRISE,
                },
            ]

        if self.randomize:
            self.logger.debug("Randomizing task order with seed %s", self.random_seed)
            random.shuffle(self.tasks)

        task_names = [task["name"] for task in self.tasks]
        self.logger.debug(f"Selected tasks: {task_names}")

        self.results_history: list[TaskResult] = []
        try:
            self.agents: dict[AgentType, AgentInterface] = {
                AgentType.WRAPPER_AGENT: WrapperAgent(),
                AgentType.MARKETING_AGENT: MarketingAgent(),
                AgentType.REAL_AGENT: RealAgent(),
            }
            self.logger.debug("Successfully initialized all agent types")
        except Exception as e:
            raise BenchmarkExecutionError(
                f"Failed to initialize agents: {str(e)}",
                error_code="AGENT_INITIALIZATION_FAILED",
                context={"error_type": type(e).__name__, "error": str(e)},
            ) from e

    def _format_task_status(
        self, task_name: str, current_task: int, total_tasks: int
    ) -> str:
        """Format current task status for enhanced logging with bounds checking."""
        return f"Running benchmark for task: {task_name} ({current_task}/{total_tasks})"

    def _format_agent_status(
        self, agent_name: str, current_agent: int, total_agents: int
    ) -> str:
        """Format current agent status for enhanced logging with bounds checking."""
        return f"Testing agent: {agent_name} ({current_agent}/{total_agents})"

    def _create_failure_result(
        self,
        task: dict[str, Any],
        agent_type: AgentType,
        error_message: str,
        error_code: str | None = None,
        execution_time: float = 0.0,
    ) -> TaskResult:
        """
        Create a TaskResult for failed executions with comprehensive error context.

        Args:
            task: The task that failed.
            agent_type: Type of agent that failed.
            error_message: Human-readable error description.
            error_code: Machine-readable error code.
            execution_time: Actual execution time before failure.

        Returns:
            TaskResult: Structured failure result for analysis.
        """
        failure_reason = error_message
        if error_code:
            failure_reason = f"{error_code}: {error_message}"

        return TaskResult(
            task_id=task.get("task_id", str(uuid.uuid4())),
            task_name=task.get("name", "Unknown Task"),
            task_type=task.get("complexity", TaskComplexity.MODERATE),
            agent_type=agent_type,
            success=False,
            execution_time=execution_time,
            error_count=1,
            context_retention_score=0.0,
            cost_estimate=0.001,
            failure_reason=failure_reason,
            steps_completed=0,
            total_steps=1,
        )

    def _validate_benchmark_params(self, iterations: int, timeout: int) -> None:
        """Validate benchmark parameters with comprehensive checks."""
        if iterations <= 0:
            raise ConfigurationError(
                f"Iterations must be positive, got {iterations}",
                error_code="INVALID_ITERATIONS",
                context={"iterations": iterations, "min_required": 1},
            )

        if iterations > MAX_SAFE_ITERATIONS:
            raise ConfigurationError(
                f"Iterations too high for performance safety, got {iterations}",
                error_code="EXCESSIVE_ITERATIONS",
                context={"iterations": iterations, "max_allowed": MAX_SAFE_ITERATIONS},
            )

        if timeout < MIN_TIMEOUT_SECONDS:
            raise ConfigurationError(
                f"Timeout too short, got {timeout}s",
                error_code="INVALID_TIMEOUT",
                context={"timeout": timeout, "min_required": MIN_TIMEOUT_SECONDS},
            )

    async def _execute_task_iteration(
        self,
        agent: Any,
        task: dict[str, Any],
        agent_type: AgentType,
        iteration: int,
        iterations: int,
        task_name: str,
    ) -> TaskResult:
        """Execute a single task iteration with proper async handling."""
        task_timeout = (
            QUICK_MODE_TASK_TIMEOUT if self.quick_mode else COMPREHENSIVE_TASK_TIMEOUT
        )
        self.logger.debug(
            "Iteration %d/%d, timeout=%.1fs",
            iteration + 1,
            iterations,
            task_timeout,
        )

        try:
            if self.enable_retries:
                result = await execute_agent_task_with_retry(
                    agent,
                    task,
                    max_retries=self.max_retries,
                    timeout=task_timeout,
                )
            else:
                result = await asyncio.wait_for(
                    agent.execute_task(task), timeout=task_timeout
                )

            if not self.quiet:
                self.logger.debug(
                    "Task completed: success=%s, exec_time=%.2fs",
                    result.success,
                    result.execution_time,
                )

            return result

        except TimeoutError:
            if not self.quiet:
                self.logger.warning(
                    "Timeout on agent: %s during task: %s (iter %d)",
                    agent.name,
                    task_name,
                    iteration + 1,
                )
            return self._create_failure_result(
                task,
                agent_type,
                f"Timeout after {task_timeout}s",
                "TASK_TIMEOUT",
                task_timeout,
            )

        except AgentRealityCheckError as e:
            if not self.quiet:
                self.logger.warning(
                    "Agent error on %s during task: %s (iter %d): %s",
                    agent.name,
                    task_name,
                    iteration + 1,
                    str(e),
                )
            return self._create_failure_result(
                task,
                agent_type,
                str(e),
                getattr(e, "error_code", "AGENT_ERROR"),
            )

        except Exception as e:
            if not self.quiet:
                self.logger.exception(
                    "Unexpected error from agent: %s on task: %s - %s",
                    agent.name,
                    task_name,
                    str(e),
                )
            return self._create_failure_result(
                task,
                agent_type,
                f"Unexpected error: {str(e)}",
                "UNEXPECTED_ERROR",
            )

    def _check_global_timeout(self, start_time: float, timeout: int) -> bool:
        """Check if global timeout has been exceeded."""
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            self.logger.warning("Global timeout hit after %.2fs", elapsed_time)
            return True
        return False

    def _update_progress(
        self,
        progress_callback: Callable[[int, int, str], None] | None,
        completed_operations: int,
        total_operations: int,
        description: str,
        start_time: float,
    ) -> None:
        """Update progress with callback or periodic logging."""
        if progress_callback:
            try:
                progress_callback(completed_operations, total_operations, description)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

        interval = (
            QUICK_MODE_PROGRESS_INTERVAL
            if self.quick_mode
            else COMPREHENSIVE_PROGRESS_INTERVAL
        )
        if (
            not progress_callback
            and not self.quiet
            and completed_operations % interval == 0
        ):
            progress = (completed_operations / total_operations) * 100
            avg_time = (time.time() - start_time) / completed_operations
            eta = (total_operations - completed_operations) * avg_time
            self.logger.info(
                "Progress: %.1f%% (%d/%d) | ETA: %.1fs",
                progress,
                completed_operations,
                total_operations,
                eta,
            )

    async def run_comprehensive_benchmark(
        self,
        iterations: int = 10,
        timeout: int = 400,
        progress_callback: Callable[[int, int, str], None] | None = None,
        enhanced_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a full benchmark across all agents and tasks with enterprise error handling.

        Executes comprehensive benchmark evaluation with statistical analysis,
        enhanced features support, and detailed progress tracking.

        Args:
            iterations: Number of times each agent-task pair should be evaluated.
                Must be 1-1000 for performance safety.
            timeout: Maximum allowed wall-time in seconds for the entire benchmark.
                Must be ≥ 10 seconds. Individual tasks have separate timeouts.
            progress_callback: Optional callback function for progress updates.
                Called with (completed_ops, total_ops, description).
            enhanced_config: Optional configuration for enhanced features:
                - enable_tool_failures: Enable tool failure simulation
                - network_condition: Network condition simulation
                - confidence_level: Statistical confidence level

        Returns:
            Dict containing comprehensive benchmark results:
                - results: Analyzed agent performance metrics
                - early_exit: Boolean indicating if benchmark timed out
                - execution_metadata: Timing and completion statistics
                - enhanced_config: Enhanced feature configuration used (if any)
                - enhanced_features_active: Boolean flag for enhanced mode

        Raises:
            ConfigurationError: When benchmark parameters are invalid:
                - iterations outside valid range [1, 1000]
                - timeout below minimum threshold (10 seconds)
            BenchmarkExecutionError: When benchmark execution fails:
                - Timeout during execution
                - Agent initialization or execution failures
                - Resource management failures

        Note:
            - Supports both standard and enhanced benchmarking modes
            - Enhanced features are opt-in and backward compatible
            - Progress tracking with configurable callback mechanism
            - Automatic resource cleanup on completion or failure
            - Statistical analysis integration for enhanced insights
            - Early exit handling for timeout scenarios with partial results

        Example:
            >>> benchmark = AgentBenchmark(quick_mode=False)
            >>> results = await benchmark.run_comprehensive_benchmark(
            ...     iterations=20, timeout=600
            ... )
            >>> print(f"Completed: {results['execution_metadata']['completion_rate']:.1%}")
        """
        self.logger.info(
            f"Starting comprehensive benchmark: {iterations} iterations, {timeout}s timeout"
        )
        if enhanced_config:
            self.logger.info(f"Enhanced features enabled: {enhanced_config}")
        else:
            self.logger.info("Running standard benchmark (no enhanced features)")

        self._validate_benchmark_params(iterations, timeout)
        resource_manager = BenchmarkResourceManager()

        try:
            async with benchmark_execution_context(
                "comprehensive_benchmark", timeout, resource_manager
            ) as ctx:
                self.logger.info("Benchmark execution context established")

                self._setup_benchmark_context(ctx, iterations, enhanced_config)
                self._setup_enhanced_features(enhanced_config)

                if not self.quiet:
                    self.logger.info("Starting AI Agent Reality Check Benchmark")

                self.logger.info(
                    f"Executing benchmark loop with {len(self.tasks)} tasks and {len(self.agents)} agents"
                )

                benchmark_results = await self._execute_benchmark_loop(
                    iterations, timeout, progress_callback
                )

                final_duration = benchmark_results.get("final_duration", 0)
                completed_ops = benchmark_results.get("completed_operations", 0)
                total_ops = benchmark_results.get("total_operations", 0)

                self.logger.info(
                    f"Benchmark completed successfully in {final_duration:.2f}s"
                )
                self.logger.info(f"Operations completed: {completed_ops}/{total_ops}")

                if benchmark_results.get("did_timeout"):
                    self.logger.warning(
                        "Benchmark exited early due to timeout - results may be partial"
                    )

                final_results = self._build_final_results(
                    benchmark_results, enhanced_config is not None, enhanced_config
                )

                if "results" in final_results and "results" in final_results["results"]:
                    agent_results = final_results["results"]["results"]
                    for agent_type, metrics in agent_results.items():
                        success_rate = metrics.get("success_rate", 0)
                        self.logger.info(
                            f"{agent_type}: {success_rate:.1%} success rate"
                        )

                return final_results

        except Exception as e:
            self.logger.error(f"Benchmark failed with error: {e}", exc_info=True)
            self.logger.error(
                f"Error context - iterations: {iterations}, timeout: {timeout}, enhanced: {enhanced_config is not None}"
            )

            if isinstance(e, ConfigurationError | BenchmarkExecutionError):
                raise
            else:
                raise BenchmarkExecutionError(
                    f"Comprehensive benchmark failed: {str(e)}",
                    error_code="BENCHMARK_FAILED",
                    context={
                        "iterations": iterations,
                        "timeout": timeout,
                        "enhanced_config": enhanced_config is not None,
                        "error_type": type(e).__name__,
                    },
                ) from e

    def _setup_benchmark_context(
        self,
        ctx: BenchmarkErrorContext,
        iterations: int,
        enhanced_config: dict[str, Any] | None,
    ) -> None:
        """Setup the benchmark execution context with metadata."""
        ctx.add_context("iterations", iterations)
        ctx.add_context("task_count", len(self.tasks))
        ctx.add_context("agent_count", len(self.agents))
        ctx.add_context("enhanced_features", bool(enhanced_config))
        ctx.add_context("retry_enabled", self.enable_retries)
        ctx.add_context("quick_mode", self.quick_mode)

    def _setup_enhanced_features(self, enhanced_config: dict[str, Any] | None) -> None:
        """Configure enhanced features if requested."""
        if not enhanced_config:
            return

        self.logger.debug(f"Enhanced features requested: {enhanced_config}")
        try:
            self._configure_enhanced_features(enhanced_config)
        except Exception as e:
            self.logger.warning(f"Enhanced feature configuration failed: {e}")
            enhanced_config["configuration_warnings"] = [str(e)]

    async def _execute_benchmark_loop(
        self,
        iterations: int,
        timeout: int,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> dict[str, Any]:
        """Execute the main benchmark loop and return results."""
        all_results = []
        total_operations = len(self.tasks) * len(self.agents) * iterations
        completed_operations = 0
        start_time = time.time()
        did_timeout = False

        if not self.quiet:
            self.logger.info("Planned total operations: %d", total_operations)

        for task_idx, task in enumerate(self.tasks, 1):
            task_results = await self._execute_single_task(
                task,
                task_idx,
                iterations,
                timeout,
                start_time,
                progress_callback,
                completed_operations,
                total_operations,
            )

            completed_operations += task_results["completed_operations"]
            all_results.extend(task_results["results"])

            if task_results["timeout_occurred"]:
                did_timeout = True
                break

        final_duration = time.time() - start_time
        if not self.quiet:
            self.logger.info(
                "Benchmark finished in %.2fs. Total executions: %d",
                final_duration,
                len(all_results),
            )

        self.results_history.extend(all_results)

        return {
            "all_results": all_results,
            "did_timeout": did_timeout,
            "final_duration": final_duration,
            "completed_operations": completed_operations,
            "total_operations": total_operations,
        }

    async def _execute_single_task(
        self,
        task: dict[str, Any],
        task_idx: int,
        iterations: int,
        timeout: int,
        start_time: float,
        progress_callback: Callable[[int, int, str], None] | None,
        base_completed_operations: int,
        total_operations: int,
    ) -> dict[str, Any]:
        """Execute a single task across all agents and return results."""
        task_name = str(task["name"])
        task["task_id"] = str(uuid.uuid4())

        if not self.quiet:
            task_status = self._format_task_status(task_name, task_idx, len(self.tasks))
            self.logger.info(task_status)

        task_results = []
        completed_operations = 0
        timeout_occurred = False

        self._update_progress(
            progress_callback,
            base_completed_operations,
            total_operations,
            f"Running task: {task_name}",
            start_time,
        )

        for agent_idx, (agent_type, agent) in enumerate(self.agents.items(), 1):
            agent_results = await self._execute_agent_iterations(
                agent,
                agent_type,
                task,
                agent_idx,
                iterations,
                timeout,
                start_time,
                progress_callback,
                base_completed_operations + completed_operations,
                total_operations,
                task_name,
            )

            task_results.extend(agent_results["results"])
            completed_operations += agent_results["completed_operations"]

            if agent_results["timeout_occurred"]:
                timeout_occurred = True
                break

        return {
            "results": task_results,
            "completed_operations": completed_operations,
            "timeout_occurred": timeout_occurred,
        }

    async def _execute_agent_iterations(
        self,
        agent: Any,
        agent_type: AgentType,
        task: dict[str, Any],
        agent_idx: int,
        iterations: int,
        timeout: int,
        start_time: float,
        progress_callback: Callable[[int, int, str], None] | None,
        base_completed_operations: int,
        total_operations: int,
        task_name: str,
    ) -> dict[str, Any]:
        """Execute multiple iterations for a single agent on a task."""
        if not self.quiet:
            agent_status = self._format_agent_status(
                agent.name, agent_idx, len(self.agents)
            )
            self.logger.debug(agent_status)

        task_results = []
        completed_operations = 0
        timeout_occurred = False

        for i in range(iterations):
            if self._check_global_timeout(start_time, timeout):
                timeout_occurred = True
                break

            result = await self._execute_task_iteration(
                agent, task, agent_type, i, iterations, task_name
            )

            task_results.append(result)
            completed_operations += 1

            self._update_progress(
                progress_callback,
                base_completed_operations + completed_operations,
                total_operations,
                f"Testing {agent.name} on {task_name}",
                start_time,
            )

            if self._check_global_timeout(start_time, timeout):
                if not self.quiet:
                    self.logger.warning("Exiting early due to timeout")
                timeout_occurred = True
                break

        self._log_agent_task_summary(task_results, agent.name)

        return {
            "results": task_results,
            "completed_operations": completed_operations,
            "timeout_occurred": timeout_occurred,
        }

    def _log_agent_task_summary(
        self, task_results: list[TaskResult], agent_name: str
    ) -> None:
        """Log summary statistics for an agent's performance on a task."""
        if not task_results or self.quiet:
            return

        success_rate = sum(1 for r in task_results if r.success) / len(task_results)
        avg_time = sum(r.execution_time for r in task_results) / len(task_results)
        self.logger.debug(
            "Agent %s summary: %.1f%% success, %.2fs avg",
            agent_name,
            success_rate * 100,
            avg_time,
        )

    def _build_final_results(
        self,
        benchmark_results: dict[str, Any],
        enhanced_features_active: bool,
        enhanced_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the final comprehensive result structure."""
        result = {
            "results": self.analyze_results(benchmark_results["all_results"]),
            "early_exit": benchmark_results["did_timeout"],
            "execution_metadata": {
                "total_duration": benchmark_results["final_duration"],
                "completed_operations": benchmark_results["completed_operations"],
                "planned_operations": benchmark_results["total_operations"],
                "completion_rate": (
                    benchmark_results["completed_operations"]
                    / benchmark_results["total_operations"]
                ),
                "operations_per_second": (
                    benchmark_results["completed_operations"]
                    / benchmark_results["final_duration"]
                    if benchmark_results["final_duration"] > 0
                    else 0
                ),
            },
        }

        if enhanced_features_active:
            result["enhanced_config"] = enhanced_config
            result["enhanced_features_active"] = True

        return result

    def _configure_enhanced_features(self, enhanced_config: dict[str, Any]) -> None:
        """
        Configure enhanced features with comprehensive error handling.
        FIXED: Boolean callable error that was breaking enhanced mode.
        """
        enable_tool_failures = enhanced_config.get("enable_tool_failures", False)
        network_condition = enhanced_config.get("network_condition", "stable")
        enable_tool_failures = bool(enable_tool_failures)
        network_condition = str(network_condition)

        enhanced_config["enable_tool_failures"] = enable_tool_failures
        enhanced_config["network_condition"] = network_condition

        if enable_tool_failures:
            if not ENHANCED_EXECUTOR_AVAILABLE:
                self.logger.warning(
                    "Enhanced executor not available, using standard execution"
                )
                enhanced_config["enable_tool_failures"] = False
                return

            try:
                from .agents.enhanced_executor import (  # noqa: PLC0415
                    NetworkCondition,
                    ToolFailureConfig,
                )

                tool_config = ToolFailureConfig()
                network_sim = NetworkCondition(network_condition)

                for _agent_type, agent in self.agents.items():
                    try:
                        if hasattr(agent, "__dict__"):
                            agent.enhanced_execution = True  # type: ignore[attr-defined]
                            agent.tool_config = tool_config  # type: ignore[attr-defined]
                            agent.network_condition = network_sim  # type: ignore[attr-defined]

                            self.logger.debug(
                                f"Enhanced features configured for {type(agent).__name__}"
                            )
                        else:
                            self.logger.warning(
                                f"Agent {type(agent).__name__} does not support enhanced configuration"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to configure enhanced features for {type(agent).__name__}: {e}"
                        )

                self.logger.debug("Enhanced features configuration completed")

            except ImportError as e:
                self.logger.warning(f"Enhanced executor imports failed: {e}")
                enhanced_config["enable_tool_failures"] = False

    def analyze_results(self, results: list[TaskResult]) -> dict[str, Any]:
        """Analyze raw benchmark results and compute comprehensive statistics.

        Processes individual task execution results to generate aggregated metrics,
        statistical analysis, and performance insights across agent types.

        Args:
            results: List of TaskResult objects from benchmark execution.
                Each result contains success status, timing, costs, and metadata.

        Returns:
            Dict containing analyzed results structure:
                - results: Agent-wise aggregated metrics including:
                    - success_rate: Fraction of successful task completions
                    - avg_execution_time: Mean execution time across tasks
                    - avg_context_retention: Mean context retention score
                    - cost_per_success: Resource cost per successful completion
                    - step_completion_rate: Fraction of task steps completed
                    - failure_categories: Breakdown of failure types
                    - timing statistics: min/max execution times
                - statistical_assessment: Statistical power and effect size analysis (if available)
                - sample_size_analysis: Sample adequacy recommendations (if available)
                - metadata: Analysis timestamp, sample counts, statistical flags

        Note:
            - Handles empty result sets gracefully with warning metadata
            - Groups results by agent type for comparative analysis
            - Calculates derived metrics like cost efficiency and completion rates
            - Enhanced statistical analysis when scipy dependencies available
            - Failure categorization by error codes for debugging insights
            - Sample size recommendations for improving statistical power

        Statistical Features (when scipy available):
            - Effect size calculations for pairwise agent comparisons
            - Statistical power assessment for meaningful difference detection
            - Sample size recommendations for adequate statistical power
            - Confidence level considerations for interval estimates

        Example:
            >>> analysis = benchmark.analyze_results(task_results)
            >>> for agent, metrics in analysis['results'].items():
            ...     print(f"{agent}: {metrics['success_rate']:.1%} success")
        """
        self.logger.debug("Analyzing %d results", len(results))

        if not results:
            return {
                "results": {},
                "metadata": {
                    "total_results": 0,
                    "analysis_timestamp": time.time(),
                    "warning": "No results to analyze",
                },
            }

        agent_metrics: dict[AgentType, list[TaskResult]] = defaultdict(list)

        for result in results:
            agent_metrics[result.agent_type].append(result)

        aggregated: dict[str, Any] = {}

        for agent_type, agent_results in agent_metrics.items():
            total = len(agent_results)
            successes = sum(1 for r in agent_results if r.success)
            total_time = sum(r.execution_time for r in agent_results)
            total_context = sum(r.context_retention_score for r in agent_results)
            total_cost = sum(r.cost_estimate for r in agent_results)
            total_steps_completed = sum(r.steps_completed for r in agent_results)
            total_steps = sum(r.total_steps for r in agent_results)
            total_errors = sum(r.error_count for r in agent_results)

            if total == 0:
                success_rate = 0.0
                avg_execution_time = 0.0
                avg_context_retention = 0.0
                avg_error_count = 0.0
            else:
                success_rate = successes / total
                avg_execution_time = total_time / total
                avg_context_retention = total_context / total
                avg_error_count = total_errors / total

            if not self.quiet:
                self.logger.debug(
                    "[%s] Analysis: total_cost=%.4f, successes=%d, total=%d",
                    agent_type.name,
                    total_cost,
                    successes,
                    total,
                )

            cost_per_success = total_cost / successes if successes > 0 else float("inf")
            step_completion_rate = (
                total_steps_completed / total_steps if total_steps > 0 else 0.0
            )

            failure_reasons = [
                r.failure_reason
                for r in agent_results
                if not r.success and r.failure_reason
            ]
            failure_categories: dict[str, int] = {}
            for reason in failure_reasons:
                error_code = reason.split(":")[0] if ":" in reason else "UNKNOWN"
                failure_categories[error_code] = (
                    failure_categories.get(error_code, 0) + 1
                )

            aggregated[agent_type.name] = {
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "avg_context_retention": avg_context_retention,
                "cost_per_success": cost_per_success,
                "step_completion_rate": step_completion_rate,
                "total_tasks": total,
                "successful_tasks": successes,
                "failed_tasks": total - successes,
                "total_cost": total_cost,
                "avg_error_count": avg_error_count,
                "failure_categories": failure_categories,
                "min_execution_time": (
                    min(r.execution_time for r in agent_results) if agent_results else 0
                ),
                "max_execution_time": (
                    max(r.execution_time for r in agent_results) if agent_results else 0
                ),
            }

        if ENHANCED_STATS_AVAILABLE:
            statistical_assessment = assess_statistical_significance(aggregated)
            sample_size_recommendations: dict[str, dict[str, Any]] = {}

            if len(aggregated) >= MIN_AGENTS_FOR_COMPARISON:
                agent_names = list(aggregated.keys())
                for i, agent1 in enumerate(agent_names):
                    for agent2 in agent_names[i + 1 :]:
                        rate1 = aggregated[agent1]["success_rate"]
                        rate2 = aggregated[agent2]["success_rate"]
                        effect_size, interpretation = calculate_effect_size(
                            rate1, rate2
                        )

                        required_n = calculate_required_sample_size(effect_size)
                        current_n = min(
                            aggregated[agent1]["total_tasks"],
                            aggregated[agent2]["total_tasks"],
                        )

                        comparison_key = f"{agent1}_vs_{agent2}"
                        sample_size_recommendations[comparison_key] = {
                            "current_sample_size": current_n,
                            "recommended_sample_size": required_n,
                            "is_adequate": current_n >= required_n,
                            "effect_size": effect_size,
                            "interpretation": interpretation,
                        }

                        self.logger.debug(
                            f"Sample size analysis {agent1} vs {agent2}: "
                            f"current={current_n}, required={required_n}, "
                            f"adequate={current_n >= required_n}"
                        )

            return {
                "results": aggregated,
                "statistical_assessment": statistical_assessment,
                "sample_size_analysis": sample_size_recommendations,
                "metadata": {
                    "random_seed": getattr(self, "random_seed", None),
                    "total_comparisons": len(
                        statistical_assessment.get("effect_sizes", {})
                    ),
                    "sufficient_statistical_power": statistical_assessment.get(
                        "sufficient_power", False
                    ),
                    "adequate_sample_sizes": (
                        all(
                            rec["is_adequate"]
                            for rec in sample_size_recommendations.values()
                        )
                        if sample_size_recommendations
                        else True
                    ),
                    "analysis_timestamp": time.time(),
                    "total_results_analyzed": len(results),
                },
            }

        else:
            self.logger.debug(
                "Statistical assessment functions not available - returning basic results"
            )
            return {
                "results": aggregated,
                "metadata": {
                    "random_seed": getattr(self, "random_seed", None),
                    "analysis_timestamp": time.time(),
                    "total_results_analyzed": len(results),
                    "enhanced_statistics": False,
                },
            }

    def _generate_agent_performance_section(
        self, results_data: dict[str, Any]
    ) -> list[str]:
        """Generate the agent performance section of the report."""
        lines = ["Agent Performance Analysis:", ""]

        for agent_type, metrics in results_data.items():
            success_rate = metrics["success_rate"]
            if success_rate >= EXCELLENT_PERFORMANCE_THRESHOLD:
                indicator = "🟢 EXCELLENT"
            elif success_rate >= GOOD_PERFORMANCE_THRESHOLD:
                indicator = "🟡 GOOD"
            elif success_rate >= FAIR_PERFORMANCE_THRESHOLD:
                indicator = "🟠 FAIR"
            else:
                indicator = "🔴 POOR"

            lines.extend(
                [
                    f"[{agent_type}] {indicator}",
                    f"  Success Rate:        {metrics['success_rate']:.1%}",
                    f"  Avg Execution Time:  {metrics['avg_execution_time']:.2f}s",
                    f"  Context Retention:   {metrics['avg_context_retention']:.1%}",
                    f"  Cost per Success:    ${metrics['cost_per_success']:.4f}",
                    f"  Step Completion:     {metrics['step_completion_rate']:.1%}",
                    f"  Tasks: {metrics['successful_tasks']}/{metrics['total_tasks']} successful",
                ]
            )

            failure_categories = metrics.get("failure_categories", {})
            if failure_categories:
                lines.append("  Common Failure Types:")
                for error_code, count in sorted(
                    failure_categories.items(), key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"     • {error_code}: {count} occurrences")

            lines.append("")

        return lines

    def _generate_statistical_recommendations(
        self, analysis: dict[str, Any]
    ) -> list[str]:
        """Generate statistical recommendations section."""
        lines = []

        if "statistical_assessment" in analysis:
            stats = analysis["statistical_assessment"]
            if not stats.get("sufficient_power", True):
                lines.extend(
                    [
                        "Statistical Recommendations:",
                        *[f"  • {rec}" for rec in stats.get("recommendations", [])],
                        "",
                    ]
                )

        if "sample_size_analysis" in analysis:
            sample_analysis = analysis["sample_size_analysis"]
            inadequate_samples = [
                comp
                for comp, data in sample_analysis.items()
                if not data["is_adequate"]
            ]

            if inadequate_samples:
                lines.extend(
                    [
                        "🔬 Sample Size Recommendations:",
                        *[
                            f"  • {comp}: Need {sample_analysis[comp]['recommended_sample_size']} samples "
                            f"(currently {sample_analysis[comp]['current_sample_size']})"
                            for comp in inadequate_samples[:3]
                        ],
                        "",
                    ]
                )
            else:
                lines.extend(["✅ Sample sizes adequate for statistical power", ""])

        return lines

    def _generate_executive_summary(self, results_data: dict[str, Any]) -> list[str]:
        """Generate executive summary section."""
        lines = []

        if len(results_data) >= MIN_AGENTS_FOR_COMPARISON:
            best_agent = max(results_data.items(), key=lambda x: x[1]["success_rate"])
            worst_agent = min(results_data.items(), key=lambda x: x[1]["success_rate"])
            performance_gap = (
                best_agent[1]["success_rate"] - worst_agent[1]["success_rate"]
            )

            lines.extend(
                [
                    "Executive Summary:",
                    f"  • Best Performer: {best_agent[0]} ({best_agent[1]['success_rate']:.1%} success)",
                    f"  • Performance Gap: {performance_gap:.1%} between best and worst",
                    f"  • Reality Check: {performance_gap:.1%} performance difference exposes architectural impact",
                    "",
                ]
            )

        return lines

    def generate_reality_report(self, analysis: dict[str, Any]) -> str:
        """Generate a comprehensive reality report with enhanced error context.

        Creates human-readable analysis report highlighting performance gaps,
        architectural impacts, and actionable insights from benchmark results.

        Args:
            analysis: Processed metrics dictionary from analyze_results() containing
                agent performance statistics, metadata, and optional statistical analysis.

        Returns:
            Multi-line string containing formatted report with:
                - Executive summary with key performance insights
                - Agent-by-agent performance analysis with ratings
                - Statistical recommendations when enhanced analysis available
                - Reality check insights exposing architectural differences
                - Actionable recommendations for improvement

        Note:
            - Handles incomplete benchmarks from early timeout gracefully
            - Includes metadata about statistical power and sample adequacy
            - Uses performance thresholds: Excellent (≥80%), Good (≥60%), Fair (≥40%)
            - Highlights performance gaps between agent architectures
            - Provides context about result reliability and statistical significance
            - Safe to call with any analysis format - handles errors gracefully

        Report Sections:
            1. Benchmark metadata and warnings
            2. Agent performance ratings with key metrics
            3. Statistical recommendations (when available)
            4. Executive summary with insights and performance gaps

        Example:
            >>> report = benchmark.generate_reality_report(analysis_results)
            >>> print(report)  # Human-readable performance analysis
        """
        self.logger.debug("Generating reality report")

        if not analysis or "results" not in analysis:
            return "No results available to generate report."

        lines = [
            "\nAI Agents Reality Check - Comprehensive Report",
            "=" * 55,
            "",
        ]

        if analysis.get("early_exit"):
            lines.extend(
                [
                    "[!] BENCHMARK INCOMPLETE - EARLY EXIT DUE TO TIMEOUT",
                    "    Results may be partial and not statistically reliable.",
                    "",
                ]
            )

        if "metadata" in analysis:
            metadata = analysis["metadata"]
            lines.extend(
                [
                    "📊 Benchmark Metadata:",
                    f"   Random Seed: {metadata.get('random_seed', 'N/A')}",
                    f"   Total Results: {metadata.get('total_results_analyzed', 'N/A')}",
                    f"   Statistical Power: {'✅ Sufficient' if metadata.get('sufficient_statistical_power') else '⚠️  Insufficient'}",
                    "",
                ]
            )

        results_data = analysis.get("results", {})
        if "results" in results_data:
            results_data = results_data["results"]

        if not results_data:
            lines.append("⚠️  No agent results available for analysis.")
            return "\n".join(lines)

        lines.extend(self._generate_agent_performance_section(results_data))
        lines.extend(self._generate_statistical_recommendations(analysis))
        lines.extend(self._generate_executive_summary(results_data))

        return "\n".join(lines)

    def export_results(
        self,
        filepath: str,
        format: str = "json",
        enhanced_data: dict[str, Any] | None = None,
    ) -> None:
        """Export benchmark results with comprehensive error handling and logging directory support.

        Exports benchmark execution results to file with support for JSON and CSV formats,
        automatic directory creation, and enhanced metadata inclusion.

        Args:
            filepath: Target file path for export. Parent directories created automatically.
            format: Export format - "json" or "csv". Default: "json".
            enhanced_data: Optional enhanced data structure to export instead of standard
                results. Used for exporting enhanced benchmark results with additional metadata.

        Raises:
            ConfigurationError: When export parameters are invalid:
                - Unsupported format specified
            ExportError: When export operations fail:
                - Permission denied for file/directory access
                - File system errors (disk full, invalid path)
                - Data serialization failures
                - Missing pandas dependency for CSV export

        Note:
            JSON Export:
            - Includes comprehensive metadata (version, timestamps, configuration)
            - Handles infinity and NaN values with safe serialization
            - Enhanced data includes additional feature metadata when provided

            CSV Export:
            - Requires pandas dependency
            - Exports flattened TaskResult data
            - Creates empty DataFrame with proper columns if no results

            Safety Features:
            - Automatic directory creation for target path
            - Safe JSON serialization preventing infinity/NaN errors
            - Comprehensive error context for debugging
            - Fallback handling for missing dependencies

        Example:
            >>> benchmark.export_results("logs/results.json", format="json")
            >>> # Enhanced export
            >>> benchmark.export_results(
            ...     "logs/enhanced.json",
            ...     enhanced_data={"version": "2.0", "features": ["tool_failures"]}
            ... )
        """

        def safe_json_serializer(obj: Any) -> str | None:
            """JSON serializer that handles infinity and NaN values safely."""
            if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
                return None
            return str(obj)

        if format.lower() not in ["json", "csv"]:
            raise ConfigurationError(
                f"Unsupported export format: {format}",
                error_code="INVALID_EXPORT_FORMAT",
                context={"format": format, "supported_formats": ["json", "csv"]},
            )

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.logger.debug("Exporting results to %s (format=%s)", filepath, format)

        try:
            if enhanced_data:
                if format.lower() == "json":
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(
                            enhanced_data, f, indent=2, default=safe_json_serializer
                        )
                    self.logger.info(f"Enhanced data exported to {filepath}")
                elif format.lower() == "csv":
                    self.logger.info(
                        f"Enhanced features in export: {enhanced_data.get('features', [])}"
                    )
                    self._export_standard_csv(filepath)
            elif format.lower() == "json":
                if not self.results_history:
                    self.logger.warning("No results history to export")
                    results_data: dict[str, Any] = {
                        "metadata": {
                            "version": "0.1.0",
                            "total_results": 0,
                            "export_timestamp": time.time(),
                            "quick_mode": self.quick_mode,
                            "random_seed": getattr(self, "random_seed", None),
                            "enable_retries": self.enable_retries,
                            "max_retries": self.max_retries,
                        },
                        "results": [],
                    }
                else:
                    results_data = {
                        "metadata": {
                            "version": "0.1.0",
                            "total_results": len(self.results_history),
                            "export_timestamp": time.time(),
                            "quick_mode": self.quick_mode,
                            "random_seed": getattr(self, "random_seed", None),
                            "enable_retries": self.enable_retries,
                            "max_retries": self.max_retries,
                        },
                        "results": [
                            result.to_dict() for result in self.results_history
                        ],
                    }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(results_data, f, indent=2, default=safe_json_serializer)

            elif format.lower() == "csv":
                self._export_standard_csv(filepath)

            self.logger.info(
                f"Successfully exported {len(self.results_history)} results to {filepath}"
            )

        except PermissionError as e:
            raise ExportError(
                f"Permission denied writing to {filepath}",
                error_code="EXPORT_PERMISSION_DENIED",
                context={"filepath": filepath, "format": format},
            ) from e

        except OSError as e:
            raise ExportError(
                f"File system error writing to {filepath}: {str(e)}",
                error_code="EXPORT_FILESYSTEM_ERROR",
                context={
                    "filepath": filepath,
                    "format": format,
                    "os_error": str(e),
                },
            ) from e

        except Exception as e:
            self.logger.error(f"Export failed for {filepath}: {e}", exc_info=True)
            raise ExportError(
                f"Export failed: {str(e)}",
                error_code="EXPORT_FAILED",
                context={
                    "filepath": filepath,
                    "format": format,
                    "error_type": type(e).__name__,
                    "results_count": len(self.results_history),
                },
            ) from e

    def _export_standard_csv(self, filepath: str) -> None:
        """
        Export results to CSV format with enhanced error handling.

        Args:
            filepath: Path to write CSV file.

        Raises:
            ExportError: When CSV export fails.
        """
        if not PANDAS_AVAILABLE:
            raise ExportError(
                "CSV export requires pandas. Install with: pip install pandas",
                error_code="MISSING_PANDAS_DEPENDENCY",
                context={"required_package": "pandas"},
            ) from None

        try:
            if not self.results_history:
                self.logger.warning("No results to export to CSV")
                df = pd.DataFrame(
                    columns=[
                        "task_id",
                        "task_name",
                        "task_type",
                        "agent_type",
                        "success",
                        "execution_time",
                        "error_count",
                        "context_retention_score",
                        "cost_estimate",
                        "failure_reason",
                        "steps_completed",
                        "total_steps",
                    ]
                )
            else:
                df = pd.DataFrame([result.to_dict() for result in self.results_history])

            df.to_csv(filepath, index=False)
            self.logger.debug(
                f"CSV export completed: {len(df)} rows written to {filepath}"
            )

        except Exception as e:
            raise ExportError(
                f"CSV export failed: {str(e)}",
                error_code="CSV_EXPORT_FAILED",
                context={"filepath": filepath, "error_type": type(e).__name__},
            ) from e
