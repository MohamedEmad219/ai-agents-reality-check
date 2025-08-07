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

Module: ai_agents_reality_check/agents/marketing/marketing_agent.py

MarketingAgent implementation representing the "demo-quality" architectural tier.
This agent demonstrates basic planning and memory capabilities that exceed wrapper
agents but fall short of production-grade real agents due to shallow implementation
and limited error recovery mechanisms.

The MarketingAgent serves as an intermediate benchmark point, achieving 60-75%
success rates through orchestrated execution while maintaining cost efficiency
trade-offs typical of marketing-driven agent implementations.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import time
import uuid
from typing import Any

from ai_agents_reality_check.agents.tracer import trace_marketing_result
from ai_agents_reality_check.types import AgentType, TaskComplexity, TaskResult

from .marketing_executor import execute_plan
from .marketing_memory import EphemeralMemory
from .marketing_planner import construct_plan

# Constants for magic numbers
MIN_COMPLETION_RATE_THRESHOLD = 0.3
MAX_SUCCESS_HISTORY_LENGTH = 20


class MarketingAgent:
    """Marketing Promise Agent representing demo-quality agent architecture.

    This class implements an agent that bridges the gap between simple wrapper agents
    and sophisticated real agents. It demonstrates basic planning capabilities,
    ephemeral memory management, and coordinated task execution while maintaining
    the limitations typical of marketing-driven implementations.

    The MarketingAgent is designed to achieve 60-75% success rates across task
    complexities, consistently outperforming wrapper agents while remaining
    significantly less capable than production-ready real agents.

    Attributes:
        name (str): Human-readable agent identifier.
        api_call_cost (float): Base cost per API operation.
        execution_history (List[TaskResult]): Historical task execution results.
        memory (EphemeralMemory): Volatile memory store for task context.
        task_success_history (Dict[TaskComplexity, List[bool]]): Success tracking
            by complexity level for basic learning capabilities.
    """

    def __init__(self) -> None:
        """Initialize MarketingAgent with default configuration and memory systems.

        Sets up the agent with ephemeral memory, cost parameters, and success tracking
        structures. Initializes empty execution history and complexity-based success
        tracking for basic learning capabilities.

        Args:
            None

        Returns:
            None
        """
        self.name = "Marketing Promise Agent"
        self.api_call_cost = 0.015
        self.execution_history: list[TaskResult] = []

        self.memory = EphemeralMemory()
        self.task_success_history: dict[TaskComplexity, list[bool]] = {
            complexity: [] for complexity in TaskComplexity
        }

    async def execute_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute a task using marketing-tier planning and execution capabilities.

        Processes tasks through a multi-stage pipeline including plan construction,
        coordinated execution, and memory updates. Demonstrates the marketing agent's
        ability to handle complex tasks through orchestration while maintaining
        the fragility typical of shallow architectural implementations.

        Args:
            task (Dict[str, Any]): Task specification containing:
                - name (str, optional): Human-readable task identifier
                - task_name (str, optional): Alternative task identifier
                - complexity (TaskComplexity, optional): Task difficulty level
                - task_type (TaskComplexity, optional): Alternative complexity field
                - Additional task-specific parameters

        Returns:
            TaskResult: Comprehensive execution result containing:
                - task_id (str): Unique task identifier
                - success (bool): Task completion status
                - execution_time (float): Task duration in seconds
                - error_count (int): Number of errors encountered
                - context_retention_score (float): Memory utilization metric
                - cost_estimate (float): Economic cost of execution
                - failure_reason (str, optional): Error description if failed
                - steps_completed (int): Successfully executed steps
                - total_steps (int): Total planned steps

        Raises:
            Exception: Captures and converts any execution errors into TaskResult
                failures with appropriate error logging and cost accounting.
        """
        task_id = str(uuid.uuid4())
        task["task_id"] = task_id
        task_name = task.get("name") or task.get("task_name", "Unnamed Task")
        complexity = task.get("complexity") or task.get(
            "task_type", TaskComplexity.MODERATE
        )

        start_time = time.time()

        try:
            plan = construct_plan(task)
            execution_result = await execute_plan(task, plan, self.api_call_cost)

            if execution_result.success:
                for i, subgoal in enumerate(plan.get("subgoals", [])):
                    if subgoal.get("status") == "completed":
                        self.memory.mark_completed(task_id, i, "completed")

            self._update_success_memory(complexity, execution_result.success)

            self.execution_history.append(execution_result)
            trace_marketing_result(execution_result)
            return execution_result

        except Exception as e:
            execution_time = time.time() - start_time

            failure_result = TaskResult(
                task_id=task_id,
                task_name=task_name,
                task_type=complexity,
                agent_type=AgentType.MARKETING_AGENT,
                success=False,
                execution_time=round(execution_time, 3),
                error_count=1,
                context_retention_score=0.1,
                cost_estimate=self.api_call_cost,
                failure_reason=f"Exception: {str(e)}",
                steps_completed=0,
                total_steps=1,
            )

            self.execution_history.append(failure_result)
            return failure_result

    def _build_task_result(
        self,
        task_id: str,
        task_name: str,
        complexity: TaskComplexity,
        execution_result: dict[str, Any],
        execution_time: float,
    ) -> TaskResult:
        """Construct comprehensive TaskResult from execution metadata.

        Processes raw execution data into a standardized TaskResult object with
        calculated metrics including context retention, cost estimation, and
        failure reason classification. Applies complexity-based cost multipliers
        and retention bonuses based on execution success patterns.

        Args:
            task_id (str): Unique task identifier.
            task_name (str): Human-readable task name.
            complexity (TaskComplexity): Task difficulty classification.
            execution_result (Dict[str, Any]): Raw execution metrics containing:
                - success (bool): Overall execution success
                - steps_completed (int): Completed step count
                - total_steps (int): Total planned steps
                - completion_rate (float): Step completion ratio
                - error_count (int): Error occurrence count
            execution_time (float): Task duration in seconds.

        Returns:
            TaskResult: Standardized task execution result with calculated metrics
                including context retention scores, cost estimates, and failure
                reason classification based on execution patterns.
        """

        success = execution_result["success"]
        steps_completed = execution_result["steps_completed"]
        total_steps = execution_result["total_steps"]
        base_retention = 0.40
        success_bonus = 0.20 if success else 0.0
        completion_bonus = execution_result["completion_rate"] * 0.20
        context_retention = min(0.75, base_retention + success_bonus + completion_bonus)
        base_cost = self.api_call_cost
        step_cost = total_steps * 0.002
        retry_cost = execution_result["error_count"] * 0.003

        complexity_multiplier = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.2,
            TaskComplexity.COMPLEX: 1.5,
            TaskComplexity.ENTERPRISE: 2.0,
        }.get(complexity, 1.3)

        total_cost = (base_cost + step_cost + retry_cost) * complexity_multiplier

        failure_reason = None
        if not success:
            if execution_result["completion_rate"] < MIN_COMPLETION_RATE_THRESHOLD:
                failure_reason = "Multiple step failures, insufficient completion"
            elif execution_result["error_count"] > total_steps // 2:
                failure_reason = "High error rate, retry exhaustion"
            else:
                failure_reason = "Failed to meet completion threshold"

        return TaskResult(
            task_id=task_id,
            task_name=task_name,
            task_type=complexity,
            agent_type=AgentType.MARKETING_AGENT,
            success=success,
            execution_time=round(execution_time, 3),
            error_count=execution_result["error_count"],
            context_retention_score=round(context_retention, 3),
            cost_estimate=round(total_cost, 4),
            failure_reason=failure_reason,
            steps_completed=steps_completed,
            total_steps=total_steps,
        )

    def _update_success_memory(self, complexity: TaskComplexity, success: bool) -> None:
        """Update complexity-specific success history for basic learning.

        Maintains a rolling window of success/failure outcomes organized by task
        complexity to enable rudimentary learning and adaptation. Implements
        a fixed-size history buffer to prevent unbounded memory growth.

        Args:
            complexity (TaskComplexity): Task difficulty level for categorization.
            success (bool): Task execution outcome.

        Returns:
            None

        Note:
            History is capped at MAX_SUCCESS_HISTORY_LENGTH entries per complexity
            level to maintain consistent memory usage patterns.
        """
        if complexity not in self.task_success_history:
            self.task_success_history[complexity] = []

        self.task_success_history[complexity].append(success)

        if len(self.task_success_history[complexity]) > MAX_SUCCESS_HISTORY_LENGTH:
            self.task_success_history[complexity] = self.task_success_history[
                complexity
            ][-MAX_SUCCESS_HISTORY_LENGTH:]

    def get_success_rate(self) -> float:
        """Calculate overall success rate across all historical task executions.

        Computes the percentage of successful task completions from the agent's
        execution history, providing a simple performance metric for benchmarking
        and comparison purposes.

        Args:
            None

        Returns:
            float: Success rate as a decimal between 0.0 and 1.0, where 1.0
                represents 100% success. Returns 0.0 if no tasks have been executed.
        """
        if not self.execution_history:
            return 0.0
        return sum(1 for r in self.execution_history if r.success) / len(
            self.execution_history
        )
