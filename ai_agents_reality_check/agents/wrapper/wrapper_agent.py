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

Module: ai_agents_reality_check/agents/wrapper/wrapper_agent.py

Baseline wrapper agent implementation representing the architectural reality of
most production "AI agents" - simple prompt-response patterns with minimal
sophistication, no planning capabilities, and realistic failure rates of 35-45%.

The WrapperAgent serves as the fundamental baseline for architectural comparison,
demonstrating the limitations of stateless LLM wrappers that are commonly
misrepresented as sophisticated agent systems in the market.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import asyncio
import random
from typing import Any

from ai_agents_reality_check.agents.tracer import trace_wrapper_result
from ai_agents_reality_check.types import AgentType, TaskComplexity, TaskResult

from .wrapper_executor import execute_wrapper_task
from .wrapper_memory import WrapperMemory
from .wrapper_utils import WRAPPER_FAILURE_REASONS

# Constants for magic values
RETRY_SUCCESS_RATE = 0.25


class WrapperAgent:
    """Stateless LLM wrapper representing the baseline agent architectural tier.

    Implements the simplest possible agent pattern through direct prompt-response
    interactions with minimal retry logic, no planning capabilities, and no
    persistent memory. This class represents the architectural reality of most
    production "AI agents" that achieve 35-45% success rates through basic
    prompt engineering rather than sophisticated agent design.

    The WrapperAgent serves as the critical baseline for benchmarking, exposing
    the performance gap between marketing claims and actual wrapper implementations
    that dominate the current agent landscape.

    Attributes:
        name (str): Human-readable agent identifier.
        api_call_cost (float): Low cost per operation reflecting simple architecture.
        execution_history (List[TaskResult]): Historical task execution results.
        call_count (int): Total number of LLM calls for experience tracking.
        memory (WrapperMemory): No-op memory system simulating stateless operation.
    """

    def __init__(self) -> None:
        """Initialize WrapperAgent with minimal capabilities and stateless design.

        Sets up the agent with cost-efficient parameters, empty execution history,
        and a no-op memory system that simulates the stateless nature of typical
        LLM wrapper implementations.

        Args:
            None

        Returns:
            None
        """
        self.name = "LLM Wrapper (Wrapper Agent)"
        self.api_call_cost = 0.002
        self.execution_history: list[TaskResult] = []
        self.call_count = 0
        self.memory = WrapperMemory()

    async def execute_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute task using minimal wrapper agent logic with single LLM call pattern.

        Implements the characteristic wrapper agent execution pattern: direct task
        processing through a single LLM interaction with basic retry logic, no
        planning phase, and no sophisticated error handling. Represents the baseline
        architectural approach that achieves 35-45% success rates.

        Args:
            task (Dict[str, Any]): Task specification containing:
                - name (str, optional): Human-readable task identifier
                - task_id (str, optional): Unique task identifier
                - complexity (TaskComplexity, optional): Task difficulty level
                - Additional task parameters (largely ignored by wrapper logic)

        Returns:
            TaskResult: Basic execution result containing:
                - task_id (str): Task identifier
                - success (bool): Single-call execution outcome
                - execution_time (float): Call duration
                - error_count (int): Basic retry count
                - context_retention_score (float): Minimal retention (0.01-0.18)
                - cost_estimate (float): Low cost reflecting simple operation
                - failure_reason (str, optional): Common wrapper failure patterns
                - steps_completed/total_steps (int): Always 1 (single-step pattern)

        Note:
            Failure rates increase significantly with task complexity due to the
            lack of planning, memory, and recovery mechanisms characteristic of
            wrapper implementations.
        """
        self.call_count += 1

        try:
            result = await execute_wrapper_task(task)

            self.memory.add_entry(f"Executed: {task.get('name', 'Unknown')}")

            trace_wrapper_result(result)

            self.execution_history.append(result)
            return result

        except Exception:
            failure_result = TaskResult(
                task_id=task.get("task_id", "unknown"),
                task_name=task.get("name", "Failed Task"),
                task_type=task.get("complexity", TaskComplexity.SIMPLE),
                agent_type=AgentType.WRAPPER_AGENT,
                success=False,
                execution_time=0.1,
                error_count=1,
                context_retention_score=0.01,
                cost_estimate=self.api_call_cost,
                failure_reason=random.choice(WRAPPER_FAILURE_REASONS),
                steps_completed=0,
                total_steps=1,
            )

            self.execution_history.append(failure_result)
            return failure_result

    async def _single_llm_call(
        self, task_name: str, complexity: TaskComplexity
    ) -> dict[str, Any]:
        """Simulate single LLM interaction with realistic success rates and retry logic.

        Models the core wrapper agent operation: a single LLM API call with
        complexity-dependent success rates and minimal retry capability.
        Includes basic experience tracking and realistic response quality modeling.

        Args:
            task_name (str): Human-readable task description.
            complexity (TaskComplexity): Task difficulty affecting success probability.

        Returns:
            Dict[str, Any]: Call result containing:
                - success (bool): LLM call outcome
                - response_quality (float): Quality metric (0.0-1.0)
                - call_time (float): Execution duration in seconds
                - retries (int): Number of retry attempts (0-1)

        Note:
            Success rates are modeled realistically by complexity:
            Simple (55%), Moderate (40%), Complex (25%), Enterprise (15%).
            Single retry attempt with 25% success rate if initial call fails.
        """
        call_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(call_time * 0.01)

        base_success_rates = {
            TaskComplexity.SIMPLE: 0.55,
            TaskComplexity.MODERATE: 0.40,
            TaskComplexity.COMPLEX: 0.25,
            TaskComplexity.ENTERPRISE: 0.15,
        }

        base_success_rate = base_success_rates.get(complexity, 0.35)
        experience_bonus = min(0.03, self.call_count * 0.0005)
        final_success_rate = base_success_rate + experience_bonus
        success = random.random() < final_success_rate
        retries = 0

        if not success:
            retries = 1
            if random.random() < RETRY_SUCCESS_RATE:
                success = True

        if success:
            response_quality = random.uniform(0.4, 0.8)
        else:
            response_quality = random.uniform(0.1, 0.3)

        return {
            "success": success,
            "response_quality": response_quality,
            "call_time": call_time,
            "retries": retries,
        }

    def _build_wrapper_result(
        self,
        task_id: str,
        task_name: str,
        complexity: TaskComplexity,
        llm_result: dict[str, Any],
        execution_time: float,
    ) -> TaskResult:
        """Construct TaskResult from single LLM call with wrapper-specific calculations.

        Processes LLM call results into standardized TaskResult format with
        wrapper-specific cost modeling, minimal context retention, and
        characteristic failure reason selection.

        Args:
            task_id (str): Unique task identifier.
            task_name (str): Human-readable task name.
            complexity (TaskComplexity): Task difficulty level.
            llm_result (Dict[str, Any]): Raw LLM call results from _single_llm_call.
            execution_time (float): Total task execution duration.

        Returns:
            TaskResult: Wrapper-specific result with minimal metrics reflecting
                the architectural limitations of stateless LLM wrapper patterns.
        """

        success = llm_result["success"]
        base_cost = self.api_call_cost

        if llm_result["retries"] > 0:
            base_cost *= 1.3

        complexity_multiplier = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.05,
            TaskComplexity.COMPLEX: 1.1,
            TaskComplexity.ENTERPRISE: 1.15,
        }.get(complexity, 1.05)

        total_cost = base_cost * complexity_multiplier
        base_retention = 0.03

        if success:
            quality_bonus = llm_result["response_quality"] * 0.12
            context_retention = base_retention + quality_bonus
        else:
            context_retention = base_retention * 0.5

        steps_completed = 1 if success else 0
        total_steps = 1
        failure_reason = None

        if not success:
            failure_reasons = [
                "Single LLM call failed",
                "Insufficient context handling",
                "No planning capability",
                "Limited retry logic exhausted",
                "Task complexity exceeded wrapper capabilities",
            ]
            failure_reason = random.choice(failure_reasons)

        return TaskResult(
            task_id=task_id,
            task_name=task_name,
            task_type=complexity,
            agent_type=AgentType.WRAPPER_AGENT,
            success=success,
            execution_time=round(execution_time, 3),
            error_count=llm_result["retries"],
            context_retention_score=round(context_retention, 3),
            cost_estimate=round(total_cost, 4),
            failure_reason=failure_reason,
            steps_completed=steps_completed,
            total_steps=total_steps,
        )

    def get_success_rate(self) -> float:
        """Calculate overall success rate across all task execution attempts.

        Computes simple success percentage from execution history for performance
        tracking and comparison with other agent architectures.

        Args:
            None

        Returns:
            float: Success rate as decimal (0.0-1.0), or 0.0 if no executions.
        """
        if not self.execution_history:
            return 0.0
        successes = sum(1 for r in self.execution_history if r.success)
        return successes / len(self.execution_history)

    def get_average_cost_per_success(self) -> float:
        """Calculate economic efficiency metric for successful task completions.

        Computes the average cost incurred per successful task completion,
        providing insight into the economic trade-offs of wrapper agent
        architectures versus more sophisticated alternatives.

        Args:
            None

        Returns:
            float: Average cost per successful execution, or float('inf') if
                no successful executions have occurred.
        """
        successes = sum(1 for r in self.execution_history if r.success)
        total_cost = sum(r.cost_estimate for r in self.execution_history)
        return total_cost / successes if successes > 0 else float("inf")
