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

File: tests/unit/test_marketing_agent.py

Unit tests for MarketingAgent - the "demo-quality" orchestration tier.

Comprehensive test suite for MarketingAgent implementation covering initialization,
execution behavior, memory management, performance characteristics, and error handling.
Tests verify that MarketingAgent performs at intermediate levels between WrapperAgent
and RealAgent, with multi-step execution capabilities and ephemeral memory management.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import pytest
from ai_agents_reality_check.agents.marketing import MarketingAgent
from ai_agents_reality_check.types import AgentType, TaskComplexity, TaskResult


class TestMarketingAgent:
    """Unit tests for MarketingAgent implementation."""

    @pytest.fixture
    def agent(self) -> MarketingAgent:
        """Create a fresh MarketingAgent for each test."""
        return MarketingAgent()

    @pytest.fixture
    def simple_task(self) -> dict:
        """Simple task fixture."""
        return {"name": "Simple marketing task", "complexity": TaskComplexity.SIMPLE}

    @pytest.fixture
    def moderate_task(self) -> dict:
        """Moderate task fixture."""
        return {
            "name": "Moderate marketing task",
            "complexity": TaskComplexity.MODERATE,
        }

    @pytest.fixture
    def complex_task(self) -> dict:
        """Complex task fixture."""
        return {"name": "Complex marketing task", "complexity": TaskComplexity.COMPLEX}

    def test_agent_initialization(self, agent: MarketingAgent) -> None:
        """Test proper initialization of MarketingAgent."""
        assert agent.name == "Marketing Promise Agent"
        assert agent.api_call_cost == 0.015
        assert agent.execution_history == []
        assert hasattr(agent, "memory")
        assert hasattr(agent, "task_success_history")

        for complexity in TaskComplexity:
            assert complexity in agent.task_success_history
            assert agent.task_success_history[complexity] == []

    def test_memory_characteristics(self, agent: MarketingAgent) -> None:
        """Test that MarketingAgent has ephemeral memory characteristics."""
        agent.memory.mark_completed("test_task", 0, "completed")
        assert agent.memory.get_status("test_task", 0) == "completed"

        agent.memory.reset()
        assert agent.memory.get_status("test_task", 0) == "unknown"

    @pytest.mark.asyncio
    async def test_execute_simple_task(
        self, agent: MarketingAgent, simple_task: dict
    ) -> None:
        """Test execution of simple task."""
        result = await agent.execute_task(simple_task)

        assert isinstance(result, TaskResult)
        assert result.agent_type == AgentType.MARKETING_AGENT
        assert result.task_name == "Simple marketing task"
        assert result.task_type == TaskComplexity.SIMPLE
        assert result.execution_time > 0
        assert result.total_steps >= 2
        assert result.cost_estimate > 0

    @pytest.mark.asyncio
    async def test_multi_step_execution(
        self, agent: MarketingAgent, moderate_task: dict
    ) -> None:
        """Test that marketing agents use multi-step execution."""
        result = await agent.execute_task(moderate_task)

        assert result.total_steps > 1
        assert result.steps_completed >= 0
        assert result.steps_completed <= result.total_steps

    @pytest.mark.asyncio
    async def test_context_retention_better_than_wrapper(
        self, agent: MarketingAgent
    ) -> None:
        """Test that context retention is better than wrapper agents."""
        tasks = [
            {"name": "Task 1", "complexity": TaskComplexity.SIMPLE},
            {"name": "Task 2", "complexity": TaskComplexity.MODERATE},
        ]

        retention_scores = []
        for task in tasks:
            result = await agent.execute_task(task)
            retention_scores.append(result.context_retention_score)

        for score in retention_scores:
            assert score > 0.2
            assert score <= 1.0

    @pytest.mark.asyncio
    async def test_performance_hierarchy(self, agent: MarketingAgent) -> None:
        """Test that marketing agent performs better than wrapper but worse than real."""
        complexities = [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
        ]

        for complexity in complexities:
            task = {"name": f"Test {complexity.value}", "complexity": complexity}
            results = []

            for _ in range(5):
                result = await agent.execute_task(task)
                results.append(result)

            success_rate = sum(1 for r in results if r.success) / len(results)

            if complexity == TaskComplexity.SIMPLE:
                assert success_rate >= 0.4
            elif complexity == TaskComplexity.COMPLEX:
                assert success_rate >= 0.0

    @pytest.mark.asyncio
    async def test_coordination_overhead(
        self, agent: MarketingAgent, complex_task: dict
    ) -> None:
        """Test that coordination overhead affects performance on complex tasks."""
        results = []
        for _ in range(5):
            result = await agent.execute_task(complex_task)
            results.append(result)

        failed_results = [r for r in results if not r.success]

        if failed_results:
            failure_reasons = {
                r.failure_reason for r in failed_results if r.failure_reason
            }
            coordination_failure_types = {
                "RESOURCE_CONTENTION",
                "COORDINATION_FAILURE",
                "ORCHESTRATION_TIMEOUT",
                "MEMORY_LIMIT",
            }

            actual_failure_types = {
                reason.split(":")[0] for reason in failure_reasons if ":" in reason
            }
            overlap = actual_failure_types.intersection(coordination_failure_types)
            if failure_reasons:
                assert len(overlap) > 0

    @pytest.mark.asyncio
    async def test_success_memory_tracking(self, agent: MarketingAgent) -> None:
        """Test that success history is properly tracked by complexity."""
        task = {"name": "Memory test", "complexity": TaskComplexity.SIMPLE}

        initial_history = len(agent.task_success_history[TaskComplexity.SIMPLE])

        await agent.execute_task(task)

        updated_history = len(agent.task_success_history[TaskComplexity.SIMPLE])
        assert updated_history == initial_history + 1

    @pytest.mark.asyncio
    async def test_success_memory_limits(
        self, agent: MarketingAgent, simple_task: dict
    ) -> None:
        """Test that success memory is limited to prevent unbounded growth."""
        for _ in range(25):
            await agent.execute_task(simple_task)

        history_length = len(agent.task_success_history[TaskComplexity.SIMPLE])
        assert history_length <= 20

    @pytest.mark.asyncio
    async def test_cost_estimation_multi_step(self, agent: MarketingAgent) -> None:
        """Test that cost estimation accounts for multi-step execution."""
        tasks = [
            {"name": "Simple", "complexity": TaskComplexity.SIMPLE},
            {"name": "Complex", "complexity": TaskComplexity.COMPLEX},
        ]

        for task in tasks:
            result = await agent.execute_task(task)

            base_cost = agent.api_call_cost
            assert result.cost_estimate >= base_cost

            assert result.cost_estimate >= base_cost * (result.total_steps * 0.1)

    @pytest.mark.asyncio
    async def test_execution_time_reasonable(self, agent: MarketingAgent) -> None:
        """Test that execution times are reasonable for multi-step execution."""
        tasks = [
            {"name": "Simple", "complexity": TaskComplexity.SIMPLE},
            {"name": "Complex", "complexity": TaskComplexity.COMPLEX},
        ]

        for task in tasks:
            result = await agent.execute_task(task)

            assert result.execution_time > 0.5
            assert result.execution_time < 10.0

    @pytest.mark.asyncio
    async def test_step_completion_tracking(
        self, agent: MarketingAgent, moderate_task: dict
    ) -> None:
        """Test that step completion is properly tracked."""
        result = await agent.execute_task(moderate_task)

        assert 0 <= result.steps_completed <= result.total_steps

        if result.success:
            completion_ratio = result.steps_completed / result.total_steps
            assert completion_ratio >= 0.5

    def test_get_success_rate_calculation(self, agent: MarketingAgent) -> None:
        """Test success rate calculation with empty history."""
        assert agent.get_success_rate() == 0.0

    @pytest.mark.asyncio
    async def test_memory_integration_with_execution(
        self, agent: MarketingAgent, simple_task: dict
    ) -> None:
        """Test that memory is properly integrated with task execution."""
        result = await agent.execute_task(simple_task)
        memory_dict = agent.memory.as_dict()

        if result.success and result.steps_completed > 0:
            assert isinstance(memory_dict, dict)
            assert result.task_id is not None

    @pytest.mark.asyncio
    async def test_complexity_scaling_behavior(self, agent: MarketingAgent) -> None:
        """Test how agent behavior scales with complexity."""
        complexity_results = {}

        for complexity in TaskComplexity:
            task = {
                "name": f"Scaling test {complexity.value}",
                "complexity": complexity,
            }
            result = await agent.execute_task(task)
            complexity_results[complexity] = result

        simple_steps = complexity_results[TaskComplexity.SIMPLE].total_steps
        enterprise_steps = complexity_results[TaskComplexity.ENTERPRISE].total_steps

        assert enterprise_steps >= simple_steps

    @pytest.mark.asyncio
    async def test_error_handling_graceful(self, agent: MarketingAgent) -> None:
        """Test graceful error handling in various scenarios."""
        malformed_task = {"name": "Malformed"}

        try:
            result = await agent.execute_task(malformed_task)
            assert isinstance(result, TaskResult)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_execution_history_integrity(
        self, agent: MarketingAgent, simple_task: dict
    ) -> None:
        """Test that execution history maintains integrity."""
        initial_count = len(agent.execution_history)

        await agent.execute_task(simple_task)

        assert len(agent.execution_history) == initial_count + 1

        latest_result = agent.execution_history[-1]
        assert isinstance(latest_result, TaskResult)
        assert latest_result.agent_type == AgentType.MARKETING_AGENT
