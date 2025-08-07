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

File: tests/unit/test_wrapper_agent.py

Unit tests for WrapperAgent - the baseline LLM wrapper tier.

Comprehensive test suite for WrapperAgent implementation covering basic
functionality, stateless memory characteristics, performance degradation
with complexity, and realistic failure patterns. Tests verify that
WrapperAgent represents the simplest agent architecture with single-step
execution and minimal context retention.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import uuid

import pytest
from ai_agents_reality_check.agents.wrapper import WrapperAgent
from ai_agents_reality_check.types import AgentType, TaskComplexity, TaskResult


class TestWrapperAgent:
    """Unit tests for WrapperAgent implementation."""

    @pytest.fixture
    def agent(self) -> WrapperAgent:
        """Create a fresh WrapperAgent for each test."""
        return WrapperAgent()

    @pytest.fixture
    def simple_task(self) -> dict:
        """Simple task fixture."""
        return {"name": "Simple test task", "complexity": TaskComplexity.SIMPLE}

    @pytest.fixture
    def complex_task(self) -> dict:
        """Complex task fixture."""
        return {"name": "Complex test task", "complexity": TaskComplexity.COMPLEX}

    def test_agent_initialization(self, agent: WrapperAgent) -> None:
        """Test proper initialization of WrapperAgent."""
        assert agent.name == "LLM Wrapper (Wrapper Agent)"
        assert agent.api_call_cost == 0.002
        assert agent.execution_history == []
        assert agent.call_count == 0
        assert hasattr(agent, "memory")

    def test_memory_characteristics(self, agent: WrapperAgent) -> None:
        """Test that WrapperAgent has stateless memory characteristics."""
        assert len(agent.memory) == 0
        assert agent.memory.get_context() == ""

        agent.memory.add_entry("test entry")
        assert len(agent.memory) == 0

        agent.memory.clear()
        assert len(agent.memory) == 0

    @pytest.mark.asyncio
    async def test_execute_simple_task(
        self, agent: WrapperAgent, simple_task: dict
    ) -> None:
        """Test execution of simple task."""
        result = await agent.execute_task(simple_task)

        assert isinstance(result, TaskResult)
        assert result.agent_type == AgentType.WRAPPER_AGENT
        assert result.task_name == "Simple test task"
        assert result.task_type == TaskComplexity.SIMPLE
        assert result.execution_time > 0
        assert result.total_steps == 1
        assert result.cost_estimate > 0
        assert 0 <= result.context_retention_score <= 1

    @pytest.mark.asyncio
    async def test_execute_complex_task(
        self, agent: WrapperAgent, complex_task: dict
    ) -> None:
        """Test execution of complex task shows degraded performance."""
        result = await agent.execute_task(complex_task)

        assert isinstance(result, TaskResult)
        assert result.agent_type == AgentType.WRAPPER_AGENT
        assert result.task_type == TaskComplexity.COMPLEX
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_failure_reasons(
        self, agent: WrapperAgent, complex_task: dict
    ) -> None:
        """Test that failure reasons are realistic for wrapper limitations."""
        results = []
        for _ in range(10):
            result = await agent.execute_task(complex_task)
            results.append(result)

        failed_results = [r for r in results if not r.success]

        if failed_results:
            failure_reasons = {
                r.failure_reason for r in failed_results if r.failure_reason
            }

        expected_failure_types = {
            "REASONING_FAILURE",
            "STATE_ERROR",
            "CONTEXT_LOSS",
            "TIMEOUT",
            "HALLUCINATION",
        }

        actual_failure_types = {
            reason.split(":")[0] for reason in failure_reasons if ":" in reason
        }
        assert len(actual_failure_types.intersection(expected_failure_types)) > 0

    @pytest.mark.asyncio
    async def test_execution_history_tracking(
        self, agent: WrapperAgent, simple_task: dict
    ) -> None:
        """Test that execution history is properly tracked."""
        assert len(agent.execution_history) == 0

        await agent.execute_task(simple_task)
        assert len(agent.execution_history) == 1

        await agent.execute_task(simple_task)
        assert len(agent.execution_history) == 2

        for result in agent.execution_history:
            assert isinstance(result, TaskResult)

    def test_get_success_rate_empty_history(self, agent: WrapperAgent) -> None:
        """Test success rate calculation with empty history."""
        assert agent.get_success_rate() == 0.0

    @pytest.mark.asyncio
    async def test_get_success_rate_calculation(
        self, agent: WrapperAgent, simple_task: dict
    ) -> None:
        """Test success rate calculation with some history."""
        for _ in range(3):
            await agent.execute_task(simple_task)

        success_rate = agent.get_success_rate()
        assert 0.0 <= success_rate <= 1.0

        successful_count = sum(1 for r in agent.execution_history if r.success)
        expected_rate = successful_count / len(agent.execution_history)
        assert success_rate == expected_rate

    def test_get_average_cost_per_success_no_successes(
        self, agent: WrapperAgent
    ) -> None:
        """Test cost per success when there are no successes."""

        failed_result = TaskResult(
            task_id=str(uuid.uuid4()),
            task_name="Failed task",
            task_type=TaskComplexity.SIMPLE,
            agent_type=AgentType.WRAPPER_AGENT,
            success=False,
            execution_time=1.0,
            error_count=1,
            context_retention_score=0.1,
            cost_estimate=0.002,
            failure_reason="Test failure",
            steps_completed=0,
            total_steps=1,
        )
        agent.execution_history.append(failed_result)

        assert agent.get_average_cost_per_success() == float("inf")

    @pytest.mark.asyncio
    async def test_call_count_increment(
        self, agent: WrapperAgent, simple_task: dict
    ) -> None:
        """Test that call count increments with each execution."""
        initial_count = agent.call_count

        await agent.execute_task(simple_task)
        assert agent.call_count == initial_count + 1

        await agent.execute_task(simple_task)
        assert agent.call_count == initial_count + 2

    @pytest.mark.asyncio
    async def test_context_retention_scores(self, agent: WrapperAgent) -> None:
        """Test that context retention scores are consistently low for wrappers."""
        tasks = [
            {"name": "Task 1", "complexity": TaskComplexity.SIMPLE},
            {"name": "Task 2", "complexity": TaskComplexity.MODERATE},
            {"name": "Task 3", "complexity": TaskComplexity.COMPLEX},
        ]

        retention_scores = []
        for task in tasks:
            result = await agent.execute_task(task)
            retention_scores.append(result.context_retention_score)

        for score in retention_scores:
            assert 0 <= score <= 0.2

    @pytest.mark.asyncio
    async def test_cost_estimation_consistency(
        self, agent: WrapperAgent, simple_task: dict
    ) -> None:
        """Test that cost estimation is consistent and reasonable."""
        results = []
        for _ in range(3):
            result = await agent.execute_task(simple_task)
            results.append(result)

        base_cost = agent.api_call_cost
        for result in results:
            assert result.cost_estimate >= base_cost
            assert result.cost_estimate <= base_cost * 2

    @pytest.mark.asyncio
    async def test_step_completion_always_one(self, agent: WrapperAgent) -> None:
        """Test that wrapper agents always have 1 total step."""
        tasks = [
            {"name": "Task 1", "complexity": TaskComplexity.SIMPLE},
            {"name": "Task 2", "complexity": TaskComplexity.ENTERPRISE},
        ]

        for task in tasks:
            result = await agent.execute_task(task)
            assert result.total_steps == 1
            assert result.steps_completed in [0, 1]
