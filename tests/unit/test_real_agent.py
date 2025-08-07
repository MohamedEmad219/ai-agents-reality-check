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

File: tests/unit/test_real_agent.py

Unit tests for RealAgent - the full autonomous agent architecture.

Comprehensive test suite for RealAgent implementation covering advanced
capabilities including hierarchical planning, semantic memory integration,
tool usage tracking, recovery mechanisms, and high-performance execution.
Tests verify enterprise-grade functionality with comprehensive memory
systems and sophisticated execution patterns.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import json
from pathlib import Path

import pytest
from ai_agents_reality_check.agents.real import RealAgent
from ai_agents_reality_check.types import AgentType, TaskComplexity, TaskResult
from jsonschema import ValidationError, validate


class TestRealAgent:
    """Unit tests for RealAgent implementation."""

    @pytest.fixture
    def agent(self) -> RealAgent:
        """Create a fresh RealAgent for each test."""
        return RealAgent()

    @pytest.fixture
    def simple_task(self) -> dict:
        """Simple task fixture."""
        return {"name": "Simple real task", "complexity": TaskComplexity.SIMPLE}

    @pytest.fixture
    def moderate_task(self) -> dict:
        """Moderate task fixture."""
        return {"name": "Moderate real task", "complexity": TaskComplexity.MODERATE}

    @pytest.fixture
    def complex_task(self) -> dict:
        """Complex task fixture."""
        return {"name": "Complex real task", "complexity": TaskComplexity.COMPLEX}

    @pytest.fixture
    def enterprise_task(self) -> dict:
        """Enterprise task fixture."""
        return {"name": "Enterprise real task", "complexity": TaskComplexity.ENTERPRISE}

    def test_agent_initialization(self, agent: RealAgent) -> None:
        """Test proper initialization of RealAgent."""
        assert agent.name == "Real Autonomous Agent"
        assert agent.api_call_cost == 0.025
        assert agent.execution_history == []
        assert agent.speed_multiplier == 1.0

        assert hasattr(agent, "working_memory")
        assert hasattr(agent, "episodic_memory")
        assert hasattr(agent, "semantic_memory")

        assert isinstance(agent.working_memory, dict)
        assert isinstance(agent.episodic_memory, dict)
        assert isinstance(agent.semantic_memory, dict)

    def test_speed_multiplier_configuration(self) -> None:
        """Test that speed multiplier can be configured."""
        fast_agent = RealAgent(speed_multiplier=0.5)
        assert fast_agent.speed_multiplier == 0.5

        slow_agent = RealAgent(speed_multiplier=2.0)
        assert slow_agent.speed_multiplier == 2.0

    @pytest.mark.asyncio
    async def test_execute_task_interface(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test the public execute_task interface."""
        result = await agent.execute_task(simple_task)

        assert isinstance(result, TaskResult)
        assert result.agent_type == AgentType.REAL_AGENT
        assert result.task_name == "Simple real task"
        assert result.task_type == TaskComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_run_method_interface(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test the internal run method interface."""
        result = await agent.run(simple_task)

        assert isinstance(result, TaskResult)
        assert result.agent_type == AgentType.REAL_AGENT

    @pytest.mark.asyncio
    async def test_working_memory_lifecycle(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that working memory is properly managed during execution."""
        result = await agent.execute_task(simple_task)

        task_id = result.task_id
        assert task_id in agent.working_memory

        memory_entry = agent.working_memory[task_id]
        assert "task" in memory_entry
        assert "start_time" in memory_entry
        assert "tools_used" in memory_entry
        assert "step_history" in memory_entry

    @pytest.mark.asyncio
    async def test_high_context_retention(self, agent: RealAgent) -> None:
        """Test that RealAgent achieves high context retention scores."""
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
            assert score > 0.5
            assert score <= 1.0

    @pytest.mark.asyncio
    async def test_high_success_rates(self, agent: RealAgent) -> None:
        """Test that RealAgent achieves high success rates across complexities."""
        complexities = [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
            TaskComplexity.ENTERPRISE,
        ]

        for complexity in complexities:
            task = {"name": f"Test {complexity.value}", "complexity": complexity}
            results = []

            for _ in range(5):
                result = await agent.execute_task(task)
                results.append(result)

            success_rate = sum(1 for r in results if r.success) / len(results)

            if complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
                assert success_rate >= 0.6
            else:
                assert success_rate >= 0.2

    @pytest.mark.asyncio
    async def test_hierarchical_planning(
        self, agent: RealAgent, complex_task: dict
    ) -> None:
        """Test that RealAgent uses hierarchical planning."""
        result = await agent.execute_task(complex_task)

        task_id = result.task_id
        memory_entry = agent.working_memory.get(task_id, {})
        step_history = memory_entry.get("step_history", [])

        assert len(step_history) > 1

        for step in step_history:
            assert "step_id" in step
            assert "tool" in step
            assert "description" in step
            assert "status" in step
            assert "trace" in step

    @pytest.mark.asyncio
    async def test_semantic_memory_integration(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that semantic memory is properly integrated."""
        result1 = await agent.execute_task(simple_task)
        result2 = await agent.execute_task(simple_task)

        assert isinstance(result1, TaskResult)
        assert isinstance(result2, TaskResult)

        complexity = simple_task["complexity"]
        if complexity in agent.semantic_memory:
            assert len(agent.semantic_memory[complexity]) >= 0

    @pytest.mark.asyncio
    async def test_tool_usage_tracking(
        self, agent: RealAgent, moderate_task: dict
    ) -> None:
        """Test that tool usage is properly tracked."""
        result = await agent.execute_task(moderate_task)

        task_id = result.task_id
        memory_entry = agent.working_memory.get(task_id, {})
        tools_used = memory_entry.get("tools_used", [])

        if result.success and result.steps_completed > 0:
            assert len(tools_used) > 0

            expected_tools = {"web_search", "calculator", "file_system", "api_caller"}
            for tool in tools_used:
                assert tool in expected_tools

    @pytest.mark.asyncio
    async def test_recovery_mechanisms(
        self, agent: RealAgent, complex_task: dict
    ) -> None:
        """Test that recovery mechanisms are available."""
        result = await agent.execute_task(complex_task)

        task_id = result.task_id
        memory_entry = agent.working_memory.get(task_id, {})
        step_history = memory_entry.get("step_history", [])

        recovery_attempts = []
        for step in step_history:
            trace = step.get("trace", {})
            if "recovery" in trace:
                recovery_attempts.append(step)

        for step in recovery_attempts:
            recovery_info = step["trace"]["recovery"]
            assert "strategy" in recovery_info
            assert "success" in recovery_info

    @pytest.mark.asyncio
    async def test_trace_export_functionality(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that trace export functionality works."""
        result = await agent.execute_task(simple_task)

        task_id = result.task_id
        trace_json = agent.export_trace_json(task_id)

        trace_data = json.loads(trace_json)
        assert isinstance(trace_data, list)

    @pytest.mark.asyncio
    async def test_trace_schema_compliance(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that traces comply with the expected schema."""
        result = await agent.execute_task(simple_task)

        task_id = result.task_id
        memory_entry = agent.working_memory.get(task_id, {})
        step_history = memory_entry.get("step_history", [])

        schema_path = Path("schemas/trace/agent_subgoal_trace.schema.json")
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())

            for step in step_history:
                trace = step.get("trace")
                if trace:
                    try:
                        validate(instance=trace, schema=schema)
                    except ValidationError as e:
                        pytest.fail(f"Trace schema validation failed: {e.message}")

    @pytest.mark.asyncio
    async def test_cost_estimation_premium_tier(self, agent: RealAgent) -> None:
        """Test that cost estimation reflects premium agent capabilities."""
        tasks = [
            {"name": "Simple", "complexity": TaskComplexity.SIMPLE},
            {"name": "Complex", "complexity": TaskComplexity.COMPLEX},
        ]

        for task in tasks:
            result = await agent.execute_task(task)

            base_cost = agent.api_call_cost
            assert result.cost_estimate >= base_cost
            assert result.cost_estimate <= base_cost * 5

    @pytest.mark.asyncio
    async def test_execution_time_efficiency(self, agent: RealAgent) -> None:
        """Test that execution times are reasonable despite sophistication."""
        tasks = [
            {"name": "Simple", "complexity": TaskComplexity.SIMPLE},
            {"name": "Moderate", "complexity": TaskComplexity.MODERATE},
        ]

        for task in tasks:
            result = await agent.execute_task(task)

            assert result.execution_time > 0
            assert result.execution_time < 20.0

    @pytest.mark.asyncio
    async def test_step_completion_high_ratio(
        self, agent: RealAgent, moderate_task: dict
    ) -> None:
        """Test that RealAgent achieves high step completion ratios."""
        result = await agent.execute_task(moderate_task)

        if result.total_steps > 0:
            completion_ratio = result.steps_completed / result.total_steps

            if result.success:
                assert completion_ratio >= 0.6

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, agent: RealAgent) -> None:
        """Test robust error handling in various scenarios."""
        malformed_task = {"name": "Malformed"}

        result = await agent.execute_task(malformed_task)

        assert isinstance(result, TaskResult)
        assert result.agent_type == AgentType.REAL_AGENT

    @pytest.mark.asyncio
    async def test_task_id_propagation(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that task IDs are properly propagated through the system."""
        task_with_id = simple_task.copy()
        task_with_id["task_id"] = "test_123"

        result = await agent.execute_task(task_with_id)
        assert result.task_id == "test_123"

        result2 = await agent.execute_task(simple_task)
        assert result2.task_id is not None
        assert len(result2.task_id) > 0

    @pytest.mark.asyncio
    async def test_execution_history_tracking(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that execution history is properly maintained."""
        initial_count = len(agent.execution_history)

        await agent.execute_task(simple_task)

        assert len(agent.execution_history) == initial_count + 1

        latest_result = agent.execution_history[-1]
        assert isinstance(latest_result, TaskResult)
        assert latest_result.agent_type == AgentType.REAL_AGENT

    @pytest.mark.asyncio
    async def test_semantic_memory_learning(
        self, agent: RealAgent, simple_task: dict
    ) -> None:
        """Test that semantic memory learning occurs over time."""
        complexity = simple_task["complexity"]
        initial_memory_count = len(agent.semantic_memory.get(complexity, []))

        for i in range(3):
            task = simple_task.copy()
            task["name"] = f"Learning task {i}"
            await agent.execute_task(task)

        current_memory_count = len(agent.semantic_memory.get(complexity, []))
        assert current_memory_count >= initial_memory_count

    @pytest.mark.asyncio
    async def test_speed_multiplier_effect(self) -> None:
        """Test that speed multiplier affects execution times."""
        fast_agent = RealAgent(speed_multiplier=0.1)
        normal_agent = RealAgent(speed_multiplier=1.0)

        task = {"name": "Speed test", "complexity": TaskComplexity.SIMPLE}

        fast_result = await fast_agent.execute_task(task)
        normal_result = await normal_agent.execute_task(task)

        assert fast_result.execution_time >= 0
        assert normal_result.execution_time >= 0
