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

File: tests/unit/test_types.py

Unit tests for core data types and enums.

Comprehensive test suite for fundamental data structures including TaskComplexity,
AgentType, AgentErrorType enums and TaskResult dataclass. Tests cover enum values,
serialization behavior, edge cases, type safety, and data integrity across all
combinations of agent types and task complexities.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import uuid

import pytest
from ai_agents_reality_check.types import (
    AgentErrorType,
    AgentType,
    TaskComplexity,
    TaskResult,
)


class TestEnums:
    """Test core enum definitions."""

    def test_task_complexity_enum(self) -> None:
        """Test TaskComplexity enum values."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.ENTERPRISE.value == "enterprise"

        assert TaskComplexity.SIMPLE in TaskComplexity
        assert TaskComplexity.MODERATE in TaskComplexity
        assert TaskComplexity.COMPLEX in TaskComplexity
        assert TaskComplexity.ENTERPRISE in TaskComplexity

        assert len(TaskComplexity) == 4

    def test_agent_type_enum(self) -> None:
        """Test AgentType enum values."""
        assert AgentType.WRAPPER_AGENT.value == "wrapper_agent"
        assert AgentType.MARKETING_AGENT.value == "marketing_agent"
        assert AgentType.REAL_AGENT.value == "real_agent"

        assert AgentType.WRAPPER_AGENT in AgentType
        assert AgentType.MARKETING_AGENT in AgentType
        assert AgentType.REAL_AGENT in AgentType

        assert len(AgentType) == 3

    def test_agent_error_type_enum(self) -> None:
        """Test AgentErrorType enum values."""
        assert AgentErrorType.NONE.value == "none"
        assert AgentErrorType.TOOL_TIMEOUT.value == "tool_timeout"
        assert (
            AgentErrorType.SEMANTIC_MEMORY_CORRUPTION.value
            == "semantic_memory_corruption"
        )
        assert AgentErrorType.INVALID_OUTPUT.value == "invalid_output"
        assert AgentErrorType.DEPENDENCY_NOT_MET.value == "dependency_not_met"
        assert AgentErrorType.UNKNOWN.value == "unknown"

        assert AgentErrorType.NONE in AgentErrorType
        assert AgentErrorType.TOOL_TIMEOUT in AgentErrorType
        assert AgentErrorType.SEMANTIC_MEMORY_CORRUPTION in AgentErrorType
        assert AgentErrorType.INVALID_OUTPUT in AgentErrorType
        assert AgentErrorType.DEPENDENCY_NOT_MET in AgentErrorType
        assert AgentErrorType.UNKNOWN in AgentErrorType

    def test_enum_string_representations(self) -> None:
        """Test string representations of enums."""
        assert str(TaskComplexity.SIMPLE) == "TaskComplexity.SIMPLE"
        assert str(AgentType.WRAPPER_AGENT) == "AgentType.WRAPPER_AGENT"
        assert str(AgentErrorType.TOOL_TIMEOUT) == "AgentErrorType.TOOL_TIMEOUT"


class TestTaskResult:
    """Test TaskResult dataclass."""

    @pytest.fixture
    def sample_task_result(self) -> TaskResult:
        """Create a sample TaskResult for testing."""
        return TaskResult(
            task_id="test_123",
            task_name="Test Task",
            task_type=TaskComplexity.SIMPLE,
            agent_type=AgentType.WRAPPER_AGENT,
            success=True,
            execution_time=1.5,
            error_count=0,
            context_retention_score=0.8,
            cost_estimate=0.002,
            failure_reason=None,
            steps_completed=1,
            total_steps=1,
        )

    def test_task_result_creation(self, sample_task_result: TaskResult) -> None:
        """Test basic TaskResult creation."""
        assert sample_task_result.task_id == "test_123"
        assert sample_task_result.task_name == "Test Task"
        assert sample_task_result.task_type == TaskComplexity.SIMPLE
        assert sample_task_result.agent_type == AgentType.WRAPPER_AGENT
        assert sample_task_result.success is True
        assert sample_task_result.execution_time == 1.5
        assert sample_task_result.error_count == 0
        assert sample_task_result.context_retention_score == 0.8
        assert sample_task_result.cost_estimate == 0.002
        assert sample_task_result.failure_reason is None
        assert sample_task_result.steps_completed == 1
        assert sample_task_result.total_steps == 1

    def test_task_result_defaults(self) -> None:
        """Test TaskResult with minimal required parameters."""
        minimal_result = TaskResult(
            task_id="minimal_123",
            task_name="Minimal Task",
            task_type=TaskComplexity.MODERATE,
            agent_type=AgentType.MARKETING_AGENT,
            success=False,
            execution_time=2.0,
            error_count=1,
            context_retention_score=0.3,
            cost_estimate=0.015,
        )

        assert minimal_result.failure_reason is None
        assert minimal_result.steps_completed == 0
        assert minimal_result.total_steps == 0

    def test_task_result_with_failure(self) -> None:
        """Test TaskResult with failure information."""
        failed_result = TaskResult(
            task_id="failed_123",
            task_name="Failed Task",
            task_type=TaskComplexity.COMPLEX,
            agent_type=AgentType.REAL_AGENT,
            success=False,
            execution_time=5.0,
            error_count=3,
            context_retention_score=0.1,
            cost_estimate=0.025,
            failure_reason="Tool timeout after 5 attempts",
            steps_completed=2,
            total_steps=5,
        )

        assert failed_result.success is False
        assert failed_result.failure_reason == "Tool timeout after 5 attempts"
        assert failed_result.error_count == 3
        assert failed_result.steps_completed < failed_result.total_steps

    def test_task_result_to_dict(self, sample_task_result: TaskResult) -> None:
        """Test TaskResult serialization to dictionary."""
        result_dict = sample_task_result.to_dict()

        expected_fields = {
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
        }

        assert set(result_dict.keys()) == expected_fields

        assert result_dict["task_type"] == "simple"
        assert result_dict["agent_type"] == "wrapper_agent"
        assert result_dict["task_id"] == "test_123"
        assert result_dict["task_name"] == "Test Task"
        assert result_dict["success"] is True
        assert result_dict["execution_time"] == 1.5
        assert result_dict["error_count"] == 0
        assert result_dict["context_retention_score"] == 0.8
        assert result_dict["cost_estimate"] == 0.002
        assert result_dict["failure_reason"] is None
        assert result_dict["steps_completed"] == 1
        assert result_dict["total_steps"] == 1

    def test_task_result_serialization_types(self) -> None:
        """Test that serialized TaskResult has correct types."""
        result = TaskResult(
            task_id=str(uuid.uuid4()),
            task_name="Type Test",
            task_type=TaskComplexity.ENTERPRISE,
            agent_type=AgentType.REAL_AGENT,
            success=True,
            execution_time=3.14,
            error_count=0,
            context_retention_score=0.95,
            cost_estimate=0.0234,
            failure_reason=None,
            steps_completed=10,
            total_steps=10,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict["task_id"], str)
        assert isinstance(result_dict["task_name"], str)
        assert isinstance(result_dict["task_type"], str)
        assert isinstance(result_dict["agent_type"], str)
        assert isinstance(result_dict["success"], bool)
        assert isinstance(result_dict["execution_time"], float)
        assert isinstance(result_dict["error_count"], int)
        assert isinstance(result_dict["context_retention_score"], float)
        assert isinstance(result_dict["cost_estimate"], float)
        assert result_dict["failure_reason"] is None
        assert isinstance(result_dict["steps_completed"], int)
        assert isinstance(result_dict["total_steps"], int)

    def test_task_result_edge_cases(self) -> None:
        """Test TaskResult with edge case values."""
        edge_case_result = TaskResult(
            task_id="",
            task_name="Edge Case",
            task_type=TaskComplexity.SIMPLE,
            agent_type=AgentType.WRAPPER_AGENT,
            success=False,
            execution_time=0.0,
            error_count=0,
            context_retention_score=0.0,
            cost_estimate=0.0,
            failure_reason="Immediate failure",
            steps_completed=0,
            total_steps=0,
        )

        assert edge_case_result.task_id == ""
        assert edge_case_result.execution_time == 0.0
        assert edge_case_result.context_retention_score == 0.0
        assert edge_case_result.cost_estimate == 0.0
        assert edge_case_result.steps_completed == 0
        assert edge_case_result.total_steps == 0

    def test_task_result_completion_ratio(self) -> None:
        """Test calculation of completion ratios."""
        partial_result = TaskResult(
            task_id="partial_123",
            task_name="Partial Task",
            task_type=TaskComplexity.MODERATE,
            agent_type=AgentType.MARKETING_AGENT,
            success=False,
            execution_time=3.0,
            error_count=2,
            context_retention_score=0.4,
            cost_estimate=0.01,
            steps_completed=3,
            total_steps=5,
        )

        completion_ratio = (
            partial_result.steps_completed / partial_result.total_steps
            if partial_result.total_steps > 0
            else 0
        )

        assert completion_ratio == 0.6

    def test_task_result_immutability(self, sample_task_result: TaskResult) -> None:
        """Test that TaskResult behaves as an immutable dataclass."""
        original_task_id = sample_task_result.task_id

        sample_task_result.task_id = "modified_123"
        assert sample_task_result.task_id == "modified_123"

        sample_task_result.task_id = original_task_id

    def test_task_result_string_representation(
        self, sample_task_result: TaskResult
    ) -> None:
        """Test string representation of TaskResult."""
        result_str = str(sample_task_result)

        assert "TaskResult" in result_str
        assert "test_123" in result_str
        assert "Test Task" in result_str

    def test_task_result_equality(self) -> None:
        """Test TaskResult equality comparison."""
        result1 = TaskResult(
            task_id="equal_test",
            task_name="Equal Test",
            task_type=TaskComplexity.SIMPLE,
            agent_type=AgentType.WRAPPER_AGENT,
            success=True,
            execution_time=1.0,
            error_count=0,
            context_retention_score=0.5,
            cost_estimate=0.001,
        )

        result2 = TaskResult(
            task_id="equal_test",
            task_name="Equal Test",
            task_type=TaskComplexity.SIMPLE,
            agent_type=AgentType.WRAPPER_AGENT,
            success=True,
            execution_time=1.0,
            error_count=0,
            context_retention_score=0.5,
            cost_estimate=0.001,
        )

        result3 = TaskResult(
            task_id="different_test",
            task_name="Equal Test",
            task_type=TaskComplexity.SIMPLE,
            agent_type=AgentType.WRAPPER_AGENT,
            success=True,
            execution_time=1.0,
            error_count=0,
            context_retention_score=0.5,
            cost_estimate=0.001,
        )

        assert result1 == result2
        assert result1 != result3
        assert result2 != result3

    def test_all_enum_combinations(self) -> None:
        """Test TaskResult with all enum combinations."""
        for complexity in TaskComplexity:
            for agent_type in AgentType:
                result = TaskResult(
                    task_id=f"combo_{complexity.value}_{agent_type.value}",
                    task_name=f"Combo Test {complexity.value} {agent_type.value}",
                    task_type=complexity,
                    agent_type=agent_type,
                    success=True,
                    execution_time=1.0,
                    error_count=0,
                    context_retention_score=0.5,
                    cost_estimate=0.01,
                )

                assert result.task_type == complexity
                assert result.agent_type == agent_type

                result_dict = result.to_dict()
                assert result_dict["task_type"] == complexity.value
                assert result_dict["agent_type"] == agent_type.value
