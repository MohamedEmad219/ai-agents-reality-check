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

Module: ai_agents_reality_check/agents/real/real_planner.py

Hierarchical planning engine with semantic memory integration for the RealAgent.
Implements intelligent plan construction through template reuse and fresh generation,
enabling the agent to leverage past successful executions while adapting to novel
task requirements.

The planner supports both semantic reuse of proven plan templates and dynamic
generation of new execution strategies, providing the architectural foundation
for the RealAgent's superior performance characteristics.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import random
import time
import uuid
from typing import Any

from ai_agents_reality_check.types import TaskComplexity


def create_hierarchical_plan(
    task: dict[str, Any],
    semantic_memory: dict[TaskComplexity, list[dict[str, Any]]],
    working_memory: dict[str, Any],
) -> dict[str, Any]:
    """Generate or reuse hierarchical execution plans with semantic memory integration.

    Creates structured execution plans through either semantic memory reuse of
    previously successful templates or fresh generation of new plan structures.
    The planner prioritizes reuse of proven strategies while maintaining flexibility
    for novel task requirements, forming the foundation of the RealAgent's
    planning capabilities.

    Plans include hierarchical subgoal decomposition, tool selection, execution
    control parameters, and comprehensive tracing infrastructure to support
    robust task execution and learning.

    Args:
        task (Dict[str, Any]): Task specification containing:
            - task_id (str): Unique task identifier for tracking
            - task_name (str): Human-readable task description
            - task_type (TaskComplexity): Difficulty classification for planning
            - Additional task-specific parameters for plan customization
        semantic_memory (Dict[TaskComplexity, List[Dict[str, Any]]]): Memory store
            containing successful plan templates organized by complexity level
            for intelligent reuse and optimization.
        working_memory (Dict[str, Any]): Active agent state including current
            execution context, tool usage history, and runtime parameters.

    Returns:
        Dict[str, Any]: Hierarchical execution plan containing:
            - plan_id (str): Unique plan identifier for tracking
            - task_id (str): Associated task identifier
            - task_name (str): Human-readable task description
            - complexity (TaskComplexity): Task difficulty classification
            - plan_type (str): "semantic_reuse" or "generated_fresh"
            - created_at (float): Plan creation timestamp
            - subgoals (List[Dict[str, Any]]): Ordered execution steps with
            tool assignments, descriptions, and tracing infrastructure
            - control (Dict[str, Any]): Execution parameters including retry
            limits, backoff strategies, and execution policies

    Note:
        The planner prioritizes semantic reuse when successful templates exist
        for the target complexity level, falling back to fresh generation when
        no suitable templates are available. This enables continuous learning
        and performance optimization over time.
    """
    task_id = task["task_id"]
    complexity = task["task_type"]
    task_name = task["task_name"]

    if complexity in semantic_memory:
        successful_plans = [
            p for p in semantic_memory[complexity] if p.get("status") == "success"
        ]
        if successful_plans:
            reused = random.choice(successful_plans).copy()
            reused.update(
                {
                    "plan_id": str(uuid.uuid4()),
                    "task_id": task_id,
                    "task_name": task_name,
                    "created_at": time.time(),
                    "plan_type": "semantic_reuse",
                }
            )
            return reused

    tool_pool = ["web_search", "calculator", "file_system", "api_caller"]
    num_steps = random.randint(3, 5)

    subgoals = []
    for i in range(num_steps):
        step_id = f"{task_id}_step_{i}"
        tool = random.choice(tool_pool)
        subgoals.append(
            {
                "step_id": step_id,
                "tool": tool,
                "description": f"Step {i + 1}: Use {tool} for {task_name}",
                "status": "pending",
                "trace": {
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "output": None,
                    "last_error": None,
                    "failure_reason": None,
                },
            }
        )

    plan = {
        "plan_id": str(uuid.uuid4()),
        "task_id": task_id,
        "task_name": task_name,
        "complexity": complexity,
        "plan_type": "generated_fresh",
        "created_at": time.time(),
        "subgoals": subgoals,
        "control": {
            "max_retries": 2,
            "retry_backoff": 0.1,
            "execution_policy": "sequential",
        },
    }

    return plan
