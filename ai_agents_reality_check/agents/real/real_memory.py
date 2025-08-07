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

Module: ai_agents_reality_check/agents/real/real_memory.py

Semantic memory management system enabling plan reuse and continual learning
for production-grade agent architectures. Implements intelligent plan template
storage, success-based filtering, and complexity-organized retrieval that
enables RealAgent's performance optimization over time.

This module provides the persistent learning infrastructure that differentiates
sophisticated agent systems from shallow implementations, enabling continuous
improvement through experience accumulation and successful pattern reuse.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

from typing import Any

from ai_agents_reality_check.types import TaskComplexity


async def verify_and_learn(
    task_id: str,
    execution_result: dict[str, Any],
    semantic_memory: dict[TaskComplexity, list[dict[str, Any]]],
) -> bool:
    """Extract and store successful execution patterns in semantic memory for reuse.

    Analyzes completed task executions to identify successful plan templates worthy
    of preservation in semantic memory. Implements success-based filtering, plan
    template cleaning, and duplicate detection to build a curated repository of
    proven execution strategies organized by task complexity.

    The learning process requires a minimum success threshold (75% step completion)
    to ensure only reliable patterns are preserved. Successful plans are cleaned
    of execution-specific details while retaining structural information and tool
    selection patterns that enable effective reuse in future similar tasks.

    Args:
        task_id (str): Unique identifier for the completed task execution, used
            for tracing and debugging purposes during the learning process.
        execution_result (Dict[str, Any]): Complete execution plan with results:
            - subgoals (List[Dict[str, Any]]): Executed steps with status and traces
            - complexity (TaskComplexity): Task difficulty level for organization
            - task_name (str): Task description for template identification
            - control (Dict[str, Any], optional): Execution control parameters
            - Additional execution metadata and timing information
        semantic_memory (Dict[TaskComplexity, List[Dict[str, Any]]]): Persistent
            memory store organized by complexity levels containing successful
            plan templates for future reuse and optimization.

    Returns:
        bool: True if learning occurred and a new plan template was stored in
            semantic memory, False if no learning took place due to insufficient
            success rate, missing data, or duplicate pattern detection.

    Note:
        Learning threshold requires 75% step completion rate to ensure only
        reliable execution patterns are preserved. Plan templates are cleaned
        to remove execution-specific details while preserving structural patterns,
        tool selections, and coordination strategies that enable effective reuse.

        Duplicate detection prevents memory bloat by comparing subgoal structures
        before storage, ensuring semantic memory contains only unique successful
        patterns for each complexity level.

    Examples:
        Successful learning scenario:
            - Task completes with 4/5 steps successful (80% > 75% threshold)
            - Plan structure cleaned and stored in appropriate complexity tier
            - Returns True indicating successful learning occurred

        Failed learning scenarios:
            - Task completes with 2/5 steps successful (40% < 75% threshold)
            - Missing subgoals or execution result data
            - Identical plan structure already exists in memory
            - All scenarios return False indicating no learning occurred
    """
    if not execution_result or not execution_result.get("subgoals"):
        return False

    successful_steps = [
        sg
        for sg in execution_result["subgoals"]
        if sg.get("status") in {"completed", "recovered"}
    ]

    if len(successful_steps) < len(execution_result["subgoals"]) * 0.75:
        return False

    complexity = execution_result["complexity"]

    cleaned_subgoals = []
    for sg in execution_result["subgoals"]:
        cleaned_subgoals.append(
            {
                "step_id": sg.get("step_id"),
                "tool": sg.get("tool"),
                "description": sg.get("description"),
                "status": sg.get("status"),
                "trace": {
                    "output": sg.get("trace", {}).get("output"),
                    "last_error": sg.get("trace", {}).get("last_error"),
                    "failure_reason": sg.get("trace", {}).get("failure_reason"),
                },
            }
        )

    reusable_plan = {
        "task_name": execution_result["task_name"],
        "complexity": complexity,
        "subgoals": cleaned_subgoals,
        "control": execution_result.get("control", {}),
        "status": "success",
        "plan_type": "learned_semantic",
    }

    existing_plans = semantic_memory.setdefault(complexity, [])
    if not any(p["subgoals"] == reusable_plan["subgoals"] for p in existing_plans):
        existing_plans.append(reusable_plan)
        return True

    return False
