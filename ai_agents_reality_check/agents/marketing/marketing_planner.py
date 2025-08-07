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

Module: ai_agents_reality_check/agents/marketing/marketing_planner.py

Static planning implementation simulating the shallow planning logic characteristic
of marketing-driven agent architectures. Provides basic subgoal decomposition based
on task complexity without the dynamic adaptation, tool integration, or recursive
refinement found in production-grade planning systems.

This module represents the planning limitations that enable marketing agents to
exceed wrapper performance while remaining fundamentally constrained compared to
hierarchical planning systems that enable continuous optimization and adaptation.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import uuid
from typing import Any

from ai_agents_reality_check.types import TaskComplexity


def construct_plan(task: dict[str, Any]) -> dict[str, Any]:
    """Generate static execution plan based on task complexity mapping.

    Creates a basic execution plan through simple complexity-to-step-count mapping
    without dynamic analysis, tool selection, or adaptive refinement. Represents
    the shallow planning approach typical of marketing-driven agents that provide
    enough structure to exceed wrapper performance while lacking the sophisticated
    planning capabilities required for production-grade reliability.

    The planning process uses fixed step counts per complexity level and generates
    generic subgoal descriptions, simulating the static planning horizons that
    limit marketing agents to intermediate performance tiers despite appearing
    more sophisticated than simple wrapper approaches.

    Args:
        task (Dict[str, Any]): Task specification containing:
            - complexity (TaskComplexity): Required task difficulty level used
            for step count mapping and plan structure determination
            - name (str, optional): Human-readable task identifier, defaults
            to "Unnamed Task" if not provided
            - Additional task parameters (unused by static planning logic)

    Returns:
        Dict[str, Any]: Static execution plan containing:
            - plan_id (str): Unique identifier for plan tracking and reference
            - task_name (str): Task description extracted from input or default
            - complexity (TaskComplexity): Task difficulty level for execution
            - subgoals (List[Dict[str, Any]]): Ordered list of execution steps with:
            - step_id (str): Unique step identifier for tracking
            - description (str): Generic step description
            - status (str): Initial "pending" status for execution tracking

    Note:
        Step counts are determined by static complexity mapping:
        Simple (2 steps), Moderate (4 steps), Complex (6 steps), Enterprise (10 steps).

        This approach lacks the dynamic adaptation, tool integration, and recursive
        refinement that characterize production-grade planning systems, representing
        the architectural limitations that constrain marketing agents to intermediate
        performance despite superficial planning capabilities.

    Raises:
        KeyError: If the required 'complexity' key is missing from the task
            specification, as the static mapping requires explicit complexity
            classification for plan generation.
    """
    complexity = task["complexity"]

    step_map = {
        TaskComplexity.SIMPLE: 2,
        TaskComplexity.MODERATE: 4,
        TaskComplexity.COMPLEX: 6,
        TaskComplexity.ENTERPRISE: 10,
    }

    total_steps = step_map.get(complexity, 2)
    plan_id = str(uuid.uuid4())

    subgoals = [
        {
            "step_id": f"{plan_id}_step_{i}",
            "description": f"Marketing subgoal {i + 1}",
            "status": "pending",
        }
        for i in range(total_steps)
    ]

    return {
        "plan_id": plan_id,
        "task_name": task.get("name", "Unnamed Task"),
        "complexity": complexity,
        "subgoals": subgoals,
    }
