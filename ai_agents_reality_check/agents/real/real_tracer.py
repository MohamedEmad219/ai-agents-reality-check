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

Module: ai_agents_reality_check/agents/real/real_tracer.py

Comprehensive execution tracing infrastructure for production-grade agent
architectures. Provides structured metadata annotation for subgoal-level
execution including timing, error tracking, recovery status, and extensible
tagging systems that enable detailed performance analysis and debugging.

This module delivers the observability infrastructure required for sophisticated
agent systems, enabling comprehensive execution analysis, performance optimization,
and behavioral debugging through structured trace data capture.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

from typing import Any


def annotate_trace(
    subgoal: dict[str, Any],
    status: str,
    start_time: float,
    end_time: float,
    tool: str = "builtin",
    last_error: str | None = None,
    recovered: bool = False,
    retries: int = 0,
    notes: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Enrich subgoal execution with comprehensive structured trace metadata.

    Augments subgoal dictionaries with detailed execution tracing information
    including timing data, error tracking, recovery status, and extensible
    metadata systems. Provides the observability foundation required for
    production-grade agent analysis, debugging, and performance optimization.

    The tracing system captures execution timing with millisecond precision,
    comprehensive error information, recovery attempt tracking, and flexible
    tagging systems that enable sophisticated downstream analysis and pattern
    recognition across agent execution histories.

    Args:
        subgoal (Dict[str, Any]): Target subgoal dictionary to annotate with
            trace metadata. Modified in-place with execution status and
            comprehensive trace information.
        status (str): Execution status classification such as "completed" for
            successful execution, "failed" for unrecoverable failures,
            "recovered" for successful error recovery, or custom status strings
            for specialized execution outcomes.
        start_time (float): Execution start timestamp as Unix epoch seconds,
            typically captured using time.time() at execution initiation.
        end_time (float): Execution completion timestamp as Unix epoch seconds,
            captured at execution termination for accurate duration calculation.
        tool (str, optional): Tool identifier used during subgoal execution,
            defaults to "builtin" for internal operations. Enables tool-specific
            analysis and performance tracking across different execution paths.
        last_error (str | None, optional): Error message or exception details
            if execution failure occurred. None for successful executions.
            Preserved for debugging and failure pattern analysis.
        recovered (bool, optional): Recovery mechanism invocation indicator,
            True if error recovery was attempted regardless of outcome,
            False for executions without recovery attempts. Defaults to False.
        retries (int, optional): Number of retry attempts performed during
            execution, enabling analysis of execution reliability and retry
            effectiveness patterns. Defaults to 0 for single-attempt executions.
        notes (str | None, optional): Human-readable annotations or contextual
            information for debugging and analysis purposes. Empty string if
            no notes provided, enabling flexible documentation of execution context.
        tags (List[str] | None, optional): Extensible metadata tags for
            classification, filtering, and analysis purposes. Empty list if
            no tags provided, supporting flexible categorization schemes.

    Returns:
        None: Function modifies the subgoal dictionary in-place, adding
            comprehensive trace metadata and updating execution status.

    Note:
        The function performs in-place modification of the subgoal dictionary,
        adding two key fields:

        - **status**: Updated with the provided execution status string
        - **trace**: Comprehensive metadata dictionary containing:
        - tool: Tool identifier for execution tracking
        - start_time/end_time: Precise timing boundaries
        - duration: Calculated execution duration (rounded to 3 decimal places)
        - recovered: Recovery mechanism invocation indicator
        - retries: Retry attempt count for reliability analysis
        - last_error: Error details for failure analysis (None if successful)
        - notes: Human-readable annotations for context
        - tags: Extensible metadata tags for classification

        Duration calculation automatically handles precision rounding to 3 decimal
        places (millisecond accuracy) for consistent timing analysis across
        execution traces.

    Examples:
        Successful execution annotation:
            annotate_trace(
                subgoal=step_dict,
                status="completed",
                start_time=1704067200.123,
                end_time=1704067201.456,
                tool="web_search",
                notes="Successfully retrieved search results"
            )

        Failed execution with recovery:
            annotate_trace(
                subgoal=step_dict,
                status="recovered",
                start_time=1704067200.123,
                end_time=1704067202.789,
                tool="api_caller",
                last_error="Connection timeout",
                recovered=True,
                retries=2,
                tags=["network_error", "retry_success"]
            )
    """
    trace: dict[str, Any] = {
        "tool": tool,
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(end_time - start_time, 3),
        "recovered": recovered,
        "retries": retries,
        "last_error": last_error or None,
        "notes": notes or "",
        "tags": tags or [],
    }

    subgoal["status"] = status
    subgoal["trace"] = trace
