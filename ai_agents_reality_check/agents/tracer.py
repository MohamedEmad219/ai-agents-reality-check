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

Module: ai_agents_reality_check/agents/tracer.py

Centralized execution tracing infrastructure supporting multiple storage
backends and agent-specific trace formats. Provides comprehensive logging
capabilities including memory-based structured traces for production-grade
agents and file-based logging for analysis and debugging purposes.

This module delivers unified tracing infrastructure that enables detailed
execution analysis, performance debugging, and behavioral pattern recognition
across different agent architectures while maintaining appropriate storage
strategies for each agent type's sophistication level.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import json
import os
from typing import Any

from ai_agents_reality_check.logging_config import trace_to_file
from ai_agents_reality_check.types import TaskResult


def log_trace_to_memory(
    working_memory: dict[str, Any], task_id: str, trace_entry: dict[str, Any]
) -> None:
    """Log structured trace entry to working memory with file backup logging.

    Maintains comprehensive execution traces in agent working memory while
    simultaneously writing human-readable trace entries to persistent log
    files. Provides the dual-storage tracing infrastructure required for
    sophisticated agent analysis and debugging capabilities.

    The memory-based traces enable real-time execution analysis and recovery
    decision making, while file-based logging provides persistent records
    for post-execution analysis, debugging, and performance optimization
    across multiple execution sessions.

    Args:
        working_memory (Dict[str, Any]): Agent's active memory system containing:
            - Task-specific execution context and state information
            - Structured trace accumulation for real-time analysis
            - Runtime parameters and coordination metadata
        task_id (str): Unique task identifier for trace organization and
            correlation across execution phases and analysis systems.
        trace_entry (Dict[str, Any]): Structured trace data containing:
            - tool (str, optional): Tool identifier for execution tracking
            - status (str, optional): Execution status for pattern analysis
            - description (str, optional): Human-readable step description
            - output (str, optional): Execution result for debugging
            - Additional execution metadata and timing information

    Returns:
        None: Function modifies working memory in-place and writes to log files
            for comprehensive trace capture and persistent storage.

    Note:
        Trace storage strategy:
        - Memory traces: Structured data for real-time analysis and recovery
        - File traces: Human-readable logs with truncated output (80 chars)
        - Dual storage: Enables both automated analysis and human debugging

        Working memory enhancement creates nested structure:
        working_memory[task_id]["structured_trace"] = [trace_entries...]

        File logging includes formatted messages with key execution metadata:
        "[STATUS] Description | Tool: tool_name | Output: truncated_output"
    """
    working_memory.setdefault(task_id, {}).setdefault("structured_trace", []).append(
        trace_entry
    )

    tool = trace_entry.get("tool", "unknown_tool")
    status = trace_entry.get("status", "unknown_status")
    description = trace_entry.get("description", "No description provided")
    output = trace_entry.get("output", "")

    log_msg = (
        f"[{status.upper()}] {description} | Tool: {tool} | Output: {str(output)[:80]}"
    )
    trace_to_file(agent_name="real_agent", task_id=task_id, message=log_msg)


def trace_marketing_result(result: TaskResult) -> None:
    """Log comprehensive execution trace for MarketingAgent with dual storage.

    Captures marketing agent execution results through both human-readable
    log files and structured JSON exports that enable comprehensive analysis
    of the intermediate performance tier between wrapper and production-grade
    agents. Provides detailed execution metrics for benchmarking and optimization.

    Marketing agent traces emphasize coordination complexity, cost efficiency,
    and context retention patterns that characterize demo-quality agent
    architectures with basic orchestration capabilities but limited sophistication.

    Args:
        result (TaskResult): Comprehensive execution result containing:
            - task_name (str): Human-readable task description
            - success (bool): Overall execution outcome
            - task_type (TaskComplexity): Difficulty classification
            - error_count (int): Accumulated error count during execution
            - cost_estimate (float): Economic cost of execution
            - context_retention_score (float): Memory utilization metric
            - task_id (str): Unique identifier for trace correlation
            - Additional performance and debugging metadata

    Returns:
        None: Function writes to multiple trace destinations including log
            files and structured JSON exports for comprehensive analysis.

    Note:
        Trace format emphasizes marketing agent characteristics:
        - Call count calculation includes successful and failed attempts
        - Cost tracking reflects orchestration overhead patterns
        - Context retention highlights memory utilization limitations
        - Complexity awareness shows performance degradation patterns

        Dual storage strategy:
        - Log files: Human-readable execution summaries for debugging
        - JSON exports: Structured data for automated analysis and benchmarking
    """
    call_count = result.error_count + (1 if result.success else 0)

    message = (
        f"Task='{result.task_name}' | "
        f"Success={result.success} | "
        f"Complexity={result.task_type.value} | "
        f"Calls={call_count} | "
        f"Errors={result.error_count} | "
        f"Cost=${result.cost_estimate:.4f} | "
        f"CtxRet={result.context_retention_score:.2f}"
    )

    trace_to_file(agent_name="marketing_agent", task_id=result.task_id, message=message)
    export_structured_trace(result)


def export_trace_json(working_memory: dict[str, Any], task_id: str) -> str:
    """Export structured execution trace from RealAgent working memory as JSON.

    Serializes the complete structured trace accumulated during RealAgent
    execution into formatted JSON for analysis, debugging, and performance
    evaluation. Provides comprehensive execution history including timing,
    tool usage, error patterns, and recovery attempts.

    The JSON export enables sophisticated post-execution analysis including
    performance optimization, behavioral pattern recognition, and execution
    replay for debugging production-grade agent architectures.

    Args:
        working_memory (Dict[str, Any]): RealAgent's working memory containing:
            - Task-specific structured trace accumulation
            - Comprehensive execution metadata and timing information
            - Tool usage patterns and error recovery details
            - Additional runtime context for analysis purposes
        task_id (str): Unique task identifier for trace extraction and
            correlation with execution results and performance metrics.

    Returns:
        str: Formatted JSON string containing complete structured trace with
            indented formatting for human readability and automated parsing.
            Returns JSON representation of trace array or empty array if
            no structured trace exists for the specified task.

    Note:
        Structured traces contain comprehensive execution metadata:
        - Step-by-step execution details with timing information
        - Tool usage patterns and performance characteristics
        - Error occurrences and recovery attempt outcomes
        - Memory state changes and context evolution patterns

        JSON formatting includes 2-space indentation for readability while
        maintaining machine-parseable structure for automated analysis tools.
    """
    trace = working_memory[task_id].get("structured_trace", [])
    return json.dumps(trace, indent=2)


def trace_wrapper_result(result: TaskResult) -> None:
    """Log comprehensive execution trace for WrapperAgent with dual storage.

    Captures wrapper agent execution results through both human-readable log
    files and structured JSON exports that enable analysis of baseline agent
    performance characteristics. Provides detailed execution metrics for
    benchmarking the architectural limitations of stateless LLM wrapper
    implementations commonly misrepresented as sophisticated agents.

    Wrapper agent traces emphasize simplicity patterns, cost efficiency,
    and architectural constraints that limit performance to 35-45% success
    rates despite economic advantages over more sophisticated alternatives.

    Args:
        result (TaskResult): Execution result from wrapper agent containing:
            - task_name (str): Human-readable task description
            - success (bool): Single-shot execution outcome
            - task_type (TaskComplexity): Task difficulty classification
            - error_count (int): Basic error count (typically 0-1)
            - cost_estimate (float): Low cost reflecting simple architecture
            - context_retention_score (float): Minimal retention (0.05-0.18)
            - task_id (str): Unique identifier for trace correlation
            - Additional baseline performance metadata

    Returns:
        None: Function writes to multiple trace destinations including log
            files and structured JSON exports for comprehensive baseline
            analysis and architectural comparison purposes.

    Note:
        Trace format reflects wrapper agent characteristics:
        - Call count typically 1 (single-shot execution pattern)
        - Low cost estimates reflecting architectural simplicity
        - Minimal context retention showing stateless limitations
        - Error patterns reflecting architectural constraints

        Dual storage enables:
        - Log files: Human-readable baseline performance summaries
        - JSON exports: Structured baseline data for comparative benchmarking
    """
    call_count = result.error_count + (1 if result.success else 0)

    message = (
        f"Task='{result.task_name}' | "
        f"Success={result.success} | "
        f"Complexity={result.task_type.value} | "
        f"Calls={call_count} | "
        f"Errors={result.error_count} | "
        f"Cost=${result.cost_estimate:.4f} | "
        f"CtxRet={result.context_retention_score:.2f}"
    )

    trace_to_file(agent_name="wrapper_agent", task_id=result.task_id, message=message)
    export_structured_trace(result)


def export_structured_trace(result: TaskResult) -> None:
    """Export TaskResult as structured JSON trace file for persistent storage.

    Creates persistent JSON trace files containing complete TaskResult data
    for post-execution analysis, debugging, and performance evaluation.
    Provides detailed execution records that enable comprehensive analysis
    of agent behavior patterns and architectural performance characteristics.

    The structured trace export enables sophisticated analysis tools to
    process execution data for pattern recognition, performance optimization,
    and comparative evaluation across different agent architectures and
    execution scenarios.

    Args:
        result (TaskResult): Complete task execution result containing:
            - Comprehensive execution metadata and performance metrics
            - Error information and failure categorization details
            - Cost estimates and efficiency calculations
            - Context retention and architectural fidelity measurements
            - Additional debugging and analysis information

    Returns:
        None: Function creates persistent JSON trace file in organized
            directory structure for systematic analysis and storage.

    Note:
        File organization strategy:
        - Directory: traces/wrapper/ for systematic organization
        - Filename: wrapper_trace_{task_id}.json for unique identification
        - Format: Indented JSON (2 spaces) for human readability
        - Content: Complete TaskResult dictionary with all metadata

        Directory creation includes error-safe makedirs with exist_ok=True
        to handle concurrent execution and repeated runs without conflicts.

        JSON structure uses TaskResult.to_dict() method ensuring consistent
        serialization format across all trace exports and analysis tools.
    """
    trace_dir = os.path.join("traces", "wrapper")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"wrapper_trace_{result.task_id}.json")
    with open(trace_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
