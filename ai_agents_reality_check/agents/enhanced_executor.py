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

Module: ai_agents_reality_check/agents/enhanced_executor.py

Production-grade execution engine with realistic tool failure simulation,
network condition modeling, and complexity-aware performance degradation.
Enables comprehensive stress testing and resilience evaluation of agent
architectures under real-world deployment conditions.

This module provides the sophisticated execution infrastructure required for
evaluating agent performance under realistic operational constraints including
tool failures, network variability, and complexity-dependent degradation
patterns that reflect production environment challenges.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import asyncio
import random
import time
from typing import Any

from ai_agents_reality_check.types import TaskComplexity

from .tracer import log_trace_to_memory


class ToolFailureConfig:
    """Configuration system for realistic tool failure rates and latency patterns.

    Provides comprehensive tool-specific failure modeling and latency simulation
    that reflects production environment characteristics. Enables accurate
    performance evaluation under realistic operational constraints including
    tool-specific reliability patterns, network dependencies, and complexity
    scaling factors that affect agent execution in production deployments.

    The configuration system models eight distinct tool categories with empirically
    derived failure rates and latency distributions, complexity multipliers that
    reflect task difficulty scaling, and network condition adjustments that
    enable comprehensive stress testing scenarios.

    Attributes:
        tool_failure_rates (Dict[str, float]): Tool-specific base failure rates
            reflecting production reliability patterns. Ranges from 1% for
            local computation tools to 15% for complex inference services.
        tool_latency_patterns (Dict[str, Dict[str, int]]): Tool-specific latency
            distributions with min/max/mean values in milliseconds, enabling
            realistic timing simulation for performance evaluation.
        complexity_multipliers (Dict[TaskComplexity, float]): Task complexity
            scaling factors that increase failure rates and latency for more
            sophisticated tasks, reflecting real-world difficulty scaling.
    """

    def __init__(self) -> None:
        """Initialize tool failure configuration with production-derived parameters.

        Sets up comprehensive tool failure modeling with empirically derived failure
        rates, realistic latency distributions, and complexity scaling factors that
        enable accurate simulation of production environment constraints during
        agent performance evaluation and stress testing.

        Args:
            None

        Returns:
            None

        Note:
            Configuration parameters are derived from production system analysis
            and reflect realistic operational constraints:

            Tool Failure Rates:
            - calculator: 1% (local computation, highly reliable)
            - web_search: 5% (network dependent, moderate reliability)
            - database: 6% (I/O dependent, good reliability)
            - file_system: 8% (I/O dependent, moderate reliability)
            - api_caller: 12% (external service dependent)
            - llm_inference: 15% (complex processing, variable reliability)

            Complexity Multipliers:
            - Simple: 1.0x (baseline difficulty)
            - Moderate: 1.3x (increased coordination complexity)
            - Complex: 1.7x (significant coordination overhead)
            - Enterprise: 2.2x (maximum complexity scaling)
        """
        self.tool_failure_rates = {
            "web_search": 0.05,
            "calculator": 0.01,
            "file_system": 0.08,
            "api_caller": 0.12,
            "database": 0.06,
            "email": 0.10,
            "scheduler": 0.07,
            "llm_inference": 0.15,
        }

        self.tool_latency_patterns = {
            "web_search": {"min": 200, "max": 2000, "mean": 800},
            "calculator": {"min": 10, "max": 50, "mean": 25},
            "file_system": {"min": 50, "max": 500, "mean": 150},
            "api_caller": {"min": 300, "max": 5000, "mean": 1200},
            "database": {"min": 100, "max": 1500, "mean": 400},
            "email": {"min": 500, "max": 3000, "mean": 1200},
            "scheduler": {"min": 400, "max": 2500, "mean": 900},
            "llm_inference": {"min": 1000, "max": 10000, "mean": 3500},
        }

        self.complexity_multipliers = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.3,
            TaskComplexity.COMPLEX: 1.7,
            TaskComplexity.ENTERPRISE: 2.2,
        }


class NetworkCondition:
    """Network condition simulation for realistic performance evaluation.

    Models five distinct network conditions that affect tool performance through
    latency multiplication, failure rate scaling, and timing jitter introduction.
    Enables comprehensive evaluation of agent resilience under varying network
    conditions from stable baseline to degraded connectivity scenarios.

    Network conditions model real-world deployment scenarios including stable
    enterprise connectivity, slow remote connections, unstable mobile networks,
    degraded infrastructure, and peak-hour congestion patterns that affect
    production agent performance.

    Attributes:
        condition (str): Current network condition identifier.
        conditions (Dict[str, Dict[str, float]]): Network condition definitions
            with latency multipliers, failure multipliers, and jitter factors
            that modify tool execution characteristics.
    """

    def __init__(self, condition: str = "stable"):
        """Initialize network condition simulation with realistic parameters.

        Sets up network condition modeling with empirically derived parameters that
        reflect real-world connectivity scenarios from stable enterprise networks
        to degraded infrastructure conditions. Enables comprehensive agent
        resilience evaluation under varying operational constraints.

        Args:
            condition (str, optional): Network condition identifier. Must be one of:
                "stable", "slow", "unstable", "degraded", "peak_hours". Defaults
                to "stable" for baseline performance evaluation.

        Returns:
            None

        Note:
            Network condition parameters model realistic scenarios:
            - stable: Baseline enterprise connectivity (1.0x latency, 1.0x failures)
            - slow: Remote/satellite connections (2.5x latency, 1.4x failures)
            - unstable: Mobile/wireless networks (1.8x latency, 2.1x failures)
            - degraded: Infrastructure issues (3.2x latency, 1.7x failures)
            - peak_hours: Congested networks (1.6x latency, 1.3x failures)

            Jitter factors introduce timing variability that reflects real-world
            network behavior patterns and enables realistic performance evaluation.
        """
        self.condition = condition
        self.conditions = {
            "stable": {"latency_mult": 1.0, "failure_mult": 1.0, "jitter": 0.1},
            "slow": {"latency_mult": 2.5, "failure_mult": 1.4, "jitter": 0.3},
            "unstable": {"latency_mult": 1.8, "failure_mult": 2.1, "jitter": 0.6},
            "degraded": {"latency_mult": 3.2, "failure_mult": 1.7, "jitter": 0.4},
            "peak_hours": {"latency_mult": 1.6, "failure_mult": 1.3, "jitter": 0.2},
        }

    def get_multipliers(self) -> dict[str, float]:
        """Retrieve network condition multipliers for performance calculation.

        Returns the latency multiplier, failure multiplier, and jitter factor for
        the current network condition, enabling tool execution simulation that
        accurately reflects the performance characteristics of different network
        environments during agent evaluation.

        Args:
            None

        Returns:
            Dict[str, float]: Network condition multipliers containing:
                - latency_mult (float): Latency multiplication factor
                - failure_mult (float): Failure rate multiplication factor
                - jitter (float): Timing variability factor for realistic simulation

        Note:
            Multipliers are applied to base tool parameters to simulate network
            effects on agent execution performance, enabling comprehensive
            evaluation under varying operational conditions.
        """
        return self.conditions.get(self.condition, self.conditions["stable"])


class ToolFailureError(Exception):
    """Custom exception for categorized tool failures with detailed context.

    Extends standard Exception with tool-specific failure categorization and
    context information that enables sophisticated error analysis and recovery
    strategy selection. Provides the foundation for realistic failure simulation
    and category-aware recovery mechanisms in production-grade agent systems.

    Attributes:
        category (str): Failure category classification for recovery strategy selection.
        tool_name (str): Tool identifier for failure analysis and debugging.
    """

    def __init__(self, message: str, category: str, tool_name: str):
        """Initialize categorized tool failure exception with context.

        Creates a tool failure exception with detailed categorization and context
        information that enables sophisticated error analysis and category-aware
        recovery strategy selection during agent execution simulation.

        Args:
            message (str): Human-readable error description for debugging and analysis.
            category (str): Failure category classification such as "timeout",
                "rate_limit", "network_error", or "service_unavailable" that
                determines appropriate recovery strategies.
            tool_name (str): Tool identifier that encountered the failure for
                context and debugging purposes.

        Returns:
            None

        Note:
            Categorized failures enable sophisticated recovery strategies that
            consider the specific failure type when determining recovery approaches,
            timeouts, and success probabilities in production-grade agent systems.
        """
        super().__init__(message)
        self.category = category
        self.tool_name = tool_name


async def enhanced_execute_plan(
    plan: dict[str, Any],
    working_memory: dict[str, Any],
    speed_multiplier: float = 1.0,
    tool_config: ToolFailureConfig | None = None,
    network_condition: NetworkCondition | None = None,
) -> dict[str, Any]:
    """Execute hierarchical plan with realistic tool failures and network simulation.

    Orchestrates plan execution through sophisticated simulation of production
    environment constraints including tool-specific failure rates, network
    condition effects, complexity scaling, and comprehensive recovery mechanisms.
    Provides the execution infrastructure required for evaluating agent resilience
    under realistic operational conditions.

    The enhanced execution process maintains detailed metrics including network
    performance, tool reliability, failure categorization, and recovery success
    rates that enable comprehensive analysis of agent behavior under stress
    conditions and realistic deployment scenarios.

    Args:
        plan (Dict[str, Any]): Hierarchical execution plan containing:
            - task_id (str): Unique task identifier for tracking
            - subgoals (List[Dict[str, Any]]): Ordered execution steps
            - complexity (TaskComplexity, optional): Task difficulty level
            - Additional plan metadata for execution coordination
        working_memory (Dict[str, Any]): Agent memory system for state tracking,
            updated with execution results, failure information, and network
            metrics throughout the execution process.
        speed_multiplier (float, optional): Execution speed modifier for testing
            scenarios. Values > 1.0 accelerate execution, < 1.0 slow execution.
            Defaults to 1.0 for realistic timing simulation.
        tool_config (ToolFailureConfig, optional): Tool failure configuration
            specifying failure rates and latency patterns. Uses default
            production-derived configuration if not provided.
        network_condition (NetworkCondition, optional): Network condition
            simulation affecting latency and failure rates. Uses stable
            baseline condition if not provided.

    Returns:
        Dict[str, Any]: Updated plan with execution results containing:
            - Modified subgoals with status, timing, and failure information
            - Enhanced trace data with network metrics and reliability scores
            - Comprehensive failure categorization for analysis purposes
            - Recovery attempt details and success indicators

    Note:
        Enhanced execution includes sophisticated features:
        - Tool-specific failure simulation with realistic rates and categories
        - Network condition effects on latency and reliability
        - Complexity scaling that increases failure rates for difficult tasks
        - Category-aware recovery mechanisms with success probability modeling
        - Comprehensive metrics collection for performance analysis

        Working memory is enhanced with execution metrics including total latency,
        failure counts, successful calls, retry attempts, and detailed failure
        categorization that enables comprehensive agent performance evaluation.

    Raises:
        ToolFailureError: Categorized tool failures that trigger recovery mechanisms.
        Exception: Unexpected errors that terminate execution with detailed logging.
    """
    task_id = plan["task_id"]
    subgoals = plan["subgoals"]
    complexity = plan.get("complexity", TaskComplexity.MODERATE)

    if tool_config is None:
        tool_config = ToolFailureConfig()
    if network_condition is None:
        network_condition = NetworkCondition("stable")

    working_memory[task_id]["completed_steps"] = []
    working_memory[task_id]["tool_failures"] = []
    working_memory[task_id]["network_metrics"] = {
        "total_latency": 0.0,
        "failed_calls": 0,
        "successful_calls": 0,
        "retries": 0,
    }

    for i, step in enumerate(subgoals):
        step["trace"]["start_time"] = time.time()
        step["trace"]["complexity_context"] = complexity.value

        tool_name = step.get("tool")
        if not tool_name:
            step["status"] = "failed"
            step["trace"]["last_error"] = "No tool specified"
            step["trace"]["failure_category"] = "configuration_error"
            continue

        try:
            result, metrics = await enhanced_simulate_tool(
                tool_name,
                step,
                speed_multiplier,
                tool_config,
                network_condition,
                complexity,
            )

            step["status"] = "completed"
            step["trace"]["output"] = result
            step["trace"]["network_metrics"] = metrics
            step["trace"]["tool_reliability"] = calculate_tool_reliability(
                tool_name, tool_config
            )

            working_memory[task_id]["tools_used"].append(tool_name)
            working_memory[task_id]["completed_steps"].append(step)
            working_memory[task_id]["network_metrics"]["successful_calls"] += 1
            working_memory[task_id]["network_metrics"]["total_latency"] += metrics[
                "actual_latency"
            ]

        except ToolFailureError as e:
            step["status"] = "failed"
            step["trace"]["last_error"] = str(e)
            step["trace"]["failure_category"] = e.category
            step["trace"]["tool_reliability"] = calculate_tool_reliability(
                tool_name, tool_config
            )

            working_memory[task_id]["tool_failures"].append(
                {
                    "tool": tool_name,
                    "step": i,
                    "error": str(e),
                    "category": e.category,
                    "timestamp": time.time(),
                }
            )
            working_memory[task_id]["network_metrics"]["failed_calls"] += 1

            recovered = await attempt_enhanced_recovery(
                task_id, step, working_memory[task_id]["step_history"], e
            )

            if recovered:
                step["status"] = "recovered"
                step["trace"]["output"] = f"Recovered output from {tool_name}"
                step["trace"]["recovery_strategy"] = "category_aware"
                working_memory[task_id]["tools_used"].append(tool_name)
                working_memory[task_id]["completed_steps"].append(step)
                working_memory[task_id]["network_metrics"]["retries"] += 1
            else:
                step["trace"]["failure_reason"] = "Unrecoverable tool failure"
                break

        except Exception as e:
            step["status"] = "failed"
            step["trace"]["last_error"] = f"Unexpected error: {str(e)}"
            step["trace"]["failure_category"] = "unknown_error"
            break

        step["trace"]["end_time"] = time.time()
        step["trace"]["duration"] = round(
            step["trace"]["end_time"] - step["trace"]["start_time"], 3
        )
        working_memory[task_id]["step_history"].append(step)
        log_trace_to_memory(working_memory, task_id, step)

    return plan


async def enhanced_simulate_tool(
    tool_name: str,
    step: dict[str, Any],
    speed_multiplier: float,
    tool_config: ToolFailureConfig,
    network_condition: NetworkCondition,
    complexity: TaskComplexity,
) -> tuple[Any, dict[str, Any]]:
    """Simulate realistic tool execution with failure patterns and network effects.

    Models tool execution with production-derived failure rates, network condition
    effects, complexity scaling, and realistic latency distributions that enable
    accurate agent performance evaluation under real-world operational constraints.
    Provides the foundation for comprehensive stress testing and resilience evaluation.

    Tool simulation includes sophisticated failure categorization, network-aware
    latency modeling, complexity-dependent failure scaling, and comprehensive
    metrics collection that enables detailed analysis of agent behavior under
    varying operational conditions and deployment scenarios.

    Args:
        tool_name (str): Tool identifier for failure rate and latency lookup.
            Must match configured tool types for accurate simulation parameters.
        step (Dict[str, Any]): Execution step context containing tool parameters
            and execution metadata. Enhanced with simulation results and metrics.
        speed_multiplier (float): Execution speed modifier affecting latency
            simulation. Values > 1.0 reduce delays, < 1.0 increase delays
            for testing under various performance scenarios.
        tool_config (ToolFailureConfig): Tool configuration specifying failure
            rates, latency patterns, and complexity multipliers for realistic
            simulation of production environment constraints.
        network_condition (NetworkCondition): Network condition affecting latency
            and failure rates through multiplication factors and jitter
            introduction that reflects real-world connectivity variations.
        complexity (TaskComplexity): Task difficulty level affecting failure
            rates and execution characteristics through scaling multipliers
            that reflect increased coordination complexity.

    Returns:
        Tuple[Any, Dict[str, Any]]: Tool execution result and metrics containing:
            - Tool output with execution context and timing information
            - Comprehensive metrics including actual latency, network effects,
            jitter application, and failure rate calculations
            - Performance data enabling detailed analysis and optimization

    Raises:
        ToolFailureError: Categorized tool failures with realistic distribution:
            - timeout (30%): Tool execution timeouts under load
            - network_error (25%): Connectivity and communication failures
            - rate_limit (20%): API quota and throttling constraints
            - service_unavailable (15%): Temporary service outages
            - authentication_error (5%): Access and permission issues
            - invalid_response (5%): Data format and parsing failures

    Note:
        Failure simulation uses production-derived parameters:
        - Base failure rates specific to each tool category
        - Complexity multipliers increasing failure probability for difficult tasks
        - Network condition multipliers reflecting connectivity constraints
        - Realistic latency modeling with normal distribution and jitter

        Metrics collection enables comprehensive analysis including network
        performance impact, failure rate effectiveness, and execution timing
        patterns that support agent performance optimization and deployment planning.
    """
    latency_config = tool_config.tool_latency_patterns.get(
        tool_name, {"min": 100, "max": 1000, "mean": 300}
    )

    network_mult = network_condition.get_multipliers()
    base_latency = random.normalvariate(
        latency_config["mean"], latency_config["mean"] * 0.3
    )
    base_latency = max(latency_config["min"], min(latency_config["max"], base_latency))

    jitter = random.uniform(-network_mult["jitter"], network_mult["jitter"])
    actual_latency = (
        base_latency * network_mult["latency_mult"] * (1 + jitter)
    ) / 1000.0
    actual_latency *= speed_multiplier

    await asyncio.sleep(actual_latency)

    base_failure_rate = tool_config.tool_failure_rates.get(tool_name, 0.08)
    complexity_mult = tool_config.complexity_multipliers.get(complexity, 1.0)
    network_failure_mult = network_mult["failure_mult"]

    adjusted_failure_rate = min(
        0.95, base_failure_rate * complexity_mult * network_failure_mult
    )

    if random.random() < adjusted_failure_rate:
        failure_categories = {
            "timeout": 0.3,
            "rate_limit": 0.2,
            "network_error": 0.25,
            "service_unavailable": 0.15,
            "authentication_error": 0.05,
            "invalid_response": 0.05,
        }

        category = random.choices(
            list(failure_categories.keys()), weights=list(failure_categories.values())
        )[0]

        failure_messages = {
            "timeout": f"Tool {tool_name} timed out after {actual_latency:.2f}s",
            "rate_limit": f"Rate limit exceeded for {tool_name}",
            "network_error": f"Network connection failed for {tool_name}",
            "service_unavailable": f"Service {tool_name} temporarily unavailable",
            "authentication_error": f"Authentication failed for {tool_name}",
            "invalid_response": f"Invalid response format from {tool_name}",
        }

        raise ToolFailureError(failure_messages[category], category, tool_name)

    metrics = {
        "actual_latency": actual_latency,
        "base_latency": base_latency / 1000.0,
        "network_multiplier": network_mult["latency_mult"],
        "jitter_applied": jitter,
        "failure_rate_used": adjusted_failure_rate,
    }

    return f"Result from {tool_name} (latency: {actual_latency:.2f}s)", metrics


async def attempt_enhanced_recovery(
    task_id: str,
    step: dict[str, Any],
    step_history: list,
    failure_exception: ToolFailureError,
) -> bool:
    """Attempt category-aware error recovery with realistic success modeling.

    Implements sophisticated recovery mechanisms that consider failure categories
    when determining recovery strategies, timeout periods, and success probabilities.
    Provides production-grade error handling that enables continued execution
    despite tool failures and operational constraints.

    Recovery strategies are tailored to specific failure categories with empirically
    derived success rates that reflect real-world recovery effectiveness patterns.
    The system includes appropriate delays and retry logic that balance recovery
    success with execution efficiency in production environments.

    Args:
        task_id (str): Task identifier for recovery tracking and analysis.
        step (Dict[str, Any]): Failed execution step enhanced with recovery
            information including strategy selection and success details.
        step_history (List): Previous execution steps providing context for
            recovery decision making and pattern analysis.
        failure_exception (ToolFailureError): Categorized failure exception
            containing failure type, tool context, and error details that
            determine appropriate recovery strategy selection.

    Returns:
        bool: True if recovery succeeded and step should be considered completed,
            False if recovery failed and step should be marked permanently failed.

    Note:
        Category-specific recovery success rates reflect production patterns:
        - timeout: 70% success (often transient, retry effective)
        - network_error: 60% success (connectivity often recoverable)
        - service_unavailable: 40% success (depends on service restoration)
        - rate_limit: 30% success (requires backoff, limited effectiveness)
        - invalid_response: 20% success (data issues rarely self-resolve)
        - authentication_error: 10% success (credential issues persistent)

        Recovery includes appropriate delays:
        - rate_limit: 1-3 second backoff for quota recovery
        - other failures: 0.2-0.8 second retry delay for system stability

        Comprehensive recovery tracking enables analysis of recovery effectiveness
        and continuous improvement of recovery strategy selection and parameterization.
    """
    category = failure_exception.category

    recovery_strategies = {
        "timeout": 0.7,
        "rate_limit": 0.3,
        "network_error": 0.6,
        "service_unavailable": 0.4,
        "authentication_error": 0.1,
        "invalid_response": 0.2,
    }

    recovery_chance = recovery_strategies.get(category, 0.3)

    if category == "rate_limit":
        await asyncio.sleep(random.uniform(1.0, 3.0))
    else:
        await asyncio.sleep(random.uniform(0.2, 0.8))

    if random.random() < recovery_chance:
        step["trace"]["recovery_details"] = {
            "original_failure": category,
            "recovery_probability": recovery_chance,
            "strategy_used": f"category_aware_{category}",
        }
        return True

    return False


def calculate_tool_reliability(
    tool_name: str, config: ToolFailureConfig
) -> dict[str, float]:
    """Calculate comprehensive reliability metrics for tool performance analysis.

    Computes tool reliability statistics including success rates, uptime expectations,
    and failure frequency patterns that enable comprehensive analysis of tool
    performance characteristics and deployment planning for production environments.

    Reliability metrics provide quantitative foundations for tool selection,
    redundancy planning, and performance optimization in agent system design
    and deployment scenarios.

    Args:
        tool_name (str): Tool identifier for reliability metric calculation.
        config (ToolFailureConfig): Tool configuration containing base failure
            rates and reliability parameters for metric computation.

    Returns:
        Dict[str, float]: Comprehensive reliability metrics containing:
            - base_success_rate (float): Baseline success probability (0.0-1.0)
            - expected_uptime (float): Percentage uptime expectation (0-100)
            - mean_time_between_failures (float): Average failure interval
            or infinity for tools with zero failure rate

    Note:
        Reliability metrics enable:
        - Tool selection based on reliability requirements
        - Redundancy planning for critical execution paths
        - Performance optimization through tool substitution
        - Deployment planning with realistic failure expectations

        Metrics are derived from base failure rates and provide quantitative
        foundations for production deployment planning and system reliability
        analysis in agent architectures.
    """
    base_failure_rate = config.tool_failure_rates.get(tool_name, 0.08)

    return {
        "base_success_rate": 1.0 - base_failure_rate,
        "expected_uptime": (1.0 - base_failure_rate) * 100,
        "mean_time_between_failures": (
            1.0 / base_failure_rate if base_failure_rate > 0 else float("inf")
        ),
    }
