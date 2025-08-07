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

Module: ai_agents_reality_check/types.py

Defines shared data types and enums used throughout the AI Agents Reality Check system.

This includes agent classifications, task complexity levels, and result representation
via dataclasses. These types standardize communication across benchmarks, analysis,
reporting, and agent logic with comprehensive error handling and enterprise-grade
structured error management

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


class TaskComplexity(Enum):
    """Defines a level of task complexity used to evaluate agent capabilities.

    Provides standardized complexity classification for benchmarking tasks
    that enables consistent evaluation across different agent architectures.

    Attributes:
        SIMPLE: Basic tasks requiring minimal reasoning or memory. Examples:
            - Simple calculations or data formatting
            - Single-step operations with clear inputs/outputs
            - Tasks completable with basic prompt engineering

        MODERATE: Tasks with intermediate steps and dependencies. Examples:
            - Multi-step workflows with decision points
            - Tasks requiring context retention between steps
            - Integration of multiple information sources

        COMPLEX: Multi-step tasks requiring planning or tool use. Examples:
            - Research synthesis with citation requirements
            - Debugging and problem-solving workflows
            - Tasks requiring strategic planning and execution

        ENTERPRISE: High-complexity tasks with branching logic and persistent state. Examples:
            - Multi-system integration workflows
            - Compliance audits with documentation requirements
            - Complex financial analysis with risk assessment

    Note:
        - Used throughout the system for task classification and analysis
        - Enables complexity-based performance analysis and reporting
        - Supports difficulty-based agent capability assessment
        - Essential for statistical analysis and effect size calculations
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class AgentType(Enum):
    """Categorizes the type of agent being evaluated for architectural analysis.

    Provides standardized agent classification that enables systematic comparison
    of different architectural approaches to AI agent implementation.

    Attributes:
        WRAPPER_AGENT: Simple LLM prompt-only wrappers with no internal state.
            - Basic prompt engineering with minimal logic
            - No persistent memory or context management
            - Limited error handling and recovery capabilities
            - Representative of many "AI agent" implementations in practice

        MARKETING_AGENT: More complex pipelines, often promoted as intelligent agents.
            - Enhanced prompt chains with some logic flow
            - Basic tool integration and response formatting
            - Limited memory and context retention
            - Representative of marketed "AI agent" solutions

        REAL_AGENT: Agents with planning, memory, and execution capabilities.
            - Comprehensive architecture with planning and memory systems
            - Advanced tool use with error handling and recovery
            - Persistent state management and context retention
            - Representative of production-grade agent implementations

    Note:
        - Central to the "reality check" mission of exposing architectural differences
        - Used for all performance analysis and statistical comparisons
        - Enables identification of performance gaps between approaches
        - Critical for understanding the impact of agent architecture on success rates
    """

    WRAPPER_AGENT = "wrapper_agent"
    MARKETING_AGENT = "marketing_agent"
    REAL_AGENT = "real_agent"


class AgentErrorType(Enum):
    """Categorizes different types of errors that can occur during agent execution.

    Provides standardized error classification for debugging, analysis, and
    improvement of agent implementations.

    Attributes:
        NONE: No error occurred - successful execution.
        TOOL_TIMEOUT: Tool execution exceeded allowed time limits.
        SEMANTIC_MEMORY_CORRUPTION: Memory system integrity compromised.
        INVALID_OUTPUT: Agent produced malformed or invalid output.
        DEPENDENCY_NOT_MET: Required dependencies or preconditions not satisfied.
        UNKNOWN: Unclassified error type for unexpected failures.

    Note:
        - Used for error tracking and failure mode analysis
        - Enables categorization of failure patterns across agent types
        - Supports debugging and architectural improvement recommendations
        - Essential for understanding failure modes in different agent architectures
    """

    NONE = "none"
    TOOL_TIMEOUT = "tool_timeout"
    SEMANTIC_MEMORY_CORRUPTION = "semantic_memory_corruption"
    INVALID_OUTPUT = "invalid_output"
    DEPENDENCY_NOT_MET = "dependency_not_met"
    UNKNOWN = "unknown"


@dataclass
class TaskResult:
    """Stores structured results for a single agent-task execution.

    Provides comprehensive execution results with timing, success metrics,
    cost tracking, and failure analysis for benchmark evaluation.

    Attributes:
        task_id: Unique identifier for the task instance.
        task_name: Human-readable name of the task for reporting.
        task_type: Complexity classification from TaskComplexity enum.
        agent_type: The category of agent from AgentType enum.
        success: Whether the task was successfully completed.
        execution_time: Duration of task execution in seconds.
        error_count: Number of errors encountered during execution.
        context_retention_score: Score (0.0-1.0) reflecting context maintenance quality.
        cost_estimate: Estimated token or resource cost for the execution.
        failure_reason: Optional description of failure cause if success=False.
        steps_completed: Number of task steps successfully completed.
        total_steps: Total expected steps for complete task execution.

    Methods:
        to_dict(): Converts to serializable dictionary with enum values flattened.

    Note:
        - Central data structure for all benchmark result tracking
        - Used extensively in analysis, reporting, and statistical calculations
        - Provides consistent structure across different agent implementations
        - Essential for comparative analysis and performance insights
    """

    task_id: str
    task_name: str
    task_type: TaskComplexity
    agent_type: AgentType
    success: bool
    execution_time: float
    error_count: int
    context_retention_score: float
    cost_estimate: float
    failure_reason: str | None = None
    steps_completed: int = 0
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Converts the TaskResult object into a serializable dictionary format.

        Transforms the TaskResult instance into a dictionary suitable for JSON
        serialization, CSV export, and other data interchange formats.

        Returns:
            Dict containing all TaskResult fields with enum values converted to strings:
                - All numeric and boolean fields preserved as-is
                - task_type converted to string value (e.g., "complex")
                - agent_type converted to string value (e.g., "real_agent")
                - failure_reason included if present, None otherwise

        Note:
            - Essential for data export and persistence
            - Enum values flattened to strings for compatibility
            - Maintains all original data with type information
            - Used by export functions for JSON and CSV generation
            - Safe for JSON serialization without custom encoders

        Example:
            >>> result = TaskResult(task_id="123", task_name="test", ...)
            >>> data = result.to_dict()
            >>> data['agent_type']  # Returns "real_agent" not AgentType.REAL_AGENT
        """

        result = asdict(self)
        result["task_type"] = self.task_type.value
        result["agent_type"] = self.agent_type.value
        return result


class AgentRealityCheckError(Exception):
    """Base exception for all AI Agents Reality Check errors.

    Provides structured error handling with error codes, context data,
    and enhanced debugging information for enterprise deployments.

    Attributes:
        message: Human-readable error message for user display.
        error_code: Optional machine-readable error code for programmatic handling.
        context: Additional context data dict for debugging and monitoring.

    Methods:
        to_dict(): Serialize error information for logging/monitoring systems.

    Error Code Patterns:
        - BENCHMARK_*: Benchmark execution errors
        - AGENT_*: Agent-specific execution errors
        - CONFIGURATION_*: Configuration and validation errors
        - EXPORT_*: Data export and file operation errors
        - STATISTICAL_*: Statistical analysis errors

    Example:
        >>> raise AgentRealityCheckError(
        ...     "Benchmark failed",
        ...     error_code="BENCHMARK_FAILED",
        ...     context={"iterations": 10, "timeout": 30}
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize structured error with context.

        Args:
            message: Human-readable error description for display to users.
            error_code: Optional machine-readable error identifier for programmatic handling.
            context: Optional additional debugging context dictionary.

        Note:
            - Context dictionary is initialized empty if None provided
            - Error code enables programmatic error handling and monitoring
            - Message should be user-friendly while context provides technical details
            - Base class for all application-specific exceptions
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message with context."""
        base_msg = self.message
        if self.error_code:
            base_msg += f" (Code: {self.error_code})"
        return base_msg

    def to_dict(self) -> dict[str, Any]:
        """Serialize error to dictionary for logging/monitoring.

        Converts error information into structured format suitable for
        logging systems, monitoring tools, and error analysis.

        Returns:
            Dict containing error information:
                - error_type: Exception class name for classification
                - message: Human-readable error description
                - error_code: Machine-readable error code (if present)
                - context: Additional debugging context dictionary

        Note:
            - Used by logging and monitoring systems for structured error tracking
            - Enables error analysis and pattern detection in production
            - Safe for JSON serialization and database storage
            - Maintains full error context for debugging purposes

        Example:
            >>> error = AgentRealityCheckError("Test error", "TEST_ERROR", {"key": "value"})
            >>> error_data = error.to_dict()
            >>> print(error_data['error_type'])  # "AgentRealityCheckError"
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class BenchmarkExecutionError(AgentRealityCheckError):
    """Raised when benchmark execution fails.

    Covers failures during benchmark initialization, execution,
    or result processing phases with specific error code patterns.

    Common Error Codes:
        - BENCHMARK_INIT_FAILED: Benchmark setup and initialization failed
        - BENCHMARK_TIMEOUT: Execution exceeded configured time limits
        - BENCHMARK_EXECUTION_FAILED: Runtime execution error during benchmark
        - AGENT_INITIALIZATION_FAILED: Agent setup failed during benchmark init

    Usage Scenarios:
        - Agent initialization failures during benchmark setup
        - Timeout scenarios in benchmark execution
        - Resource management failures during execution
        - Unexpected runtime errors in benchmark orchestration

    Note:
        - Inherits structured error handling from AgentRealityCheckError
        - Provides specific context for benchmark-related failures
        - Used by CLI error handling for appropriate exit codes and messages
        - Essential for debugging benchmark execution issues
    """

    pass


class AgentExecutionError(AgentRealityCheckError):
    """Raised when individual agent task execution fails.

    Covers failures within specific agent implementations during
    task processing with agent-specific error context.

    Common Error Codes:
        - AGENT_TIMEOUT: Agent execution timed out during task processing
        - TOOL_FAILURE: Agent tool execution failed or returned invalid results
        - MEMORY_ERROR: Agent memory operation failed or corrupted
        - PLANNING_FAILED: Agent planning phase failed or produced invalid plan

    Usage Scenarios:
        - Individual agent task execution timeouts
        - Tool integration failures within agents
        - Memory or state management failures
        - Planning and reasoning failures in complex agents

    Note:
        - Distinguishes agent-level failures from benchmark-level failures
        - Provides specific context for debugging agent implementations
        - Used in retry logic and failure analysis
        - Critical for understanding agent-specific failure modes
    """

    pass


class ConfigurationError(AgentRealityCheckError):
    """Raised when configuration parameters are invalid.

    Covers validation failures for CLI arguments, benchmark settings,
    and system configuration with specific guidance for resolution.

    Common Error Codes:
        - INVALID_ITERATIONS: Iteration count outside valid range [1, 1000]
        - INVALID_TIMEOUT: Timeout value below minimum threshold or invalid
        - INVALID_CONFIDENCE: Confidence level outside statistical range [0.8, 0.99]
        - INVALID_OUTPUT_PATH: Output path invalid, inaccessible, or malformed
        - OUTPUT_PERMISSION_DENIED: Cannot write to specified output location

    Usage Scenarios:
        - CLI argument validation failures
        - Configuration file parsing errors
        - Environment setup validation failures
        - Parameter range and type validation errors

    Note:
        - Provides specific guidance for fixing configuration issues
        - Used by CLI validation functions with detailed error context
        - Essential for user-friendly error messages and debugging
        - Includes suggested value ranges and valid options in context
    """

    pass


class ExportError(AgentRealityCheckError):
    """Raised when result export operations fail.

    Covers failures during JSON, CSV, or other format export operations
    with specific context for debugging file and serialization issues.

    Common Error Codes:
        - EXPORT_FAILED: General export operation failure
        - FILE_WRITE_ERROR: Cannot write to specified file or directory
        - SERIALIZATION_ERROR: Data serialization failed during export
        - INVALID_FORMAT: Unsupported or invalid export format specified

    Usage Scenarios:
        - File system permission errors during export
        - Disk space exhaustion during large result exports
        - Data serialization failures with complex result structures
        - Format specification errors and unsupported formats

    Note:
        - Provides detailed file system and serialization error context
        - Used by export functions with specific failure details
        - Essential for debugging file operations and data persistence
        - Includes file paths and format information in error context
    """

    pass


class StatisticalAnalysisError(AgentRealityCheckError):
    """Raised when statistical analysis operations fail.

    Covers failures in statistical computations, confidence interval
    calculations, or enhanced analysis features with mathematical context.

    Common Error Codes:
        - INSUFFICIENT_DATA: Not enough data points for meaningful analysis
        - STATISTICAL_COMPUTATION_ERROR: Mathematical/statistical calculation failed
        - MISSING_DEPENDENCIES: Required statistical packages (scipy/numpy) missing
        - INVALID_STATISTICAL_PARAMETERS: Analysis parameters outside valid ranges

    Usage Scenarios:
        - Insufficient sample sizes for statistical tests
        - Missing scipy/numpy dependencies for advanced statistics
        - Invalid confidence levels or statistical parameters
        - Mathematical computation failures in statistics calculations

    Note:
        - Provides specific guidance for statistical analysis requirements
        - Used by enhanced statistics features with mathematical context
        - Essential for debugging statistical computation issues
        - Includes sample size requirements and dependency information
    """

    pass


class EnhancedFeatureError(AgentRealityCheckError):
    """
    Raised when enhanced feature operations fail.

    Covers failures in optional advanced features like
    ensemble benchmarks, network simulation, etc.

    Common error codes:
        - FEATURE_NOT_AVAILABLE: Enhanced feature not installed
        - ENSEMBLE_FAILED: Ensemble benchmark failed
        - NETWORK_SIMULATION_ERROR: Network condition simulation failed
        - TOOL_FAILURE_SIMULATION_ERROR: Tool failure simulation failed
    """

    pass


@dataclass
class EnhancedTaskResult(TaskResult):
    """Enhanced version of TaskResult with additional error tracking and context.

    Extends the base TaskResult with enterprise-grade error information,
    performance metrics, and debugging context for production deployments.

    Additional Attributes:
        error_context: Detailed error context and debugging information dict.
        performance_metrics: Additional performance measurements dict.
        execution_metadata: Execution environment and configuration dict.

    Methods:
        add_error_context(key, value): Add debugging context for error analysis.
        add_performance_metric(metric_name, value): Add performance measurement.
        set_execution_metadata(metadata): Set execution environment metadata.

    Note:
        - Used in enhanced benchmarking modes for detailed analysis
        - Provides additional context for debugging and monitoring
        - Maintains compatibility with base TaskResult interface
        - Essential for enterprise deployments requiring detailed metrics
    """

    error_context: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Enhanced dictionary conversion with error context.

        Converts EnhancedTaskResult to dictionary format including all base
        TaskResult fields plus enhanced enterprise tracking fields.

        Returns:
            Dict containing complete task result data:
                - All base TaskResult fields (inherited)
                - error_context: Dict with debugging and error analysis information
                - performance_metrics: Dict with additional performance measurements
                - execution_metadata: Dict with execution environment details
                - enhanced_result: Boolean flag indicating enhanced result format

        Note:
            - Maintains full compatibility with base TaskResult.to_dict()
            - Enhanced fields provide additional context for enterprise monitoring
            - Used for advanced analysis and debugging in production environments
            - Safe for JSON serialization with all enhanced context preserved

        Example:
            >>> result = EnhancedTaskResult(task_id="123", ...)
            >>> result.add_error_context("retry_attempts", 2)
            >>> data = result.to_dict()
            >>> data['enhanced_result']  # True
            >>> data['error_context']['retry_attempts']  # 2
        """
        base_dict = super().to_dict()
        base_dict.update(
            {
                "error_context": self.error_context,
                "performance_metrics": self.performance_metrics,
                "execution_metadata": self.execution_metadata,
                "enhanced_result": True,
            }
        )
        return base_dict

    def add_error_context(self, key: str, value: Any) -> None:
        """Add debugging context for error analysis.

        Adds key-value pairs to the error context dictionary for detailed
        debugging and error analysis in enterprise deployments.

        Args:
            key: Context key name for the debugging information.
            value: Context value - can be any JSON-serializable type.

        Note:
            - Used for capturing detailed execution context during failures
            - Enables detailed post-mortem analysis of agent execution issues
            - Safe to call multiple times - accumulates context information
            - Essential for debugging complex agent execution scenarios

        Example:
            >>> result.add_error_context("tool_used", "web_search")
            >>> result.add_error_context("retry_count", 3)
        """
        self.error_context[key] = value

    def add_performance_metric(self, metric_name: str, value: float) -> None:
        """Add performance measurement for monitoring.

        Adds custom performance metrics beyond standard TaskResult measurements
        for detailed performance monitoring and analysis.

        Args:
            metric_name: Name of the performance metric being tracked.
            value: Numeric value of the performance measurement.

        Note:
            - Used for tracking custom performance indicators
            - Enables detailed performance analysis beyond standard metrics
            - Safe to call multiple times - accumulates performance data
            - Essential for monitoring agent performance in production

        Example:
            >>> result.add_performance_metric("memory_usage_mb", 156.7)
            >>> result.add_performance_metric("tool_latency_ms", 234.5)
        """
        self.performance_metrics[metric_name] = value

    def set_execution_metadata(self, metadata: dict[str, Any]) -> None:
        """Set execution environment metadata.

        Updates the execution metadata dictionary with environment and
        configuration information for production monitoring.

        Args:
            metadata: Dictionary containing execution environment details
                such as system info, configuration, versions, etc.

        Note:
            - Updates existing metadata rather than replacing it
            - Used for capturing execution environment context
            - Essential for debugging environment-specific issues
            - Supports compliance and audit requirements in enterprise deployments

        Example:
            >>> result.set_execution_metadata({
            ...     "python_version": "3.11.5",
            ...     "system": "Linux",
            ...     "benchmark_version": "0.1.0"
            ... })
        """
        self.execution_metadata.update(metadata)


def handle_agent_errors(
    error_context: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to provide consistent error handling for agent methods.

    Provides standardized error handling with context preservation for agent
    implementation methods, ensuring consistent error reporting and debugging.

    Args:
        error_context: Optional additional context to include in errors.
            Merged with runtime context during error handling.

    Returns:
        Decorator function that wraps agent methods with error handling.

    Error Handling:
        - Converts TimeoutError to AgentExecutionError with AGENT_TIMEOUT code
        - Preserves existing AgentRealityCheckError instances with enhanced context
        - Wraps unexpected exceptions as AgentExecutionError with UNEXPECTED_AGENT_ERROR code
        - Captures function name, arguments summary, and execution context

    Example:
        >>> @handle_agent_errors({"agent_type": "RealAgent"})
        ... async def execute_task(self, task):
        ...     # method implementation
        ...     return await self._execute_with_tools(task)

    Note:
        - Designed specifically for agent method error handling
        - Preserves full error context chain for debugging
        - Safe to use on any async agent method
        - Essential for consistent error reporting across agent implementations
    """

    def decorator(func: F) -> F:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except TimeoutError as timeout_err:
                context = error_context or {}
                context.update(
                    {
                        "function": func.__name__,
                        "timeout_occurred": True,
                        "args_summary": str(args)[:100] if args else None,
                    }
                )
                raise AgentExecutionError(
                    f"Agent execution timed out in {func.__name__}",
                    error_code="AGENT_TIMEOUT",
                    context=context,
                ) from timeout_err
            except Exception as e:
                if isinstance(e, AgentRealityCheckError):
                    raise

                context = error_context or {}
                context.update(
                    {
                        "function": func.__name__,
                        "original_error": str(e),
                        "original_error_type": type(e).__name__,
                    }
                )
                raise AgentExecutionError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    error_code="UNEXPECTED_AGENT_ERROR",
                    context=context,
                ) from e

        return cast(F, wrapper)

    return decorator


class BenchmarkErrorContext:
    """Context manager for tracking benchmark operation context.

    Provides structured error context collection during benchmark execution
    for enhanced debugging, monitoring, and error analysis in enterprise deployments.

    Methods:
        add_context(key, value): Add context information for error tracking.

    Context Management:
        - Automatically captures operation name and timing
        - Enhances any raised exceptions with accumulated context
        - Converts standard exceptions to structured AgentRealityCheckError types
        - Preserves original error information while adding operational context

    Example:
        >>> with BenchmarkErrorContext("benchmark_execution") as ctx:
        ...     ctx.add_context("iterations", 10)
        ...     ctx.add_context("agent_count", 3)
        ...     result = await run_benchmark()

    Note:
        - Used throughout benchmark execution for consistent error context
        - Provides detailed context for debugging benchmark failures
        - Essential for enterprise monitoring and error analysis
        - Safe to nest - contexts accumulate without conflicts
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.context: dict[str, Any] = {"operation": operation}

    def __enter__(self) -> "BenchmarkErrorContext":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        if exc_type and issubclass(exc_type, Exception):
            if isinstance(exc_val, AgentRealityCheckError):
                exc_val.context.update(self.context)
            elif exc_type == asyncio.TimeoutError:
                raise BenchmarkExecutionError(
                    f"Timeout during {self.operation}",
                    error_code="BENCHMARK_TIMEOUT",
                    context=self.context,
                ) from exc_val
            else:
                raise BenchmarkExecutionError(
                    f"Error during {self.operation}: {exc_val}",
                    error_code="BENCHMARK_EXECUTION_FAILED",
                    context=self.context,
                ) from exc_val

        return False

    def add_context(self, key: str, value: Any) -> None:
        """Add context information for error tracking.

        Adds key-value pairs to the context dictionary for enhanced error
        reporting and debugging when exceptions occur within the context.

        Args:
            key: Context key name for the operational information.
            value: Context value - any JSON-serializable data.

        Note:
            - Context is automatically included in any raised exceptions
            - Safe to call multiple times - accumulates context information
            - Used for detailed operational debugging and monitoring
            - Essential for tracking complex benchmark execution scenarios

        Example:
            >>> with BenchmarkErrorContext("agent_execution") as ctx:
            ...     ctx.add_context("agent_type", "RealAgent")
            ...     ctx.add_context("task_complexity", "COMPLEX")
            ...     # context automatically added to any exceptions
        """
        self.context[key] = value
