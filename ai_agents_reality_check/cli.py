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

Module: ai_agents_reality_check/cli.py

Enhanced CLI with proper quiet mode handling for Rich console output.

Provides comprehensive command-line interface for AI agent benchmarking with
Rich-formatted output, statistical analysis integration, ensemble benchmarking,
network simulation, and robust error handling with appropriate exit codes and
user guidance.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .agent_benchmark import AgentBenchmark
from .logging_config import suppress_statistical_warnings
from .types import AgentType, TaskComplexity, TaskResult

# Constants for magic numbers
MIN_TIMEOUT_SECONDS = 10
MAX_ITERATIONS = 1000
MIN_CONFIDENCE_LEVEL = 0.8
MAX_CONFIDENCE_LEVEL = 0.99
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SAMPLE_SIZE_FOR_STATS = 2
MIN_SAMPLE_SIZE_FOR_CI = 4
MIN_SAMPLE_SIZE_FOR_OUTLIERS = 4
LARGE_SAMPLE_THRESHOLD = 5000
MIN_SAMPLE_SIZE_FOR_T_TEST = 30
SMALL_SAMPLE_SIZE = 20

# Performance rating thresholds
EXCELLENT_THRESHOLD = 0.9
GOOD_THRESHOLD = 0.7
FAIR_THRESHOLD = 0.5
MODERATE_THRESHOLD = 0.6
POOR_THRESHOLD = 0.4

# Analysis thresholds
ENSEMBLE_ADVANTAGE_THRESHOLD = 0.1
HIGH_RESILIENCE_THRESHOLD = 0.9
MEDIUM_RESILIENCE_THRESHOLD = 0.7
HIGH_SYNERGY_THRESHOLD = 0.7
MODERATE_SYNERGY_THRESHOLD = 0.4

console = Console()
try:
    from .logging_config import get_rich_console, setup_benchmark_logger
    from .types import AgentRealityCheckError as _AgentRealityCheckError
    from .types import BenchmarkExecutionError as _BenchmarkExecutionError
    from .types import ConfigurationError as _ConfigurationError
    from .types import ExportError as _ExportError
    from .types import StatisticalAnalysisError as _StatisticalAnalysisError

    logger = setup_benchmark_logger(quiet=False)
    TYPES_AVAILABLE = True

    AgentRealityCheckError = _AgentRealityCheckError  # type: ignore[no-redef]
    BenchmarkExecutionError = _BenchmarkExecutionError  # type: ignore[no-redef]
    ConfigurationError = _ConfigurationError  # type: ignore[no-redef]
    ExportError = _ExportError  # type: ignore[no-redef]
    StatisticalAnalysisError = _StatisticalAnalysisError  # type: ignore[no-redef]

except ImportError as e:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import logging_config or types: {e}")

    def setup_benchmark_logger(quiet: bool = False) -> logging.Logger:
        return logger

    def get_rich_console() -> Console:
        return console

    class AgentRealityCheckError(Exception):  # type: ignore[no-redef]
        def __init__(
            self,
            message: str,
            error_code: str | None = None,
            context: dict[str, Any] | None = None,
        ) -> None:
            self.message = message
            self.error_code = error_code
            self.context = context or {}
            super().__init__(message)

    class ConfigurationError(AgentRealityCheckError):  # type: ignore[no-redef]
        pass

    class BenchmarkExecutionError(AgentRealityCheckError):  # type: ignore[no-redef]
        pass

    class ExportError(AgentRealityCheckError):  # type: ignore[no-redef]
        pass

    class StatisticalAnalysisError(AgentRealityCheckError):  # type: ignore[no-redef]
        pass

    TYPES_AVAILABLE = False

CustomExceptionTypes = (  # type: ignore[assignment]
    AgentRealityCheckError,
    BenchmarkExecutionError,
    ConfigurationError,
    ExportError,
    StatisticalAnalysisError,
)

try:
    from .ensemble.ensemble_benchmark import (
        EnsembleBenchmark,  # type: ignore[misc]
        EnsemblePattern,  # type: ignore[attr-defined,misc]
    )

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logger.debug("Ensemble features not available")

    class EnsembleBenchmark:  # type: ignore[misc,no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Ensemble features not available")

        async def run_ensemble_benchmark(
            self,
            tasks: list[dict[str, Any]],
            patterns: list[Any],
            agent_combinations: list[list[Any]],
            iterations: int = 5,
        ) -> dict[str, Any]:
            """Stub method that raises ImportError"""
            raise ImportError("Ensemble features not available")

        def generate_ensemble_report(self, analysis: dict[str, Any]) -> str:
            """Stub method that raises ImportError"""
            raise ImportError("Ensemble features not available")

    class EnsemblePattern(Enum):  # type: ignore[misc,no-redef]
        PIPELINE = "pipeline"
        PARALLEL = "parallel"
        HIERARCHICAL = "hierarchical"
        CONSENSUS = "consensus"
        SPECIALIZATION = "specialization"


try:
    from .utils.enhanced_statistics import (
        StatisticalAnalysis,
        generate_enhanced_statistical_report,
    )

    ENHANCED_STATS_AVAILABLE = True
except ImportError:
    ENHANCED_STATS_AVAILABLE = False
    logger.debug("Enhanced statistics not available")

try:
    from . import __author__, __version__
except ImportError:
    __author__ = "Unknown"
    __version__ = "Unknown"

with contextlib.suppress(ImportError):
    pass

app = typer.Typer(
    name="agent-benchmark",
    help="AI Agents Reality Check - Exposing the failure rate",
    add_completion=False,
    rich_markup_mode="rich",
)

ENABLE_CONSOLE_TRACES = False
TRACE_OUTPUT_FILE = "agent_traces.log"


def _get_enhanced_features_list(enhanced_config: dict[str, Any] | None) -> list[str]:
    """Extract list of enhanced features from config."""
    if not enhanced_config:
        return []

    features_list: list[str] = []
    if enhanced_config.get("enable_tool_failures"):
        features_list.append("tool_failures")
    if enhanced_config.get("network_condition", "stable") != "stable":
        features_list.append("network_simulation")
    if (
        enhanced_config.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)
        != DEFAULT_CONFIDENCE_LEVEL
    ):
        features_list.append("enhanced_statistics")

    return features_list


def _enhance_results_with_statistics(
    results: dict[str, Any], statistical_analyzer: Any
) -> dict[str, Any]:
    """Enhance results with statistical analysis if available."""
    try:
        return results
    except Exception:
        return results


class CLIErrorHandler:
    """CLI error handling with proper exit codes and user guidance."""

    EXIT_SUCCESS = 0
    EXIT_GENERAL_ERROR = 1
    EXIT_MISUSE = 2
    EXIT_PERMISSION_DENIED = 3
    EXIT_FILE_NOT_FOUND = 4
    EXIT_TIMEOUT = 124
    EXIT_INTERRUPTED = 130

    @staticmethod
    def _handle_configuration_errors(
        error: ConfigurationError, command: str
    ) -> tuple[str, str, int]:
        """Handle configuration-specific errors."""
        error_msg = f"[red]Configuration Error in {command}:[/red] {error.message}"
        error_code = CLIErrorHandler.EXIT_MISUSE
        hint_msg = ""

        if hasattr(error, "error_code"):
            if error.error_code == "INVALID_ITERATIONS":
                hint_msg = f"[yellow]Hint:[/yellow] Use --iterations with a value between 1 and {MAX_ITERATIONS}"
            elif error.error_code == "INVALID_TIMEOUT":
                hint_msg = f"[yellow]Hint:[/yellow] Use --timeout with a value of at least {MIN_TIMEOUT_SECONDS} seconds"

        return error_msg, hint_msg, error_code

    @staticmethod
    def _handle_benchmark_errors(
        error: BenchmarkExecutionError, command: str
    ) -> tuple[str, str, int]:
        """Handle benchmark execution errors."""
        error_msg = f"[red]Benchmark Failed in {command}:[/red] {error.message}"
        error_code = CLIErrorHandler.EXIT_GENERAL_ERROR
        hint_msg = ""

        if "timeout" in error.message.lower():
            error_code = CLIErrorHandler.EXIT_TIMEOUT
            hint_msg = (
                "[yellow]Hint:[/yellow] Try --quick-mode or increase --timeout value"
            )
        elif "memory" in error.message.lower():
            hint_msg = "[yellow]Hint:[/yellow] Reduce --iterations or use --quick-mode"

        return error_msg, hint_msg, error_code

    @staticmethod
    def _handle_standard_errors(error: Exception, command: str) -> tuple[str, str, int]:
        """Handle standard Python errors."""
        error_msg = ""
        hint_msg = ""
        error_code = CLIErrorHandler.EXIT_GENERAL_ERROR

        if isinstance(error, FileNotFoundError):
            error_msg = f"[red]File not found:[/red] {getattr(error, 'filename', None) or str(error)}"
            error_code = CLIErrorHandler.EXIT_FILE_NOT_FOUND
            hint_msg = "[yellow]Hint:[/yellow] Check file path and permissions"

        elif isinstance(error, PermissionError):
            error_msg = f"[red]Permission denied:[/red] {str(error)}"
            error_code = CLIErrorHandler.EXIT_PERMISSION_DENIED
            hint_msg = "[yellow]Hint:[/yellow] Check file/directory permissions or try a different location"

        elif isinstance(error, KeyboardInterrupt):
            error_msg = "[yellow]Operation cancelled by user[/yellow]"
            error_code = CLIErrorHandler.EXIT_INTERRUPTED

        elif isinstance(error, TimeoutError) or "timeout" in str(error).lower():
            error_msg = f"[red]Operation timed out:[/red] {str(error)}"
            error_code = CLIErrorHandler.EXIT_TIMEOUT
            hint_msg = "[yellow]Hint:[/yellow] Increase timeout or use --quick-mode"

        elif isinstance(error, ModuleNotFoundError):
            missing_module = (
                str(error).split("'")[1] if "'" in str(error) else "unknown"
            )
            error_msg = f"[red]Missing dependency:[/red] {missing_module}"

            if missing_module in ["scipy", "numpy"]:
                hint_msg = (
                    f"[yellow]Hint:[/yellow] Install with: pip install {missing_module}"
                )
            elif missing_module == "pandas":
                hint_msg = (
                    "[yellow]Hint:[/yellow] Install for CSV export: pip install pandas"
                )
            else:
                hint_msg = f"[yellow]Hint:[/yellow] Install missing dependency: pip install {missing_module}"

        elif isinstance(error, ImportError):
            error_msg = f"[red]Import error:[/red] {str(error)}"
            hint_msg = "[yellow]Hint:[/yellow] Try: make install-dev"

        else:
            error_msg = f"[red]Unexpected error in {command}:[/red] {str(error)}"

        return error_msg, hint_msg, error_code

    @staticmethod
    def _handle_custom_application_errors(
        error: Exception, command: str
    ) -> tuple[str, str, int]:
        """Handle custom application errors."""
        if isinstance(error, ConfigurationError):
            return CLIErrorHandler._handle_configuration_errors(error, command)
        elif isinstance(error, BenchmarkExecutionError):
            return CLIErrorHandler._handle_benchmark_errors(error, command)
        elif isinstance(error, ExportError):
            error_msg = f"[red]Export Failed in {command}:[/red] {error.message}"
            hint_msg = "[yellow]Hint:[/yellow] Check file permissions and disk space"
            return error_msg, hint_msg, CLIErrorHandler.EXIT_GENERAL_ERROR
        elif isinstance(error, StatisticalAnalysisError):
            error_msg = (
                f"[red]Statistical Analysis Failed in {command}:[/red] {error.message}"
            )
            hint_msg = "[yellow]Hint:[/yellow] Install enhanced dependencies: pip install scipy"
            return error_msg, hint_msg, CLIErrorHandler.EXIT_GENERAL_ERROR
        elif isinstance(error, AgentRealityCheckError):
            error_msg = f"[red]Error in {command}:[/red] {error.message}"
            return error_msg, "", CLIErrorHandler.EXIT_GENERAL_ERROR
        else:
            return "", "", CLIErrorHandler.EXIT_GENERAL_ERROR

    @staticmethod
    def _display_debug_information(error: Exception, debug: bool) -> None:
        """Display debug information if requested."""
        if not debug:
            return

        try:
            if isinstance(
                error,
                AgentRealityCheckError
                | BenchmarkExecutionError
                | ConfigurationError
                | ExportError
                | StatisticalAnalysisError,
            ):
                if hasattr(error, "context") and getattr(error, "context", None):
                    console.print(f"[dim]Context: {error.context}[/dim]")
                if hasattr(error, "error_code") and getattr(error, "error_code", None):
                    console.print(f"[dim]Error Code: {error.error_code}[/dim]")
        except Exception:
            pass

        console.print(f"[dim]{traceback.format_exc()}[/dim]")

    @staticmethod
    def handle_error(error: Exception, command: str, debug: bool = False) -> NoReturn:
        """Handle CLI errors with appropriate user messaging and exit codes.

        Provides error handling with user-friendly messages,
        contextual hints, appropriate exit codes, and optional debug information.

        Args:
            error: The exception that occurred during command execution.
            command: The CLI command that failed (used for help text and context).
            debug: Whether to show detailed debug information including stack traces
                and error context. Default: False.

        Note:
            This method never returns - it always calls sys.exit() with appropriate code.

        Exit Codes:
            - 0: Success (not used in error handler)
            - 1: General error
            - 2: Misuse (invalid arguments/configuration)
            - 3: Permission denied
            - 4: File not found
            - 124: Timeout
            - 130: Interrupted (Ctrl+C)

        Error Types Handled:
            - Custom application errors (ConfigurationError, BenchmarkExecutionError, etc.)
            - Standard Python errors (FileNotFoundError, PermissionError, etc.)
            - Timeout and interruption scenarios
            - Missing dependency errors with installation hints

        Debug Information (when enabled):
            - Full stack trace
            - Error context data for custom exceptions
            - Error codes for structured errors
            - Additional debugging metadata

        Example:
            >>> try:
            ...     run_benchmark()
            ... except Exception as e:
            ...     CLIErrorHandler.handle_error(e, "benchmark", debug=True)
        """
        error_msg, hint_msg, error_code = (
            CLIErrorHandler._handle_custom_application_errors(error, command)
        )

        if not error_msg:
            error_msg, hint_msg, error_code = CLIErrorHandler._handle_standard_errors(
                error, command
            )

        console.print(error_msg)

        CLIErrorHandler._display_debug_information(error, debug)

        if hint_msg:
            console.print(hint_msg)

        if error_code in [
            CLIErrorHandler.EXIT_MISUSE,
            CLIErrorHandler.EXIT_GENERAL_ERROR,
        ]:
            console.print(
                f"[dim]For help, run:[/dim] [bold]agent-benchmark {command} --help[/bold]"
            )

        sys.exit(error_code)

    @staticmethod
    def validate_benchmark_args(
        iterations: int,
        timeout: int,
        confidence_level: float | None = None,
        output: Path | None = None,
    ) -> None:
        """Validate benchmark arguments with validation.

        Performs comprehensive validation of CLI arguments with detailed error
        messages and context for debugging and user guidance.

        Args:
            iterations: Number of benchmark iterations. Must be 1-1000.
            timeout: Timeout in seconds. Must be ≥ 10 seconds.
            confidence_level: Statistical confidence level. Must be 0.8-0.99 if provided.
            output: Output file path. Validated for writeability if provided.

        Raises:
            ConfigurationError: When arguments are invalid with specific error codes:
                - INVALID_ITERATIONS: iterations outside valid range
                - EXCESSIVE_ITERATIONS: iterations above performance limit
                - INVALID_TIMEOUT: timeout below minimum threshold
                - INVALID_CONFIDENCE: confidence_level outside valid range
                - INVALID_OUTPUT_PATH: output path issues
                - OUTPUT_PERMISSION_DENIED: cannot write to output directory

        Validation Checks:
            - Iterations: 1 ≤ iterations ≤ 1000 (performance safety)
            - Timeout: ≥ 10 seconds (minimum for meaningful execution)
            - Confidence: 0.8 ≤ confidence ≤ 0.99 (statistical validity)
            - Output: Path writeability and directory creation

        Note:
            - Creates parent directories for output path if needed
            - Tests write permission with temporary file
            - Provides specific error codes for programmatic handling
            - Includes suggested value ranges in error messages

        Example:
            >>> try:
            ...     CLIErrorHandler.validate_benchmark_args(50, 300, 0.95, Path("results.json"))
            ... except ConfigurationError as e:
            ...     print(f"Validation failed: {e.message}")
        """

        if iterations < 1:
            raise ConfigurationError(
                "Iterations must be at least 1",
                error_code="INVALID_ITERATIONS",
                context={"provided_value": iterations},
            )

        if iterations > MAX_ITERATIONS:
            raise ConfigurationError(
                f"Iterations cannot exceed {MAX_ITERATIONS} for performance reasons",
                error_code="EXCESSIVE_ITERATIONS",
                context={"provided_value": iterations, "max_allowed": MAX_ITERATIONS},
            )

        if timeout < MIN_TIMEOUT_SECONDS:
            raise ConfigurationError(
                f"Timeout must be at least {MIN_TIMEOUT_SECONDS} seconds",
                error_code="INVALID_TIMEOUT",
                context={
                    "provided_value": timeout,
                    "min_required": MIN_TIMEOUT_SECONDS,
                },
            )

        if confidence_level is not None and not (
            MIN_CONFIDENCE_LEVEL <= confidence_level <= MAX_CONFIDENCE_LEVEL
        ):
            raise ConfigurationError(
                f"Confidence level must be between {MIN_CONFIDENCE_LEVEL} and {MAX_CONFIDENCE_LEVEL}",
                error_code="INVALID_CONFIDENCE",
                context={
                    "provided_value": confidence_level,
                    "valid_range": f"{MIN_CONFIDENCE_LEVEL}-{MAX_CONFIDENCE_LEVEL}",
                },
            )

        if output is not None:
            if output.exists() and not output.is_file():
                raise ConfigurationError(
                    f"Output path exists but is not a file: {output}",
                    error_code="INVALID_OUTPUT_PATH",
                    context={"path": str(output), "path_type": "directory"},
                )

            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                test_file = output.parent / ".write_test"
                test_file.touch()
                test_file.unlink()
            except PermissionError as e:
                raise ConfigurationError(
                    f"Cannot write to output directory: {output.parent}",
                    error_code="OUTPUT_PERMISSION_DENIED",
                    context={"directory": str(output.parent)},
                ) from e
            except Exception as e:
                raise ConfigurationError(
                    f"Output directory validation failed: {e}",
                    error_code="OUTPUT_DIRECTORY_ERROR",
                    context={"directory": str(output.parent), "error": str(e)},
                ) from e


def set_trace_mode(
    console_traces: bool = False, file_output: str = "agent_traces.log"
) -> None:
    """Configure trace output mode."""
    global ENABLE_CONSOLE_TRACES, TRACE_OUTPUT_FILE  # noqa: PLW0603
    ENABLE_CONSOLE_TRACES = console_traces
    TRACE_OUTPUT_FILE = file_output


def _create_main_title_panel(
    enhanced: bool = False, mode: str = "Comprehensive"
) -> Panel:
    """Create the main title panel with proper width and styling.

    Generates Rich Panel for CLI header with mode indication and enhancement status.

    Args:
        enhanced: Whether enhanced features are active. Changes subtitle messaging.
        mode: Execution mode description (e.g., "Comprehensive", "Quick Mode").

    Returns:
        Rich Panel object ready for console display with appropriate styling and content.

    Note:
        - Enhanced mode shows "Enhanced with Complexity Dimensions" subtitle
        - Standard mode shows "Exposing Failure That Nobody Talks About" subtitle
        - Uses bright red styling to emphasize reality check theme
        - Mode displayed in panel subtitle for context
    """
    if enhanced:
        sub_title = "Enhanced with Complexity Dimensions"
    else:
        sub_title = "Exposing Failure That Nobody Talks About"

    title_panel = Panel.fit(
        f"\n[bold red]AI Agent Reality Check[/bold red]\n\n{sub_title}\n",
        subtitle=f"Mode: {mode}",
        border_style="bright_red",
    )

    return title_panel


def display_project_banner() -> None:
    """Display project banner with version and branding."""
    console.print()
    try:
        terminal_width = console.size.width
    except Exception:
        terminal_width = 80

    banner = Text(justify="center")
    banner.append("AI Agents Reality Check", style="bold bright_white")
    banner.append(f"  v{__version__}\n", style="bright_green")
    banner.append("Exposing Architectural Theater\n", style="bold bright_cyan")

    panel = Panel(
        banner,
        border_style="bright_cyan",
        title="[bold bright_white]Welcome to AI Agents Reality Check",
        subtitle="[dim bright_black]by ByteStack Labs",
        padding=(1, 4),
        expand=True,
        width=terminal_width,
    )

    console.print(panel)
    console.print()


def display_project_footer() -> None:
    """Display clean project footer for all screenshots."""
    console.print(
        "\nAI Agents Reality Check | github.com/Cre4T3Tiv3/ai-agents-reality-check",
        style="dim",
        justify="center",
    )


def _create_progress_context(description: str, quiet: bool = False) -> Progress:
    """Create consistent progress context for all commands.

    Sets up Rich Progress bar with consistent styling and information display
    across all CLI commands.

    Args:
        description: Initial description text for the progress bar.
        quiet: Quiet mode flag. Note: quiet=True suppresses logging conflicts,
            NOT progress bars themselves.

    Returns:
        Rich Progress context manager with spinner, bar, percentage, and elapsed time.

    Note:
        - Progress bars always show unless explicitly disabled
        - Quiet mode only affects logging conflicts with Rich console
        - Consistent styling across all benchmark operations
        - Includes spinner, progress bar, percentage, and elapsed time
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        disable=False,
    )


def _should_suppress_console_output(quiet: bool) -> bool:
    """
    Determine if console output should be suppressed.

    quiet=True means suppress logging conflicts with Rich progress bars,
    but NOT suppress the Rich console output itself.
    """
    return False


def _get_status_icon(success_rate: float) -> str:
    """Return a colored status icon based on success rate."""
    if success_rate >= EXCELLENT_THRESHOLD:
        return "🟢"
    elif success_rate >= GOOD_THRESHOLD:
        return "🟡"
    elif success_rate >= FAIR_THRESHOLD:
        return "🟠"
    else:
        return "🔴"


def _get_performance_rating(success_rate: float) -> str:
    """Return a performance label based on success rate."""
    if success_rate >= EXCELLENT_THRESHOLD:
        return "Excellent"
    elif success_rate >= GOOD_THRESHOLD:
        return "Good"
    elif success_rate >= FAIR_THRESHOLD:
        return "Fair"
    else:
        return "Poor"


def _create_performance_cards(agents_data: dict[str, Any]) -> Columns:
    """Create styled Rich panels for agent performance."""
    cards = []

    for agent_type, data in agents_data.items():
        status = _get_status_icon(data["success_rate"])
        rating = _get_performance_rating(data["success_rate"])
        agent_name = agent_type.replace("_", " ").title()

        card_content = f"""
{status} [bold]{data["success_rate"]:.1%}[/bold] Success Rate
{data["avg_execution_time"]:.2f}s Average Time  
{data["avg_context_retention"]:.1%} Context Retention
${data["cost_per_success"]:.4f} Cost per Success
{data["step_completion_rate"]:.1%} Step Completion
{rating}
        """

        if data["success_rate"] >= EXCELLENT_THRESHOLD:
            border_style = "green"
        elif data["success_rate"] >= GOOD_THRESHOLD:
            border_style = "yellow"
        elif data["success_rate"] >= FAIR_THRESHOLD:
            border_style = "bright_blue"
        else:
            border_style = "red"

        card = Panel(
            card_content.strip(),
            title=f"[cyan]{agent_name}[/cyan]",
            border_style=border_style,
            width=32,
        )
        cards.append(card)

    return Columns(cards, equal=True, expand=True)


def _create_insights_panel(agents_data: dict[str, Any]) -> Panel:
    """Create a summary insights panel showing top agent performance stats."""
    if not agents_data:
        return Panel(
            "No data available for insights", title="Insights", border_style="red"
        )

    best_agent = max(agents_data.items(), key=lambda x: x[1]["success_rate"])
    worst_agent = min(agents_data.items(), key=lambda x: x[1]["success_rate"])

    valid_costs = {
        k: v for k, v in agents_data.items() if v["cost_per_success"] != float("inf")
    }
    if valid_costs:
        cheapest_agent = min(
            valid_costs.items(), key=lambda x: x[1]["cost_per_success"]
        )
        cost_text = f"• Cost Leader: [yellow]{cheapest_agent[0].replace('_', ' ').title()}[/yellow] (${cheapest_agent[1]['cost_per_success']:.4f}/success)"
    else:
        cost_text = "• Cost analysis: No successful completions for cost comparison"

    fastest_agent = min(agents_data.items(), key=lambda x: x[1]["avg_execution_time"])
    best_rate = round(best_agent[1]["success_rate"] * 100, 1)
    worst_rate = round(worst_agent[1]["success_rate"] * 100, 1)
    gap = best_rate - worst_rate

    insights_content = f"""
[bold]Key Insights:[/bold]
• Best Performer: [green]{best_agent[0].replace("_", " ").title()}[/green] ({best_rate:.1f}% success)
• Biggest Gap: [red]{gap:.1f}%[/red] between {best_agent[0].replace("_", " ").title()} and {worst_agent[0].replace("_", " ").title()}
{cost_text}
• Speed Champion: [blue]{fastest_agent[0].replace("_", " ").title()}[/blue] ({fastest_agent[1]["avg_execution_time"]:.2f}s avg)
    """

    return Panel(
        insights_content.strip(), title="Analysis", border_style="bright_green"
    )


def _create_performance_chart(agents_data: dict[str, Any]) -> Panel:
    """Generate an ASCII bar chart comparing success rates."""
    if not agents_data:
        return Panel(
            "No data available for chart",
            title="Performance Chart",
            border_style="red",
        )

    chart_lines = ["Success Rate Comparison:", ""]
    max_rate = max(data["success_rate"] for data in agents_data.values())

    for agent_type, data in agents_data.items():
        rate = data["success_rate"]
        bar_length = int((rate / max_rate) * 30) if max_rate > 0 else 0
        bar = "█" * bar_length + "▒" * (30 - bar_length)
        agent_name = agent_type.replace("_", " ").title()
        chart_lines.append(f"{agent_name:15} │{bar}│ {rate:.1%}")

    return Panel(
        "\n".join(chart_lines),
        title="Performance Visualization",
        border_style="cyan",
    )


def _display_rich_benchmark_output(
    results: dict[str, Any],
    enhanced_config: dict[str, Any] | None = None,
    quiet: bool = False,
) -> None:
    """Display benchmark results with consistent Rich formatting.

    Renders comprehensive benchmark results using Rich components with
    performance cards, insights, and statistical analysis.

    Args:
        results: Benchmark results dictionary with agent performance metrics.
        enhanced_config: Optional enhanced configuration used in benchmark.
        quiet: Whether to suppress console output (respects quiet mode).

    Display Components:
        - Main title panel with completion status
        - Enhanced feature summary (if applicable)
        - Agent performance summary table
        - Statistical insights and shocking performance gaps
        - Recommendations and warnings

    Note:
        - Gracefully handles missing or incomplete results
        - Enhanced features get special highlighting when active
        - Performance gaps prominently displayed to expose architectural differences
        - Safe to call with any results format
    """
    if _should_suppress_console_output(quiet):
        return

    console.print()
    console.print(
        Panel.fit(
            "AI Agent Reality Check - Benchmark Report",
            subtitle="Performance Analysis Complete",
            border_style="bright_blue",
        )
    )

    if enhanced_config:
        _display_enhanced_summary(results, enhanced_config, quiet)

    _display_summary_table(results, quiet)
    _display_shocking_stats(results, quiet)


def _initialize_benchmark_with_validation(
    quick_mode: bool, quiet: bool, random_seed: int | None, randomize: bool
) -> AgentBenchmark:
    """Initialize benchmark with proper error handling."""
    try:
        return AgentBenchmark(
            quick_mode=quick_mode,
            quiet=quiet,
            random_seed=random_seed,
            randomize=randomize,
        )
    except Exception as e:
        raise BenchmarkExecutionError(
            f"Failed to initialize benchmark: {e}",
            error_code="BENCHMARK_INIT_FAILED",
            context={
                "quick_mode": quick_mode,
                "random_seed": random_seed,
                "error": str(e),
            },
        ) from e


def _setup_enhanced_config(
    enable_tool_failures: bool, network_condition: str, confidence_level: float
) -> dict[str, Any] | None:
    """Setup enhanced configuration if any enhanced features are enabled."""
    using_enhanced = enable_tool_failures or network_condition != "stable"

    if using_enhanced:
        return {
            "enable_tool_failures": enable_tool_failures,
            "network_condition": network_condition,
            "confidence_level": confidence_level,
        }
    return None


async def _run_benchmark_with_progress(
    benchmark_runner: AgentBenchmark,
    iterations: int,
    timeout: int,
    enhanced_config: dict[str, Any] | None,
    enhanced_stats: bool,
    confidence_level: float,
    update_progress_bar: Any,
) -> dict[str, Any]:
    """Run benchmark with proper error handling and progress updates."""
    try:
        logger.debug("Starting benchmark execution...")

        if enhanced_config:
            result = await benchmark_runner.run_comprehensive_benchmark(
                iterations=iterations,
                timeout=timeout,
                enhanced_config=enhanced_config,
                progress_callback=update_progress_bar,
            )
        else:
            result = await benchmark_runner.run_comprehensive_benchmark(
                iterations=iterations,
                timeout=timeout,
                progress_callback=update_progress_bar,
            )

        if result.get("early_exit"):
            logger.warning(
                "Benchmark exited early due to timeout. Results may be partial."
            )

        if enhanced_stats and ENHANCED_STATS_AVAILABLE:
            statistical_analyzer = StatisticalAnalysis(
                confidence_level=confidence_level
            )
            result = _enhance_results_with_statistics(result, statistical_analyzer)
        elif enhanced_stats and not ENHANCED_STATS_AVAILABLE:
            console.print(
                "[yellow]Warning: Enhanced statistics not available. Install with: pip install scipy[/yellow]"
            )

        return result

    except TimeoutError as e:
        raise BenchmarkExecutionError(
            f"Benchmark timed out after {timeout} seconds",
            error_code="BENCHMARK_TIMEOUT",
            context={"timeout": timeout, "iterations": iterations},
        ) from e

    except Exception as e:
        raise BenchmarkExecutionError(
            f"Benchmark execution failed: {e}",
            error_code="BENCHMARK_EXECUTION_FAILED",
            context={
                "iterations": iterations,
                "enhanced_config": enhanced_config,
            },
        ) from e


def _display_benchmark_results(
    results: dict[str, Any],
    benchmark_runner: AgentBenchmark,
    enhanced_stats: bool,
    enhanced_config: dict[str, Any] | None,
    confidence_level: float,
    show_console: bool,
    quiet: bool,
) -> None:
    """Display benchmark results with proper error handling."""
    try:
        if show_console:
            if enhanced_stats and ENHANCED_STATS_AVAILABLE and enhanced_config:
                enhanced_report = generate_enhanced_statistical_report(
                    results, confidence_level
                )
                console.print(enhanced_report)
            else:
                report = benchmark_runner.generate_reality_report(results)
                console.print(report)

            _display_rich_benchmark_output(results, enhanced_config, quiet)
    except Exception as e:
        raise BenchmarkExecutionError(
            f"Failed to display results: {e}",
            error_code="DISPLAY_FAILED",
            context={"enhanced_stats": enhanced_stats, "error": str(e)},
        ) from e


def _save_benchmark_results(
    results: dict[str, Any],
    benchmark_runner: AgentBenchmark,
    output: Path | None,
    format: str,
    enhanced_config: dict[str, Any] | None,
    show_console: bool,
) -> None:
    """Save benchmark results with proper error handling."""
    try:
        if output:
            _save_results(
                benchmark_runner, results, str(output), format, enhanced_config
            )
            if show_console:
                console.print(f"\nResults saved to: [bold blue]{output}[/bold blue]")
        else:
            logs_dir = "logs"
            os.makedirs(logs_dir, exist_ok=True)

            auto_save_path = os.path.join(logs_dir, "results.json")

            try:
                benchmark_runner.export_results(
                    auto_save_path,
                    format="json",
                    enhanced_data=(
                        None
                        if not enhanced_config
                        else {
                            **results,
                            "enhanced_config": enhanced_config,
                            "version": "2.0",
                            "features": _get_enhanced_features_list(enhanced_config),
                        }
                    ),
                )

                if show_console:
                    console.print(
                        f"\nAuto-saved results to: [bold blue]{auto_save_path}[/bold blue]"
                    )

            except Exception as e:
                if show_console:
                    console.print(
                        f"[yellow]Warning: Could not save results to {auto_save_path}: {e}[/yellow]"
                    )

                fallback_path = os.path.join(logs_dir, "results_backup.json")
                try:
                    benchmark_runner.export_results(fallback_path, format="json")
                    if show_console:
                        console.print(
                            f"[yellow]Results saved to fallback location: {fallback_path}[/yellow]"
                        )
                except Exception as fallback_error:
                    if show_console:
                        console.print(
                            f"[red]Failed to save results: {fallback_error}[/red]"
                        )

    except Exception as e:
        raise ExportError(
            f"Failed to save results: {e}",
            error_code="EXPORT_FAILED",
            context={
                "output_path": (str(output) if output else "results_backup.json"),
                "format": format,
            },
        ) from e


def _setup_benchmark_environment(
    suppress_warnings: bool, quiet: bool, console_traces: bool
) -> None:
    """Setup the benchmark environment with proper error handling."""
    if suppress_warnings:
        suppress_statistical_warnings()

    try:
        _ = setup_benchmark_logger(quiet=quiet)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to setup logging system: {e}",
            error_code="LOGGING_SETUP_FAILED",
            context={"quiet_mode": quiet, "error": str(e)},
        ) from e

    set_trace_mode(console_traces, "benchmark_traces.log")


@app.command()
def benchmark(
    quiet: bool = typer.Option(False, "--quiet", help="Suppress intermediate logs"),  # noqa: B008
    iterations: int = typer.Option(10, "--iterations", "-i", min=1, max=100),  # noqa: B008
    output: Path | None = None,
    format: str = typer.Option("json", "--format", "-f"),  # noqa: B008
    quick_mode: bool = typer.Option(False, "--quick-mode"),  # noqa: B008
    timeout: int = typer.Option(400, "--timeout"),
    random_seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducible results"
    ),
    randomize: bool = typer.Option(
        False, "--randomize", help="Shuffle task order using the seed"
    ),
    enable_tool_failures: bool = typer.Option(
        False,
        "--tool-failures/--no-tool-failures",
        help="Enable realistic tool failure simulation",
    ),
    network_condition: str = typer.Option(
        "stable",
        "--network",
        help="Network condition: stable, slow, unstable, degraded, peak_hours",
    ),
    confidence_level: float = typer.Option(
        DEFAULT_CONFIDENCE_LEVEL,
        "--confidence",
        help="Confidence level for statistical intervals",
    ),
    enhanced_stats: bool = typer.Option(
        False, "--enhanced-stats", help="Enable enhanced statistical analysis with CI"
    ),
    console_traces: bool = typer.Option(
        False,
        "--console-traces",
        help="Enable console trace output (default: file only)",
    ),
    suppress_warnings: bool = typer.Option(
        True,
        "--suppress-warnings/--show-warnings",
        help="Suppress expected statistical warnings (default: True)",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed error information"
    ),
) -> None:
    """Run the full benchmark suite with error handling.

    Enhanced with tool failures, network simulation, and statistical analysis.
    All enhanced features are opt-in and backward compatible with standard benchmarking.

    Args:
        quiet: Suppress intermediate logging to reduce console noise. Does not affect
            progress bars or final results display.
        iterations: Number of times to execute each agent-task combination. Higher values
            provide better statistical power but increase execution time.
        output: Output file path for results. Auto-saves to logs/ if not specified.
        format: Export format - "json" (default) or "csv". CSV requires pandas.
        quick_mode: Use reduced task set and shorter timeouts for faster execution.
        timeout: Maximum benchmark execution time in seconds. Individual tasks have
            separate timeouts.
        random_seed: Seed for random number generation to ensure reproducible results
            across benchmark runs.
        randomize: Randomize task execution order using the specified seed.
        enable_tool_failures: Enable realistic tool failure simulation for enhanced
            analysis of agent resilience.
        network_condition: Simulate network conditions - stable, slow, unstable,
            degraded, or peak_hours.
        confidence_level: Statistical confidence level for interval calculations (0.8-0.99).
        enhanced_stats: Enable advanced statistical analysis with confidence intervals
            and effect size calculations.
        console_traces: Output agent execution traces to console in addition to files.
        suppress_warnings: Hide expected statistical warnings to reduce noise.
        debug: Show detailed error information including stack traces and context.

    Enhanced Features:
        - Tool failure simulation with configurable failure rates and recovery
        - Network condition simulation affecting agent performance
        - Enhanced statistical analysis with confidence intervals and effect sizes
        - Advanced error handling with structured context and debugging information

    Output:
        - Comprehensive performance analysis with agent comparisons
        - Statistical insights and recommendations when enhanced features enabled
        - Detailed performance cards and visualizations
        - Auto-saved results in JSON format to logs/ directory

    Example:
        >>> # Standard benchmark
        >>> agent-benchmark benchmark --iterations 20

        >>> # Enhanced benchmark with tool failures
        >>> agent-benchmark benchmark --iterations 15 --tool-failures --network degraded --enhanced-stats
    """
    try:
        CLIErrorHandler.validate_benchmark_args(
            iterations, timeout, confidence_level, output
        )

        _setup_benchmark_environment(suppress_warnings, quiet, console_traces)

        enhanced_config = _setup_enhanced_config(
            enable_tool_failures, network_condition, confidence_level
        )
        using_enhanced = enhanced_config is not None
        show_console = not _should_suppress_console_output(quiet)

        if show_console:
            display_project_banner()
            mode_text = "Quick Mode" if quick_mode else "Comprehensive Mode"
            console.print()
            title_panel = _create_main_title_panel(using_enhanced, mode_text)
            console.print(title_panel)
            console.print()

            console.print(
                f"Running benchmark with [bold]{iterations}[/bold] iterations per task..."
            )
            if quick_mode:
                console.print("[yellow]⚡ Quick mode: Using reduced task set[/yellow]")
            console.print()

        benchmark_runner = _initialize_benchmark_with_validation(
            quick_mode, quiet, random_seed, randomize
        )

        with _create_progress_context("Running benchmark...", quiet) as progress:
            task = progress.add_task(
                (
                    "Running enhanced benchmark..."
                    if using_enhanced
                    else "Running comprehensive benchmark..."
                ),
                total=100,
            )

            def update_progress_bar(
                completed: int, total: int, description: str
            ) -> None:
                try:
                    percentage = (completed / total * 100) if total > 0 else 0
                    progress.update(
                        task, completed=percentage, total=100, description=description
                    )
                except Exception:
                    pass

            try:
                results = asyncio.run(
                    _run_benchmark_with_progress(
                        benchmark_runner,
                        iterations,
                        timeout,
                        enhanced_config,
                        enhanced_stats,
                        confidence_level,
                        update_progress_bar,
                    )
                )
                progress.update(task, completed=100, description="Benchmark completed!")
            except Exception as e:
                progress.update(task, completed=100, description="Benchmark failed!")
                raise e

        _display_benchmark_results(
            results,
            benchmark_runner,
            enhanced_stats,
            enhanced_config,
            confidence_level,
            show_console,
            quiet,
        )

        if not output:
            output = Path("logs/enhanced_results.json")

        _save_benchmark_results(
            results, benchmark_runner, output, format, enhanced_config, show_console
        )

        if show_console:
            display_project_footer()

    except Exception as e:
        CLIErrorHandler.handle_error(e, "benchmark", debug=debug)


@app.command()
def ensemble(
    iterations: int = typer.Option(5, "--iterations", "-i", min=1, max=50),
    quiet: bool = typer.Option(False, "--quiet"),
    patterns: str = typer.Option(
        "all",
        "--patterns",
        help="Ensemble patterns: all, pipeline, parallel, hierarchical, consensus, specialization",
    ),
    combinations: str = typer.Option(
        "all", "--combinations", help="Agent combinations: all, mixed, homogeneous"
    ),
    output: Path | None = None,
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed error information"
    ),
) -> None:
    """Run agent ensemble benchmarks with error handling.

    Evaluates collaborative agent performance across different ensemble patterns
    and agent combinations to assess multi-agent coordination capabilities.

    Args:
        iterations: Number of iterations per ensemble pattern-combination pair.
            Lower than individual benchmarks due to increased complexity.
        quiet: Suppress intermediate logging output.
        patterns: Ensemble patterns to evaluate:
            - all: Test all available patterns
            - pipeline: Sequential agent coordination
            - parallel: Concurrent agent execution
            - hierarchical: Multi-level agent organization
            - consensus: Agreement-based decision making
            - specialization: Role-based agent coordination
        combinations: Agent combination strategies:
            - all: Test all meaningful combinations
            - mixed: Heterogeneous agent teams
            - homogeneous: Same-type agent teams
        output: Output file for ensemble results. Auto-saves to logs/ if not specified.
        debug: Show detailed error information and execution context.

    Requires:
        Ensemble feature dependencies must be installed. The command will fail
        with helpful error message if ensemble components are not available.

    Ensemble Patterns:
        - Pipeline: Agents work in sequence, each building on previous output
        - Parallel: Agents work independently on same task, results combined
        - Hierarchical: Multi-level coordination with supervisor-worker relationships
        - Consensus: Agents collaborate to reach agreement on solutions
        - Specialization: Agents take specialized roles based on expertise

    Output Analysis:
        - Performance comparison across ensemble patterns
        - Coordination overhead analysis
        - Synergy detection (ensemble vs individual performance)
        - Collaboration effectiveness metrics

    Example:
        >>> # Test all patterns with mixed agent teams
        >>> agent-benchmark ensemble --patterns all --combinations mixed --iterations 10

        >>> # Focus on pipeline and consensus patterns
        >>> agent-benchmark ensemble --patterns "pipeline,consensus" --iterations 15
    """
    try:
        CLIErrorHandler.validate_benchmark_args(iterations, 300)

        _ = setup_benchmark_logger(quiet=quiet)
        show_console = not _should_suppress_console_output(quiet)

        if not ENSEMBLE_AVAILABLE:
            raise ConfigurationError(
                "Ensemble features not available - missing dependencies",
                error_code="FEATURE_NOT_AVAILABLE",
                context={
                    "feature": "ensemble",
                    "available_features": ["core_benchmarking"],
                },
            )

        if show_console:
            display_project_banner()
            console.print()
            title_panel = Panel.fit(
                "\n[bold blue]Agent Ensemble Benchmark[/bold blue]\n"
                "\nEvaluating collaborative agent performance across patterns\n",
                border_style="bright_blue",
            )
            console.print(title_panel)
            console.print()

        selected_patterns = _parse_ensemble_patterns(patterns)
        agent_combinations = _parse_agent_combinations(combinations)

        ensemble_tasks = [
            {
                "name": "Collaborative data analysis",
                "complexity": TaskComplexity.MODERATE,
            },
            {
                "name": "Multi-step problem solving",
                "complexity": TaskComplexity.COMPLEX,
            },
            {
                "name": "Distributed system coordination",
                "complexity": TaskComplexity.ENTERPRISE,
            },
        ]

        try:
            ensemble_benchmark = EnsembleBenchmark(quiet=quiet)
        except Exception as e:
            raise BenchmarkExecutionError(
                f"Failed to initialize ensemble benchmark: {e}",
                error_code="ENSEMBLE_INIT_FAILED",
                context={"error": str(e)},
            ) from e

        with _create_progress_context(
            "Running ensemble benchmark...", quiet
        ) as progress:
            task = progress.add_task("Running ensemble benchmark...", total=100)

            def update_progress_bar(
                completed: int, total: int, description: str
            ) -> None:
                try:
                    percentage = (completed / total * 100) if total > 0 else 0
                    progress.update(
                        task, completed=percentage, total=100, description=description
                    )
                except Exception:
                    pass

            async def run_ensemble() -> dict[str, Any]:
                try:
                    update_progress_bar(10, 100, "Initializing ensemble benchmark...")

                    result = await ensemble_benchmark.run_ensemble_benchmark(
                        tasks=ensemble_tasks,
                        patterns=selected_patterns,
                        agent_combinations=agent_combinations,
                        iterations=iterations,
                    )

                    update_progress_bar(100, 100, "Ensemble benchmark completed!")
                    return result
                except Exception as e:
                    raise BenchmarkExecutionError(
                        f"Ensemble benchmark execution failed: {e}",
                        error_code="ENSEMBLE_EXECUTION_FAILED",
                        context={
                            "patterns": [p.value for p in selected_patterns],
                            "iterations": iterations,
                            "error": str(e),
                        },
                    ) from e

            try:
                results = asyncio.run(run_ensemble())
                progress.update(
                    task, completed=100, description="Ensemble benchmark completed!"
                )
            except Exception as e:
                progress.update(
                    task, completed=100, description="Ensemble benchmark failed!"
                )
                raise e

        if show_console:
            console.print()
            console.print(
                Panel.fit(
                    "\nEnsemble Benchmark Results\n",
                    subtitle="Multi-Agent Collaboration",
                    border_style="bright_blue",
                )
            )

            ensemble_report = ensemble_benchmark.generate_ensemble_report(
                results["results"]
            )
            console.print(ensemble_report)
            _display_ensemble_summary_table(results, quiet)

        if not output:
            output = Path("logs/ensemble_results.json")

        if output:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                with open(output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                if show_console:
                    console.print(
                        f"\nEnsemble results saved to: [bold blue]{output}[/bold blue]"
                    )
                    display_project_footer()
            except Exception as e:
                raise ExportError(
                    f"Failed to save ensemble results: {e}",
                    error_code="ENSEMBLE_EXPORT_FAILED",
                    context={"output_path": str(output), "error": str(e)},
                ) from e

    except Exception as e:
        CLIErrorHandler.handle_error(e, "ensemble", debug=debug)


def _parse_ensemble_patterns(patterns: str) -> list[Any]:
    """Parse and validate ensemble patterns."""
    if patterns == "all":
        return list(EnsemblePattern)

    pattern_map = {
        "pipeline": EnsemblePattern.PIPELINE,
        "parallel": EnsemblePattern.PARALLEL,
        "hierarchical": EnsemblePattern.HIERARCHICAL,
        "consensus": EnsemblePattern.CONSENSUS,
        "specialization": EnsemblePattern.SPECIALIZATION,
    }
    selected_patterns = []
    for pattern_str in patterns.split(","):
        pattern_name = pattern_str.strip()
        if pattern_name in pattern_map:
            selected_patterns.append(pattern_map[pattern_name])
        else:
            raise ConfigurationError(
                f"Invalid ensemble pattern: {pattern_name}",
                error_code="INVALID_ENSEMBLE_PATTERN",
                context={
                    "provided": pattern_name,
                    "valid_patterns": list(pattern_map.keys()),
                },
            )
    return selected_patterns


def _parse_agent_combinations(combinations: str) -> list[list[Any]]:
    """Parse and validate agent combinations."""
    if combinations == "all":
        return [
            [AgentType.WRAPPER_AGENT, AgentType.MARKETING_AGENT],
            [AgentType.MARKETING_AGENT, AgentType.REAL_AGENT],
            [AgentType.WRAPPER_AGENT, AgentType.REAL_AGENT],
            [
                AgentType.WRAPPER_AGENT,
                AgentType.MARKETING_AGENT,
                AgentType.REAL_AGENT,
            ],
        ]
    elif combinations == "mixed":
        return [
            [
                AgentType.WRAPPER_AGENT,
                AgentType.MARKETING_AGENT,
                AgentType.REAL_AGENT,
            ],
        ]
    elif combinations == "homogeneous":
        return [
            [AgentType.WRAPPER_AGENT, AgentType.WRAPPER_AGENT],
            [AgentType.MARKETING_AGENT, AgentType.MARKETING_AGENT],
            [AgentType.REAL_AGENT, AgentType.REAL_AGENT],
        ]
    else:
        raise ConfigurationError(
            f"Invalid combinations option: {combinations}",
            error_code="INVALID_AGENT_COMBINATIONS",
            context={
                "provided": combinations,
                "valid_options": ["all", "mixed", "homogeneous"],
            },
        )


@app.command()
def network_test(
    condition: str = typer.Argument(..., help="Network condition to simulate"),
    iterations: int = typer.Option(20, "--iterations", "-i"),
    quiet: bool = typer.Option(False, "--quiet"),
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed error information"
    ),
) -> None:
    """Run network condition simulation with error handling.

    Tests agent resilience and performance under various network conditions
    to evaluate real-world deployment readiness.

    Args:
        condition: Network condition to simulate. Must be one of:
            - stable: Optimal network conditions (baseline)
            - slow: High latency but reliable connections
            - unstable: Variable latency and intermittent failures
            - degraded: Poor performance with frequent timeouts
            - peak_hours: Congested network with variable performance
        iterations: Number of test iterations. Higher values provide better
            resilience assessment but increase execution time.
        quiet: Suppress intermediate logging output.
        debug: Show detailed error information and network simulation details.

    Network Simulation:
        Each condition simulates realistic network characteristics:
        - Latency variations and packet loss patterns
        - Connection timeout probabilities
        - Bandwidth limitations and throughput variations
        - Recovery time patterns after failures

    Analysis Output:
        - Agent resilience ratings under network stress
        - Performance degradation analysis compared to stable conditions
        - Recovery pattern assessment and failure mode analysis
        - Recommendations for production deployment considerations

    Example:
        >>> # Test agent resilience under unstable network conditions
        >>> agent-benchmark network-test unstable --iterations 25

        >>> # Baseline performance under stable conditions
        >>> agent-benchmark network-test stable --iterations 30
    """
    try:
        CLIErrorHandler.validate_benchmark_args(iterations, 300)

        _ = setup_benchmark_logger(quiet=quiet)
        show_console = not _should_suppress_console_output(quiet)

        valid_conditions = ["stable", "slow", "unstable", "degraded", "peak_hours"]
        if condition not in valid_conditions:
            raise ConfigurationError(
                f"Invalid network condition: {condition}",
                error_code="INVALID_NETWORK_CONDITION",
                context={
                    "provided": condition,
                    "valid_conditions": valid_conditions,
                },
            )

        if show_console:
            display_project_banner()
            console.print()
            title_panel = Panel.fit(
                f"\n[bold cyan]Network Simulation: {condition.title()}[/bold cyan]\n"
                f"\nTesting agent resilience under {condition} network conditions\n",
                border_style="cyan",
                width=100,
            )
            console.print(title_panel)
            console.print()

        try:
            benchmark_runner = AgentBenchmark(quick_mode=True, quiet=quiet)
        except Exception as e:
            raise BenchmarkExecutionError(
                f"Failed to initialize network test: {e}",
                error_code="NETWORK_TEST_INIT_FAILED",
                context={"condition": condition, "error": str(e)},
            ) from e

        with _create_progress_context(
            f"Testing {condition} network conditions...", quiet
        ) as progress:
            task = progress.add_task("Running network simulation...", total=100)

            def update_progress_bar(
                completed: int, total: int, description: str
            ) -> None:
                try:
                    percentage = (completed / total * 100) if total > 0 else 0
                    progress.update(
                        task, completed=percentage, total=100, description=description
                    )
                except Exception:
                    pass

            async def run_network_test() -> dict[str, Any]:
                try:
                    enhanced_config = {
                        "enable_tool_failures": True,
                        "network_condition": condition,
                        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
                    }

                    return await benchmark_runner.run_comprehensive_benchmark(
                        iterations=iterations,
                        timeout=300,
                        enhanced_config=enhanced_config,
                        progress_callback=update_progress_bar,
                    )
                except Exception as e:
                    raise BenchmarkExecutionError(
                        f"Network simulation failed: {e}",
                        error_code="NETWORK_SIMULATION_FAILED",
                        context={
                            "condition": condition,
                            "iterations": iterations,
                            "error": str(e),
                        },
                    ) from e

            try:
                results = asyncio.run(run_network_test())
                progress.update(
                    task, completed=100, description="Network simulation completed!"
                )
            except Exception as e:
                progress.update(
                    task, completed=100, description="Network simulation failed!"
                )
                raise e

        if show_console:
            console.print()
            console.print(
                Panel.fit(
                    "Network Resilience Analysis",
                    subtitle=f"Performance under {condition.title()} conditions",
                    border_style="bright_blue",
                )
            )
            _display_network_analysis(results, condition, quiet)

            if show_console:
                display_project_footer()

    except Exception as e:
        CLIErrorHandler.handle_error(e, "network-test", debug=debug)


def _load_and_validate_results_file(results_file: Path) -> dict[str, Any]:
    """Load and validate results file with proper error handling."""
    if not results_file.exists():
        raise ConfigurationError(
            f"Results file not found: {results_file}",
            error_code="RESULTS_FILE_NOT_FOUND",
            context={"file_path": str(results_file)},
        )

    try:
        with open(results_file) as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON in results file: {e}",
            error_code="INVALID_RESULTS_FILE",
            context={"file_path": str(results_file), "json_error": str(e)},
        ) from e

    if "results" not in data:
        raise ConfigurationError(
            "Invalid results file format - missing 'results' key",
            error_code="INVALID_RESULTS_FORMAT",
            context={
                "file_path": str(results_file),
                "available_keys": list(data.keys()),
            },
        )

    return data


def _process_analyze_results(
    data: dict[str, Any],
    confidence_level: float,
    show_console: bool,
    show_outliers: bool,
    export_stats: Path | None,
    quiet: bool,
) -> None:
    """Process and display analysis results - extracted to reduce complexity."""
    enhanced_config = data.get("enhanced_config")
    if enhanced_config and show_console:
        console.print(
            f"Enhanced features detected: {', '.join(data.get('features', []))}"
        )

    if ENHANCED_STATS_AVAILABLE:
        try:
            statistical_analyzer = StatisticalAnalysis(
                confidence_level=confidence_level
            )
            enhanced_data = _enhance_results_with_statistics(data, statistical_analyzer)

            if show_console:
                enhanced_report = generate_enhanced_statistical_report(
                    enhanced_data, confidence_level
                )
                console.print(enhanced_report)

                if show_outliers:
                    _display_outlier_analysis(enhanced_data, quiet)

            if export_stats:
                with open(export_stats, "w") as f:
                    json.dump(enhanced_data, f, indent=2, default=str)
                if show_console:
                    console.print(
                        f"\nEnhanced statistics exported to: [bold blue]{export_stats}[/bold blue]"
                    )
        except Exception as e:
            raise StatisticalAnalysisError(
                f"Statistical analysis failed: {e}",
                error_code="STATS_ANALYSIS_FAILED",
                context={"confidence_level": confidence_level, "error": str(e)},
            ) from e
    elif show_console:
        console.print(
            "[yellow]Enhanced statistics not available. Install with: pip install scipy[/yellow]"
        )
        _display_summary_table(data, quiet)


def _convert_raw_results_to_analysis(
    data: dict[str, Any], show_console: bool
) -> dict[str, Any]:
    """Convert raw results to analysis format - extracted to reduce complexity."""
    if "results" in data and isinstance(data["results"], list):
        if show_console:
            console.print(f"Found {len(data['results'])} raw task results")
            console.print("Converting raw results to analysis format...")

        task_results = []
        for result_dict in data["results"]:
            try:
                agent_type = AgentType(result_dict["agent_type"])
                task_type = TaskComplexity(result_dict["task_type"])

                task_result = TaskResult(
                    task_id=result_dict["task_id"],
                    task_name=result_dict["task_name"],
                    task_type=task_type,
                    agent_type=agent_type,
                    success=result_dict["success"],
                    execution_time=result_dict["execution_time"],
                    error_count=result_dict["error_count"],
                    context_retention_score=result_dict["context_retention_score"],
                    cost_estimate=result_dict["cost_estimate"],
                    failure_reason=result_dict.get("failure_reason"),
                    steps_completed=result_dict.get("steps_completed", 0),
                    total_steps=result_dict.get("total_steps", 0),
                )
                task_results.append(task_result)
            except (KeyError, ValueError) as e:
                if show_console:
                    console.print(
                        f"[yellow]Warning: Skipping malformed result: {e}[/yellow]"
                    )
                continue

        if not task_results:
            raise StatisticalAnalysisError(
                "No valid task results found in file",
                error_code="NO_VALID_RESULTS",
                context={"total_entries": len(data["results"])},
            )

        benchmark = AgentBenchmark(quiet=True)
        analyzed_data = benchmark.analyze_results(task_results)
        return {"results": analyzed_data}

    return data


@app.command()
def analyze(
    results_file: Annotated[
        Path | None,
        typer.Argument(
            help="Path to results.json file (default: auto-detect in logs/)"
        ),
    ] = None,
    confidence_level: Annotated[
        float, typer.Option("--confidence")
    ] = DEFAULT_CONFIDENCE_LEVEL,
    show_outliers: Annotated[bool, typer.Option("--outliers")] = False,
    export_stats: Annotated[Path | None, typer.Option("--export-stats")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    suppress_warnings: Annotated[
        bool,
        typer.Option(
            "--suppress-warnings/--show-warnings",
            help="Suppress expected statistical warnings (default: True)",
        ),
    ] = True,
    debug: Annotated[
        bool, typer.Option("--debug", help="Show detailed error information")
    ] = False,
) -> None:
    """Load and analyze results file with error handling.

    Performs post-hoc analysis of benchmark results with enhanced statistical
    analysis, outlier detection, and detailed reporting capabilities.

    Args:
        results_file: Path to benchmark results JSON file. If None, auto-detects
            from common locations (logs/results.json, logs/results_backup.json, etc.).
        confidence_level: Confidence level for statistical intervals (0.8-0.99).
            Default: 0.95 (95% confidence intervals).
        show_outliers: Include outlier detection and analysis in output.
        export_stats: Export enhanced statistical analysis to specified file path.
        quiet: Suppress intermediate output during analysis.
        suppress_warnings: Hide expected statistical warnings to reduce console noise.
        debug: Show detailed error information including file parsing details.

    File Format Support:
        - Standard benchmark results (aggregated metrics)
        - Raw task results (converts automatically to analysis format)
        - Enhanced results with additional metadata
        - Legacy result formats with graceful handling

    Analysis Features:
        - Enhanced statistical analysis with confidence intervals
        - Effect size calculations and interpretations
        - Statistical power assessment and sample size recommendations
        - Outlier detection and anomaly analysis
        - Performance trend analysis and insights

    Auto-Detection:
        Searches for results files in order:
        1. logs/results.json
        2. logs/results_backup.json
        3. logs/ensemble_results.json
        4. results.json (current directory)

    Example:
        >>> # Analyze latest results with enhanced statistics
        >>> agent-benchmark analyze --confidence 0.99 --outliers --enhanced-stats

        >>> # Analyze specific file and export detailed statistics
        >>> agent-benchmark analyze results.json --export-stats detailed_stats.json
    """
    try:
        CLIErrorHandler.validate_benchmark_args(10, 60, confidence_level, export_stats)

        if suppress_warnings:
            suppress_statistical_warnings()

        _ = setup_benchmark_logger(quiet=quiet)
        show_console = not _should_suppress_console_output(quiet)

        if results_file is None:
            results_file = _auto_detect_results_file(show_console)

        data = _load_and_validate_results_file(results_file)

        if show_console:
            display_project_banner()
            console.print()
            title_panel = Panel.fit(
                "Benchmark Results Analysis",
                subtitle=f"Analyzing {results_file}",
                border_style="bright_blue",
            )
            console.print(title_panel)

        data = _convert_raw_results_to_analysis(data, show_console)
        _process_analyze_results(
            data, confidence_level, show_console, show_outliers, export_stats, quiet
        )

        if show_console:
            display_project_footer()

    except Exception as e:
        CLIErrorHandler.handle_error(e, "analyze", debug=debug)


def _auto_detect_results_file(show_console: bool) -> Path:
    """Auto-detect results file from common locations."""
    candidates = [
        Path("logs/results.json"),
        Path("logs/results_backup.json"),
        Path("logs/ensemble_results.json"),
        Path("results.json"),
    ]

    for candidate in candidates:
        if candidate.exists():
            if show_console:
                console.print(f"[yellow]Auto-detected: {candidate}[/yellow]")
            return candidate

    raise ConfigurationError(
        "No results file found. Run a benchmark first.",
        error_code="NO_RESULTS_FILE",
        context={"searched_paths": [str(c) for c in candidates]},
    )


@app.command()
def quick(
    quiet: bool = typer.Option(False, "--quiet"),
    iterations: int = typer.Option(5),
    random_seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducible results"
    ),
    randomize: bool = typer.Option(
        False, "--randomize", help="Shuffle task order using the seed"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed error information"
    ),
) -> None:
    """Run a simplified benchmark with error handling.

    Provides fast benchmark execution with reduced task set and shorter timeouts,
    ideal for development, testing, and quick performance checks.

    Args:
        quiet: Suppress intermediate logging output during execution.
        iterations: Number of iterations per agent-task pair. Lower default (5)
            for faster execution while maintaining statistical validity.
        random_seed: Random seed for reproducible results across quick benchmark runs.
        randomize: Randomize task execution order using the specified seed.
        debug: Show detailed error information and execution context.

    Quick Mode Features:
        - Reduced task set (3 representative tasks vs 8 full tasks)
        - Shorter timeouts (30s vs 60s per task)
        - Streamlined output focusing on key metrics
        - Faster agent initialization and cleanup
        - Essential performance insights without extended analysis

    Task Selection:
        Quick mode uses representative tasks across complexity levels:
        - 1 Simple task: Basic functionality validation
        - 1 Moderate task: Intermediate capability testing
        - 1 Complex task: Advanced reasoning assessment

    Output:
        - Performance comparison across agent types
        - Success rate and execution time analysis
        - Cost efficiency assessment
        - Quick insights into architectural performance differences

    Use Cases:
        - Development workflow validation
        - CI/CD pipeline integration
        - Quick agent comparison
        - Pre-deployment sanity checks
        - Debugging agent implementations

    Example:
        >>> # Quick performance check
        >>> agent-benchmark quick --iterations 3

        >>> # Reproducible quick benchmark
        >>> agent-benchmark quick --seed 12345 --randomize --iterations 10
    """
    try:
        CLIErrorHandler.validate_benchmark_args(iterations, 400)

        _ = setup_benchmark_logger(quiet=quiet)
        show_console = not _should_suppress_console_output(quiet)

        if show_console:
            display_project_banner()
            console.print()
            title_panel = Panel.fit(
                "\n AI Agent Reality Check \n",
                subtitle="Agent Performance",
                border_style="bright_yellow",
            )
            console.print(title_panel)
            console.print()

        try:
            benchmark_runner = AgentBenchmark(
                quick_mode=True,
                quiet=True,
                random_seed=random_seed,
                randomize=randomize,
            )
        except Exception as e:
            raise BenchmarkExecutionError(
                f"Failed to initialize quick benchmark: {e}",
                error_code="QUICK_BENCHMARK_INIT_FAILED",
                context={
                    "random_seed": random_seed,
                    "randomize": randomize,
                    "error": str(e),
                },
            ) from e

        with _create_progress_context("Starting benchmark...", quiet) as progress:
            task = progress.add_task("Starting benchmark...", total=45)

            def update_progress(completed: int, total: int, description: str) -> None:
                progress.update(
                    task, completed=completed, total=total, description=description
                )

            async def run_benchmark() -> dict[str, Any]:
                try:
                    result = await benchmark_runner.run_comprehensive_benchmark(
                        iterations=iterations,
                        timeout=400,
                        progress_callback=update_progress,
                    )

                    if result.get("early_exit"):
                        logger.warning(
                            "Benchmark exited early due to timeout. Results may be partial."
                        )

                    return result
                except Exception as e:
                    raise BenchmarkExecutionError(
                        f"Quick benchmark execution failed: {e}",
                        error_code="QUICK_BENCHMARK_FAILED",
                        context={"iterations": iterations, "error": str(e)},
                    ) from e

            try:
                results = asyncio.run(run_benchmark())
                progress.update(
                    task,
                    completed=45,
                    total=45,
                    description="Benchmark completed!",
                )
            except Exception as e:
                progress.update(
                    task, completed=45, total=45, description="Benchmark failed!"
                )
                raise e

        if show_console:
            try:
                console.print()
                report = benchmark_runner.generate_reality_report(results)
                console.print(report)
                _display_rich_benchmark_output(results, None, quiet)
                filename = (
                    "quick_random_results.json" if randomize else "quick_results.json"
                )
                output_path = Path(f"logs/{filename}")
                _save_benchmark_results(
                    results, benchmark_runner, output_path, "json", None, show_console
                )

                if show_console:
                    display_project_footer()

            except Exception as e:
                raise BenchmarkExecutionError(
                    f"Failed to display or save quick benchmark results: {e}",
                    error_code="QUICK_BENCHMARK_DISPLAY_FAILED",
                    context={"error": str(e)},
                ) from e

    except Exception as e:
        CLIErrorHandler.handle_error(e, "quick", debug=debug)


@app.command()
def compare(
    agent1: str = typer.Argument(...),
    agent2: str = typer.Argument(...),
    iterations: int = typer.Option(10),
    quick_mode: bool = typer.Option(False, "--quick-mode"),
    quiet: bool = typer.Option(False, "--quiet"),
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed error information"
    ),
) -> None:
    """Compare two agents head-to-head with error handling.

    Performs focused comparison between two specific agent types with detailed
    performance analysis and statistical comparison.

    Args:
        agent1: First agent type to compare. Must be one of: WRAPPER_AGENT,
            MARKETING_AGENT, REAL_AGENT.
        agent2: Second agent type to compare. Must be different from agent1.
        iterations: Number of iterations for each agent-task pair comparison.
        quick_mode: Use reduced task set and shorter timeouts for faster comparison.
        quiet: Suppress intermediate logging output during execution.
        debug: Show detailed error information and comparison context.

    Comparison Analysis:
        - Head-to-head performance metrics
        - Statistical significance of performance differences
        - Effect size calculation and interpretation
        - Cost efficiency comparison
        - Execution time analysis
        - Success rate differential assessment

    Agent Types:
        - WRAPPER_AGENT: Simple LLM prompt wrappers
        - MARKETING_AGENT: Enhanced pipelines marketed as intelligent agents
        - REAL_AGENT: Full agent architectures with memory, planning, tools

    Output Format:
        - Side-by-side performance comparison
        - Statistical significance indicators
        - Performance gap quantification
        - Architectural impact analysis
        - Recommendations based on comparison results

    Example:
        >>> # Compare wrapper vs real agent performance
        >>> agent-benchmark compare WRAPPER_AGENT REAL_AGENT --iterations 15

        >>> # Quick comparison for development
        >>> agent-benchmark compare MARKETING_AGENT REAL_AGENT --quick-mode
    """
    try:
        CLIErrorHandler.validate_benchmark_args(iterations, 300 if quick_mode else 400)

        _ = setup_benchmark_logger(quiet=quiet)
        show_console = not _should_suppress_console_output(quiet)

        valid_types = ["WRAPPER_AGENT", "MARKETING_AGENT", "REAL_AGENT"]
        if agent1 not in valid_types or agent2 not in valid_types:
            raise ConfigurationError(
                f"Invalid agent types: {agent1}, {agent2}",
                error_code="INVALID_AGENT_TYPES",
                context={"provided": [agent1, agent2], "valid_types": valid_types},
            )

        if show_console:
            display_project_banner()
            console.print()
            title_panel = Panel.fit(
                f"Head-to-Head Comparison\n{agent1.replace('_', ' ').title()} vs {agent2.replace('_', ' ').title()}",
                border_style="bright_blue",
            )
            console.print(title_panel)

            if quick_mode:
                console.print(
                    "[yellow]⚡ Using quick mode for faster execution[/yellow]"
                )
            console.print()

        try:
            benchmark_runner = AgentBenchmark(
                quick_mode=quick_mode,
                quiet=True,
            )
        except Exception as e:
            raise BenchmarkExecutionError(
                f"Failed to initialize comparison benchmark: {e}",
                error_code="COMPARE_BENCHMARK_INIT_FAILED",
                context={
                    "agents": [agent1, agent2],
                    "quick_mode": quick_mode,
                    "error": str(e),
                },
            ) from e

        async def run_benchmark() -> dict[str, Any]:
            try:
                result = await benchmark_runner.run_comprehensive_benchmark(
                    iterations=iterations, timeout=300 if quick_mode else 400
                )

                if result.get("early_exit"):
                    logger.warning(
                        "Benchmark exited early due to timeout. Results may be partial."
                    )

                return result
            except Exception as e:
                raise BenchmarkExecutionError(
                    f"Comparison benchmark execution failed: {e}",
                    error_code="COMPARE_BENCHMARK_FAILED",
                    context={
                        "agents": [agent1, agent2],
                        "iterations": iterations,
                        "error": str(e),
                    },
                ) from e

        with _create_progress_context(
            "Running comparison benchmark...", quiet
        ) as progress:
            task = progress.add_task("Running comparison benchmark...", total=100)
            try:
                results = asyncio.run(run_benchmark())
                progress.update(
                    task, completed=100, description="Comparison completed!"
                )
            except Exception as e:
                progress.update(task, completed=100, description="Comparison failed!")
                raise e

        agent_data = results.get("results", {}).get("results", {})
        filtered_data = {k: v for k, v in agent_data.items() if k in [agent1, agent2]}
        filtered_results = {"results": {"results": filtered_data}}

        if show_console:
            console.print()
            console.print(
                Panel.fit(
                    "Head-to-Head Comparison Results",
                    subtitle="Performance Analysis Complete",
                    border_style="bright_blue",
                )
            )
            _display_summary_table(filtered_results, quiet)

            if show_console:
                display_project_footer()

    except Exception as e:
        CLIErrorHandler.handle_error(e, "compare", debug=debug)


def _display_enhanced_summary(
    results: dict[str, Any], config: dict[str, Any], quiet: bool = False
) -> None:
    """Display enhanced summary with new features highlighted."""
    if _should_suppress_console_output(quiet):
        return

    enhancements = []
    if config.get("enable_tool_failures"):
        enhancements.append("Tool Failure Simulation")

    network_condition = config.get("network_condition", "stable")
    if network_condition != "stable":
        enhancements.append(f"Network: {network_condition.title()}")

    if (
        config.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)
        != DEFAULT_CONFIDENCE_LEVEL
    ):
        enhancements.append(f"CI: {config['confidence_level']:.0%}")

    if enhancements:
        enhancement_panel = Panel(
            " | ".join(enhancements),
            title="Enhanced Features Active",
            border_style="bright_green",
        )
        console.print(enhancement_panel)


@app.command()
def version(
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed error information"
    ),
) -> None:
    """Display current version and package metadata with error handling.

    Shows comprehensive version information including available features,
    dependencies, and repository information for debugging and support.

    Args:
        debug: Show detailed error information if version detection fails.

    Information Displayed:
        - Package version and author information
        - Available feature modules (Core, Ensemble, Enhanced Statistics)
        - Build information and tooling details
        - Repository URL for documentation and issues
        - License and copyright information

    Feature Detection:
        - Core Benchmarking: Always available
        - Ensemble Collaboration: Available if ensemble dependencies installed
        - Enhanced Statistics: Available if scipy/numpy dependencies installed

    Example:
        >>> agent-benchmark version
        AI Agents Reality Check v0.1.0
        Author: Jesse Moses (@Cre4T3Tiv3)
        Features: Core Benchmarking, Enhanced Statistics
        Built with UV and modern Python tooling
    """
    try:
        features = ["Core Benchmarking"]
        if ENSEMBLE_AVAILABLE:
            features.append("Ensemble Collaboration")
        if ENHANCED_STATS_AVAILABLE:
            features.append("Enhanced Statistics")

        version_panel = Panel(
            f"AI Agents Reality Check v{__version__}\n"
            f"Author: {__author__}\n"
            f"Features: {', '.join(features)}\n"
            f"Built with UV and modern Python tooling\n"
            f"Repository: https://github.com/Cre4T3Tiv3/ai-agents-reality-check",
            title="Version Information",
            style="blue",
        )
        console.print(version_panel)

    except Exception as e:
        CLIErrorHandler.handle_error(e, "version", debug=debug)


def _display_ensemble_summary_table(
    results: dict[str, Any], quiet: bool = False
) -> None:
    """Display ensemble results in a formatted table."""
    if _should_suppress_console_output(quiet):
        return

    analysis = results.get("results", {})
    pattern_analysis = analysis.get("by_pattern", {})

    if not pattern_analysis:
        return

    console.print()
    table = Table(title="Ensemble Pattern Performance")
    table.add_column("Pattern", style="cyan", no_wrap=True)
    table.add_column("Success Rate", justify="right", style="green")
    table.add_column("Avg Cost", justify="right", style="yellow")
    table.add_column("Coordination", justify="right", style="blue")
    table.add_column("Advantage", justify="right", style="magenta")

    sorted_patterns = sorted(
        pattern_analysis.items(), key=lambda x: x[1]["success_rate"], reverse=True
    )

    for pattern, metrics in sorted_patterns:
        pattern_display = pattern.replace("_", " ").title()

        success_rate = metrics["success_rate"]
        if success_rate >= EXCELLENT_THRESHOLD:
            success_indicator = "🟢"
        elif success_rate >= MODERATE_THRESHOLD:
            success_indicator = "🟡"
        else:
            success_indicator = "🔴"

        advantage = metrics.get("avg_ensemble_advantage", 0)
        advantage_indicator = (
            "↗"
            if advantage > ENSEMBLE_ADVANTAGE_THRESHOLD
            else "↘"
            if advantage < -ENSEMBLE_ADVANTAGE_THRESHOLD
            else "→"
        )

        avg_total_cost = metrics.get("avg_total_cost", 0)
        cost_display = f"${avg_total_cost:.3f}" if avg_total_cost is not None else "N/A"

        coordination_overhead = metrics.get("avg_coordination_overhead", 0)
        coord_display = (
            f"{coordination_overhead:.2f}s"
            if coordination_overhead is not None
            else "N/A"
        )

        table.add_row(
            f"{success_indicator} {pattern_display}",
            f"{success_rate:.1%}",
            cost_display,
            coord_display,
            f"{advantage_indicator} {advantage:+.1%}",
        )

    console.print(table)


def _display_network_analysis(
    results: dict[str, Any], condition: str, quiet: bool = False
) -> None:
    """Display network condition specific analysis."""
    if _should_suppress_console_output(quiet):
        return

    agent_data = results.get("results", {}).get("results", {})

    console.print(f"[bold]Impact of {condition.title()} Network Conditions[/bold]")
    console.print()

    network_table = Table(
        title=f"Agent Resilience Under {condition.title()} Conditions"
    )
    network_table.add_column("Agent Type", style="cyan")
    network_table.add_column("Success Rate", justify="right", style="green")
    network_table.add_column("Avg Time", justify="right", style="yellow")
    network_table.add_column("Resilience", justify="right", style="magenta")

    stable_baselines = {
        "WRAPPER_AGENT": {"success": 0.45, "time": 2.0},
        "MARKETING_AGENT": {"success": 0.65, "time": 4.0},
        "REAL_AGENT": {"success": 0.85, "time": 6.0},
    }

    for agent_type, metrics in agent_data.items():
        current_success = metrics.get("success_rate", 0)
        current_time = metrics.get("avg_execution_time", 0)

        baseline = stable_baselines.get(agent_type, {"success": 0.5, "time": 3.0})

        success_resilience = (
            current_success / baseline["success"] if baseline["success"] > 0 else 0
        )
        time_resilience = baseline["time"] / current_time if current_time > 0 else 0
        overall_resilience = (success_resilience + time_resilience) / 2

        if overall_resilience >= HIGH_RESILIENCE_THRESHOLD:
            resilience_indicator = "🟢 High"
        elif overall_resilience >= MEDIUM_RESILIENCE_THRESHOLD:
            resilience_indicator = "🟡 Medium"
        else:
            resilience_indicator = "🔴  Low"

        network_table.add_row(
            agent_type.replace("_", " ").title(),
            f"{current_success:.1%}",
            f"{current_time:.2f}s",
            f"{resilience_indicator} ({overall_resilience:.2f})",
        )

    console.print(network_table)


def _display_outlier_analysis(results: dict[str, Any], quiet: bool = False) -> None:
    """Display outlier detection analysis."""
    if _should_suppress_console_output(quiet):
        return

    console.print("\n[bold]Outlier Detection Analysis[/bold]")
    console.print()

    agent_data = results.get("results", {}).get("results", {})

    for agent_type, metrics in agent_data.items():
        enhanced_stats = metrics.get("enhanced_statistics", {})
        dist_analysis = enhanced_stats.get("distribution_analysis", {})

        outlier_count = dist_analysis.get("outlier_count", 0)
        if outlier_count > 0:
            console.print(
                f"{agent_type.replace('_', ' ').title()}: {outlier_count} outliers detected"
            )
        else:
            console.print(
                f"{agent_type.replace('_', ' ').title()}: No outliers detected"
            )


def _save_results(
    benchmark_runner: AgentBenchmark,
    results: dict[str, Any],
    filepath: str,
    format: str,
    enhanced_config: dict[str, Any] | None,
) -> None:
    """Save results with enhanced metadata if applicable."""

    def safe_json_serializer(obj: Any) -> str | None:
        """JSON serializer that handles infinity and NaN values safely."""
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return None
        return str(obj)

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if enhanced_config:
        features_list = _get_enhanced_features_list(enhanced_config)

        enhanced_export = {
            **results,
            "enhanced_config": enhanced_config,
            "version": "2.0",
            "features": features_list,
        }

        if format.lower() == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(enhanced_export, f, indent=2, default=safe_json_serializer)
        else:
            benchmark_runner.export_results(
                filepath, format=format, enhanced_data=enhanced_export
            )
    else:
        benchmark_runner.export_results(filepath, format=format)


def _display_summary_table(results: dict[str, Any], quiet: bool = False) -> None:
    """Render benchmark results as enhanced performance cards and insights.

    Creates Rich-formatted performance visualization with cards, insights,
    and performance charts for comprehensive results display.

    Args:
        results: Benchmark results with agent performance data.
        quiet: Whether to suppress console output.

    Display Elements:
        - Performance cards with success rates, timing, and cost metrics
        - Color-coded performance indicators (🟢🟡🟠🔴)
        - Insights panel with best performers and cost leaders
        - ASCII performance chart for visual comparison
        - Performance alerts for low-performing agents
        - Architectural recommendations

    Performance Ratings:
        - Excellent: ≥90% success rate (🟢)
        - Good: ≥70% success rate (🟡)
        - Fair: ≥50% success rate (🟠)
        - Poor: <50% success rate (🔴)

    Note:
        - Handles empty results gracefully with appropriate messaging
        - Provides actionable recommendations for poor performers
        - Highlights architectural differences between agent types
        - Safe to call repeatedly with consistent formatting
    """
    if _should_suppress_console_output(quiet):
        return

    agents_data = results.get("results", {}).get("results", {})
    if not agents_data:
        console.print("[red]No agent data available for display[/red]")
        return

    console.print()
    console.print("[bold]Agent Performance Summary[/bold]")
    console.print()
    performance_cards = _create_performance_cards(agents_data)
    console.print(performance_cards)

    console.print()
    insights_panel = _create_insights_panel(agents_data)
    console.print(insights_panel)

    console.print()
    chart_panel = _create_performance_chart(agents_data)
    console.print(chart_panel)

    low_performers = [
        agent
        for agent, data in agents_data.items()
        if data["success_rate"] < MODERATE_THRESHOLD
    ]
    if low_performers:
        warning_content = (
            "[yellow]Performance Alert:[/yellow] Agents with poor success rates often rely on minimal wrappers.\n"
            "To improve reliability: Adopt full agent architectures with memory, planning, and tool-use; not just prompt chaining."
        )
        console.print()
        console.print(
            Panel(warning_content, title="Recommendations", border_style="yellow")
        )


def _display_shocking_stats(results: dict[str, Any], quiet: bool = False) -> None:
    """Highlight dramatic gaps between wrapper, marketing, and real agents.

    Emphasizes the reality check theme by prominently displaying performance
    gaps that expose the differences between agent architectures.

    Args:
        results: Benchmark results with agent performance data.
        quiet: Whether to suppress console output.

    Highlighted Statistics:
        - Wrapper agent failure rates vs real agent success
        - Marketing claim vs actual wrapper performance gap ("illusion gap")
        - Cost efficiency comparison between architectures
        - Success rate differentials with bold formatting

    Reality Check Messages:
        - Exposes low success rates of LLM wrappers
        - Quantifies performance gaps between architectures
        - Highlights cost implications of different approaches
        - Provides context about marketing vs reality

    Note:
        - Only displays when all three agent types are present
        - Uses dramatic formatting to emphasize key insights
        - Handles edge cases like zero success rates gracefully
        - Central to the "reality check" mission of exposing architectural differences
    """
    if _should_suppress_console_output(quiet):
        return

    agents_data = results.get("results", {}).get("results", {})

    wrapper_data = agents_data.get("WRAPPER_AGENT", {})
    real_data = agents_data.get("REAL_AGENT", {})
    marketing_data = agents_data.get("MARKETING_AGENT", {})

    if not all([wrapper_data, real_data, marketing_data]):
        return

    wrapper_success = round(wrapper_data.get("success_rate", 0) * 100, 1)
    real_success = round(real_data.get("success_rate", 0) * 100, 1)
    marketing_success = round(marketing_data.get("success_rate", 0) * 100, 1)

    wrapper_cost = wrapper_data.get("cost_per_success", 0) or 0.0
    real_cost = real_data.get("cost_per_success", 0) or 0.0

    if wrapper_cost in (0, float("inf")) or math.isnan(wrapper_cost):
        wrapper_cost_display = "N/A"
    else:
        wrapper_cost_display = f"${wrapper_cost:.4f}"

    if (
        wrapper_cost in (0, float("inf"))
        or real_cost in (0, float("inf"))
        or math.isnan(wrapper_cost)
        or math.isnan(real_cost)
    ):
        cost_efficiency_text = "N/A"
    else:
        cost_efficiency = wrapper_cost / real_cost
        cost_efficiency_text = f"{cost_efficiency:.1f}× more cost-efficient"

    if wrapper_success == 0.0:
        success_comparison = (
            "Real agents achieved success where LLM wrappers failed entirely"
        )
    else:
        success_gap = real_success - wrapper_success
        success_comparison = (
            f"Real agents outperform LLM wrappers by [bold green]{success_gap:.1f}%[/bold green] "
            f"(success: {real_success:.1f}% vs {wrapper_success:.1f}%)"
        )

    illusion_gap = marketing_success - wrapper_success

    shocking_content = (
        f"• [bold red]{wrapper_success:.1f}%[/bold red] success rate for most 'AI agents' ([dim]LLM wrappers[/dim])\n"
        f"• {success_comparison}\n"
        f"• [bold yellow]{illusion_gap:.1f}[/bold yellow] percentage point illusion gap "
        f"between marketing claims and wrapper performance\n"
        f"• Real agents are [bold green]{cost_efficiency_text}[/bold green] per success "
        f"(wrapper: {wrapper_cost_display})\n"
    )

    shocking_panel = Panel(
        shocking_content,
        title="Reality Check",
        border_style="red",
    )
    console.print()
    console.print(shocking_panel)


def main() -> None:
    """Launch the CLI entry point when invoked as a script.

    Sets up and executes the Typer CLI application with comprehensive
    error handling and proper exit behavior.

    Note:
        - Called when module executed directly or via console script
        - Handles all CLI routing and command dispatch
        - Provides consistent error handling across all commands
        - Safe entry point for packaging and distribution
    """
    app()


if __name__ == "__main__":
    main()
