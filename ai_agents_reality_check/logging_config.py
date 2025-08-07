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

Module: ai_agents_reality_check/logging_config.py

Logging configuration and utilities for the AI Agents Reality Check system.

Provides comprehensive logging setup with Rich formatting, file output, warning
suppression, and trace functionality designed for both development and production
environments.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import logging
import os
import warnings
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class LogConfig:
    DEFAULT_LOG_LEVEL = logging.INFO
    LOG_DIR = "logs"

    @staticmethod
    def setup_logger(
        name: str = "ai_agents_logger",
        level: int = DEFAULT_LOG_LEVEL,
        quiet: bool = False,
    ) -> logging.Logger:
        """Setup a logger with optional Rich formatting and warning suppression.

        Creates a comprehensive logger with console and file output, Rich formatting
        for enhanced readability, and configurable warning filters.

        Args:
            name: Logger name for identification and hierarchy. Default: "ai_agents_logger".
            level: Logging level using standard logging constants. Default: logging.INFO.
            quiet: Suppress console output to avoid Rich console conflicts. File logging
                continues regardless. Default: False.

        Returns:
            Configured Logger instance ready for use throughout the application.

        Logger Features:
            Console Output (when not quiet):
            - Rich formatting with colors, markup, and enhanced tracebacks
            - Simplified format without timestamps (uses Rich's built-in timing)
            - Stderr output to avoid conflicts with stdout data
            - Fallback to standard console handler if Rich unavailable

            File Output (always enabled):
            - Timestamped log files in logs/ directory
            - Comprehensive format with timestamps, levels, and logger names
            - DEBUG level for detailed file logging regardless of console level
            - UTF-8 encoding for international character support

            Warning Management:
            - Automatic configuration of statistical warning filters
            - Suppression of expected scipy/numpy computation warnings
            - Clean console output while maintaining file logging

        Note:
            - Creates logs/ directory automatically if needed
            - Clears existing handlers to prevent duplicate output
            - Disables propagation to avoid duplicate logging
            - Safe to call multiple times - reconfigures existing loggers
            - Handles missing dependencies gracefully with fallbacks

        Example:
            >>> logger = LogConfig.setup_logger("benchmark", logging.DEBUG, quiet=True)
            >>> logger.info("Benchmark starting...")  # File only if quiet=True
        """
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(level)

        LogConfig._configure_warning_filters()

        os.makedirs(LogConfig.LOG_DIR, exist_ok=True)

        if not quiet:
            try:
                console = Console(stderr=True, force_terminal=True)
                rich_handler = RichHandler(
                    console=console,
                    show_path=False,
                    show_time=False,
                    rich_tracebacks=True,
                    markup=True,
                )
                rich_handler.setFormatter(logging.Formatter("%(message)s"))
                rich_handler.setLevel(level)
                logger.addHandler(rich_handler)
            except Exception:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
                )
                console_handler.setLevel(level)
                logger.addHandler(console_handler)

        try:
            log_filename = f"benchmark_{RUN_TIMESTAMP}.log"
            log_filepath = os.path.join(LogConfig.LOG_DIR, log_filename)

            file_handler = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        except Exception:
            pass

        logger.propagate = False

        return logger

    @staticmethod
    def _configure_warning_filters() -> None:
        """Configure warning filters to suppress expected statistical warnings."""
        warnings.filterwarnings(
            "ignore",
            message=".*Input data has range zero.*",
            category=UserWarning,
            module="scipy.stats.*",
        )

        warnings.filterwarnings(
            "ignore",
            message=".*Results may not be accurate.*",
            category=UserWarning,
            module="scipy.stats.*",
        )

        warnings.filterwarnings(
            "ignore",
            message=".*Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
            module="ai_agents_reality_check.utils.enhanced_statistics",
        )

        warnings.filterwarnings(
            "ignore",
            message=".*catastrophic cancellation.*",
            category=RuntimeWarning,
            module="ai_agents_reality_check.utils.enhanced_statistics",
        )

        logging.getLogger("scipy.stats").setLevel(logging.ERROR)


def trace_to_file(agent_name: str, task_id: str, message: str) -> None:
    """Log a trace message to a timestamped file.

    Provides specialized trace logging for agent execution debugging with
    structured format and automatic file management.

    Args:
        agent_name: Name of the agent generating the trace (e.g., "RealAgent").
        task_id: Unique identifier for the task being executed.
        message: Trace message content describing the execution step or event.

    File Output:
        - Separate trace files per agent: {agent_name_lower}_{timestamp}.log
        - Structured format: "[AGENT_NAME Trace] Task={task_id} :: {message}"
        - UTF-8 encoding for international character support
        - Automatic directory creation in logs/ folder

    Note:
        - Designed for high-frequency trace logging during agent execution
        - Silent failure handling - traces don't interrupt execution on I/O errors
        - Timestamped files prevent conflicts across benchmark runs
        - Useful for debugging agent decision-making and execution flow

    Example:
        >>> trace_to_file("RealAgent", "task_123", "Starting tool execution")
        >>> # Creates logs/realagent_20250807_143022.log with structured entry
    """
    try:
        log_dir = LogConfig.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        log_line = f"[{agent_name.upper()} Trace] Task={task_id} :: {message}\n"
        log_file_path = os.path.join(
            log_dir, f"{agent_name.lower()}_{RUN_TIMESTAMP}.log"
        )
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception:
        pass


console = Console()

_temp_logger = LogConfig.setup_logger("temp_init", quiet=True)
logger = LogConfig.setup_logger()


def get_rich_console() -> Console:
    """Get a shared Rich console instance.

    Provides access to the globally configured Rich Console instance for
    consistent formatting and output across the application.

    Returns:
        Rich Console instance configured for application use.

    Note:
        - Shared instance ensures consistent formatting
        - Used by CLI commands and Rich display functions
        - Configured for optimal terminal compatibility
        - Safe to use for all Rich output operations

    Example:
        >>> console = get_rich_console()
        >>> console.print("[bold red]Error:[/bold red] Task failed")
    """
    return console


def setup_benchmark_logger(quiet: bool = False) -> logging.Logger:
    """Setup logger specifically for benchmark runs.

    Creates a specialized logger for benchmark operations with appropriate
    level configuration and quiet mode handling.

    Args:
        quiet: Enable quiet mode to suppress console conflicts with Rich output.
            File logging continues with DEBUG level. Default: False.

    Returns:
        Configured Logger instance optimized for benchmark execution logging.

    Logger Configuration:
        - DEBUG level when not quiet (detailed console output)
        - INFO level when quiet (essential messages only)
        - Always DEBUG level for file output
        - Benchmark-specific logger name for filtering

    Note:
        - Uses LogConfig.setup_logger with benchmark-specific defaults
        - Quiet mode prevents logging conflicts with Rich progress bars
        - Essential for clean CLI output while maintaining file traces
        - Safe to call multiple times during benchmark execution

    Example:
        >>> logger = setup_benchmark_logger(quiet=True)  # File logging only
        >>> logger.debug("Detailed benchmark step")  # File only
        >>> logger.error("Critical benchmark error")  # File only in quiet mode
    """
    benchmark_logger = LogConfig.setup_logger(
        "ai_agents_benchmark",
        level=logging.DEBUG if not quiet else logging.INFO,
        quiet=quiet,
    )

    return benchmark_logger


def suppress_statistical_warnings() -> None:
    """Explicitly suppress expected statistical warnings for production runs.

    Configures warning filters to suppress expected statistical computation
    warnings that are normal during benchmark analysis but create noise in output.

    Suppressed Warnings:
        - Scipy warnings about zero-range data (normal for identical values)
        - Statistical accuracy warnings for edge cases
        - Precision loss warnings from moment calculations
        - Catastrophic cancellation warnings in statistics

    Warning Categories:
        - UserWarning from scipy.stats modules
        - RuntimeWarning from enhanced statistics calculations
        - Sets scipy.stats logger to ERROR level

    Note:
        - Called automatically by LogConfig.setup_logger()
        - Can be called explicitly for additional suppression
        - Only suppresses expected/normal warnings, not errors
        - Maintains error and critical message visibility
        - Safe to call multiple times - filters are idempotent

    Example:
        >>> suppress_statistical_warnings()  # Clean up statistical noise
        >>> # Statistical calculations now run without expected warning noise
    """
    LogConfig._configure_warning_filters()
