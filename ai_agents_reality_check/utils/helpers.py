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

Module: ai_agents_reality_check/utils/helpers.py

Helper functions and utilities for the AI Agents Reality Check system.

This includes statistics computation, formatting utilities, configuration validation,
logging setup, outlier detection, and benchmark ID generation. Provides core utility
functions used throughout the benchmarking system with robust error handling and
comprehensive documentation.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import math
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_agents_reality_check.types import TaskComplexity

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ai_agents_reality_check.logging_config import logger
from ai_agents_reality_check.types import TaskResult

# Constants for magic values
MIN_QUARTILE_SAMPLES = 4
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
POWER_90_Z_BETA = 1.28
POWER_95_Z_BETA = 1.645
EFFECT_SIZE_NEGLIGIBLE_THRESHOLD = 0.2
EFFECT_SIZE_SMALL_THRESHOLD = 0.5
EFFECT_SIZE_MEDIUM_THRESHOLD = 0.8
EFFECT_SIZE_LARGE_THRESHOLD = 1.2
MAX_ITERATIONS = 1000
MIN_CI_SAMPLES = 2
CONFIDENCE_95_Z_SCORE = 1.96
CONFIDENCE_99_Z_SCORE = 2.576
MIN_OUTLIER_SAMPLES = 4
ZSCORE_OUTLIER_THRESHOLD = 3
BYTES_PER_KILOBYTE = 1024.0
T_DISTRIBUTION_THRESHOLD = 30
HIGH_ANOMALY_THRESHOLD = 3.0
LOW_SUCCESS_RATE_THRESHOLD = 0.3
POWER_90 = 0.9
POWER_95 = 0.95
CONFIDENCE_95 = 0.95
EFFECT_SIZE_ZERO_THRESHOLD = 0.001
MAX_REASONABLE_EFFECT_SIZE = 5.0
MIN_AGENTS_FOR_COMPARISON = 2
VERY_SMALL_EFFECT_THRESHOLD = 0.1
SIMILAR_SUCCESS_RATE_THRESHOLD = 0.05
MINIMUM_STATISTICAL_POWER = 0.8


def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate summary statistics for a list of numeric values.

    Computes basic descriptive statistics with robust error handling for edge cases
    including empty lists, single values, and invalid numeric values (NaN, infinity).

    Args:
        values: A list of numeric (float) values. Can be empty or contain invalid values.

    Returns:
        A dictionary containing statistical measures:
            - mean: Arithmetic mean of valid values
            - median: Middle value of sorted valid values
            - std_dev: Standard deviation (0.0 for single values)
            - min: Minimum value
            - max: Maximum value
            - q25: 25th percentile (first quartile)
            - q75: 75th percentile (third quartile)

    Note:
        - Filters out NaN and infinite values automatically
        - Returns zero-filled dict for empty or all-invalid input
        - Quartiles use index-based approximation for small samples
        - Single values return that value for all measures except std_dev (0.0)

    Example:
        >>> stats = calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> print(f"Mean: {stats['mean']}, Std: {stats['std_dev']:.2f}")
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }

    n = len(values)
    if n == 1:
        val = values[0]
        return {
            "mean": val,
            "median": val,
            "std_dev": 0.0,
            "min": val,
            "max": val,
            "q25": val,
            "q75": val,
        }

    clean_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]

    if not clean_values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }

    sorted_values = sorted(clean_values)
    n = len(sorted_values)

    return {
        "mean": statistics.mean(sorted_values),
        "median": statistics.median(sorted_values),
        "std_dev": statistics.stdev(sorted_values) if n > 1 else 0.0,
        "min": min(sorted_values),
        "max": max(sorted_values),
        "q25": sorted_values[max(0, int(0.25 * (n - 1)))],
        "q75": sorted_values[min(n - 1, int(0.75 * (n - 1)))],
    }


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Converts numeric seconds into appropriate time units with intelligent
    formatting for different scales from milliseconds to hours.

    Args:
        seconds: Time duration in seconds as a float. Can be fractional.

    Returns:
        A formatted string with appropriate units:
            - Milliseconds for durations < 1 second (e.g., "300ms")
            - Seconds with 1 decimal for 1s to 1 minute (e.g., "2.5s")
            - Minutes and seconds for 1 minute to 1 hour (e.g., "4m 15s")
            - Hours and minutes for durations ≥ 1 hour (e.g., "1h 3m")

    Example:
        >>> format_duration(0.5)
        '500ms'
        >>> format_duration(125.7)
        '2m 6s'
        >>> format_duration(3661)
        '1h 1m'
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    elif seconds < SECONDS_PER_HOUR:
        minutes = int(seconds // SECONDS_PER_MINUTE)
        remaining_seconds = seconds % SECONDS_PER_MINUTE
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // SECONDS_PER_HOUR)
        remaining_minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        return f"{hours}h {remaining_minutes}m"


def calculate_required_sample_size(
    effect_size: float, power: float = 0.8, alpha: float = 0.05, two_tailed: bool = True
) -> int:
    """Calculate required sample size for detecting a given effect size.

    Uses Cohen's power analysis for comparing two proportions to determine
    minimum sample size needed to achieve desired statistical power.

    Args:
        effect_size: Expected effect size (Cohen's h for proportions, d for means).
            Must be positive. Values > 5.0 are clamped to prevent overflow.
        power: Desired statistical power (probability of detecting true effect).
            Default is 0.8. Supported values: 0.8, 0.9, 0.95.
        alpha: Significance level (Type I error rate). Default is 0.05.
        two_tailed: Whether to use two-tailed test. Default is True.

    Returns:
        Minimum required sample size per group. Range: 5-1000 with 20% buffer applied.
        Returns 999 for negligible effect sizes where comparison is not meaningful.

    Note:
        - Based on standard power analysis formulas for proportion comparison
        - Includes 20% buffer for practical considerations
        - Handles edge cases: zero effect size returns maximum sample size
        - Clamps unrealistic effect sizes to prevent numerical overflow

    Example:
        >>> calculate_required_sample_size(0.5)  # Medium effect size
        64
        >>> calculate_required_sample_size(0.2, power=0.9)  # Small effect, high power
        192
    """
    logger.debug(
        f"Calculating sample size for effect_size={effect_size}, power={power}"
    )

    if effect_size <= EFFECT_SIZE_ZERO_THRESHOLD:
        logger.warning(
            "Effect size is essentially zero — statistical comparison not meaningful. "
            "Returning maximum practical sample size."
        )
        return 999

    if effect_size > MAX_REASONABLE_EFFECT_SIZE:
        logger.warning(
            f"Effect size {effect_size} is unrealistically large, clamping to 5.0"
        )
        effect_size = 5.0

    z_alpha = 1.96 if two_tailed else 1.645
    z_beta_map = {0.8: 0.84, 0.9: 1.28, 0.95: 1.645}
    z_beta = z_beta_map.get(power, 0.84)

    try:
        n_raw = ((z_alpha + z_beta) ** 2) / (effect_size**2)
        n_with_buffer = int(math.ceil(n_raw * 1.2))
        n_final = max(5, min(1000, n_with_buffer))

        logger.debug(f"Sample size calculation: raw={n_raw:.1f}, final={n_final}")
        return n_final

    except (OverflowError, ValueError) as e:
        logger.warning(f"Sample size calculation overflow: {e}")
        return 100


def calculate_effect_size(
    success_rate_1: float,
    success_rate_2: float,
    n1: int | None = None,
    n2: int | None = None,
) -> tuple[float, str]:
    """Calculate effect size between two success rates using Cohen's h.

    Computes Cohen's h effect size measure for comparing two proportions,
    with domain protection to prevent mathematical errors from extreme values.

    Args:
        success_rate_1: Success rate for group 1 as decimal (0.0-1.0).
        success_rate_2: Success rate for group 2 as decimal (0.0-1.0).
        n1: Sample size for group 1. Currently unused but reserved for future enhancements.
        n2: Sample size for group 2. Currently unused but reserved for future enhancements.

    Returns:
        Tuple containing:
            - effect_size: Absolute Cohen's h value (always positive)
            - interpretation: Human-readable effect size category:
                - "Negligible": < 0.2
                - "Small": 0.2 to < 0.5
                - "Medium": 0.5 to < 0.8
                - "Large": 0.8 to < 1.2
                - "Very Large": ≥ 1.2

    Note:
        - Input rates are clamped to [0.001, 0.999] to prevent asin domain errors
        - Uses arcsine transformation: h = 2 * (asin(√p2) - asin(√p1))
        - Returns (0.0, "Invalid") for calculation failures
        - Effect size interpretation based on Cohen's conventions

    Example:
        >>> calculate_effect_size(0.3, 0.8)
        (1.09, 'Large')
        >>> calculate_effect_size(0.45, 0.55)
        (0.20, 'Small')
    """
    sr1 = max(0.001, min(0.999, success_rate_1))
    sr2 = max(0.001, min(0.999, success_rate_2))

    try:
        h = 2 * (math.asin(math.sqrt(sr2)) - math.asin(math.sqrt(sr1)))
        h_abs = abs(h)

        if h_abs < EFFECT_SIZE_NEGLIGIBLE_THRESHOLD:
            interpretation = "Negligible"
        elif h_abs < EFFECT_SIZE_SMALL_THRESHOLD:
            interpretation = "Small"
        elif h_abs < EFFECT_SIZE_MEDIUM_THRESHOLD:
            interpretation = "Medium"
        elif h_abs < EFFECT_SIZE_LARGE_THRESHOLD:
            interpretation = "Large"
        else:
            interpretation = "Very Large"

        logger.debug(
            f"Effect size between {sr1:.1%} and {sr2:.1%}: {h_abs:.3f} ({interpretation})"
        )
        return h_abs, interpretation

    except (ValueError, ZeroDivisionError) as e:
        logger.warning(
            f"Effect size calculation failed for rates {success_rate_1:.3f}, {success_rate_2:.3f}: {e}"
        )

        return 0.0, "Invalid"


def assess_statistical_significance(
    results: dict[str, Any], min_effect_size: float = 0.3
) -> dict[str, Any]:
    """Assess whether benchmark results have sufficient statistical power.

    Analyzes benchmark results to determine if sample sizes are adequate for
    detecting meaningful differences between agents, providing recommendations
    for improving statistical validity.

    Args:
        results: Benchmark results dictionary containing agent performance metrics.
            Expected keys: agent names with nested dicts containing 'success_rate',
            'total_tasks', etc.
        min_effect_size: Minimum meaningful effect size to detect. Default is 0.3
            (small to medium effect per Cohen's conventions).

    Returns:
        Dict containing statistical assessment:
            - sufficient_power: Boolean indicating if overall power is adequate
            - recommendations: List of actionable recommendations for improvement
            - effect_sizes: Dict of pairwise effect size calculations
            - sample_adequacy: Dict of adequacy flags for each comparison
            - statistical_warnings: List of concerns about data validity
            - power_summary: Overall summary statistics

    Note:
        - Requires minimum 2 agents for meaningful comparison
        - Calculates all pairwise agent comparisons
        - Identifies very small effect sizes that may not be meaningful
        - Flags similar success rates that may need more challenging tasks
        - Minimum 80% power threshold for adequacy assessment

    Example:
        >>> assessment = assess_statistical_significance(benchmark_results)
        >>> if not assessment['sufficient_power']:
        ...     for rec in assessment['recommendations']:
        ...         print(f"Recommendation: {rec}")
    """
    assessment: dict[str, Any] = {
        "sufficient_power": True,
        "recommendations": [],
        "effect_sizes": {},
        "sample_adequacy": {},
        "statistical_warnings": [],
    }

    agents = list(results.keys())

    if len(agents) < MIN_AGENTS_FOR_COMPARISON:
        assessment["statistical_warnings"].append("Insufficient agents for comparison")
        return assessment

    for i, agent1 in enumerate(agents):
        for agent2 in agents[i + 1 :]:
            if agent1 in results and agent2 in results:
                rate1 = results[agent1].get("success_rate", 0)
                rate2 = results[agent2].get("success_rate", 0)
                n1 = results[agent1].get("total_tasks", 0)
                n2 = results[agent2].get("total_tasks", 0)

                effect_size, interpretation = calculate_effect_size(rate1, rate2)
                required_n = calculate_required_sample_size(effect_size)

                comparison_key = f"{agent1}_vs_{agent2}"
                assessment["effect_sizes"][comparison_key] = {
                    "effect_size": effect_size,
                    "interpretation": interpretation,
                    "rates": [rate1, rate2],
                    "sample_sizes": [n1, n2],
                    "required_sample_size": required_n,
                    "rate_difference": abs(rate2 - rate1),
                }

                min_n = min(n1, n2) if n1 and n2 else 0
                is_adequate = min_n >= required_n
                assessment["sample_adequacy"][comparison_key] = is_adequate

                if not is_adequate:
                    assessment["sufficient_power"] = False
                    assessment["recommendations"].append(
                        f"Increase iterations for {comparison_key}: "
                        f"{min_n} current vs {required_n} required "
                        f"(effect size: {effect_size:.3f} - {interpretation})"
                    )

                if effect_size < VERY_SMALL_EFFECT_THRESHOLD:
                    assessment["statistical_warnings"].append(
                        f"Very small effect size ({effect_size:.3f}) between {agent1} and {agent2} - "
                        "differences may not be practically meaningful"
                    )

                if abs(rate1 - rate2) < SIMILAR_SUCCESS_RATE_THRESHOLD:
                    assessment["statistical_warnings"].append(
                        f"Success rates very similar between {agent1} ({rate1:.1%}) and "
                        f"{agent2} ({rate2:.1%}) - consider testing with more challenging tasks"
                    )

    total_comparisons = len(assessment["effect_sizes"])
    adequate_comparisons = sum(assessment["sample_adequacy"].values())

    assessment["power_summary"] = {
        "total_comparisons": total_comparisons,
        "adequately_powered": adequate_comparisons,
        "power_rate": (
            adequate_comparisons / total_comparisons if total_comparisons > 0 else 0
        ),
    }

    if assessment["power_summary"]["power_rate"] < MINIMUM_STATISTICAL_POWER:
        assessment["recommendations"].append(
            f"Only {adequate_comparisons}/{total_comparisons} comparisons have adequate statistical power. "
            "Consider increasing iterations or using more challenging tasks to create larger effect sizes."
        )

    return assessment


def validate_benchmark_config(config: dict[str, Any]) -> bool:
    """Validate configuration dictionary for running benchmarks.

    Performs comprehensive validation of benchmark configuration parameters
    with informative error messages for debugging and user guidance.

    Args:
        config: Dictionary containing benchmark settings. Required keys:
            - iterations: Number of benchmark iterations (int)
            - tasks: Task definitions or count
            - agents: Agent definitions or count

    Returns:
        True if the configuration is valid and ready for benchmark execution.

    Raises:
        ValueError: If required keys are missing or values are invalid.
            Error messages include specific details about validation failures.

    Note:
        - Iterations must be between 1 and 1000 for performance safety
        - Validates presence of all required configuration keys
        - Provides specific error messages for each validation failure
        - Logs validation events for debugging and audit purposes

    Example:
        >>> config = {"iterations": 10, "tasks": ["task1"], "agents": ["agent1"]}
        >>> try:
        ...     validate_benchmark_config(config)
        ...     print("Configuration valid!")
        ... except ValueError as e:
        ...     print(f"Invalid config: {e}")
    """
    required_keys = ["iterations", "tasks", "agents"]

    for key in required_keys:
        if key not in config:
            logger.warning(f"Missing required configuration key: {key}")
            raise ValueError(f"Missing required configuration key: {key}")

    if config["iterations"] < 1:
        logger.warning("Iterations must be at least 1")
        raise ValueError("Iterations must be at least 1")

    if config["iterations"] > MAX_ITERATIONS:
        logger.warning(f"Iterations exceeds upper bound of {MAX_ITERATIONS}")
        raise ValueError(
            f"Iterations cannot exceed {MAX_ITERATIONS} (performance limit)"
        )

    logger.debug("Benchmark configuration validated successfully")
    return True


def calculate_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> dict[str, float]:
    """Compute confidence interval for a list of values using normal approximation.

    Calculates confidence intervals for the sample mean using standard normal
    approximation with support for common confidence levels.

    Args:
        values: A list of numeric values for confidence interval calculation.
            Must contain at least 2 values for meaningful results.
        confidence: Confidence level as decimal. Supported values: 0.95 (default), 0.99.
            Other values will use 0.95 z-score as fallback.

    Returns:
        Dictionary containing confidence interval information:
            - lower: Lower bound of the confidence interval
            - upper: Upper bound of the confidence interval
            - margin: Margin of error (half-width of interval)

    Note:
        - Uses normal approximation (not t-distribution) for simplicity
        - Requires minimum 2 samples; returns zeros for insufficient data
        - Standard error calculated as sample_std / sqrt(n)
        - Z-scores: 1.96 for 95% confidence, 2.576 for 99% confidence

    Example:
        >>> ci = calculate_confidence_interval([10, 12, 14, 16, 18])
        >>> print(f"95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    """
    if len(values) < MIN_CI_SAMPLES:
        logger.debug("Not enough data to compute confidence interval")
        return {"lower": 0.0, "upper": 0.0, "margin": 0.0}

    mean = statistics.mean(values)
    std_error = statistics.stdev(values) / math.sqrt(len(values))

    z_score = (
        CONFIDENCE_95_Z_SCORE if confidence == CONFIDENCE_95 else CONFIDENCE_99_Z_SCORE
    )

    margin = z_score * std_error

    logger.debug(
        f"Computed confidence interval (confidence={confidence}) for {len(values)} values"
    )
    return {"lower": mean - margin, "upper": mean + margin, "margin": margin}


def detect_outliers(values: list[float], method: str = "iqr") -> list[int]:
    """Detect outlier indices in a numeric list using IQR or Z-score method.

    Identifies outliers using either interquartile range (IQR) or Z-score methods
    with standard statistical thresholds for outlier classification.

    Args:
        values: List of numeric values to analyze for outliers.
        method: Outlier detection method. Options:
            - "iqr": Interquartile range method (default)
            - "zscore": Z-score method with threshold of 3

    Returns:
        List of zero-based indices corresponding to outlier values in the original list.
        Empty list if no outliers detected or insufficient data.

    Raises:
        ValueError: If an unknown method is specified.

    Note:
        IQR Method:
        - Uses 1.5×IQR fence: outliers are < Q1-1.5×IQR or > Q3+1.5×IQR
        - Quartiles calculated using simple index approximation

        Z-score Method:
        - Uses threshold of 3 standard deviations from mean
        - More sensitive to distribution assumptions than IQR

        Both methods require minimum 4 samples for reliable detection.

    Example:
        >>> outliers = detect_outliers([1, 2, 3, 4, 100], method="iqr")
        >>> print(f"Outlier indices: {outliers}")  # [4]
    """
    if len(values) < MIN_OUTLIER_SAMPLES:
        logger.debug("Not enough values to detect outliers")
        return []

    if method == "iqr":
        sorted_values = sorted(values)
        n = len(values)
        q1 = sorted_values[n // MIN_QUARTILE_SAMPLES]
        q3 = sorted_values[3 * n // MIN_QUARTILE_SAMPLES]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [
            i
            for i, value in enumerate(values)
            if value < lower_bound or value > upper_bound
        ]

        logger.debug(f"Detected {len(outliers)} outliers using IQR")
        return outliers

    elif method == "zscore":
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        outliers = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0
            if z_score > ZSCORE_OUTLIER_THRESHOLD:
                outliers.append(i)

        logger.debug(f"Detected {len(outliers)} outliers using Z-score")
        return outliers

    else:
        logger.warning(f"Unknown outlier detection method: {method}")
        raise ValueError(f"Unknown outlier detection method: {method}")


def generate_benchmark_id() -> str:
    """Generate a timestamp-based unique identifier for a benchmark run.

    Creates a unique identifier incorporating microsecond-precision timestamp
    for tracking and organizing benchmark executions.

    Returns:
        A string identifier in format "benchmark_{microsecond_timestamp}".
        Microsecond precision ensures uniqueness even for rapid successive calls.

    Note:
        - Uses current time in microseconds since epoch
        - Format: "benchmark_" + 16-digit timestamp
        - Suitable for filenames and database keys
        - Logs generated ID for debugging and tracking

    Example:
        >>> bench_id = generate_benchmark_id()
        >>> print(bench_id)
        'benchmark_1704067200123456'
    """
    timestamp = int(time.time() * 1000000)
    benchmark_id = f"benchmark_{timestamp}"
    logger.debug(f"Generated benchmark ID: {benchmark_id}")
    return benchmark_id


def format_file_size(bytes_size: float) -> str:
    """Convert raw file size into human-readable format.

    Converts byte counts to appropriate units using binary prefixes (1024-based)
    with intelligent unit selection and decimal precision.

    Args:
        bytes_size: File size in bytes as float. Can be fractional.

    Returns:
        A string formatted with appropriate unit (B, KB, MB, GB, TB).
        Uses 1 decimal place precision for values ≥ 1 in each unit.

    Note:
        - Uses binary units (1024 bytes = 1 KB) not decimal (1000)
        - Automatically selects most appropriate unit for readability
        - Supports up to terabytes with consistent formatting
        - Handles fractional byte sizes appropriately

    Example:
        >>> format_file_size(1536)
        '1.5 KB'
        >>> format_file_size(5242880)
        '5.0 MB'
        >>> format_file_size(500)
        '500.0 B'
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < BYTES_PER_KILOBYTE:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= BYTES_PER_KILOBYTE

    return f"{bytes_size:.1f} TB"


def create_results_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Generate a compact summary report from benchmark results.

    Creates executive-level summary with key insights and performance gaps
    across all tested agents for high-level reporting and analysis.

    Args:
        results: Dictionary containing agent performance results. Expected structure:
            {agent_name: {success_rate: float, cost_per_success: float, ...}, ...}

    Returns:
        Dictionary containing summary statistics and metadata:
            - timestamp: When summary was generated
            - total_agents_tested: Count of agents in analysis
            - overall_insights: Dict with key performance metrics including:
                - best_success_rate: Highest success rate achieved
                - worst_success_rate: Lowest success rate
                - avg_success_rate: Average across all agents
                - most_cost_effective: Lowest cost per success
                - least_cost_effective: Highest cost per success
                - performance_gap: Difference between best and worst performance

    Note:
        - Handles empty results gracefully with appropriate logging
        - Timestamp in Unix epoch format for consistent sorting
        - Performance gap highlights architectural impact of different agent types
        - Cost analysis excludes infinite values from failed agents

    Example:
        >>> summary = create_results_summary(benchmark_results)
        >>> gap = summary['overall_insights']['performance_gap']
        >>> print(f"Performance gap: {gap:.1%}")
    """
    summary = {
        "timestamp": time.time(),
        "total_agents_tested": len(results),
        "overall_insights": {},
    }

    if results:
        success_rates = [data["success_rate"] for data in results.values()]
        costs = [data["cost_per_success"] for data in results.values()]

        summary["overall_insights"] = {
            "best_success_rate": max(success_rates),
            "worst_success_rate": min(success_rates),
            "avg_success_rate": statistics.mean(success_rates),
            "most_cost_effective": min(costs),
            "least_cost_effective": max(costs),
            "performance_gap": max(success_rates) - min(success_rates),
        }

        logger.debug(f"Generated results summary for {len(results)} agents")
    else:
        logger.debug("No results provided to create_results_summary()")

    return summary


def calculate_enhanced_confidence_intervals(
    values: list[float], confidence: float = 0.95, method: str = "auto"
) -> dict[str, Any]:
    """Enhanced confidence interval calculation with multiple methods.

    Provides advanced confidence interval calculation with automatic method
    selection and fallback handling for missing statistical dependencies.

    Args:
        values: List of numeric values for confidence interval calculation.
        confidence: Confidence level as decimal (default 0.95 for 95%).
        method: Calculation method. Options:
            - "auto": Automatic selection based on sample size (default)
            - "t": Student's t-distribution (recommended for n < 30)
            - "normal": Normal approximation (suitable for n ≥ 30)
            - "bootstrap": Bootstrap resampling (requires numpy)

    Returns:
        Dict containing confidence interval and method information:
            - lower: Lower bound of confidence interval
            - upper: Upper bound of confidence interval
            - margin: Margin of error (half-width)
            - method: Actual method used (may differ from requested due to fallbacks)

    Note:
        Auto method selection:
        - Uses t-distribution for sample sizes < 30
        - Uses normal approximation for sample sizes ≥ 30

        Fallback behavior:
        - Falls back to normal approximation if scipy unavailable for t-distribution
        - Falls back to normal if numpy unavailable for bootstrap
        - Returns zeros with "insufficient_data" for < 2 samples

    Example:
        >>> ci = calculate_enhanced_confidence_intervals([1,2,3,4,5], method="t")
        >>> print(f"Method: {ci['method']}, CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    """
    if len(values) < MIN_CI_SAMPLES:
        return {
            "lower": 0.0,
            "upper": 0.0,
            "margin": 0.0,
            "method": "insufficient_data",
        }

    n = len(values)
    mean_val = statistics.mean(values)

    if method == "auto":
        method = "t" if n < T_DISTRIBUTION_THRESHOLD else "normal"

    if method == "t":
        if HAS_SCIPY:
            std_error = statistics.stdev(values) / math.sqrt(n)
            t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
            margin = t_critical * std_error
        else:
            std_error = statistics.stdev(values) / math.sqrt(n)
            z_critical = (
                CONFIDENCE_95_Z_SCORE
                if confidence == CONFIDENCE_95
                else CONFIDENCE_99_Z_SCORE
            )
            margin = z_critical * std_error
            method = "normal_fallback"

    elif method == "normal":
        std_error = statistics.stdev(values) / math.sqrt(n)
        z_critical = (
            CONFIDENCE_95_Z_SCORE
            if confidence == CONFIDENCE_95
            else CONFIDENCE_99_Z_SCORE
        )
        margin = z_critical * std_error

    elif method == "bootstrap":
        if HAS_NUMPY:
            bootstrap_means = []
            for _ in range(1000):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))

            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower = np.percentile(bootstrap_means, lower_percentile)
            upper = np.percentile(bootstrap_means, upper_percentile)
            margin = (upper - lower) / 2

            return {
                "lower": float(lower),
                "upper": float(upper),
                "margin": float(margin),
                "method": "bootstrap",
            }
        else:
            return calculate_enhanced_confidence_intervals(values, confidence, "normal")

    return {
        "lower": mean_val - margin,
        "upper": mean_val + margin,
        "margin": margin,
        "method": method,
    }


def detect_performance_anomalies(
    agent_results: dict[str, Any], sensitivity: float = 2.0
) -> dict[str, Any]:
    """Detect performance anomalies across agent types.

    Identifies agents with unusual performance patterns using Z-score analysis
    and provides recommendations for investigation and improvement.

    Args:
        agent_results: Dictionary of agent performance metrics. Expected structure:
            {agent_type: {success_rate: float, cost_per_success: float, ...}, ...}
        sensitivity: Z-score threshold for anomaly detection. Default is 2.0
            (approximately 95% confidence). Higher values = less sensitive.

    Returns:
        Dictionary containing anomaly analysis:
            - detected_anomalies: List of anomalous agents with details:
                - agent_type: Name of anomalous agent
                - metric: Which metric triggered detection
                - value: Actual metric value
                - z_score: Standard deviations from mean
                - severity: "high" (z > 3) or "moderate" (2 < z ≤ 3)
            - performance_warnings: List of specific performance concerns
            - recommendations: List of actionable improvement suggestions

    Note:
        - Requires minimum 2 agents for meaningful comparison
        - Focuses on success rate anomalies as primary indicator
        - Identifies agents with zero successes (infinite cost per success)
        - Flags very low success rates (< 30%) for special attention
        - Severity levels help prioritize investigation efforts

    Example:
        >>> anomalies = detect_performance_anomalies(results, sensitivity=2.5)
        >>> for anomaly in anomalies['detected_anomalies']:
        ...     print(f"Anomaly: {anomaly['agent_type']} - {anomaly['severity']}")
    """
    anomalies: dict[str, Any] = {
        "detected_anomalies": [],
        "performance_warnings": [],
        "recommendations": [],
    }

    success_rates = {}
    for agent_type, metrics in agent_results.items():
        if isinstance(metrics, dict) and "success_rate" in metrics:
            success_rates[agent_type] = metrics["success_rate"]

    if len(success_rates) < MIN_CI_SAMPLES:
        return anomalies

    rates = list(success_rates.values())
    mean_rate = statistics.mean(rates)
    std_rate = statistics.stdev(rates) if len(rates) > 1 else 0

    if std_rate > 0:
        for agent_type, rate in success_rates.items():
            z_score = abs((rate - mean_rate) / std_rate)

            if z_score > sensitivity:
                anomalies["detected_anomalies"].append(
                    {
                        "agent_type": agent_type,
                        "metric": "success_rate",
                        "value": rate,
                        "z_score": z_score,
                        "severity": (
                            "high" if z_score > HIGH_ANOMALY_THRESHOLD else "moderate"
                        ),
                    }
                )

    for agent_type, metrics in agent_results.items():
        if isinstance(metrics, dict):
            success_rate = metrics.get("success_rate", 0)
            cost_per_success = metrics.get("cost_per_success", 0)

            if success_rate < LOW_SUCCESS_RATE_THRESHOLD:
                anomalies["performance_warnings"].append(
                    f"{agent_type}: Very low success rate ({success_rate:.1%})"
                )

            if cost_per_success == float("inf"):
                anomalies["performance_warnings"].append(
                    f"{agent_type}: No successful completions - infinite cost"
                )

    if anomalies["detected_anomalies"]:
        anomalies["recommendations"].append(
            "Investigate agents with anomalous performance patterns"
        )

    if len(anomalies["performance_warnings"]) > 0:
        anomalies["recommendations"].append(
            "Consider architectural improvements for poorly performing agents"
        )

    return anomalies


def create_challenging_task_set() -> list[dict[str, Any]]:
    """Create enhanced task set designed for realistic performance differentiation.

    Generates a balanced set of tasks across complexity levels designed to prevent
    artificial perfect success rates and create meaningful performance differences
    between agent architectures.

    Returns:
        List of task dictionaries, each containing:
            - name: Descriptive task name
            - complexity: TaskComplexity enum value
            - description: Detailed description of task requirements

    Note:
        Task complexity distribution:
        - 2 Simple tasks: Basic functionality testing
        - 2 Moderate tasks: Intermediate capabilities
        - 3 Complex tasks: Advanced reasoning and tool use
        - 1 Enterprise task: Production-level complexity

        Tasks specifically designed to:
        - Challenge wrapper agents with multi-step requirements
        - Test context retention and memory management
        - Require planning and coordination capabilities
        - Include error recovery and state management
        - Expose architectural differences between agent types

    Example:
        >>> tasks = create_challenging_task_set()
        >>> for task in tasks:
        ...     print(f"{task['name']}: {task['complexity'].value}")
    """
    return [
        {
            "name": "Multi-step calculation with validation",
            "complexity": TaskComplexity.SIMPLE,
            "description": "Simple task that still challenges wrapper agents",
        },
        {
            "name": "Context-dependent analysis with references",
            "complexity": TaskComplexity.MODERATE,
            "description": "Moderate task requiring context retention",
        },
        {
            "name": "Complex reasoning with tool coordination",
            "complexity": TaskComplexity.COMPLEX,
            "description": "Complex task that tests planning and execution",
        },
        {
            "name": "Multi-system integration with error recovery",
            "complexity": TaskComplexity.ENTERPRISE,
            "description": "Enterprise task that challenges even real agents",
        },
        {
            "name": "Edge case handling in data processing",
            "complexity": TaskComplexity.COMPLEX,
            "description": "Designed to test robustness and recovery mechanisms",
        },
        {
            "name": "Sequential dependency resolution",
            "complexity": TaskComplexity.MODERATE,
            "description": "Moderate task testing dependency management",
        },
        {
            "name": "Error recovery with state restoration",
            "complexity": TaskComplexity.ENTERPRISE,
            "description": "Enterprise task requiring sophisticated recovery",
        },
        {
            "name": "Context switching with memory retention",
            "complexity": TaskComplexity.COMPLEX,
            "description": "Complex task testing memory and context management",
        },
    ]


@dataclass
class EnhancedTaskResult(TaskResult):
    """Enhanced dictionary conversion with new fields.

    Converts EnhancedTaskResult to dictionary format including all base TaskResult
    fields plus enhanced tracking fields for enterprise deployments.

    Returns:
        Dict containing complete task result data including:
            - All base TaskResult fields (task_id, success, execution_time, etc.)
            - tool_failures: List of tool failure events with details
            - network_latency_avg: Average network latency during execution
            - network_condition: Network condition during execution
            - recovery_attempts: Number of retry/recovery attempts made
            - confidence_intervals: Statistical confidence interval data
            - outlier_detected: Boolean flag for outlier classification
            - performance_anomaly: Boolean flag for performance anomaly
            - enhanced_result: Boolean flag indicating enhanced result format

    Note:
        - Maintains compatibility with base TaskResult.to_dict() format
        - Enhanced fields default to appropriate empty/zero values
        - Used for advanced monitoring and analysis in enterprise deployments
        - Supports detailed post-execution analysis and debugging
    """

    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    network_latency_avg: float = 0.0
    network_condition: str = "stable"
    recovery_attempts: int = 0
    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    outlier_detected: bool = False
    performance_anomaly: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Enhanced dictionary conversion with new fields."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "tool_failures": self.tool_failures,
                "network_latency_avg": self.network_latency_avg,
                "network_condition": self.network_condition,
                "recovery_attempts": self.recovery_attempts,
                "confidence_intervals": self.confidence_intervals,
                "outlier_detected": self.outlier_detected,
                "performance_anomaly": self.performance_anomaly,
                "enhanced_result": True,
            }
        )
        return base_dict


class ComplexityDimension(Enum):
    """Extended complexity dimensions beyond basic task complexity.

    Defines additional dimensions of task complexity that affect agent performance
    beyond the basic SIMPLE/MODERATE/COMPLEX/ENTERPRISE classification.

    Attributes:
        TOOL_RELIABILITY: Challenges related to tool failure and recovery
        NETWORK_CONDITIONS: Impact of network latency and reliability
        COORDINATION_OVERHEAD: Complexity from multi-agent coordination
        MEMORY_REQUIREMENTS: Challenges from memory and context management
        RECOVERY_DIFFICULTY: Complexity of error recovery and state restoration

    Note:
        - Used in enhanced benchmarking modes for detailed analysis
        - Provides granular insight into specific failure modes
        - Enables targeted performance optimization recommendations
        - Supports advanced agent architecture evaluation
    """

    TOOL_RELIABILITY = "tool_reliability"
    NETWORK_CONDITIONS = "network_conditions"
    COORDINATION_OVERHEAD = "coordination"
    MEMORY_REQUIREMENTS = "memory"
    RECOVERY_DIFFICULTY = "recovery"
