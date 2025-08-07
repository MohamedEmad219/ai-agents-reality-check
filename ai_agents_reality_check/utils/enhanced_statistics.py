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

Module: ai_agents_reality_check/utils/enhanced_statistics.py

Enhanced statistical analysis with confidence intervals, effect size interpretations,
and advanced reporting for CLI output and research documentation.

This module provides comprehensive statistical analysis capabilities for AI agent
benchmarking results, including robust error handling, confidence interval calculations,
normality testing, outlier detection, and detailed interpretation of statistical measures.
Designed for enterprise-grade benchmarking with production-ready error handling and logging.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import math
import statistics
import warnings
from typing import Any

import numpy as np
from scipy import stats

from ai_agents_reality_check.logging_config import logger

# Constants for magic values
MIN_SAMPLES_FOR_QUARTILES = 4
MIN_SAMPLES_FOR_PERCENTILES = 20
MIN_SAMPLES_FOR_SKEWNESS = 3
MIN_SAMPLES_FOR_KURTOSIS = 4
MIN_SAMPLES_FOR_NORMALITY = 3
LARGE_SAMPLE_THRESHOLD = 5000
NORMALITY_ALPHA = 0.05
MIN_SAMPLES_FOR_CI = 2
T_DISTRIBUTION_THRESHOLD = 30
MIN_SAMPLES_FOR_OUTLIERS = 4
LOW_VARIABILITY_THRESHOLD = 0.1
MODERATE_VARIABILITY_THRESHOLD = 0.3
HIGH_VARIABILITY_THRESHOLD = 0.5
SYMMETRIC_SKEW_THRESHOLD = 0.5
POSITIVE_SKEW_THRESHOLD = 0.1
NEGATIVE_SKEW_THRESHOLD = -0.1


class StatisticalAnalysis:
    """Enhanced statistical analysis for agent benchmarking results."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical analysis with specified confidence level.

        Args:
            confidence_level: Confidence level for interval calculations. Must be between 0.8 and 0.99.
                Default is 0.95 (95% confidence intervals).

        Raises:
            ValueError: If confidence_level is outside valid range [0.8, 0.99].
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_enhanced_statistics(
        self, values: list[float], metric_name: str = "metric"
    ) -> dict[str, Any]:
        """Calculate comprehensive statistics with robust edge case handling.

        Computes descriptive statistics, confidence intervals, normality tests, outlier detection,
        and statistical interpretations for agent performance metrics. Handles edge cases including
        empty datasets, zero variance, and extreme values with appropriate fallbacks.

        Args:
            values: List of numeric values to analyze. Can be empty or contain identical values.
            metric_name: Human-readable name for the metric being analyzed. Used in logging
                and error messages. Default is "metric".

        Returns:
            Dict containing comprehensive statistical analysis:
                - Basic statistics: n, mean, median, std_dev, min, max, q25, q75, p05, p95
                - Derived measures: coefficient_of_variation, skewness, kurtosis
                - Confidence intervals: mean and median intervals with methods used
                - Distribution analysis: normality test results, outlier detection
                - Interpretations: Human-readable descriptions of variability and distribution shape

        Example:
            >>> analyzer = StatisticalAnalysis(confidence_level=0.95)
            >>> results = analyzer.calculate_enhanced_statistics([1.2, 1.5, 1.8, 2.1, 1.9])
            >>> print(f"Mean: {results['mean']:.2f}, CI: [{results['confidence_intervals']['mean']['lower']:.2f}, {results['confidence_intervals']['mean']['upper']:.2f}]")
        """
        if not values:
            logger.debug(f"Empty value list for {metric_name}")
            return self._empty_stats_dict()

        n = len(values)
        sorted_values = sorted(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)

        if len(set(values)) == 1:
            std_dev = 0.0
            cv = 0.0
            skewness = 0.0
            kurtosis = 0.0
            is_normal = True
            normality_p_value = 1.0
            outliers: list[int] = []

            logger.debug(
                f"Zero variance detected for {metric_name} - all values identical"
            )
        else:
            std_dev = statistics.stdev(values) if n > 1 else 0.0
            cv = (std_dev / mean_val) if mean_val != 0 else 0.0
            skewness, kurtosis = self._calculate_robust_moments(values)
            normality_result = self._test_normality_robust(values)
            is_normal = bool(normality_result["is_normal"])
            normality_p_value = float(normality_result["p_value"])
            outliers = self._detect_outliers_iqr(values)

        ci_mean = self._calculate_mean_confidence_interval_robust(values)
        ci_median = self._bootstrap_median_confidence_interval_robust(values)

        if n >= MIN_SAMPLES_FOR_QUARTILES:
            q25 = float(np.percentile(values, 25))
            q75 = float(np.percentile(values, 75))
        else:
            q25 = float(sorted_values[0])
            q75 = float(sorted_values[-1])

        if n >= MIN_SAMPLES_FOR_PERCENTILES:
            p95 = float(np.percentile(values, 95))
            p05 = float(np.percentile(values, 5))
        else:
            p95 = float(sorted_values[-1])
            p05 = float(sorted_values[0])

        logger.debug(f"Enhanced statistics calculated for {metric_name} (n={n})")

        return {
            "n": n,
            "mean": mean_val,
            "median": median_val,
            "std_dev": std_dev,
            "min": min(values),
            "max": max(values),
            "q25": q25,
            "q75": q75,
            "p05": p05,
            "p95": p95,
            "coefficient_of_variation": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "confidence_intervals": {
                "mean": ci_mean,
                "median": ci_median,
                "level": self.confidence_level,
            },
            "distribution_analysis": {
                "is_normal": is_normal,
                "normality_p_value": normality_p_value,
                "outlier_count": len(outliers),
                "outlier_indices": outliers,
                "zero_variance": len(set(values)) == 1,
            },
            "interpretation": self._interpret_statistics(
                mean_val, std_dev, cv, skewness
            ),
        }

    def _calculate_robust_moments(self, values: list[float]) -> tuple[float, float]:
        """Calculate skewness and kurtosis with robust error handling.

        Uses scipy.stats for moment calculations with comprehensive error handling for
        edge cases including zero variance, insufficient data, and numerical precision issues.

        Args:
            values: List of numeric values for moment calculation.

        Returns:
            Tuple of (skewness, kurtosis) as floats. Returns (0.0, 0.0) for edge cases
            or calculation failures.

        Note:
            - Requires minimum 3 samples for skewness calculation
            - Requires minimum 4 samples for kurtosis calculation
            - Handles precision loss and catastrophic cancellation warnings
            - Returns 0.0 for non-finite results
        """
        try:
            if len(set(values)) <= 1:
                return 0.0, 0.0

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Precision loss occurred.*catastrophic cancellation",
                )
                warnings.filterwarnings("ignore", message="invalid value encountered")

                if len(values) >= MIN_SAMPLES_FOR_SKEWNESS:
                    skewness_val = float(stats.skew(values))
                    if not np.isfinite(skewness_val):
                        skewness_val = 0.0
                else:
                    skewness_val = 0.0

                if len(values) >= MIN_SAMPLES_FOR_KURTOSIS:
                    kurtosis_val = float(stats.kurtosis(values))
                    if not np.isfinite(kurtosis_val):
                        kurtosis_val = 0.0
                else:
                    kurtosis_val = 0.0

            return skewness_val, kurtosis_val

        except Exception as e:
            logger.debug(f"Moment calculation failed, using defaults: {e}")
            return 0.0, 0.0

    def _test_normality_robust(
        self, values: list[float]
    ) -> dict[str, bool | float | str]:
        """Test for normality with robust handling of edge cases.

        Performs normality testing using either Shapiro-Wilk (small samples) or
        Kolmogorov-Smirnov (large samples) tests with comprehensive error handling.

        Args:
            values: List of numeric values to test for normality.

        Returns:
            Dict containing normality test results:
                - is_normal: Boolean indicating if data appears normally distributed
                - p_value: P-value from the statistical test
                - statistic: Test statistic value (if available)
                - test: Name of test used ("shapiro_wilk", "kolmogorov_smirnov", etc.)

        Note:
            - Uses Shapiro-Wilk for samples ≤ 5000
            - Uses Kolmogorov-Smirnov for samples > 5000
            - Returns insufficient_data for < 3 samples
            - Returns zero_variance for identical values
            - Alpha level of 0.05 for significance testing
        """
        if len(values) < MIN_SAMPLES_FOR_NORMALITY:
            return {"is_normal": False, "p_value": 0.0, "test": "insufficient_data"}

        if len(set(values)) == 1:
            return {"is_normal": True, "p_value": 1.0, "test": "zero_variance"}

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*Input data has range zero.*"
                )
                warnings.filterwarnings(
                    "ignore", message=".*Results may not be accurate.*"
                )

                if len(values) > LARGE_SAMPLE_THRESHOLD:
                    statistic, p_value = stats.kstest(
                        values, "norm", args=(np.mean(values), np.std(values))
                    )
                    test_name = "kolmogorov_smirnov"
                else:
                    statistic, p_value = stats.shapiro(values)
                    test_name = "shapiro_wilk"

                if not np.isfinite(p_value):
                    p_value = 0.0

                is_normal = p_value > NORMALITY_ALPHA

        except Exception as e:
            logger.debug(f"Normality test failed: {e}")
            return {"is_normal": False, "p_value": 0.0, "test": "failed"}

        return {
            "is_normal": is_normal,
            "p_value": float(p_value),
            "statistic": float(statistic) if np.isfinite(statistic) else 0.0,
            "test": test_name,
        }

    def _calculate_mean_confidence_interval_robust(
        self, values: list[float]
    ) -> dict[str, float | str]:
        """Calculate confidence interval with robust edge case handling.

        Computes confidence intervals for the mean using either t-distribution (small samples)
        or normal approximation (large samples) with comprehensive error handling for edge cases.

        Args:
            values: List of numeric values for confidence interval calculation.

        Returns:
            Dict containing confidence interval information:
                - lower: Lower bound of confidence interval
                - upper: Upper bound of confidence interval
                - margin: Margin of error (half-width of interval)
                - method: Method used ("t_distribution", "normal_approximation", etc.)

        Note:
            - Uses t-distribution for samples < 30
            - Uses normal approximation for samples ≥ 30
            - Handles insufficient data (< 2 samples) and zero variance
            - Returns point estimates for edge cases
        """
        """Calculate confidence interval with robust edge case handling."""
        if len(values) < MIN_SAMPLES_FOR_CI:
            return {
                "lower": 0.0,
                "upper": 0.0,
                "margin": 0.0,
                "method": "insufficient_data",
            }

        if len(set(values)) == 1:
            mean_val = values[0]
            return {
                "lower": mean_val,
                "upper": mean_val,
                "margin": 0.0,
                "method": "zero_variance",
            }

        try:
            n = len(values)
            mean_val = statistics.mean(values)
            std_error = statistics.stdev(values) / math.sqrt(n)

            if n < T_DISTRIBUTION_THRESHOLD:
                t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df=n - 1)
                margin = t_critical * std_error
                method = "t_distribution"
            else:
                z_critical = stats.norm.ppf((1 + self.confidence_level) / 2)
                margin = z_critical * std_error
                method = "normal_approximation"

            return {
                "lower": float(mean_val - margin),
                "upper": float(mean_val + margin),
                "margin": float(margin),
                "method": method,
            }

        except Exception as e:
            logger.debug(f"Confidence interval calculation failed: {e}")
            mean_val = statistics.mean(values)
            return {
                "lower": mean_val,
                "upper": mean_val,
                "margin": 0.0,
                "method": "fallback",
            }

    def _bootstrap_median_confidence_interval_robust(
        self, values: list[float], n_bootstrap: int = 1000
    ) -> dict[str, float | str]:
        """Calculate bootstrap confidence interval with robust handling.

        Uses bootstrap resampling to estimate confidence intervals for the median,
        providing robust estimates even for non-normal distributions.

        Args:
            values: List of numeric values for bootstrap confidence interval.
            n_bootstrap: Number of bootstrap samples to generate. Default is 1000.

        Returns:
            Dict containing bootstrap confidence interval:
                - lower: Lower bound from bootstrap percentile method
                - upper: Upper bound from bootstrap percentile method
                - margin: Half-width of confidence interval
                - method: "bootstrap" or fallback method used

        Note:
            - Uses percentile method for bootstrap confidence intervals
            - Handles insufficient data and zero variance cases
            - Falls back to point estimates on calculation failures
            - Bootstrap samples with replacement from original data
        """
        if len(values) < MIN_SAMPLES_FOR_CI:
            return {
                "lower": 0.0,
                "upper": 0.0,
                "margin": 0.0,
                "method": "insufficient_data",
            }

        if len(set(values)) == 1:
            median_val = values[0]
            return {
                "lower": median_val,
                "upper": median_val,
                "margin": 0.0,
                "method": "zero_variance",
            }

        try:
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_medians.append(np.median(sample))

            alpha_half = self.alpha / 2
            lower_percentile = (alpha_half) * 100
            upper_percentile = (1 - alpha_half) * 100

            lower = np.percentile(bootstrap_medians, lower_percentile)
            upper = np.percentile(bootstrap_medians, upper_percentile)

            return {
                "lower": float(lower),
                "upper": float(upper),
                "margin": float((upper - lower) / 2),
                "method": "bootstrap",
            }

        except Exception as e:
            logger.debug(f"Bootstrap confidence interval failed: {e}")
            median_val = statistics.median(values)
            return {
                "lower": median_val,
                "upper": median_val,
                "margin": 0.0,
                "method": "fallback",
            }

    def _detect_outliers_iqr(self, values: list[float]) -> list[int]:
        """Detect outliers using IQR method with edge case handling.

        Identifies outliers using the interquartile range (IQR) method with 1.5×IQR fences.
        Robust handling for edge cases including insufficient data and zero variance.

        Args:
            values: List of numeric values to check for outliers.

        Returns:
            List of indices corresponding to outlier values in the original list.
            Empty list if no outliers found or insufficient data.

        Note:
            - Uses standard 1.5×IQR outlier definition
            - Requires minimum 4 samples for meaningful detection
            - Returns empty list for zero variance (all identical values)
            - Outliers are values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
        """
        if len(values) < MIN_SAMPLES_FOR_OUTLIERS or len(set(values)) == 1:
            return []

        try:
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)
            iqr = q75 - q25

            if iqr == 0:
                return []

            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outliers = [
                i
                for i, value in enumerate(values)
                if value < lower_bound or value > upper_bound
            ]

            return outliers

        except Exception as e:
            logger.debug(f"Outlier detection failed: {e}")
            return []

    def _interpret_statistics(
        self, mean: float, std_dev: float, cv: float, skewness: float
    ) -> dict[str, str]:
        """Provide interpretations of statistical measures.

        Generates human-readable interpretations of statistical measures for use in reports
        and analysis summaries. Handles edge cases including zero variance.

        Args:
            mean: Sample mean value.
            std_dev: Standard deviation.
            cv: Coefficient of variation.
            skewness: Skewness measure.

        Returns:
            Dict containing interpretations:
                - variability: Description of data variability level
                - distribution_shape: Description of distribution symmetry/skewness
                - skew_direction: Direction and magnitude of skew

        Note:
            - Variability categories: Low (<0.1), Moderate (<0.3), High (<0.5), Very High (≥0.5)
            - Skewness categories: Symmetric (<0.5), Moderate (0.5-1.0), High (>1.0)
            - Special handling for zero variance case
        """
        interpretations = {}

        if std_dev == 0:
            interpretations["variability"] = "No variability (all values identical)"
            interpretations["distribution_shape"] = "Degenerate (single value)"
            interpretations["skew_direction"] = "No skew (no variance)"
            return interpretations

        if cv < LOW_VARIABILITY_THRESHOLD:
            interpretations["variability"] = "Low variability"
        elif cv < MODERATE_VARIABILITY_THRESHOLD:
            interpretations["variability"] = "Moderate variability"
        elif cv < HIGH_VARIABILITY_THRESHOLD:
            interpretations["variability"] = "High variability"
        else:
            interpretations["variability"] = "Very high variability"

        if abs(skewness) < SYMMETRIC_SKEW_THRESHOLD:
            interpretations["distribution_shape"] = "Approximately symmetric"
        elif abs(skewness) < 1.0:
            interpretations["distribution_shape"] = "Moderately skewed"
        else:
            interpretations["distribution_shape"] = "Highly skewed"

        if skewness > POSITIVE_SKEW_THRESHOLD:
            interpretations["skew_direction"] = "Right-skewed (positive)"
        elif skewness < NEGATIVE_SKEW_THRESHOLD:
            interpretations["skew_direction"] = "Left-skewed (negative)"
        else:
            interpretations["skew_direction"] = "No skew"

        return interpretations

    def _empty_stats_dict(self) -> dict[str, Any]:
        """Return empty statistics dictionary for edge cases.

        Provides a properly structured statistics dictionary with zero/default values
        for cases where no data is available for analysis.

        Returns:
            Dict with complete statistics structure using default/zero values.
            Maintains consistency with calculate_enhanced_statistics return format.

        Note:
            - Used when input data is empty or invalid
            - Ensures consistent return structure for downstream processing
            - All numeric values set to 0.0, boolean values to False
            - Method indicators set to "no_data"
        """
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "coefficient_of_variation": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "confidence_intervals": {
                "mean": {
                    "lower": 0.0,
                    "upper": 0.0,
                    "margin": 0.0,
                    "method": "no_data",
                },
                "median": {
                    "lower": 0.0,
                    "upper": 0.0,
                    "margin": 0.0,
                    "method": "no_data",
                },
                "level": self.confidence_level,
            },
            "distribution_analysis": {
                "is_normal": False,
                "normality_p_value": 0.0,
                "outlier_count": 0,
                "outlier_indices": [],
                "zero_variance": False,
            },
            "interpretation": {
                "variability": "No data",
                "distribution_shape": "No data",
                "skew_direction": "No data",
            },
        }


def _generate_report_header(confidence_level: float) -> list[str]:
    """Generate the report header section.

    Creates formatted header lines for enhanced statistical analysis reports
    with confidence level information.

    Args:
        confidence_level: Confidence level used in analysis (e.g., 0.95 for 95%).

    Returns:
        List of formatted header lines ready for report output.

    Note:
        - Formats confidence level as percentage (e.g., "95%" from 0.95)
        - Includes separator line and spacing for visual formatting
        - Designed for console and file output compatibility
    """
    return [
        f"\nEnhanced Statistical Analysis (CI: {confidence_level:.0%})",
        "=" * 60,
        "",
    ]


def _format_agent_metrics(agent_type: str, metrics: dict[str, Any]) -> list[str]:
    """Format individual agent metrics section.

    Formats performance metrics for a single agent type into human-readable
    report lines with proper formatting and units.

    Args:
        agent_type: Agent type identifier (e.g., "wrapper_agent").
        metrics: Dict containing agent performance metrics and statistics.

    Returns:
        List of formatted lines for the agent metrics section.

    Note:
        - Converts underscored names to title case (e.g., "Wrapper Agent")
        - Includes basic metrics: success rate, execution time, cost, context retention
        - Calls enhanced formatting if enhanced statistics are available
        - Handles missing metrics gracefully with default values
    """
    lines = [f"\n{agent_type.replace('_', ' ').title()}", "-" * 40]

    success_rate = metrics.get("success_rate", 0)
    avg_time = metrics.get("avg_execution_time", 0)

    if "enhanced_statistics" in metrics:
        lines.extend(
            _format_enhanced_statistics(
                success_rate, avg_time, metrics["enhanced_statistics"]
            )
        )
    else:
        lines.extend(_format_basic_statistics(success_rate, avg_time))

    lines.extend(
        [
            f"  Cost per Success: ${metrics.get('cost_per_success', 0):.4f}",
            f"  Context Retention: {metrics.get('avg_context_retention', 0):.1%}",
        ]
    )

    return lines


def _format_enhanced_statistics(
    success_rate: float, avg_time: float, stats: dict[str, Any]
) -> list[str]:
    """Format enhanced statistics section.

    Creates formatted output for enhanced statistical analysis including
    confidence intervals, variability assessments, and outlier detection.

    Args:
        success_rate: Agent success rate as decimal (0.0-1.0).
        avg_time: Average execution time in seconds.
        stats: Dict containing enhanced statistical analysis results.

    Returns:
        List of formatted lines for enhanced statistics display.

    Note:
        - Formats confidence intervals with brackets notation
        - Includes variability assessment and outlier detection results
        - Handles edge cases like zero variance and insufficient data
        - Provides context for perfect consistency cases
    """
    lines = []

    success_ci = stats.get("success_rate_ci", {})
    if success_ci.get("method") not in ["insufficient_data", "zero_variance"]:
        lines.append(
            f"  Success Rate:     {success_rate:.1%} "
            f"[{success_ci.get('lower', 0):.1%} - {success_ci.get('upper', 0):.1%}]"
        )
    else:
        lines.append(f"  Success Rate:     {success_rate:.1%}")
        if success_ci.get("method") == "zero_variance":
            lines.append("                    (Perfect consistency - no variance)")

    time_ci = stats.get("execution_time_ci", {})
    if time_ci.get("method") not in ["insufficient_data", "zero_variance"]:
        lines.append(
            f"  Avg Exec Time:    {avg_time:.2f}s "
            f"[{time_ci.get('lower', 0):.2f}s - {time_ci.get('upper', 0):.2f}s]"
        )
    else:
        lines.append(f"  Avg Exec Time:    {avg_time:.2f}s")

    dist_analysis = stats.get("distribution_analysis", {})
    if dist_analysis.get("zero_variance"):
        lines.append("  Variability:      Perfect consistency (zero variance)")
    else:
        variability = stats.get("interpretation", {}).get("variability", "Unknown")
        lines.append(f"  Variability:      {variability}")

    outlier_count = dist_analysis.get("outlier_count", 0)
    if outlier_count > 0:
        lines.append(f"  Outliers:         {outlier_count} detected")
    elif not dist_analysis.get("zero_variance"):
        lines.append("  Outliers:         None detected")

    return lines


def _format_basic_statistics(success_rate: float, avg_time: float) -> list[str]:
    """Format basic statistics section.

    Creates simple formatted output for basic metrics when enhanced
    statistics are not available or applicable.

    Args:
        success_rate: Agent success rate as decimal (0.0-1.0).
        avg_time: Average execution time in seconds.

    Returns:
        List of formatted lines for basic statistics display.

    Note:
        - Fallback formatting when enhanced analysis unavailable
        - Simple percentage and time formatting
        - Consistent with enhanced formatting style for compatibility
    """
    return [
        f"  Success Rate:     {success_rate:.1%}",
        f"  Avg Exec Time:    {avg_time:.2f}s",
    ]


def generate_enhanced_statistical_report(
    analysis_results: dict[str, Any], confidence_level: float = 0.95
) -> str:
    """Generate comprehensive statistical report with robust formatting.

    Creates a complete statistical analysis report from benchmark results with
    enhanced statistics, confidence intervals, and interpretations.

    Args:
        analysis_results: Complete analysis results from benchmark execution.
        confidence_level: Confidence level for statistical intervals. Default is 0.95.

    Returns:
        Formatted string containing complete statistical report ready for display or file output.

    Raises:
        No exceptions raised - handles all error cases gracefully with informative messages.

    Note:
        - Handles various result formats (dict vs list) automatically
        - Provides helpful guidance for unsupported formats
        - Includes suggestions for proper benchmark execution
        - Maintains formatting consistency across different data types
    """
    lines = _generate_report_header(confidence_level)
    results = analysis_results.get("results", {})

    if isinstance(results, dict):
        agent_data = results.get("results", results)
    elif isinstance(results, list):
        logger.warning(
            "Results data is in list format, cannot generate enhanced report"
        )
        return "\n".join(
            [
                "Enhanced Statistical Analysis",
                "=" * 60,
                "",
                "Results data is in list format",
                "   Enhanced statistical analysis requires aggregated metrics",
                "   Run the benchmark first to generate proper results structure",
                "",
                f"   Found {len(results)} individual task results",
                "   Use 'make run' or 'make run-enhanced' to generate analyzable data",
            ]
        )
    else:
        logger.warning(f"Unexpected results format: {type(results)}")
        return "\n".join(
            [
                "Enhanced Statistical Analysis",
                "=" * 60,
                "",
                "Unexpected results data format",
                f"   Expected dict, got {type(results).__name__}",
                "   Please run the benchmark again to generate proper results",
            ]
        )

    if not isinstance(agent_data, dict) or not agent_data:
        return "\n".join(
            [
                "Enhanced Statistical Analysis",
                "=" * 60,
                "",
                "No agent performance data found",
                "   Results file may be from raw task execution rather than analysis",
                "   Try running: make run-enhanced",
            ]
        )

    for agent_type, metrics in agent_data.items():
        lines.extend(_format_agent_metrics(agent_type, metrics))

    return "\n".join(lines)
