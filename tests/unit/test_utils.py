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

File: tests/unit/test_utils.py

Unit tests for utility functions.

Comprehensive test suite for utility functions including statistical calculations,
formatting helpers, outlier detection, confidence intervals, effect size calculations,
and results summarization. Tests cover edge cases, error conditions, performance
with large datasets, and optional advanced statistical features.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import time

import pytest
from ai_agents_reality_check.utils.helpers import (
    calculate_confidence_interval,
    calculate_effect_size,
    calculate_required_sample_size,
    calculate_statistics,
    create_results_summary,
    detect_outliers,
    format_duration,
    generate_benchmark_id,
)

try:
    from ai_agents_reality_check.utils.helpers import assess_statistical_significance

    HAS_STATISTICAL_SIGNIFICANCE = True
except ImportError:
    HAS_STATISTICAL_SIGNIFICANCE = False

try:
    from ai_agents_reality_check.utils.helpers import (
        calculate_enhanced_confidence_intervals,
    )

    HAS_ENHANCED_CI = True
except ImportError:
    HAS_ENHANCED_CI = False

try:
    from ai_agents_reality_check.utils.helpers import detect_performance_anomalies

    HAS_ANOMALY_DETECTION = True
except ImportError:
    HAS_ANOMALY_DETECTION = False


class TestStatisticalFunctions:
    """Test statistical utility functions."""

    def test_calculate_statistics_empty_list(self) -> None:
        """Test calculate_statistics with empty input."""
        result = calculate_statistics([])

        expected_keys = {"mean", "median", "std_dev", "min", "max", "q25", "q75"}
        assert set(result.keys()) == expected_keys

        for value in result.values():
            assert value == 0.0

    def test_calculate_statistics_single_value(self) -> None:
        """Test calculate_statistics with single value."""
        result = calculate_statistics([5.0])

        assert result["mean"] == 5.0
        assert result["median"] == 5.0
        assert result["std_dev"] == 0.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["q25"] == 5.0
        assert result["q75"] == 5.0

    def test_calculate_statistics_multiple_values(self) -> None:
        """Test calculate_statistics with multiple values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_statistics(values)

        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["std_dev"] > 0

    def test_calculate_statistics_quartiles(self) -> None:
        """Test quartile calculations."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = calculate_statistics(values)

        assert result["q25"] == 2.0
        assert result["q75"] == 6.0

    def test_calculate_effect_size_zero_difference(self) -> None:
        """Test effect size calculation with identical proportions."""
        effect_size, interpretation = calculate_effect_size(0.5, 0.5)

        assert effect_size == 0.0
        assert interpretation == "Negligible"

    def test_calculate_effect_size_small_difference(self) -> None:
        """Test effect size calculation with small difference."""
        effect_size, interpretation = calculate_effect_size(0.3, 0.4)

        assert effect_size > 0
        assert effect_size < 0.5
        assert interpretation in ["Negligible", "Small"]

    def test_calculate_effect_size_large_difference(self) -> None:
        """Test effect size calculation with large difference."""
        effect_size, interpretation = calculate_effect_size(0.2, 0.8)

        assert effect_size > 0.8
        assert interpretation == "Very Large"

    def test_calculate_effect_size_edge_cases(self) -> None:
        """Test effect size calculation with edge cases."""
        effect_size, interpretation = calculate_effect_size(0.0, 1.0)
        assert effect_size > 0
        assert interpretation == "Very Large"

        effect_size, interpretation = calculate_effect_size(0.001, 0.002)
        assert effect_size >= 0
        assert interpretation in ["Negligible", "Small"]

    def test_calculate_required_sample_size_zero_effect(self) -> None:
        """Test sample size calculation with zero effect size."""
        sample_size = calculate_required_sample_size(0.0)

        assert sample_size == 999

    def test_calculate_required_sample_size_small_effect(self) -> None:
        """Test sample size calculation with small effect size."""
        sample_size = calculate_required_sample_size(0.2)

        assert sample_size > 100
        assert sample_size < 10000

    def test_calculate_required_sample_size_large_effect(self) -> None:
        """Test sample size calculation with large effect size."""
        sample_size = calculate_required_sample_size(0.8)

        assert sample_size < 100
        assert sample_size > 0

    def test_calculate_required_sample_size_power_levels(self) -> None:
        """Test sample size calculation with different power levels."""
        effect_size = 0.5

        size_80 = calculate_required_sample_size(effect_size, power=0.8)
        size_90 = calculate_required_sample_size(effect_size, power=0.9)
        size_95 = calculate_required_sample_size(effect_size, power=0.95)

        assert size_90 > size_80
        assert size_95 > size_90

    def test_calculate_confidence_interval_empty(self) -> None:
        """Test confidence interval with empty data."""
        result = calculate_confidence_interval([])

        assert result["lower"] == 0.0
        assert result["upper"] == 0.0
        assert result["margin"] == 0.0

    def test_calculate_confidence_interval_single_value(self) -> None:
        """Test confidence interval with single value."""
        result = calculate_confidence_interval([5.0])

        assert result["lower"] == 0.0
        assert result["upper"] == 0.0
        assert result["margin"] == 0.0

    def test_calculate_confidence_interval_multiple_values(self) -> None:
        """Test confidence interval with multiple values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_confidence_interval(values)

        assert result["lower"] < result["upper"]
        assert result["margin"] > 0
        assert abs(result["upper"] - result["lower"] - 2 * result["margin"]) < 0.001

    def test_calculate_confidence_interval_confidence_levels(self) -> None:
        """Test confidence interval with different confidence levels."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        ci_95 = calculate_confidence_interval(values, confidence=0.95)
        ci_99 = calculate_confidence_interval(values, confidence=0.99)

        assert ci_99["margin"] > ci_95["margin"]


class TestOutlierDetection:
    """Test outlier detection functions."""

    def test_detect_outliers_empty(self) -> None:
        """Test outlier detection with empty data."""
        outliers = detect_outliers([])
        assert outliers == []

    def test_detect_outliers_insufficient_data(self) -> None:
        """Test outlier detection with insufficient data."""
        outliers = detect_outliers([1.0, 2.0, 3.0])
        assert outliers == []

    def test_detect_outliers_no_outliers(self) -> None:
        """Test outlier detection with no outliers."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        outliers = detect_outliers(values, method="iqr")
        assert outliers == []

    def test_detect_outliers_with_outliers(self) -> None:
        """Test outlier detection with clear outliers."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        outliers = detect_outliers(values, method="iqr")

        assert 5 in outliers

    def test_detect_outliers_invalid_method(self) -> None:
        """Test outlier detection with invalid method."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        with pytest.raises(ValueError):
            detect_outliers(values, method="invalid_method")

    def test_detect_outliers_identical_values(self) -> None:
        """Test outlier detection with identical values."""
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        outliers = detect_outliers(values, method="iqr")

        assert outliers == []


class TestFormatting:
    """Test formatting utility functions."""

    def test_format_duration_milliseconds(self) -> None:
        """Test duration formatting for milliseconds."""
        assert format_duration(0.1) == "100ms"
        assert format_duration(0.5) == "500ms"
        assert format_duration(0.999) == "999ms"

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting for seconds."""
        assert format_duration(1.0) == "1.0s"
        assert format_duration(2.5) == "2.5s"
        assert format_duration(59.9) == "59.9s"

    def test_format_duration_minutes(self) -> None:
        """Test duration formatting for minutes."""
        assert format_duration(60.0) == "1m 0s"
        assert format_duration(90.0) == "1m 30s"
        assert format_duration(3599.0) == "59m 59s"

    def test_format_duration_hours(self) -> None:
        """Test duration formatting for hours."""
        assert format_duration(3600.0) == "1h 0m"
        assert format_duration(7200.0) == "2h 0m"
        assert format_duration(3660.0) == "1h 1m"

    def test_format_duration_edge_cases(self) -> None:
        """Test duration formatting edge cases."""
        assert format_duration(0.0) == "0ms"
        assert format_duration(0.001) == "1ms"


class TestResultsSummary:
    """Test results summary generation."""

    def test_create_results_summary_empty(self) -> None:
        """Test results summary with empty data."""
        summary = create_results_summary({})

        assert "timestamp" in summary
        assert summary["total_agents_tested"] == 0
        assert "overall_insights" in summary

    def test_create_results_summary_single_agent(self) -> None:
        """Test results summary with single agent."""
        results = {
            "WRAPPER_AGENT": {
                "success_rate": 0.3,
                "cost_per_success": 0.006,
            }
        }

        summary = create_results_summary(results)

        assert summary["total_agents_tested"] == 1
        assert summary["overall_insights"]["best_success_rate"] == 0.3
        assert summary["overall_insights"]["worst_success_rate"] == 0.3
        assert summary["overall_insights"]["performance_gap"] == 0.0

    def test_create_results_summary_multiple_agents(self) -> None:
        """Test results summary with multiple agents."""
        results = {
            "WRAPPER_AGENT": {
                "success_rate": 0.3,
                "cost_per_success": 0.006,
            },
            "MARKETING_AGENT": {
                "success_rate": 0.7,
                "cost_per_success": 0.08,
            },
            "REAL_AGENT": {
                "success_rate": 0.9,
                "cost_per_success": 0.03,
            },
        }

        summary = create_results_summary(results)

        assert summary["total_agents_tested"] == 3
        assert summary["overall_insights"]["best_success_rate"] == 0.9
        assert summary["overall_insights"]["worst_success_rate"] == 0.3

        performance_gap = summary["overall_insights"]["performance_gap"]
        assert abs(performance_gap - 0.6) < 1e-10


class TestUtilityFunctions:
    def test_generate_benchmark_id(self) -> None:
        """Test benchmark ID generation."""
        id1 = generate_benchmark_id()
        time.sleep(0.001)
        id2 = generate_benchmark_id()

        assert isinstance(id1, str)
        assert isinstance(id2, str)

        assert id1.startswith("benchmark_")
        assert id2.startswith("benchmark_")

        assert id1 != id2

    def test_benchmark_id_format(self) -> None:
        """Test benchmark ID format."""
        benchmark_id = generate_benchmark_id()

        parts = benchmark_id.split("_")
        assert len(parts) == 2
        assert parts[0] == "benchmark"

        assert parts[1].isdigit()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_calculate_statistics_nan_values(self) -> None:
        """Test calculate_statistics with NaN values."""
        values = [1.0, 2.0, float("nan"), 4.0, 5.0]

        try:
            result = calculate_statistics(values)
            assert isinstance(result, dict)
        except (ValueError, TypeError):
            pass

    def test_calculate_statistics_infinite_values(self) -> None:
        """Test calculate_statistics with infinite values."""
        values = [1.0, 2.0, float("inf"), 4.0, 5.0]

        try:
            result = calculate_statistics(values)
            assert isinstance(result, dict)
        except (ValueError, TypeError, OverflowError):
            pass

    def test_effect_size_boundary_values(self) -> None:
        """Test effect size calculation with boundary values."""
        effect_size, interpretation = calculate_effect_size(0.0, 1.0)
        assert effect_size >= 0
        assert interpretation in [
            "Negligible",
            "Small",
            "Medium",
            "Large",
            "Very Large",
        ]

        effect_size, interpretation = calculate_effect_size(1.0, 1.0)
        assert effect_size == 0.0
        assert interpretation == "Negligible"

    def test_confidence_interval_identical_values(self) -> None:
        """Test confidence interval with identical values."""
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = calculate_confidence_interval(values)

        assert result["margin"] == 0.0
        assert result["lower"] == 5.0
        assert result["upper"] == 5.0

    def test_large_dataset_performance(self) -> None:
        """Test utility functions with large datasets."""
        large_values = [float(i) for i in range(10000)]
        result = calculate_statistics(large_values)
        assert isinstance(result, dict)

        outliers = detect_outliers(large_values, method="iqr")
        assert isinstance(outliers, list)

    def test_negative_values_handling(self) -> None:
        """Test utility functions with negative values."""
        values = [-5.0, -2.0, 0.0, 2.0, 5.0]

        result = calculate_statistics(values)
        assert result["mean"] == 0.0
        assert result["min"] == -5.0
        assert result["max"] == 5.0

        outliers = detect_outliers(values, method="iqr")
        assert isinstance(outliers, list)


class TestAdvancedStatistics:
    """Test advanced statistical utility functions if available."""

    @pytest.mark.skipif(
        not HAS_STATISTICAL_SIGNIFICANCE,
        reason="Statistical significance assessment not available",
    )
    def test_assess_statistical_significance_available(self) -> None:
        """Test statistical significance assessment if function is available."""
        mock_results = {
            "WRAPPER_AGENT": {
                "success_rate": 0.3,
                "total_tasks": 20,
            },
            "REAL_AGENT": {
                "success_rate": 0.9,
                "total_tasks": 20,
            },
        }

        assessment = assess_statistical_significance(mock_results)

        assert "sufficient_power" in assessment
        assert "effect_sizes" in assessment
        assert isinstance(assessment["sufficient_power"], bool)

    @pytest.mark.skipif(
        not HAS_ENHANCED_CI, reason="Enhanced confidence intervals not available"
    )
    def test_enhanced_confidence_intervals_available(self) -> None:
        """Test enhanced confidence intervals if function is available."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_enhanced_confidence_intervals(values)

        assert "lower" in result
        assert "upper" in result
        assert "margin" in result
        assert "method" in result

    @pytest.mark.skipif(
        not HAS_ANOMALY_DETECTION, reason="Performance anomaly detection not available"
    )
    def test_performance_anomaly_detection_available(self) -> None:
        """Test performance anomaly detection if function is available."""
        mock_results = {
            "WRAPPER_AGENT": {"success_rate": 0.3},
            "MARKETING_AGENT": {"success_rate": 0.7},
            "REAL_AGENT": {"success_rate": 0.9},
        }

        anomalies = detect_performance_anomalies(mock_results)

        assert "detected_anomalies" in anomalies
        assert "performance_warnings" in anomalies
        assert "recommendations" in anomalies
