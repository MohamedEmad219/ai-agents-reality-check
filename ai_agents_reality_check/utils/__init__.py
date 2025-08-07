"""Copyright 2025 ByteStack Labs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Module: ai_agents_reality_check/utils/__init__.py

Provides convenient access to core utility functions used throughout the AI Agents Reality Check system.
This includes statistical calculations, summary generation, duration formatting, and logging setup.

Enhanced features are optional and will gracefully degrade if dependencies are missing.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

from .helpers import (
    calculate_statistics,
    create_challenging_task_set,
    create_results_summary,
    format_duration,
)

__all__ = [
    "calculate_statistics",
    "format_duration",
    "create_results_summary",
    "create_challenging_task_set",
]

STATS_AVAILABLE = False
try:
    from .helpers import (
        assess_statistical_significance,  # noqa: F401
        calculate_effect_size,  # noqa: F401
        calculate_required_sample_size,  # noqa: F401
    )

    __all__.extend(
        [
            "assess_statistical_significance",
            "calculate_effect_size",
            "calculate_required_sample_size",
        ]
    )

    STATS_AVAILABLE = True

except ImportError:
    pass

ENHANCED_STATS_AVAILABLE = False
try:
    from .enhanced_statistics import (
        StatisticalAnalysis,  # noqa: F401
        generate_enhanced_statistical_report,  # noqa: F401
    )

    __all__.extend(
        [
            "StatisticalAnalysis",
            "generate_enhanced_statistical_report",
        ]
    )

    ENHANCED_STATS_AVAILABLE = True

except ImportError:
    pass

__all__.extend(["STATS_AVAILABLE", "ENHANCED_STATS_AVAILABLE"])
