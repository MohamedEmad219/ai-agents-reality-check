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

Subpackage: ensemble
Module: ai_agents_reality_check/ensemble/__init__.py

Agent ensemble benchmarking and coordination systems.
Provides multi-agent collaboration patterns and performance evaluation.
This module is optional and will gracefully degrade if dependencies are missing.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

import warnings

try:
    from .ensemble_benchmark import (
        EnsembleBenchmark,
        EnsembleCoordinator,
        EnsemblePattern,
        EnsembleResult,
    )

    __all__ = [
        "EnsembleBenchmark",
        "EnsemblePattern",
        "EnsembleResult",
        "EnsembleCoordinator",
    ]

except ImportError as e:
    warnings.warn(
        f"Ensemble features not available: {e}. "
        "Install optional dependencies for full functionality.",
        ImportWarning,
        stacklevel=2,
    )

    __all__ = []
