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

Module: ai_agents_reality_check/__init__.py

Package initializer for AI Agents Reality Check.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Jesse Moses (@Cre4T3Tiv3)"
__email__ = "jesse@bytestacklabs.com"

from .agent_benchmark import AgentBenchmark
from .agents import MarketingAgent, RealAgent, WrapperAgent
from .types import AgentType, TaskComplexity, TaskResult

__all__ = [
    "AgentBenchmark",
    "TaskResult",
    "TaskComplexity",
    "AgentType",
    "WrapperAgent",
    "MarketingAgent",
    "RealAgent",
]
