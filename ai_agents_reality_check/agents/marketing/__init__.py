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

Subpackage: agents.marketing
Module: ai_agents_reality_check/agents/marketing/__init__.py

Contains the MarketingAgent and its modular components:
planner, executor, memory, and retry logic.
Represents fragile SaaS-style agents with shallow planning and orchestration.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

from .marketing_agent import MarketingAgent

__all__ = ["MarketingAgent"]
