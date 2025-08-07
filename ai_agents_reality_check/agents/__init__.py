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

Module: ai_agents_reality_check/agents/__init__.py

Package interface for the three-tier agent architecture benchmark system,
exposing the fundamental agent implementations used to evaluate the performance
gap between marketing claims and architectural reality in AI agent systems.

This module provides unified access to the agent implementations that represent
the complete spectrum of agent architectural sophistication: from stateless
wrappers to production-grade autonomous systems. The exposed agents enable
comprehensive benchmarking of the performance gaps that separate
shallow implementations from sophisticated agent architectures.

Exposed Agent Types:
    WrapperAgent: Baseline stateless LLM wrapper
    MarketingAgent: Intermediate demo-quality orchestration
    RealAgent: Production-grade autonomous system

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 0.1.0
License: Apache 2.0
"""

from .marketing.marketing_agent import MarketingAgent
from .real.real_agent import RealAgent
from .wrapper.wrapper_agent import WrapperAgent

__all__ = ["WrapperAgent", "MarketingAgent", "RealAgent"]
