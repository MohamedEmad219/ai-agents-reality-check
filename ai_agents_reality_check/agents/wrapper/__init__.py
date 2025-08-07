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

Subpackage: agents.wrapper
Module: ai_agents_reality_check/agents/wrapper/__init__.py

Contains the implementation of the WrapperAgent,
which simulates a trivial LLM wrapper agent with no planning,
memory, or recovery mechanisms.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

from .wrapper_agent import WrapperAgent

__all__ = ["WrapperAgent"]
