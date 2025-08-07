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

Module: ai_agents_reality_check/agents/wrapper/wrapper_utils.py

Failure classification constants for wrapper agent architectures, defining
the characteristic failure patterns observed in stateless LLM wrapper
implementations commonly misrepresented as sophisticated agent systems.

This module provides the failure taxonomy that enables realistic benchmarking
of wrapper agent performance, capturing the specific error patterns that
result from architectural limitations rather than model capabilities.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

"""Comprehensive taxonomy of failure patterns characteristic of wrapper agents.

Defines the specific failure modes commonly observed in stateless LLM wrapper
implementations that lack sophisticated agent architectures. These failure
classifications reflect the architectural limitations that constrain wrapper
agents to 35-45% success rates rather than model-specific performance issues.

The taxonomy captures failure patterns resulting from:
- Lack of planning and multi-step reasoning capabilities
- Absence of persistent memory and context management
- Missing error recovery and retry mechanisms
- Vulnerability to prompt manipulation and context overflow
- Inability to maintain state across execution boundaries

These failure classifications enable realistic performance modeling and
benchmarking that accurately represents the architectural constraints of
wrapper implementations commonly deployed as "AI agents" in production.

Failure Categories:
    HALLUCINATION: Generated non-existent resources or capabilities
        - Reflects lack of grounding and verification mechanisms
        - Common when wrappers attempt to simulate unavailable functionality
    
    REASONING_FAILURE: Inability to handle multi-step logical processes
        - Results from single-shot execution without planning architecture
        - Demonstrates architectural limitation rather than model deficiency
    
    CONTEXT_LOSS: Information loss between operations
        - Consequence of stateless design without memory persistence
        - Prevents complex task completion requiring context carryover
    
    STATE_ERROR: Inability to maintain execution state
        - Fundamental limitation of wrapper architectures
        - Prevents coordination across multiple execution steps
    
    TIMEOUT: Execution timeouts on complex reasoning
        - Results from attempting complex tasks without proper decomposition
        - Reflects lack of planning and progress management
    
    PROMPT_INJECTION: Security vulnerability exploitation
        - Consequence of direct prompt handling without sanitization
        - Architectural security limitation of wrapper approaches
    
    RATE_LIMIT: API rate limiting without retry logic
        - Demonstrates lack of sophisticated error handling
        - Results from simple request-response patterns
    
    INVALID_OUTPUT: Malformed response generation
        - Reflects absence of output validation and formatting
        - Common in wrappers lacking structured response handling
    
    MISUNDERSTANDING: Complete task requirement misinterpretation
        - Results from single-shot processing without clarification
        - Demonstrates lack of iterative understanding refinement
    
    CONTEXT_OVERFLOW: Context window limitation failures
        - Architectural constraint of direct prompt processing
        - Reflects absence of context management and summarization

Note:
    These failure patterns are architecturally determined rather than model-
    specific, representing the inherent limitations of wrapper approaches
    that process tasks through direct prompt-response patterns without
    sophisticated agent infrastructure.

Type:
    List[str]: Ordered collection of failure reason strings with category
    prefixes and descriptive explanations for comprehensive failure
    classification and analysis purposes.
"""

WRAPPER_FAILURE_REASONS = [
    "HALLUCINATION: Hallucinated non-existent API endpoint",
    "REASONING_FAILURE: Failed to handle multi-step reasoning",
    "CONTEXT_LOSS: Lost context between operations",
    "STATE_ERROR: Unable to maintain state across calls",
    "TIMEOUT: Timeout on complex reasoning",
    "PROMPT_INJECTION: Prompt injection vulnerability exploited",
    "RATE_LIMIT: Rate limit exceeded without retry logic",
    "INVALID_OUTPUT: Generated invalid JSON/code structure",
    "MISUNDERSTANDING: Misunderstood task requirements completely",
    "CONTEXT_OVERFLOW: Context window overflow on complex input",
]
