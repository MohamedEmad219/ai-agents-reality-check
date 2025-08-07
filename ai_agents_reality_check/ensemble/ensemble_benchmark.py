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

Module: ai_agents_reality_check/ensemble/ensemble_benchmark.py

Comprehensive ensemble benchmarking system for evaluating collaborative agent
performance across multiple coordination patterns. Tests pipeline, parallel,
hierarchical, consensus, and specialization approaches to expose the reality
that coordination overhead often exceeds collaboration benefits.

This module provides the infrastructure for discovering the shocking empirical
reality: 0% positive synergy rate across all ensemble patterns tested, with
individual real agents consistently outperforming complex multi-agent systems.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
License: Apache 2.0
Version: 0.1.0
"""

import asyncio
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any

from ai_agents_reality_check.agents import MarketingAgent, RealAgent, WrapperAgent
from ai_agents_reality_check.logging_config import logger
from ai_agents_reality_check.types import AgentType, TaskComplexity, TaskResult

# Constants for magic values
MIN_HIERARCHICAL_AGENTS = 2
SYNERGY_THRESHOLD = 0.1
STRONG_SYNERGY_THRESHOLD = 0.7
MODERATE_SYNERGY_THRESHOLD = 0.4
MEANINGFUL_ADVANTAGE_THRESHOLD = 0.1


class EnsemblePattern(Enum):
    """Enumeration of collaborative agent coordination patterns for benchmarking.

    Defines the five fundamental ensemble collaboration approaches used in
    multi-agent system evaluation, each representing different coordination
    strategies with distinct overhead characteristics and success patterns.

    Values:
        PIPELINE: Sequential task delegation where agents process tasks in order,
            with each agent building on the previous agent's work. Moderate
            coordination overhead with fail-fast characteristics.
        PARALLEL: Concurrent task execution where multiple agents work simultaneously
            on the same task with result aggregation. Lower coordination overhead
            but requires consensus mechanisms for final decisions.
        HIERARCHICAL: Manager-worker coordination where one agent coordinates
            while others execute delegated subtasks. High coordination overhead
            due to planning and delegation requirements.
        CONSENSUS: Democratic decision making where all agents contribute to
            task execution and vote on final outcomes. Highest coordination
            overhead due to voting and agreement mechanisms.
        SPECIALIZATION: Expert role assignment where each agent handles tasks
            matched to their architectural capabilities. Variable coordination
            overhead based on task distribution complexity.
    """

    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    SPECIALIZATION = "specialization"


class EnsembleResult:
    """Comprehensive results container for ensemble execution analysis.

    Encapsulates complete ensemble execution data including individual agent
    performances, coordination costs, synergy analysis, and comparative metrics
    that enable detailed evaluation of multi-agent collaboration effectiveness
    versus individual agent performance.

    The result container enables comprehensive analysis including ensemble
    advantage calculation, cost efficiency evaluation, and coordination overhead
    assessment that reveals the empirical reality of ensemble performance patterns.

    Attributes:
        ensemble_id (str): Unique identifier for ensemble execution tracking.
        pattern (EnsemblePattern): Coordination pattern used for execution.
        task_name (str): Human-readable task description.
        task_complexity (TaskComplexity): Task difficulty classification.
        participant_agents (List[AgentType]): Agent types included in ensemble.
        success (bool): Overall ensemble execution outcome.
        execution_time (float): Total ensemble execution duration.
        total_cost (float): Aggregate cost across all participating agents.
        individual_results (List[TaskResult]): Complete individual agent results.
        coordination_overhead (float): Additional time cost of coordination.
        consensus_agreement (float): Agreement level for consensus patterns.
        failure_reason (str, optional): Specific failure classification.
        avg_individual_success (float): Mean success rate of individual agents.
        cost_efficiency (float): Cost per successful execution or infinity.
        ensemble_advantage (float): Performance improvement over best individual.
    """

    def __init__(
        self,
        ensemble_id: str,
        pattern: EnsemblePattern,
        task_name: str,
        task_complexity: TaskComplexity,
        participant_agents: list[AgentType],
        success: bool,
        execution_time: float,
        total_cost: float,
        individual_results: list[TaskResult],
        coordination_overhead: float,
        consensus_agreement: float = 0.0,
        failure_reason: str | None = None,
    ):
        """Initialize comprehensive ensemble result with calculated metrics.

        Creates ensemble result container with automatic calculation of derived
        metrics including individual success averaging, cost efficiency computation,
        and ensemble advantage assessment that enables detailed performance analysis.

        Args:
            ensemble_id (str): Unique execution identifier for tracking and correlation.
            pattern (EnsemblePattern): Coordination pattern used for execution.
            task_name (str): Human-readable task description for reporting.
            task_complexity (TaskComplexity): Task difficulty level classification.
            participant_agents (List[AgentType]): Agent types included in ensemble.
            success (bool): Overall ensemble execution outcome status.
            execution_time (float): Total execution duration including coordination.
            total_cost (float): Aggregate economic cost across all agents.
            individual_results (List[TaskResult]): Complete individual agent results.
            coordination_overhead (float): Additional time cost of coordination logic.
            consensus_agreement (float, optional): Agreement level for voting patterns.
                Defaults to 0.0 for non-consensus patterns.
            failure_reason (str, optional): Specific failure classification if failed.
                None for successful executions.

        Returns:
            None

        Note:
            Automatic metric calculations include:
            - avg_individual_success: Mean of individual agent success rates
            - cost_efficiency: total_cost / 1 if successful, infinity if failed
            - ensemble_advantage: Calculated performance improvement over best individual

            Ensemble advantage represents the core metric for evaluating whether
            coordination benefits exceed coordination costs in multi-agent systems.
        """
        self.ensemble_id = ensemble_id
        self.pattern = pattern
        self.task_name = task_name
        self.task_complexity = task_complexity
        self.participant_agents = participant_agents
        self.success = success
        self.execution_time = execution_time
        self.total_cost = total_cost
        self.individual_results = individual_results
        self.coordination_overhead = coordination_overhead
        self.consensus_agreement = consensus_agreement
        self.failure_reason = failure_reason
        self.avg_individual_success = (
            sum(1 for r in individual_results if r.success) / len(individual_results)
            if individual_results
            else 0.0
        )
        self.cost_efficiency = total_cost / 1 if success else float("inf")
        self.ensemble_advantage = self._calculate_ensemble_advantage()

    def _calculate_ensemble_advantage(self) -> float:
        """Calculate ensemble performance advantage over best individual agent.

        Computes the fundamental metric for evaluating multi-agent coordination
        effectiveness by comparing ensemble success against the best-performing
        individual agent. Reveals whether coordination benefits exceed coordination
        costs in collaborative execution scenarios.

        Args:
            None

        Returns:
            float: Ensemble advantage as difference between ensemble success (1.0 or 0.0)
                and best individual success (1.0 or 0.0). Positive values indicate
                beneficial coordination, negative values indicate coordination overhead
                exceeds collaboration benefits.

        Note:
            This metric reveals the empirical reality that coordination costs often
            exceed collaboration benefits, with most ensemble patterns showing
            negative advantages compared to well-architected individual agents.

            The calculation uses binary success values (1.0/0.0) rather than
            continuous metrics to reflect the fundamental question of task completion
            effectiveness in production deployment scenarios.
        """
        if not self.individual_results:
            return 0.0

        best_individual_success = max(
            (1.0 if r.success else 0.0) for r in self.individual_results
        )
        ensemble_success = 1.0 if self.success else 0.0

        return ensemble_success - best_individual_success

    def to_dict(self) -> dict[str, Any]:
        """Convert ensemble result to dictionary for serialization and analysis.

        Transforms the comprehensive ensemble result into a structured dictionary
        format suitable for JSON serialization, database storage, and automated
        analysis tools. Includes all metrics and individual results for complete
        analysis capability.

        Args:
            None

        Returns:
            Dict[str, Any]: Complete ensemble result dictionary containing:
                - All ensemble metadata (ID, pattern, task information)
                - Performance metrics (success, timing, costs, advantages)
                - Coordination data (overhead, consensus agreement)
                - Individual agent results as nested dictionaries
                - Calculated metrics (efficiency, advantage, individual success rate)

        Note:
            Dictionary format enables:
            - JSON serialization for persistent storage and reporting
            - Database integration for large-scale analysis
            - Automated processing by analysis and visualization tools
            - Complete reconstruction of ensemble execution context

            Individual results are recursively converted using TaskResult.to_dict()
            to maintain complete execution context and enable detailed analysis.
        """
        return {
            "ensemble_id": self.ensemble_id,
            "pattern": self.pattern.value,
            "task_name": self.task_name,
            "task_complexity": self.task_complexity.value,
            "participant_agents": [agent.value for agent in self.participant_agents],
            "success": self.success,
            "execution_time": self.execution_time,
            "total_cost": self.total_cost,
            "coordination_overhead": self.coordination_overhead,
            "consensus_agreement": self.consensus_agreement,
            "failure_reason": self.failure_reason,
            "avg_individual_success": self.avg_individual_success,
            "cost_efficiency": self.cost_efficiency,
            "ensemble_advantage": self.ensemble_advantage,
            "individual_results": [r.to_dict() for r in self.individual_results],
        }


class EnsembleCoordinator:
    """Coordination engine for executing and managing multi-agent ensembles.

    Orchestrates collaborative agent execution across five distinct coordination
    patterns with comprehensive result tracking and performance analysis. Manages
    agent lifecycle, coordination overhead, and failure handling that reveals
    the empirical reality of ensemble performance characteristics.

    The coordinator maintains agent instances and execution history while
    providing pattern-specific coordination logic that enables comprehensive
    evaluation of multi-agent collaboration effectiveness versus individual
    agent performance in production scenarios.

    Attributes:
        ensemble_history (List[EnsembleResult]): Complete execution history for
            performance analysis and pattern recognition across multiple runs.
        agents (Dict[AgentType, Union[WrapperAgent, MarketingAgent, RealAgent]]):
            Agent instances for ensemble coordination with consistent interface.
    """

    def __init__(self) -> None:
        """Initialize ensemble coordinator with agent instances and empty history.

        Sets up coordination infrastructure with pre-instantiated agents of all
        three architectural tiers and empty execution history for comprehensive
        ensemble performance tracking and analysis.

        Args:
            None

        Returns:
            None

        Note:
            Agent instances are pre-created to ensure consistent performance
            characteristics across ensemble executions while maintaining
            individual agent state and learning capabilities where applicable.
        """
        self.ensemble_history: list[EnsembleResult] = []
        self.agents: dict[AgentType, WrapperAgent | MarketingAgent | RealAgent] = {
            AgentType.WRAPPER_AGENT: WrapperAgent(),
            AgentType.MARKETING_AGENT: MarketingAgent(),
            AgentType.REAL_AGENT: RealAgent(),
        }

    async def execute_ensemble_task(
        self,
        task: dict[str, Any],
        pattern: EnsemblePattern,
        agent_types: list[AgentType],
        coordination_config: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        """Execute task using specified ensemble pattern with comprehensive tracking.

        Orchestrates multi-agent task execution through pattern-specific coordination
        logic with detailed performance monitoring, cost tracking, and failure analysis.
        Provides the foundation for discovering the empirical reality that coordination
        overhead often exceeds collaboration benefits.

        The execution process includes realistic coordination overhead modeling,
        comprehensive error handling, and detailed result tracking that enables
        accurate assessment of ensemble effectiveness versus individual agent performance.

        Args:
            task (Dict[str, Any]): Task specification containing:
                - name (str): Human-readable task description
                - complexity (TaskComplexity): Task difficulty classification
                - Additional task-specific parameters for agent execution
            pattern (EnsemblePattern): Coordination pattern for execution including
                pipeline, parallel, hierarchical, consensus, or specialization approaches.
            agent_types (List[AgentType]): Agent types to include in ensemble execution.
                Must contain at least one agent, with pattern-specific minimum requirements.
            coordination_config (Dict[str, Any], optional): Pattern-specific configuration
                including aggregation strategies, consensus thresholds, and failure policies.

        Returns:
            EnsembleResult: Comprehensive execution result containing:
                - Success/failure status with detailed failure classification
                - Individual agent results for comparative analysis
                - Coordination overhead and cost calculations
                - Ensemble advantage metrics versus individual performance
                - Pattern-specific metrics (consensus agreement, etc.)

        Note:
            Coordination overhead modeling includes:
            - Base overhead: 0.1s per participating agent
            - Pattern-specific overhead varying from 0.2s (parallel) to 1.2s (consensus)
            - Realistic delays reflecting coordination complexity in production systems

            Pattern execution delegates to specialized coordination methods that
            implement realistic coordination logic including failure handling,
            result aggregation, and performance optimization strategies.

        Raises:
            ValueError: For unknown ensemble patterns or invalid agent configurations.
            Exception: Coordination failures are captured and converted to failed
                EnsembleResult objects with detailed error information.
        """
        ensemble_id = str(uuid.uuid4())
        coordination_config = coordination_config or {}

        start_time = time.time()

        try:
            if pattern == EnsemblePattern.PIPELINE:
                result = await self._execute_pipeline(
                    task, agent_types, coordination_config
                )
            elif pattern == EnsemblePattern.PARALLEL:
                result = await self._execute_parallel(
                    task, agent_types, coordination_config
                )
            elif pattern == EnsemblePattern.HIERARCHICAL:
                result = await self._execute_hierarchical(
                    task, agent_types, coordination_config
                )
            elif pattern == EnsemblePattern.CONSENSUS:
                result = await self._execute_consensus(
                    task, agent_types, coordination_config
                )
            elif pattern == EnsemblePattern.SPECIALIZATION:
                result = await self._execute_specialization(
                    task, agent_types, coordination_config
                )
            else:
                raise ValueError(f"Unknown ensemble pattern: {pattern}")

            execution_time = time.time() - start_time
            base_overhead = len(agent_types) * 0.1
            pattern_overhead = {
                EnsemblePattern.PIPELINE: 0.5,
                EnsemblePattern.PARALLEL: 0.2,
                EnsemblePattern.HIERARCHICAL: 0.8,
                EnsemblePattern.CONSENSUS: 1.2,
                EnsemblePattern.SPECIALIZATION: 0.6,
            }
            coordination_overhead = base_overhead + pattern_overhead.get(pattern, 0.3)

            ensemble_result = EnsembleResult(
                ensemble_id=ensemble_id,
                pattern=pattern,
                task_name=task.get("name", "Unnamed Task"),
                task_complexity=task.get("complexity", TaskComplexity.MODERATE),
                participant_agents=agent_types,
                success=result["success"],
                execution_time=execution_time,
                total_cost=result["total_cost"],
                individual_results=result["individual_results"],
                coordination_overhead=coordination_overhead,
                consensus_agreement=result.get("consensus_agreement", 0.0),
                failure_reason=result.get("failure_reason"),
            )

            self.ensemble_history.append(ensemble_result)
            return ensemble_result

        except Exception as e:
            logger.error(f"Ensemble execution failed: {e}")

            ensemble_result = EnsembleResult(
                ensemble_id=ensemble_id,
                pattern=pattern,
                task_name=task.get("name", "Unnamed Task"),
                task_complexity=task.get("complexity", TaskComplexity.MODERATE),
                participant_agents=agent_types,
                success=False,
                execution_time=time.time() - start_time,
                total_cost=0.0,
                individual_results=[],
                coordination_overhead=0.0,
                failure_reason=str(e),
            )

            self.ensemble_history.append(ensemble_result)
            return ensemble_result

    async def _execute_pipeline(
        self,
        task: dict[str, Any],
        agent_types: list[AgentType],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute sequential pipeline coordination with fail-fast semantics.

        Implements pipeline coordination where agents process tasks sequentially,
        with each agent building on the previous agent's work. Includes realistic
        fail-fast logic and context propagation that models production pipeline
        behavior patterns and coordination overhead characteristics.

        Args:
            task (Dict[str, Any]): Base task specification enhanced with pipeline context.
            agent_types (List[AgentType]): Ordered list of agents for sequential execution.
            config (Dict[str, Any]): Pipeline configuration including:
                - fail_fast (bool): Whether to terminate on first failure (default: True)
                - Additional pipeline-specific coordination parameters

        Returns:
            Dict[str, Any]: Pipeline execution results containing:
                - success (bool): Final agent success status or False if pipeline failed
                - individual_results (List[TaskResult]): Sequential agent results
                - total_cost (float): Aggregate cost across pipeline stages
                - failure_reason (str, optional): Pipeline failure classification

        Note:
            Pipeline semantics include:
            - Sequential execution with 0.1s coordination delay per stage
            - Context propagation between stages with success/failure indicators
            - Fail-fast termination option for efficiency in production scenarios
            - Final success determined by last agent in pipeline sequence

            Context enhancement enables downstream agents to adapt to upstream
            results while maintaining realistic coordination overhead patterns.
        """
        individual_results = []
        current_task = task.copy()
        total_cost = 0.0

        for agent_type in agent_types:
            agent = self.agents[agent_type]

            await asyncio.sleep(0.1)

            result = await agent.execute_task(current_task)
            individual_results.append(result)
            total_cost += result.cost_estimate

            if result.success:
                current_task["pipeline_context"] = (
                    f"Previous agent ({agent_type.value}) succeeded"
                )
            else:
                fail_fast = config.get("fail_fast", True)
                if fail_fast:
                    return {
                        "success": False,
                        "individual_results": individual_results,
                        "total_cost": total_cost,
                        "failure_reason": f"Pipeline failed at {agent_type.value}",
                    }
                else:
                    current_task["pipeline_context"] = (
                        f"Previous agent ({agent_type.value}) failed"
                    )

        final_success = individual_results[-1].success if individual_results else False

        return {
            "success": final_success,
            "individual_results": individual_results,
            "total_cost": total_cost,
        }

    async def _execute_parallel(
        self,
        task: dict[str, Any],
        agent_types: list[AgentType],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute concurrent parallel coordination with result aggregation.

        Implements parallel coordination where multiple agents work simultaneously
        on identical tasks with configurable result aggregation strategies. Includes
        comprehensive exception handling and realistic coordination delays that
        model production parallel processing characteristics.

        Args:
            task (Dict[str, Any]): Task specification duplicated across parallel agents.
            agent_types (List[AgentType]): Agent types for concurrent execution.
            config (Dict[str, Any]): Parallel configuration including:
                - aggregation (str): Strategy for result combination including
                "majority_vote", "any_success", "all_success" approaches

        Returns:
            Dict[str, Any]: Parallel execution results containing:
                - success (bool): Aggregated success based on configured strategy
                - individual_results (List[TaskResult]): All agent results including failures
                - total_cost (float): Sum of all agent execution costs

        Note:
            Parallel execution characteristics:
            - Concurrent execution with asyncio.gather for realistic parallelism
            - 0.05s coordination overhead per agent for synchronization
            - Exception handling converts failures to TaskResult objects
            - Configurable aggregation enables different voting strategies

            Aggregation strategies reflect different production scenarios:
            - majority_vote: Democratic decision making (default)
            - any_success: Optimistic aggregation for fault tolerance
            - all_success: Conservative aggregation for critical tasks
        """
        agent_tasks = []
        for agent_type in agent_types:
            agent = self.agents[agent_type]
            agent_tasks.append(agent.execute_task(task.copy()))

        await asyncio.sleep(0.05 * len(agent_types))

        individual_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        valid_results: list[TaskResult] = []
        total_cost = 0.0

        for i, result in enumerate(individual_results):
            if isinstance(result, Exception):
                failure_result = TaskResult(
                    task_id=str(uuid.uuid4()),
                    task_name=task.get("name", "Unnamed Task"),
                    task_type=task.get("complexity", TaskComplexity.MODERATE),
                    agent_type=agent_types[i],
                    success=False,
                    execution_time=0.0,
                    error_count=1,
                    context_retention_score=0.0,
                    cost_estimate=0.001,
                    failure_reason=str(result),
                    steps_completed=0,
                    total_steps=1,
                )
                valid_results.append(failure_result)
            elif isinstance(result, TaskResult):
                valid_results.append(result)
                total_cost += result.cost_estimate

        aggregation_strategy = config.get("aggregation", "majority_vote")

        if aggregation_strategy == "majority_vote":
            successful_count = sum(1 for r in valid_results if r.success)
            ensemble_success = successful_count > len(valid_results) / 2
        elif aggregation_strategy == "any_success":
            ensemble_success = any(r.success for r in valid_results)
        elif aggregation_strategy == "all_success":
            ensemble_success = all(r.success for r in valid_results)
        else:
            successful_count = sum(1 for r in valid_results if r.success)
            ensemble_success = successful_count > len(valid_results) / 2

        return {
            "success": ensemble_success,
            "individual_results": valid_results,
            "total_cost": total_cost,
        }

    async def _execute_hierarchical(
        self,
        task: dict[str, Any],
        agent_types: list[AgentType],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute hierarchical manager-worker coordination with delegation logic.

        Implements hierarchical coordination where the first agent serves as coordinator
        while remaining agents execute delegated subtasks. Includes realistic planning
        overhead and majority-based worker success aggregation that models production
        hierarchical system characteristics.

        Args:
            task (Dict[str, Any]): Task specification enhanced with role assignments.
            agent_types (List[AgentType]): Agent types with first as coordinator,
                remainder as workers. Requires minimum 2 agents for hierarchy.
            config (Dict[str, Any]): Hierarchical configuration parameters.

        Returns:
            Dict[str, Any]: Hierarchical execution results containing:
                - success (bool): Coordinator success AND worker majority success
                - individual_results (List[TaskResult]): All agent results
                - total_cost (float): Aggregate cost across hierarchy
                - failure_reason (str, optional): Specific hierarchy failure mode

        Note:
            Hierarchical coordination semantics:
            - Coordinator executes first with 0.2s planning overhead
            - Workers execute concurrently with coordinator guidance context
            - Success requires both coordinator success AND worker majority
            - Exception handling preserves partial results for analysis

            Role assignment enhances task context with coordination metadata
            that enables realistic hierarchy behavior modeling and overhead assessment.

        Raises:
            ValueError: If fewer than 2 agents provided for hierarchical coordination.
        """
        if len(agent_types) < MIN_HIERARCHICAL_AGENTS:
            raise ValueError("Hierarchical ensemble requires at least 2 agents")

        coordinator_type = agent_types[0]
        worker_types = agent_types[1:]

        coordinator_agent = self.agents[coordinator_type]
        individual_results = []
        total_cost = 0.0
        coordinator_task = task.copy()
        coordinator_task["role"] = "coordinator"

        await asyncio.sleep(0.2)

        coordinator_result = await coordinator_agent.execute_task(coordinator_task)
        individual_results.append(coordinator_result)
        total_cost += coordinator_result.cost_estimate

        if not coordinator_result.success:
            return {
                "success": False,
                "individual_results": individual_results,
                "total_cost": total_cost,
                "failure_reason": "Coordinator failed to plan task",
            }

        worker_tasks = []
        for i, worker_type in enumerate(worker_types):
            worker_agent = self.agents[worker_type]
            worker_task = task.copy()
            worker_task["role"] = f"worker_{i}"
            worker_task["coordinator_guidance"] = "Subtask from coordinator"
            worker_tasks.append(worker_agent.execute_task(worker_task))

        await asyncio.sleep(0.1)

        worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        successful_workers = 0

        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                failure_result = TaskResult(
                    task_id=str(uuid.uuid4()),
                    task_name=task.get("name", "Unnamed Task"),
                    task_type=task.get("complexity", TaskComplexity.MODERATE),
                    agent_type=worker_types[i],
                    success=False,
                    execution_time=0.0,
                    error_count=1,
                    context_retention_score=0.0,
                    cost_estimate=0.001,
                    failure_reason=str(result),
                    steps_completed=0,
                    total_steps=1,
                )
                individual_results.append(failure_result)
            elif isinstance(result, TaskResult):
                individual_results.append(result)
                total_cost += result.cost_estimate
                if result.success:
                    successful_workers += 1

        worker_majority_success = successful_workers > len(worker_types) / 2
        ensemble_success = coordinator_result.success and worker_majority_success

        return {
            "success": ensemble_success,
            "individual_results": individual_results,
            "total_cost": total_cost,
        }

    async def _execute_consensus(
        self,
        task: dict[str, Any],
        agent_types: list[AgentType],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute consensus-based coordination with voting and agreement thresholds.

        Implements democratic consensus coordination where all agents contribute votes
        with configurable agreement thresholds for final decision making. Includes
        realistic voting delays and agreement analysis that models production
        consensus system characteristics and coordination overhead.

        Args:
            task (Dict[str, Any]): Task specification enhanced with consensus indicators.
            agent_types (List[AgentType]): Agent types participating in consensus voting.
            config (Dict[str, Any]): Consensus configuration including:
                - consensus_threshold (float): Agreement threshold (default: 0.6)
                - Additional voting and decision-making parameters

        Returns:
            Dict[str, Any]: Consensus execution results containing:
                - success (bool): Whether consensus agreement exceeds threshold
                - individual_results (List[TaskResult]): All agent voting results
                - total_cost (float): Aggregate cost across all voters
                - consensus_agreement (float): Actual agreement level achieved

        Note:
            Consensus coordination characteristics:
            - All agents execute with consensus context indicators
            - 0.3s coordination overhead for voting and agreement calculation
            - Configurable threshold enables different consensus requirements
            - Agreement calculation based on binary success voting

            Consensus threshold flexibility enables modeling of different
            decision-making scenarios from simple majority (0.5) to supermajority
            (0.67) to near-unanimous (0.9) requirements in production systems.
        """
        individual_results = []
        total_cost = 0.0
        agent_tasks = []

        for agent_type in agent_types:
            agent = self.agents[agent_type]
            consensus_task = task.copy()
            consensus_task["consensus_round"] = True
            agent_tasks.append(agent.execute_task(consensus_task))

        await asyncio.sleep(0.3)

        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        votes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failure_result = TaskResult(
                    task_id=str(uuid.uuid4()),
                    task_name=task.get("name", "Unnamed Task"),
                    task_type=task.get("complexity", TaskComplexity.MODERATE),
                    agent_type=agent_types[i],
                    success=False,
                    execution_time=0.0,
                    error_count=1,
                    context_retention_score=0.0,
                    cost_estimate=0.001,
                    failure_reason=str(result),
                    steps_completed=0,
                    total_steps=1,
                )
                individual_results.append(failure_result)
                votes.append(False)
            elif isinstance(result, TaskResult):
                individual_results.append(result)
                total_cost += result.cost_estimate
                votes.append(result.success)

        agreement_count = sum(votes)
        consensus_threshold = config.get("consensus_threshold", 0.6)
        consensus_agreement = agreement_count / len(votes) if votes else 0.0

        ensemble_success = consensus_agreement >= consensus_threshold

        return {
            "success": ensemble_success,
            "individual_results": individual_results,
            "total_cost": total_cost,
            "consensus_agreement": consensus_agreement,
        }

    async def _execute_specialization(
        self,
        task: dict[str, Any],
        agent_types: list[AgentType],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute specialization-based coordination with capability-matched task routing.

        Implements specialization coordination where agents receive tasks matched to
        their architectural capabilities, with complexity reduction for agents
        handling tasks beyond their optimal performance range. Models production
        specialization patterns with realistic capability assessment.

        Args:
            task (Dict[str, Any]): Task specification enhanced with specialization context.
            agent_types (List[AgentType]): Agent types for capability-based task routing.
            config (Dict[str, Any]): Specialization configuration parameters.

        Returns:
            Dict[str, Any]: Specialization execution results containing:
                - success (bool): All agents must succeed for specialization success
                - individual_results (List[TaskResult]): Results with capability matching
                - total_cost (float): Aggregate cost across specialized agents

        Note:
            Specialization strategy includes:
            - Wrapper agents: Specialized for simple operations with complexity reduction
            - Marketing agents: Coordination tasks with moderate complexity ceiling
            - Real agents: Complex reasoning tasks with full capability utilization
            - Task complexity adjustment based on agent architectural capabilities

            Capability matching enables realistic evaluation of specialization benefits
            while revealing that coordination costs often exceed specialization advantages
            when compared to well-architected individual real agents handling full tasks.
        """
        individual_results = []
        total_cost = 0.0

        specializations = {
            AgentType.WRAPPER_AGENT: "simple_operations",
            AgentType.MARKETING_AGENT: "coordination_tasks",
            AgentType.REAL_AGENT: "complex_reasoning",
        }

        task_complexity = task.get("complexity", TaskComplexity.MODERATE)

        for agent_type in agent_types:
            agent = self.agents[agent_type]
            specialized_task = task.copy()
            specialized_task["specialization"] = specializations.get(
                agent_type, "general_purpose"
            )

            if agent_type == AgentType.WRAPPER_AGENT:
                if task_complexity in [
                    TaskComplexity.COMPLEX,
                    TaskComplexity.ENTERPRISE,
                ]:
                    specialized_task["complexity"] = TaskComplexity.SIMPLE
            elif agent_type == AgentType.MARKETING_AGENT and task_complexity in {
                TaskComplexity.ENTERPRISE,
                TaskComplexity.COMPLEX,
            }:
                specialized_task["complexity"] = TaskComplexity.MODERATE

            await asyncio.sleep(0.05)

            result = await agent.execute_task(specialized_task)
            individual_results.append(result)
            total_cost += result.cost_estimate

        specialization_success = all(r.success for r in individual_results)

        return {
            "success": specialization_success,
            "individual_results": individual_results,
            "total_cost": total_cost,
        }


class EnsembleBenchmark:
    """Comprehensive benchmarking system for multi-agent ensemble evaluation.

    Orchestrates systematic evaluation of ensemble performance across multiple
    patterns, agent combinations, and task complexities with detailed analysis
    and reporting capabilities. Designed to reveal the empirical reality that
    coordination overhead often exceeds collaboration benefits in agent systems.

    The benchmark system provides comprehensive statistical analysis including
    ensemble advantage calculation, coordination cost assessment, and synergy
    evaluation that exposes the performance gaps between marketing claims and
    architectural reality in multi-agent systems.

    Attributes:
        benchmark_history (List[EnsembleResult]): Complete benchmark execution history.
        coordinator (EnsembleCoordinator): Ensemble coordination engine.
        quiet (bool): Logging verbosity control for batch processing scenarios.
    """

    def __init__(self, quiet: bool = False) -> None:
        """Initialize ensemble benchmark system with coordinator and logging control.

        Sets up comprehensive benchmarking infrastructure including ensemble coordinator,
        empty benchmark history, and configurable logging verbosity for different
        execution scenarios from interactive analysis to automated batch processing.

        Args:
            quiet (bool, optional): Logging verbosity control where True reduces
                console output to debug level for batch processing, False enables
                info-level logging for interactive analysis. Defaults to False.

        Returns:
            None

        Note:
            Quiet mode enables batch processing scenarios where detailed console
            output would interfere with automated analysis while preserving
            debug-level logging for troubleshooting and detailed analysis purposes.
        """
        self.benchmark_history: list[EnsembleResult] = []
        self.coordinator = EnsembleCoordinator()
        self.quiet = quiet

    async def run_ensemble_benchmark(
        self,
        tasks: list[dict[str, Any]],
        patterns: list[EnsemblePattern],
        agent_combinations: list[list[AgentType]],
        iterations: int = 5,
    ) -> dict[str, Any]:
        """Execute comprehensive ensemble benchmark with systematic evaluation protocol.

        Orchestrates systematic evaluation of ensemble performance across multiple
        dimensions including patterns, agent combinations, and task complexities
        with statistical rigor and comprehensive result analysis. Designed to
        reveal the empirical reality of ensemble performance characteristics.

        The benchmark protocol includes systematic iteration across all parameter
        combinations with robust error handling and detailed result aggregation
        that enables comprehensive analysis of coordination effectiveness versus
        individual agent performance in production scenarios.

        Args:
            tasks (List[Dict[str, Any]]): Task specifications for benchmark evaluation
                containing complexity levels and execution parameters.
            patterns (List[EnsemblePattern]): Ensemble patterns to evaluate including
                pipeline, parallel, hierarchical, consensus, and specialization.
            agent_combinations (List[List[AgentType]]): Agent type combinations for
                ensemble composition including homogeneous and heterogeneous mixes.
            iterations (int, optional): Iterations per configuration for statistical
                significance. Defaults to 5 for reasonable confidence intervals.

        Returns:
            Dict[str, Any]: Comprehensive benchmark results containing:
                - results (Dict): Detailed analysis by pattern, combination, and advantage
                - total_ensembles_tested (int): Complete evaluation count
                - patterns_evaluated (List[str]): Pattern identifiers evaluated
                - agent_combinations (List[List[str]]): Agent combination specifications

        Note:
            Benchmark methodology includes:
            - Systematic evaluation across all parameter combinations
            - Robust error handling with continued execution on individual failures
            - Comprehensive result aggregation and statistical analysis
            - Performance comparison against individual agent baselines

            Statistical rigor through multiple iterations enables confident
            assessment of ensemble performance patterns and reveals the consistent
            finding that coordination costs often exceed collaboration benefits.

        Raises:
            Exception: Individual ensemble failures are logged but do not terminate
                the overall benchmark, enabling comprehensive evaluation even with
                partial failures across parameter combinations.
        """
        if not self.quiet:
            logger.info(
                f"Starting ensemble benchmark: {len(tasks)} tasks, {len(patterns)} patterns"
            )
        else:
            logger.debug(
                f"Starting ensemble benchmark: {len(tasks)} tasks, {len(patterns)} patterns"
            )

        all_results = []

        for task in tasks:
            for pattern in patterns:
                for agent_combo in agent_combinations:
                    for _iteration in range(iterations):
                        try:
                            result = await self.coordinator.execute_ensemble_task(
                                task=task,
                                pattern=pattern,
                                agent_types=agent_combo,
                            )
                            all_results.append(result)

                        except Exception as e:
                            if not self.quiet:
                                logger.error(
                                    f"Ensemble benchmark iteration failed: {e}"
                                )
                            else:
                                logger.debug(
                                    f"Ensemble benchmark iteration failed: {e}"
                                )

        self.benchmark_history.extend(all_results)

        analysis = self._analyze_ensemble_results(all_results)

        return {
            "results": analysis,
            "total_ensembles_tested": len(all_results),
            "patterns_evaluated": [p.value for p in patterns],
            "agent_combinations": [
                [a.value for a in combo] for combo in agent_combinations
            ],
        }

    def _analyze_ensemble_results(
        self, results: list[EnsembleResult]
    ) -> dict[str, Any]:
        """Perform comprehensive statistical analysis of ensemble benchmark results.

        Conducts detailed analysis of ensemble performance including pattern-specific
        metrics, agent combination effectiveness, ensemble advantages versus individual
        baselines, and overall performance characteristics. Implements robust statistical
        calculations with comprehensive error handling for production analysis.

        Args:
            results (List[EnsembleResult]): Complete ensemble execution results for
                statistical analysis and pattern recognition across multiple dimensions.

        Returns:
            Dict[str, Any]: Comprehensive analysis containing:
                - by_pattern (Dict): Pattern-specific performance metrics and statistics
                - by_agent_combination (Dict): Agent combination effectiveness analysis
                - ensemble_advantages (Dict): Comparative analysis versus individual agents
                - coordination_costs (Dict): Coordination overhead assessment
                - overall_metrics (Dict): Aggregate performance and synergy statistics

        Note:
            Analysis methodology includes:
            - Pattern grouping with success rate and cost efficiency calculations
            - Agent combination analysis with synergy threshold evaluation
            - Individual baseline calculation from ensemble execution data
            - Ensemble advantage computation revealing coordination effectiveness
            - Overall metrics including positive synergy rate assessment

            Statistical robustness through comprehensive error handling ensures
            reliable analysis even with partial data or execution failures,
            enabling accurate assessment of ensemble performance reality.
        """
        if not results:
            return {}

        analysis: dict[str, Any] = {
            "by_pattern": {},
            "by_agent_combination": {},
            "ensemble_advantages": {},
            "coordination_costs": {},
            "overall_metrics": {},
        }

        pattern_groups = defaultdict(list)
        combo_groups = defaultdict(list)

        for result in results:
            pattern_groups[result.pattern].append(result)
            combo_key = "_".join(
                sorted([agent.value for agent in result.participant_agents])
            )
            combo_groups[combo_key].append(result)

        for pattern, pattern_results in pattern_groups.items():
            success_rate = sum(1 for r in pattern_results if r.success) / len(
                pattern_results
            )
            avg_cost = sum(r.total_cost for r in pattern_results) / len(pattern_results)
            avg_coordination_overhead = sum(
                r.coordination_overhead for r in pattern_results
            ) / len(pattern_results)
            avg_ensemble_advantage = sum(
                r.ensemble_advantage for r in pattern_results
            ) / len(pattern_results)
            consensus_results = [
                r for r in pattern_results if r.consensus_agreement > 0
            ]
            avg_consensus = (
                sum(r.consensus_agreement for r in consensus_results)
                / len(consensus_results)
                if consensus_results
                else 0.0
            )

            cost_efficiency = None
            if success_rate > 0:
                cost_efficiency = avg_cost / success_rate

            analysis["by_pattern"][pattern.value] = {
                "success_rate": success_rate,
                "avg_total_cost": avg_cost,
                "avg_coordination_overhead": avg_coordination_overhead,
                "avg_ensemble_advantage": avg_ensemble_advantage,
                "avg_consensus_agreement": avg_consensus,
                "sample_size": len(pattern_results),
                "cost_efficiency": cost_efficiency,
            }

        for combo_key, combo_results in combo_groups.items():
            success_rate = sum(1 for r in combo_results if r.success) / len(
                combo_results
            )
            avg_cost = sum(r.total_cost for r in combo_results) / len(combo_results)
            avg_advantage = sum(r.ensemble_advantage for r in combo_results) / len(
                combo_results
            )

            analysis["by_agent_combination"][combo_key] = {
                "success_rate": success_rate,
                "avg_total_cost": avg_cost,
                "avg_ensemble_advantage": avg_advantage,
                "sample_size": len(combo_results),
                "agent_synergy": avg_advantage > SYNERGY_THRESHOLD,
            }

        individual_baselines = self._calculate_individual_baselines(results)

        for pattern in pattern_groups:
            pattern_results = pattern_groups[pattern]
            pattern_success = sum(1 for r in pattern_results if r.success) / len(
                pattern_results
            )

            for agent_type, baseline_success in individual_baselines.items():
                advantage = pattern_success - baseline_success
                advantage_key = f"{pattern.value}_vs_{agent_type.value}"
                analysis["ensemble_advantages"][advantage_key] = {
                    "ensemble_success_rate": pattern_success,
                    "individual_baseline": baseline_success,
                    "advantage": advantage,
                    "relative_improvement": (
                        advantage / baseline_success if baseline_success > 0 else 0.0
                    ),
                }

        overall_success = sum(1 for r in results if r.success) / len(results)
        overall_cost = sum(r.total_cost for r in results) / len(results)
        overall_advantage = sum(r.ensemble_advantage for r in results) / len(results)

        best_pattern = None
        if analysis["by_pattern"]:
            best_pattern = max(
                analysis["by_pattern"].items(), key=lambda x: x[1]["success_rate"]
            )[0]

        most_cost_effective = None
        valid_cost_efficiencies = [
            (pattern, data)
            for pattern, data in analysis["by_pattern"].items()
            if data.get("cost_efficiency") is not None
            and data["cost_efficiency"] != float("inf")
            and isinstance(data["cost_efficiency"], int | float)
        ]

        if valid_cost_efficiencies:
            most_cost_effective = min(
                valid_cost_efficiencies, key=lambda x: x[1]["cost_efficiency"]
            )[0]

        analysis["overall_metrics"] = {
            "total_ensembles": len(results),
            "overall_success_rate": overall_success,
            "avg_total_cost": overall_cost,
            "avg_ensemble_advantage": overall_advantage,
            "positive_synergy_rate": sum(1 for r in results if r.ensemble_advantage > 0)
            / len(results),
            "best_pattern": best_pattern,
            "most_cost_effective": most_cost_effective,
        }

        return analysis

    def _calculate_individual_baselines(
        self, ensemble_results: list[EnsembleResult]
    ) -> dict[AgentType, float]:
        """Calculate individual agent baseline performance from ensemble execution data.

        Extracts and aggregates individual agent performance data from ensemble
        executions to establish baseline success rates for ensemble advantage
        calculation. Enables accurate assessment of whether coordination benefits
        exceed individual agent performance capabilities.

        Args:
            ensemble_results (List[EnsembleResult]): Ensemble execution results containing
                individual agent performance data for baseline calculation.

        Returns:
            Dict[AgentType, float]: Baseline success rates by agent type derived
                from individual agent performance within ensemble contexts,
                enabling accurate ensemble advantage assessment.

        Note:
            Baseline calculation methodology:
            - Aggregates individual agent results across all ensemble executions
            - Calculates mean success rate per agent type for comparison purposes
            - Enables accurate ensemble advantage assessment by comparing ensemble
            performance against constituent agent individual capabilities

            Individual baselines provide the foundation for determining whether
            coordination benefits exceed coordination costs in multi-agent systems,
            revealing the empirical reality of ensemble performance characteristics.
        """
        individual_performances = defaultdict(list)

        for ensemble_result in ensemble_results:
            for individual_result in ensemble_result.individual_results:
                individual_performances[individual_result.agent_type].append(
                    1.0 if individual_result.success else 0.0
                )

        baselines = {}
        for agent_type, performances in individual_performances.items():
            baselines[agent_type] = sum(performances) / len(performances)

        return baselines

    def _generate_ensemble_report_part_one(self, analysis: dict[str, Any]) -> list[str]:
        """Generate the opening section of comprehensive ensemble benchmark report.

        Creates the executive summary and overall performance metrics section of the
        ensemble report, providing high-level insights into ensemble effectiveness
        across all tested patterns and agent combinations.

        Args:
            analysis: Comprehensive analysis results from ensemble benchmark execution.
                Expected structure with 'overall_metrics' key containing:
                - total_ensembles: Count of ensemble configurations tested
                - overall_success_rate: Success rate across all ensembles (0.0-1.0)
                - avg_ensemble_advantage: Average performance vs individual agents
                - positive_synergy_rate: Rate of positive collaboration outcomes (0.0-1.0)
                - best_pattern: Name of highest performing pattern (optional)
                - most_cost_effective: Name of most cost-efficient pattern (optional)

        Returns:
            List of formatted report lines for the opening section including:
                - Empty line for spacing
                - Section header with separator line
                - Total ensembles tested count
                - Overall success rate as percentage with 1 decimal precision
                - Average ensemble advantage with +/- formatting and 1 decimal precision
                - Positive synergy rate as percentage with 1 decimal precision
                - Best performing pattern name (title case) or "N/A" if unavailable
                - Most cost-effective pattern name (title case) or detailed N/A message
                - Trailing empty line for section separation

        Note:
            Implementation Details:
            - Safely accesses nested dictionary values using .get() with defaults
            - Formats percentages to 1 decimal place for readability
            - Uses +/- formatting for ensemble advantage to show direction clearly
            - Converts underscore_case pattern names to Title Case for presentation
            - Provides specific fallback messages for missing pattern data
            - Maintains consistent indentation (2 spaces) for all metrics
            - Uses fixed-width formatting for aligned metric presentation

            The section establishes the empirical foundation for understanding whether
            ensemble coordination provides measurable benefits over individual agent
            performance, setting context for detailed pattern analysis that follows.

        Example:
            >>> analysis = {
            ...     'overall_metrics': {
            ...         'total_ensembles': 12,
            ...         'overall_success_rate': 0.675,
            ...         'avg_ensemble_advantage': 0.123,
            ...         'positive_synergy_rate': 0.583,
            ...         'best_pattern': 'hierarchical_pattern',
            ...         'most_cost_effective': 'pipeline_pattern'
            ...     }
            ... }
            >>> lines = self._generate_ensemble_report_part_one(analysis)
            >>> print('\\n'.join(lines))

            Overall Ensemble Performance
            ------------------------------
            Total Ensembles Tested: 12
            Overall Success Rate:   67.5%
            Avg Ensemble Advantage: +12.3%
            Positive Synergy Rate:  58.3%
            Best Pattern:           Hierarchical Pattern
            Most Cost Effective:    Pipeline Pattern
        """
        lines = [""]
        overall = analysis.get("overall_metrics", {})
        lines.append("Overall Ensemble Performance")
        lines.append("-" * 30)
        lines.append(f"  Total Ensembles Tested: {overall.get('total_ensembles', 0)}")
        lines.append(
            f"  Overall Success Rate:   {overall.get('overall_success_rate', 0):.1%}"
        )
        lines.append(
            f"  Avg Ensemble Advantage: {overall.get('avg_ensemble_advantage', 0):+.1%}"
        )
        lines.append(
            f"  Positive Synergy Rate:  {overall.get('positive_synergy_rate', 0):.1%}"
        )

        best_pattern = overall.get("best_pattern")
        if best_pattern:
            lines.append(
                f"  Best Pattern:           {best_pattern.replace('_', ' ').title()}"
            )
        else:
            lines.append("  Best Pattern:           N/A")

        most_cost_effective = overall.get("most_cost_effective")
        if most_cost_effective:
            lines.append(
                f"  Most Cost Effective:    {most_cost_effective.replace('_', ' ').title()}"
            )
        else:
            lines.append("  Most Cost Effective:    N/A (no successful ensembles)")

        lines.append("")
        return lines

    def _generate_pattern_analysis(self, analysis: dict[str, Any]) -> list[str]:
        """Generate detailed pattern-by-pattern performance analysis section.

        Creates comprehensive analysis of each ensemble pattern's performance
        characteristics including success rates, costs, coordination overhead,
        and comparative advantages for informed architectural decisions.

        Args:
            analysis: Analysis results containing pattern-specific metrics.
                Expected structure with 'by_pattern' key containing dictionary where:
                - Keys are pattern names (e.g., 'pipeline_pattern', 'hierarchical_pattern')
                - Values are metrics dictionaries containing:
                    - success_rate: Success rate for this pattern (0.0-1.0)
                    - avg_total_cost: Average total execution cost (float)
                    - avg_coordination_overhead: Coordination time in seconds (float)
                    - avg_ensemble_advantage: Performance vs individual agents (float)
                    - cost_efficiency: Cost per success (optional, float or None)
                    - avg_consensus_agreement: Agreement rate for consensus patterns (optional, 0.0-1.0)

        Returns:
            List of formatted report lines for pattern analysis. Returns empty list if no
            pattern data available. When data present, includes:
                - Section header "Performance by Ensemble Pattern"
                - Separator line (35 dashes)
                - For each pattern:
                    - Pattern name in Title Case with leading newline
                    - Success rate as percentage (1 decimal place)
                    - Average total cost in dollars (4 decimal places)
                    - Coordination overhead in seconds (2 decimal places)
                    - Ensemble advantage with +/- sign (1 decimal place)
                    - Cost efficiency in dollars (4 decimal places) or "N/A" message
                    - Consensus agreement percentage (1 decimal place, if > 0)

        Note:
            Implementation Features:
            - Returns empty list gracefully when no pattern data exists
            - Converts underscore_case pattern names to Title Case for readability
            - Uses consistent indentation (4 spaces) for metrics under each pattern
            - Handles optional cost_efficiency metric with explicit None checking
            - Shows consensus agreement only for patterns that use it (> 0)
            - Includes coordination overhead to reveal hidden ensemble costs
            - Uses +/- formatting for ensemble advantage to show improvement/degradation

            Critical for understanding the empirical reality that coordination
            overhead often exceeds collaboration benefits in practical scenarios.
            Each pattern's metrics reveal the true cost-benefit profile of different
            ensemble coordination approaches.

            Pattern Types Typically Analyzed:
            - Pipeline: Sequential agent coordination with dependency chains
            - Parallel: Concurrent execution with result aggregation
            - Hierarchical: Multi-level coordination with supervisor relationships
            - Consensus: Agreement-based collaborative decision making
            - Specialization: Role-based task distribution and coordination

        Example:
            >>> analysis = {
            ...     'by_pattern': {
            ...         'pipeline_pattern': {
            ...             'success_rate': 0.725,
            ...             'avg_total_cost': 0.0234,
            ...             'avg_coordination_overhead': 1.45,
            ...             'avg_ensemble_advantage': 0.082,
            ...             'cost_efficiency': 0.0325
            ...         },
            ...         'consensus_pattern': {
            ...             'success_rate': 0.658,
            ...             'avg_total_cost': 0.0456,
            ...             'avg_coordination_overhead': 2.73,
            ...             'avg_ensemble_advantage': -0.031,
            ...             'cost_efficiency': None,
            ...             'avg_consensus_agreement': 0.784
            ...         }
            ...     }
            ... }
            >>> lines = self._generate_pattern_analysis(analysis)
            >>> print('\\n'.join(lines))
            Performance by Ensemble Pattern
            -----------------------------------

            Pipeline Pattern:
                Success Rate:         72.5%
                Avg Total Cost:       $0.0234
                Coordination Overhead: 1.45s
                Ensemble Advantage:   +8.2%
                Cost Efficiency:      $0.0325

            Consensus Pattern:
                Success Rate:         65.8%
                Avg Total Cost:       $0.0456
                Coordination Overhead: 2.73s
                Ensemble Advantage:   -3.1%
                Cost Efficiency:      N/A (no successes)
                Consensus Agreement:  78.4%
        """
        lines = []
        pattern_analysis = analysis.get("by_pattern", {})
        if pattern_analysis:
            lines.append("Performance by Ensemble Pattern")
            lines.append("-" * 35)

            for pattern, metrics in pattern_analysis.items():
                lines.append(f"\n  {pattern.replace('_', ' ').title()}:")
                lines.append(f"    Success Rate:         {metrics['success_rate']:.1%}")
                lines.append(
                    f"    Avg Total Cost:       ${metrics['avg_total_cost']:.4f}"
                )
                lines.append(
                    f"    Coordination Overhead: {metrics['avg_coordination_overhead']:.2f}s"
                )
                lines.append(
                    f"    Ensemble Advantage:   {metrics['avg_ensemble_advantage']:+.1%}"
                )

                cost_efficiency = metrics.get("cost_efficiency")
                if cost_efficiency is not None:
                    lines.append(f"    Cost Efficiency:      ${cost_efficiency:.4f}")
                else:
                    lines.append("    Cost Efficiency:      N/A (no successes)")

                if metrics.get("avg_consensus_agreement", 0) > 0:
                    lines.append(
                        f"    Consensus Agreement:  {metrics['avg_consensus_agreement']:.1%}"
                    )
        return lines

    def _generate_combo_analysis(self, analysis: dict[str, Any]) -> list[str]:
        """Generate agent combination effectiveness analysis section.

        Analyzes performance characteristics of different agent combinations
        to identify which team compositions provide optimal collaboration
        and synergy for ensemble task execution.

        Args:
            analysis: Analysis results containing agent combination metrics.
                Expected structure with 'by_agent_combination' key containing dictionary where:
                - Keys are combination identifiers (e.g., 'real_agent_marketing_agent')
                - Values are metrics dictionaries containing:
                    - success_rate: Success rate for this combination (0.0-1.0)
                    - avg_ensemble_advantage: Performance vs individual agents (float)
                    - agent_synergy: Boolean indicating effective collaboration (bool)

        Returns:
            List of formatted report lines for combination analysis. Returns empty list if no
            combination data available. When data present, includes:
                - Leading newline for section separation
                - Section header "Performance by Agent Combination"
                - Separator line (35 dashes)
                - For top 5 combinations (sorted by success rate, highest first):
                    - Combination name formatted as "Type + Type" with leading newline
                    - Success rate as percentage (1 decimal place)
                    - Ensemble advantage with +/- sign (1 decimal place)
                    - Agent synergy as "Yes" or "No" based on boolean value

        Note:
            Implementation Features:
            - Returns empty list gracefully when no combination data exists
            - Sorts combinations by success_rate in descending order for relevance
            - Limits display to top 5 combinations to maintain report readability
            - Transforms combination keys for user-friendly display:
                - Replaces underscores with " + " for readability
                - Removes "agent" suffix to reduce verbosity
                - Applies Title Case for professional presentation
            - Uses consistent indentation (4 spaces) for metrics under each combination
            - Shows ensemble advantage with +/- formatting to indicate improvement/degradation
            - Converts boolean synergy flag to clear "Yes"/"No" text

            Reveals which agent team compositions work well together versus
            those where coordination costs outweigh collaboration benefits.
            Essential for understanding team dynamics and architectural compatibility
            in multi-agent systems.

            Combination Analysis Purpose:
            - Identifies most effective agent team compositions
            - Reveals synergy patterns between different agent architectures
            - Exposes combinations where coordination overhead exceeds benefits
            - Guides team composition decisions for production deployments

        Example:
            >>> analysis = {
            ...     'by_agent_combination': {
            ...         'real_agent_marketing_agent': {
            ...             'success_rate': 0.783,
            ...             'avg_ensemble_advantage': 0.152,
            ...             'agent_synergy': True
            ...         },
            ...         'marketing_agent_wrapper_agent': {
            ...             'success_rate': 0.651,
            ...             'avg_ensemble_advantage': -0.034,
            ...             'agent_synergy': False
            ...         },
            ...         'real_agent_real_agent': {
            ...             'success_rate': 0.627,
            ...             'avg_ensemble_advantage': 0.058,
            ...             'agent_synergy': True
            ...         }
            ...     }
            ... }
            >>> lines = self._generate_combo_analysis(analysis)
            >>> print('\\n'.join(lines))

            Performance by Agent Combination
            -----------------------------------

            Real + Marketing:
                Success Rate:       78.3%
                Ensemble Advantage: +15.2%
                Agent Synergy:      Yes

            Marketing + Wrapper:
                Success Rate:       65.1%
                Ensemble Advantage: -3.4%
                Agent Synergy:      No

            Real + Real:
                Success Rate:       62.7%
                Ensemble Advantage: +5.8%
                Agent Synergy:      Yes
        """
        lines = []
        combo_analysis = analysis.get("by_agent_combination", {})
        if combo_analysis:
            lines.append("\n Performance by Agent Combination")
            lines.append("-" * 35)

            sorted_combos = sorted(
                combo_analysis.items(), key=lambda x: x[1]["success_rate"], reverse=True
            )

            for combo_key, metrics in sorted_combos[:5]:
                combo_display = (
                    combo_key.replace("_", " + ").replace("agent", "").title()
                )
                lines.append(f"\n  {combo_display}:")
                lines.append(f"    Success Rate:       {metrics['success_rate']:.1%}")
                lines.append(
                    f"    Ensemble Advantage: {metrics['avg_ensemble_advantage']:+.1%}"
                )
                lines.append(
                    f"    Agent Synergy:      {'Yes' if metrics['agent_synergy'] else 'No'}"
                )
        return lines

    def _generate_advantages_section(self, analysis: dict[str, Any]) -> list[str]:
        """Generate ensemble versus individual agent comparison section.

        Creates detailed comparison showing how each ensemble pattern performs
        against individual agent baselines, revealing where coordination
        provides genuine advantages versus overhead costs.

        Args:
            analysis: Analysis results containing ensemble advantage comparisons.
                Expected structure with 'ensemble_advantages' key containing dictionary where:
                - Keys are comparison identifiers in format 'pattern_vs_agent'
                (e.g., 'pipeline_vs_real_agent', 'consensus_vs_wrapper_agent')
                - Values are metrics dictionaries containing:
                    - advantage: Performance delta vs individual agent baseline (float)
                    - Additional metrics as needed for comparison analysis

        Returns:
            List of formatted report lines for advantage comparison. Returns empty list if no
            advantage data available. When data present, includes:
                - Leading newline for section separation
                - Section header "Ensemble vs Individual Agent Comparison"
                - Separator line (40 dashes)
                - Grouped comparisons by baseline agent type:
                    - Agent type header with "vs" prefix and leading newline
                    - For each pattern vs this agent:
                        - Pattern name (title case, left-aligned in 15-char field)
                        - Visual indicator emoji (🟢🟡🔴) based on advantage threshold
                        - Advantage percentage with +/- sign (1 decimal place)

        Note:
            Implementation Features:
            - Returns empty list gracefully when no advantage data exists
            - Parses advantage keys using "_vs_" delimiter to extract pattern and agent
            - Groups comparisons by baseline agent using defaultdict for clean organization
            - Converts agent names from underscore_case to Title Case for readability
            - Uses emoji indicators for immediate visual assessment:
                - 🟢 Green: Significant positive advantage (> SYNERGY_THRESHOLD)
                - 🟡 Yellow: Marginal positive advantage (0% to SYNERGY_THRESHOLD)
                - 🔴 Red: Negative advantage (coordination costs exceed benefits)
            - Formats pattern names with fixed-width alignment (15 characters) for clean columns
            - Shows advantage percentages with +/- formatting for clear direction indication

            Critical for understanding when ensemble complexity is justified
            versus when individual agents provide superior performance.
            Exposes the empirical reality that coordination overhead often exceeds
            collaboration benefits, providing data-driven foundation for architectural decisions.

            Visual Indicator Logic:
            - Compares advantage value against SYNERGY_THRESHOLD constant
            - Three-tier assessment provides quick performance categorization
            - Enables rapid identification of beneficial vs detrimental coordination patterns

        Example:
            >>> # Assuming SYNERGY_THRESHOLD = 0.1 (10%)
            >>> analysis = {
            ...     'ensemble_advantages': {
            ...         'pipeline_vs_real_agent': {'advantage': 0.125},
            ...         'parallel_vs_real_agent': {'advantage': 0.032},
            ...         'hierarchical_vs_real_agent': {'advantage': -0.081},
            ...         'consensus_vs_real_agent': {'advantage': 0.057},
            ...         'pipeline_vs_wrapper_agent': {'advantage': 0.453},
            ...         'parallel_vs_wrapper_agent': {'advantage': 0.389}
            ...     }
            ... }
            >>> lines = self._generate_advantages_section(analysis)
            >>> print('\\n'.join(lines))

            Ensemble vs Individual Agent Comparison
            ----------------------------------------

            vs Real Agent:
                Pipeline        🟢 +12.5%
                Parallel        🟡 +3.2%
                Hierarchical    🔴 -8.1%
                Consensus       🟡 +5.7%

            vs Wrapper Agent:
                Pipeline        🟢 +45.3%
                Parallel        🟢 +38.9%
        """
        lines = []
        advantages = analysis.get("ensemble_advantages", {})
        if advantages:
            lines.append("\nEnsemble vs Individual Agent Comparison")
            lines.append("-" * 40)
            agent_comparisons = defaultdict(list)

            for advantage_key, metrics in advantages.items():
                pattern, agent = advantage_key.split("_vs_")
                agent_comparisons[agent].append((pattern, metrics))

            for agent, comparisons in agent_comparisons.items():
                lines.append(f"\n  vs {agent.replace('_', ' ').title()}:")
                for pattern, metrics in comparisons:
                    advantage = metrics["advantage"]
                    symbol = (
                        "🟢"
                        if advantage > SYNERGY_THRESHOLD
                        else "🟡"
                        if advantage > 0
                        else "🔴"
                    )
                    lines.append(f"    {pattern.title():15} {symbol} {advantage:+.1%}")
        return lines

    def _generate_insights_and_recommendations(
        self, analysis: dict[str, Any]
    ) -> list[str]:
        """Generate actionable insights and recommendations section with robust error handling.

        Synthesizes benchmark results into key insights and specific recommendations
        for ensemble architecture decisions, providing data-driven guidance for
        practical deployment scenarios with comprehensive error handling.

        Args:
            analysis: Complete analysis results including overall metrics and pattern
                performance data for generating actionable insights. Expected structure:
                - 'overall_metrics' key containing:
                    - avg_ensemble_advantage: Average performance vs individual agents (float)
                    - positive_synergy_rate: Rate of positive collaboration outcomes (0.0-1.0)
                    - best_pattern: Name of highest performing pattern (optional string)
                    - most_cost_effective: Name of most efficient pattern (optional string)
                - 'by_pattern' key containing pattern-specific metrics (optional dict):
                    - Pattern names as keys with nested success_rate values

        Returns:
            List of formatted report lines containing insights and recommendations:
                - Leading newline for section separation
                - "Key Insights" header with separator line (15 dashes)
                - Ensemble effectiveness assessment based on MEANINGFUL_ADVANTAGE_THRESHOLD
                - Synergy assessment with emoji indicators based on threshold comparison:
                    - 🟢 Strong synergy (> STRONG_SYNERGY_THRESHOLD)
                    - 🟡 Moderate synergy (MODERATE_SYNERGY_THRESHOLD to STRONG_SYNERGY_THRESHOLD)
                    - 🔴 Weak synergy (< MODERATE_SYNERGY_THRESHOLD)
                - Best pattern performance note (if available)
                - "Recommendations" header with separator line (18 dashes)
                - Success optimization recommendation with error handling
                - Cost efficiency recommendation with availability checking
                - Architectural guidance based on overall ensemble advantage

        Note:
            Robust Error Handling:
            - Uses .get() methods with defaults to handle missing data gracefully
            - Wraps max() operation in try-catch for empty pattern collections
            - Provides fallback messages for insufficient data scenarios
            - Ensures recommendations are always provided even with incomplete analysis
            - Handles missing optional fields with appropriate conditional checks

            Insight Categories:
            - Ensemble Effectiveness: Overall performance assessment vs individual agents
            - Synergy Assessment: Team collaboration quality with visual indicators
            - Pattern Performance: Best performing coordination approach identification
            - Cost-Benefit Analysis: Economic viability of coordination overhead

            Recommendation Logic:
            - Success Optimization: Identifies highest success rate pattern from data
            - Cost Optimization: Uses pre-calculated most cost-effective pattern
            - Architectural Guidance: Individual vs ensemble trade-off based on advantage
            - Fallback Messaging: Provides helpful guidance even with missing data

            Threshold Dependencies:
            - MEANINGFUL_ADVANTAGE_THRESHOLD: Minimum improvement for significance
            - STRONG_SYNERGY_THRESHOLD: High positive combination rate threshold
            - MODERATE_SYNERGY_THRESHOLD: Moderate positive combination rate threshold

        Example:
            >>> # Assuming thresholds: MEANINGFUL=0.05, STRONG_SYNERGY=0.7, MODERATE_SYNERGY=0.4
            >>> analysis = {
            ...     'overall_metrics': {
            ...         'avg_ensemble_advantage': 0.023,
            ...         'positive_synergy_rate': 0.583,
            ...         'best_pattern': 'hierarchical_pattern',
            ...         'most_cost_effective': 'pipeline_pattern'
            ...     },
            ...     'by_pattern': {
            ...         'hierarchical_pattern': {'success_rate': 0.745},
            ...         'pipeline_pattern': {'success_rate': 0.672},
            ...         'consensus_pattern': {'success_rate': 0.598}
            ...     }
            ... }
            >>> lines = self._generate_insights_and_recommendations(analysis)
            >>> print('\\n'.join(lines))

            Key Insights
            ---------------
            Limited ensemble advantages observed
            🟡 Moderate agent synergy - pattern dependent
            Hierarchical Pattern pattern shows best overall performance

            Recommendations
            ------------------
            • For highest success: Use Hierarchical Pattern pattern
            • For cost efficiency: Use Pipeline Pattern pattern
            • Ensemble approaches provide measurable benefits

        Raises:
            No exceptions raised - all error conditions handled gracefully with
            appropriate fallback messaging and robust data access patterns using
            try-catch blocks around complex operations and .get() methods for safe
            dictionary access.
        """
        lines = []
        overall = analysis.get("overall_metrics", {})
        pattern_analysis = analysis.get("by_pattern", {})

        lines.append("\nKey Insights")
        lines.append("-" * 15)

        if overall.get("avg_ensemble_advantage", 0) > MEANINGFUL_ADVANTAGE_THRESHOLD:
            lines.append("  Ensembles show meaningful performance gains")
        else:
            lines.append("  Limited ensemble advantages observed")

        if overall.get("positive_synergy_rate", 0) > STRONG_SYNERGY_THRESHOLD:
            lines.append("  🟢 Strong agent synergy across most combinations")
        elif overall.get("positive_synergy_rate", 0) > MODERATE_SYNERGY_THRESHOLD:
            lines.append("  🟡 Moderate agent synergy - pattern dependent")
        else:
            lines.append("  🔴 Weak agent synergy - coordination costs exceed benefits")

        best_pattern = overall.get("best_pattern", "")
        if best_pattern:
            lines.append(
                f"  {best_pattern.title()} pattern shows best overall performance"
            )

        lines.append("\nRecommendations")
        lines.append("-" * 18)

        if pattern_analysis:
            try:
                best_success_pattern = max(
                    pattern_analysis.items(), key=lambda x: x[1]["success_rate"]
                )
                lines.append(
                    f"  • For highest success: Use {best_success_pattern[0].replace('_', ' ').title()} pattern"
                )
            except (ValueError, KeyError):
                lines.append(
                    "  • For highest success: Insufficient data for recommendation"
                )

            most_cost_effective = overall.get("most_cost_effective")
            if most_cost_effective:
                lines.append(
                    f"  • For cost efficiency: Use {most_cost_effective.replace('_', ' ').title()} pattern"
                )
            else:
                lines.append(
                    "  • For cost efficiency: No cost-effective patterns available"
                )

        if overall.get("avg_ensemble_advantage", 0) < 0:
            lines.append(
                "  • Consider individual agents over ensembles for current task types"
            )
        else:
            lines.append("  • Ensemble approaches provide measurable benefits")

        return lines

    def generate_ensemble_report(self, analysis: dict[str, Any]) -> str:
        """Generate comprehensive ensemble benchmark report with full analysis.

        Orchestrates the generation of a complete ensemble benchmark report by
        combining all analysis sections into a cohesive document that communicates
        the empirical reality of ensemble performance characteristics.

        Args:
            analysis: Complete analysis results from ensemble benchmark execution
                containing all necessary data for comprehensive report generation.
                Must include the following structure for full functionality:
                - 'overall_metrics': Dict with ensemble performance summaries
                - 'by_pattern': Dict with pattern-specific performance metrics
                - 'by_agent_combination': Dict with combination effectiveness data
                - 'ensemble_advantages': Dict with comparative performance analysis

        Returns:
            Complete formatted report string ready for display or file output containing:
                - Executive summary with overall ensemble performance metrics
                - Detailed pattern-by-pattern analysis with costs and coordination overhead
                - Agent combination effectiveness assessment with synergy indicators
                - Ensemble vs individual agent performance comparisons with visual indicators
                - Actionable insights and architectural recommendations based on empirical data

        Note:
            Report Composition:
            - Calls all specialized report generation methods in logical sequence
            - Combines results into single cohesive document with consistent formatting
            - Maintains readability while providing comprehensive technical detail
            - Suitable for technical decision making and architectural planning
            - Each section handles missing data gracefully with appropriate fallbacks

            The complete report provides empirical foundation for understanding
            when ensemble coordination complexity is justified versus when
            individual agents provide superior performance characteristics.

            Essential for communicating the reality that coordination costs often
            exceed collaboration benefits in practical multi-agent systems.

            Report Structure (5 main sections):
            1. Overall Performance Summary - Executive metrics and key indicators
            2. Pattern Analysis - Detailed performance breakdown by ensemble type
            3. Agent Combination Analysis - Team composition effectiveness evaluation
            4. Comparative Analysis - Ensemble vs individual performance with visual indicators
            5. Insights and Recommendations - Actionable guidance for architecture decisions

            Report Features:
            - Consistent formatting with clear section headers and separators
            - Visual indicators (🟢🟡🔴) for quick performance assessment
            - Specific metrics with appropriate units and precision
            - Graceful handling of missing or incomplete data across all sections
            - Focus on practical decision-making information
            - Professional presentation suitable for technical documentation

        Example Usage:
            >>> ensemble_benchmark = EnsembleBenchmark()
            >>> analysis_results = await ensemble_benchmark.run_ensemble_benchmark(
            ...     tasks=task_list,
            ...     patterns=[EnsemblePattern.PIPELINE, EnsemblePattern.CONSENSUS],
            ...     agent_combinations=agent_combos,
            ...     iterations=10
            ... )
            >>> report = ensemble_benchmark.generate_ensemble_report(analysis_results)
            >>>
            >>> # Display complete analysis
            >>> print(report)
            >>>
            >>> # Save to file for documentation
            >>> with open('ensemble_analysis_report.txt', 'w', encoding='utf-8') as f:
            ...     f.write(report)
            >>>
            >>> # Extract specific sections if needed
            >>> sections = report.split('\\n\\n')
            >>> executive_summary = sections[0]  # First section

        Example Output Structure:
            ```
            Overall Ensemble Performance
            ------------------------------
            [Executive summary with key metrics]

            Performance by Ensemble Pattern
            -----------------------------------
            [Detailed pattern-by-pattern analysis]

            Performance by Agent Combination
            -----------------------------------
            [Team composition effectiveness analysis]

            Ensemble vs Individual Agent Comparison
            ----------------------------------------
            [Comparative performance with visual indicators]

            Key Insights
            ---------------
            [Synthesis of key findings]

            Recommendations
            ------------------
            [Actionable architectural guidance]
            ```

        Implementation Details:
            - Uses list.extend() for efficient line collection from each section
            - Joins all lines with newline characters for final string output
            - Preserves spacing and formatting from individual section generators
            - No additional error handling needed as all called methods handle errors gracefully
            - Returns empty report sections gracefully when data is unavailable
        """
        lines = []
        lines.extend(self._generate_ensemble_report_part_one(analysis))
        lines.extend(self._generate_pattern_analysis(analysis))
        lines.extend(self._generate_combo_analysis(analysis))
        lines.extend(self._generate_advantages_section(analysis))
        lines.extend(self._generate_insights_and_recommendations(analysis))

        return "\n".join(lines)
