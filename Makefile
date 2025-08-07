# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------

.PHONY: help install install-dev test-unit lint lint-check format format-check type-check check run run-random clean version
.PHONY: run-enhanced run-ensemble run-network-test ensemble-quick analyze-results analyze-enhanced analyze-ensemble analyze-network show-results help-analysis

.DEFAULT_GOAL := help

# Terminal Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# Project Metadata
PROJECT_NAME := ai-agents-reality-check
REPO := https://github.com/Cre4T3Tiv3/ai-agents-reality-check
VERSION := 0.1.0
PYTHON_VERSION := 3.11+
AUTHOR := Jesse Moses (@Cre4T3Tiv3) - jesse@bytestacklabs.com

# Virtual Environment
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON := $(VENV_BIN)/python
UV := uv

# Python Path Export
export PYTHONPATH := $(PWD)/src

help: ## Show help message
	@echo "$(BLUE)AI Agents Reality Check - Enhanced Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Installation:$(NC)"
	@echo "  $(GREEN)make install$(NC)          - Install core dependencies"
	@echo "  $(GREEN)make install-dev$(NC)      - Install dev dependencies and hooks"
	@echo ""
	@echo "$(YELLOW)Testing & QA:$(NC)"
	@echo "  $(GREEN)make test-unit$(NC)        - Run all unit tests"
	@echo "  $(GREEN)make lint$(NC)             - Auto-format code (black, isort, ruff)"
	@echo "  $(GREEN)make lint-check$(NC)       - Check lint without modifying"
	@echo "  $(GREEN)make type-check$(NC)       - Run mypy type checker"
	@echo "  $(GREEN)make check$(NC)            - Full QA: lint, type-check, test"
	@echo ""
	@echo "$(YELLOW)Core Benchmarks:$(NC)"
	@echo "  $(GREEN)make run$(NC)              - Run 5-iteration standard benchmark"
	@echo "  $(GREEN)make run-random$(NC)       - Run benchmark with randomized tasks"
	@echo ""
	@echo "$(YELLOW)Enhanced Benchmarks:$(NC)"
	@echo "  $(GREEN)make run-enhanced$(NC)     - Run with tool failures + enhanced stats"
	@echo "  $(GREEN)make run-ensemble$(NC)     - Run agent ensemble collaboration tests"
	@echo "  $(GREEN)make run-network-test$(NC) - Test network resilience (unstable conditions)"
	@echo "  $(GREEN)make ensemble-quick$(NC)   - Quick ensemble test (parallel pattern)"
	@echo ""
	@echo "$(YELLOW)Analysis Commands:$(NC)"
	@echo "  $(GREEN)make analyze-results$(NC)  - Auto-analyze latest benchmark results"
	@echo "  $(GREEN)make analyze-enhanced$(NC) - Analyze enhanced results (99% CI)"
	@echo "  $(GREEN)make analyze-ensemble$(NC) - Analyze ensemble collaboration"
	@echo "  $(GREEN)make show-results$(NC)     - Show available result files"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  $(GREEN)make clean$(NC)            - Remove caches and build artifacts"
	@echo "  $(GREEN)make version$(NC)          - Show system + project version info"
	@echo "  $(GREEN)make help-analysis$(NC)    - Show detailed analysis command help"

install: ## Install core dependencies
	@echo "$(BLUE)Installing core project dependencies...$(NC)"
	$(UV) sync
	@echo "$(GREEN)Done.$(NC)"

install-dev: ## Install dev dependencies + hooks
	@echo "$(BLUE)Installing development environment...$(NC)"
	@if ! grep -q '\[.*dev.*\]' uv.lock; then \
		echo "$(YELLOW)Dev group not found in uv.lock — generating it...$(NC)"; \
		$(UV) pip install ".[dev]"; \
		$(UV) lock; \
	fi
	$(UV) sync --all-extras
	@echo "$(GREEN)Dev environment ready.$(NC)"

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(NC)"
	$(UV) run pytest tests/unit/ -v --tb=short

lint: ## Auto-format code (black + ruff, including import sorting)
	@echo "$(YELLOW)Auto-formatting code...$(NC)"
	$(UV) run black ai_agents_reality_check/ tests/
	$(UV) run ruff format ai_agents_reality_check/ tests/
	$(UV) run ruff check --fix ai_agents_reality_check/ tests/

lint-check: ## Check formatting and lint (black + ruff)
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	$(UV) run black --check ai_agents_reality_check/ tests/
	$(UV) run ruff check ai_agents_reality_check/ tests/

type-check: ## Run type checking with mypy
	@echo "$(YELLOW)Running type checking...$(NC)"
	$(UV) run mypy ai_agents_reality_check/ tests/

check: lint-check type-check test-unit ## Run full code quality suite
	@echo "$(GREEN)All quality checks passed!$(NC)"

run: ## Run short benchmark (5 iterations)
	@echo "$(CYAN)Running AI Agents Reality Check Benchmark...$(NC)"
	@echo "$(YELLOW)Mode: Quick (5 iterations)$(NC)"
	@echo ""
	$(UV) run agent-benchmark quick $(ARGS)

run-random: ## Run randomized short benchmark (5 iterations)
	@echo "$(CYAN)Running randomized AI Agents Reality Check...$(NC)"
	@echo "$(YELLOW)Mode: Quick + Randomized (seed: 2025)$(NC)"
	@echo ""
	$(UV) run agent-benchmark quick --randomize --seed 2025 $(ARGS)

run-enhanced: ## Run enhanced benchmark with tool failures + enhanced stats
	@echo "$(MAGENTA)Running enhanced AI Agents Reality Check...$(NC)"
	@echo "$(YELLOW)Mode: Enhanced (10 iterations + tool failures + stats)$(NC)"
	@echo "$(YELLOW)Features: Tool failure simulation, network conditions, CI analysis$(NC)"
	@echo ""
	$(UV) run agent-benchmark benchmark --tool-failures --enhanced-stats --confidence 0.95 --iterations 10 --quiet $(ARGS)

run-ensemble: ## Run ensemble benchmark
	@echo "$(MAGENTA)Running agent ensemble collaboration benchmark...$(NC)"
	$(UV) run agent-benchmark ensemble --iterations 3 --patterns all --combinations mixed --quiet $(ARGS)

run-network-test: ## Test network resilience (unstable conditions)
	@echo "$(MAGENTA)Testing agent resilience under unstable network conditions...$(NC)"
	$(UV) run agent-benchmark network-test unstable --iterations 15 --quiet $(ARGS)

ensemble-quick: ## Quick ensemble test (parallel pattern)
	@echo "$(CYAN)Running quick ensemble test...$(NC)"
	$(UV) run agent-benchmark ensemble --iterations 3 --patterns parallel --combinations mixed --quiet $(ARGS)

analyze-results: ## Analyze saved benchmark results (auto-detects or specify file)
	@echo "$(YELLOW)Analyzing benchmark results...$(NC)"
	@echo ""
	@if [ -n "$(FILE)" ]; then \
		echo "$(CYAN)Analyzing specified file: $(FILE)$(NC)"; \
		if [ -f "$(FILE)" ]; then \
			$(UV) run agent-benchmark analyze "$(FILE)" --confidence 0.95 --outliers; \
		else \
			echo "$(RED)File $(FILE) not found$(NC)"; \
		fi \
	elif [ -f "logs/enhanced_results.json" ]; then \
		echo "$(CYAN)Found enhanced results - analyzing with full statistical suite$(NC)"; \
		$(UV) run agent-benchmark analyze logs/enhanced_results.json --confidence 0.95 --outliers --export-stats logs/enhanced_analysis.json; \
	elif [ -f "logs/ensemble_results.json" ]; then \
		echo "$(CYAN)Found ensemble results - analyzing multi-agent collaboration$(NC)"; \
		$(UV) run agent-benchmark analyze logs/ensemble_results.json --confidence 0.95 --outliers; \
	elif [ -f "logs/quick_random_results.json" ]; then \
		echo "$(CYAN)Found randomized quick results - analyzing with seed validation$(NC)"; \
		$(UV) run agent-benchmark analyze logs/quick_random_results.json --confidence 0.95 --outliers; \
	elif [ -f "logs/quick_results.json" ]; then \
		echo "$(CYAN)Found quick results - analyzing standard benchmark$(NC)"; \
		$(UV) run agent-benchmark analyze logs/quick_results.json --confidence 0.95 --outliers; \
	elif [ -f "results.json" ]; then \
		echo "$(CYAN)Found results.json in root - analyzing$(NC)"; \
		$(UV) run agent-benchmark analyze results.json --confidence 0.95 --outliers; \
	else \
		echo "$(RED)No results files found. Available files:$(NC)"; \
		@ls -la logs/*.json 2>/dev/null || echo "  No JSON files in logs/"; \
		@ls -la results.json 2>/dev/null || echo "  No results.json in root"; \
		echo ""; \
		echo "$(YELLOW)Run one of these commands first:$(NC)"; \
		echo "  make run               # → logs/quick_results.json"; \
		echo "  make run-random        # → logs/quick_random_results.json"; \
		echo "  make run-enhanced      # → logs/enhanced_results.json"; \
		echo "  make run-ensemble      # → logs/ensemble_results.json"; \
		echo ""; \
		echo "$(YELLOW)Or specify a file:$(NC)"; \
		echo "  make analyze-results FILE=logs/your_file.json"; \
	fi

analyze-enhanced: ## Analyze enhanced benchmark results with full stats
	@echo "$(MAGENTA)Analyzing enhanced results with comprehensive statistics...$(NC)"
	@if [ -f "logs/enhanced_results.json" ]; then \
		$(UV) run agent-benchmark analyze logs/enhanced_results.json --confidence 0.99 --outliers --export-stats logs/enhanced_analysis.json; \
		echo "$(GREEN)Enhanced analysis exported to logs/enhanced_analysis.json$(NC)"; \
	else \
		echo "$(RED)No enhanced results found. Run: make run-enhanced$(NC)"; \
	fi

analyze-ensemble: ## Analyze ensemble benchmark results
	@echo "$(MAGENTA)Analyzing ensemble collaboration results...$(NC)"
	@if [ -f "logs/ensemble_results.json" ]; then \
		$(UV) run agent-benchmark analyze logs/ensemble_results.json --confidence 0.95 --outliers; \
	else \
		echo "$(RED)No ensemble results found. Run: make run-ensemble$(NC)"; \
	fi

analyze-network: ## Analyze network resilience test results
	@echo "$(MAGENTA)Analyzing network resilience results...$(NC)"
	@if [ -f "logs/network_results.json" ]; then \
		$(UV) run agent-benchmark analyze logs/network_results.json --confidence 0.95 --outliers; \
	else \
		echo "$(RED)No network results found. Run: make run-network-test$(NC)"; \
	fi

show-results: ## Show all available result files and their details
	@echo "$(CYAN)Available Result Files:$(NC)"
	@echo ""
	@if [ -f "logs/enhanced_results.json" ]; then \
		echo "$(GREEN)logs/enhanced_results.json$(NC)     - Enhanced benchmark with tool failures & stats"; \
		@stat -c "   Modified: %y (Size: %s bytes)" logs/enhanced_results.json 2>/dev/null || stat -f "   Modified: %Sm (Size: %z bytes)" logs/enhanced_results.json 2>/dev/null || echo "   (stat info unavailable)"; \
	fi
	@if [ -f "logs/ensemble_results.json" ]; then \
		echo "$(GREEN)logs/ensemble_results.json$(NC)     - Multi-agent collaboration benchmark"; \
		@stat -c "   Modified: %y (Size: %s bytes)" logs/ensemble_results.json 2>/dev/null || stat -f "   Modified: %Sm (Size: %z bytes)" logs/ensemble_results.json 2>/dev/null || echo "   (stat info unavailable)"; \
	fi
	@if [ -f "logs/quick_random_results.json" ]; then \
		echo "$(GREEN)logs/quick_random_results.json$(NC) - Quick benchmark with randomized task order"; \
		@stat -c "   Modified: %y (Size: %s bytes)" logs/quick_random_results.json 2>/dev/null || stat -f "   Modified: %Sm (Size: %z bytes)" logs/quick_random_results.json 2>/dev/null || echo "   (stat info unavailable)"; \
	fi
	@if [ -f "logs/quick_results.json" ]; then \
		echo "$(GREEN)logs/quick_results.json$(NC)        - Standard quick benchmark"; \
		@stat -c "   Modified: %y (Size: %s bytes)" logs/quick_results.json 2>/dev/null || stat -f "   Modified: %Sm (Size: %z bytes)" logs/quick_results.json 2>/dev/null || echo "   (stat info unavailable)"; \
	fi
	@if [ -f "logs/network_results.json" ]; then \
		echo "$(GREEN)logs/network_results.json$(NC)      - Network resilience test results"; \
		@stat -c "   Modified: %y (Size: %s bytes)" logs/network_results.json 2>/dev/null || stat -f "   Modified: %Sm (Size: %z bytes)" logs/network_results.json 2>/dev/null || echo "   (stat info unavailable)"; \
	fi
	@if [ -f "results.json" ]; then \
		echo "$(GREEN)results.json$(NC)                   - Generic results file"; \
		@stat -c "   Modified: %y (Size: %s bytes)" results.json 2>/dev/null || stat -f "   Modified: %Sm (Size: %z bytes)" results.json 2>/dev/null || echo "   (stat info unavailable)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Usage Examples:$(NC)"
	@echo "  make analyze-results                              # Auto-detect latest"
	@echo "  make analyze-results FILE=logs/enhanced_results.json  # Specific file"
	@echo "  make analyze-enhanced                             # Enhanced with 99% CI"
	@echo "  make analyze-ensemble                             # Ensemble-specific"

help-analysis: ## Show analysis command help
	@echo "$(BLUE)Analysis Commands Help$(NC)"
	@echo ""
	@echo "$(YELLOW)Basic Analysis:$(NC)"
	@echo "  make analyze-results                              # Auto-detect and analyze latest results"
	@echo "  make analyze-results FILE=path/to/file.json       # Analyze specific file"
	@echo "  make show-results                                 # Show all available result files"
	@echo ""
	@echo "$(YELLOW)Specialized Analysis:$(NC)"
	@echo "  make analyze-enhanced                             # Enhanced results with 99% confidence"
	@echo "  make analyze-ensemble                             # Ensemble collaboration analysis"
	@echo "  make analyze-network                              # Network resilience analysis"
	@echo ""
	@echo "$(YELLOW)Result File Types:$(NC)"
	@echo "  enhanced_results.json     - Tool failures, network sim, enhanced stats"
	@echo "  ensemble_results.json     - Multi-agent collaboration patterns"
	@echo "  quick_random_results.json - Randomized task order (seed: 2025)"
	@echo "  quick_results.json        - Standard quick benchmark"
	@echo "  network_results.json      - Network resilience under instability"

# Comprehensive benchmark with all features
run-comprehensive: ## Run comprehensive benchmark with all enhanced features
	@echo "$(MAGENTA)Running comprehensive benchmark with all enhanced features...$(NC)"
	$(UV) run agent-benchmark benchmark \
		--tool-failures \
		--enhanced-stats \
		--network unstable \
		--confidence 0.99 \
		--iterations 15 \
		--output comprehensive_results.json \
		--randomize \
		--seed 2025 \
		--quiet \
		$(ARGS)
	@echo "$(GREEN)Comprehensive benchmark completed! Results saved to comprehensive_results.json$(NC)"

# Network condition testing
run-network-stable: ## Test stable network conditions (baseline)
	$(UV) run agent-benchmark network-test stable --iterations 10 --quiet $(ARGS)

run-network-slow: ## Test slow network conditions
	$(UV) run agent-benchmark network-test slow --iterations 10 --quiet $(ARGS)

run-network-degraded: ## Test degraded network conditions
	$(UV) run agent-benchmark network-test degraded --iterations 10 --quiet $(ARGS)

run-network-peak: ## Test peak hours network conditions
	$(UV) run agent-benchmark network-test peak_hours --iterations 10 --quiet $(ARGS)

# Ensemble pattern testing
ensemble-pipeline: ## Test pipeline ensemble pattern
	$(UV) run agent-benchmark ensemble --patterns pipeline --iterations 5 --quiet $(ARGS)

ensemble-consensus: ## Test consensus ensemble pattern
	$(UV) run agent-benchmark ensemble --patterns consensus --iterations 5 --quiet $(ARGS)

ensemble-hierarchical: ## Test hierarchical ensemble pattern
	$(UV) run agent-benchmark ensemble --patterns hierarchical --iterations 5 --quiet $(ARGS)

run-debug: ## Run benchmark with debug logging (verbose mode)
	@echo "$(YELLOW)Running benchmark with debug logging...$(NC)"
	$(UV) run agent-benchmark benchmark --iterations 4 --quick-mode $(ARGS)

# Utility commands
clean: ## Remove cache and temp files
	@echo "$(YELLOW)Cleaning cache and temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .pytest_cache .mypy_cache $(VENV_DIR)
	rm -rf dist/ build/ *.egg-info/ logs/ traces/wrapper/
	@echo "$(GREEN)Cleanup completed.$(NC)"

version: ## Show system details
	@echo "$(BLUE)=== AI Agents Reality Check - System Information ===$(NC)"
	@echo "Project:    $(PROJECT_NAME)"
	@echo "Author:     $(AUTHOR)"
	@echo "Repository: $(REPO)"
	@echo "Version:    $(VERSION)"
	@echo "Python:     $(PYTHON_VERSION)"
	@echo "Platform:   $(shell python -c 'import platform; print(platform.system())')"
	@echo "UV:         $(shell uv --version 2>/dev/null || echo 'Not installed')"

# Help for specific feature sets
help-enhanced: ## Show help for enhanced features
	@echo "$(BLUE)Enhanced Features Help$(NC)"
	@echo ""
	@echo "$(YELLOW)Tool Failure Simulation:$(NC)"
	@echo "  --tool-failures          Enable realistic tool failure rates"
	@echo "  --network CONDITION      Set network condition (stable/slow/unstable/degraded/peak_hours)"
	@echo ""
	@echo "$(YELLOW)Enhanced Statistics:$(NC)"
	@echo "  --enhanced-stats         Enable confidence intervals and effect sizes"
	@echo "  --confidence LEVEL       Set confidence level (0.90, 0.95, 0.99)"
	@echo ""
	@echo "$(YELLOW)Ensemble Patterns:$(NC)"
	@echo "  --patterns LIST          Comma-separated: pipeline,parallel,hierarchical,consensus,specialization"
	@echo "  --combinations TYPE      Agent combinations: all,mixed,homogeneous"

help-examples: ## Show usage examples
	@echo "$(BLUE)Usage Examples$(NC)"
	@echo ""
	@echo "$(YELLOW)Basic Usage:$(NC)"
	@echo "  make run                 # Quick 5-iteration benchmark"
	@echo "  make run-enhanced        # Enhanced with tool failures"
	@echo ""
	@echo "$(YELLOW)Advanced Usage:$(NC)"
	@echo "  make run-network-test    # Network resilience testing"
	@echo "  make run-ensemble        # Multi-agent collaboration"
	@echo ""
	@echo "$(YELLOW)Analysis:$(NC)"
	@echo "  make analyze-results     # Analyze existing results.json"

# Verbose versions for debugging (without --quiet flag)
run-enhanced-verbose: ## Run enhanced benchmark with verbose logging
	@echo "$(MAGENTA)Running enhanced benchmark with verbose logging...$(NC)"
	$(UV) run agent-benchmark benchmark --tool-failures --enhanced-stats --confidence 0.95 --iterations 10 $(ARGS)

run-ensemble-verbose: ## Run ensemble benchmark with verbose logging
	@echo "$(MAGENTA)Running ensemble benchmark with verbose logging...$(NC)"
	$(UV) run agent-benchmark ensemble --iterations 3 --patterns all --combinations mixed $(ARGS)

run-network-test-verbose: ## Test network resilience with verbose logging
	@echo "$(MAGENTA)Testing network resilience with verbose logging...$(NC)"
	$(UV) run agent-benchmark network-test unstable --iterations 15 $(ARGS)