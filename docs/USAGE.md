# Usage Guide: AI Agents Reality Check

## Quick Start

The fastest way to run a benchmark is using the Makefile:

```bash
# Standard 5-iteration benchmark
make run

# Randomized task order (reproducible with seed)
make run-random
```

These commands wrap the internal CLI with sensible defaults for immediate results.

---

## File Output & Analysis Workflow

### Benchmark Output Files

Each benchmark command produces its own uniquely named results file in the `logs/` directory:

#### Standard Benchmarks (Auto-save)
```bash
make run              # → logs/quick_results.json
make run-random       # → logs/quick_random_results.json  
make run-enhanced     # → logs/enhanced_results.json
make run-debug        # → logs/quick_results.json (with verbose logging)
```

#### Specialized Benchmarks (Auto-save)
```bash
make run-ensemble          # → logs/ensemble_results.json
make run-network-test      # → logs/network_results.json
make run-comprehensive     # → comprehensive_results.json
make ensemble-quick        # → logs/ensemble_results.json
```

#### Network Condition Testing
```bash
make run-network-stable    # → logs/network_results.json
make run-network-slow      # → logs/network_results.json
make run-network-degraded  # → logs/network_results.json
make run-network-peak      # → logs/network_results.json
make run-network-unstable  # → logs/network_results.json (alias for run-network-test)
```

#### Ensemble Pattern Testing
```bash
make ensemble-pipeline     # → logs/ensemble_results.json
make ensemble-consensus    # → logs/ensemble_results.json
make ensemble-hierarchical # → logs/ensemble_results.json
```

#### Verbose Variants (For Debugging)
```bash
make run-enhanced-verbose     # → logs/enhanced_results.json (with verbose output)
make run-ensemble-verbose     # → logs/ensemble_results.json (with verbose output)
make run-network-test-verbose # → logs/network_results.json (with verbose output)
```

### Analysis Commands

The analysis commands **do not** generate results files - they only read and analyze existing results:

```bash
# Auto-detects most recent results file in logs/
make analyze-results

# Analyze specific results file
make analyze-results FILE=logs/enhanced_results.json

# Specialized analysis commands
make analyze-enhanced      # Enhanced results with 99% confidence
make analyze-ensemble      # Ensemble collaboration analysis
make analyze-network       # Network resilience analysis
```

#### Analysis File Detection Priority
The `analyze-results` command automatically searches for results files in this order:
1. `logs/enhanced_results.json` (enhanced benchmark)
2. `logs/ensemble_results.json` (ensemble collaboration)
3. `logs/quick_random_results.json` (randomized benchmark)
4. `logs/quick_results.json` (standard benchmark)
5. `logs/network_results.json` (network resilience)
6. `results.json` (legacy fallback)

### Show Available Results
```bash
make show-results          # List all available result files with details
```

### Workflow Examples

```bash
# Complete benchmark and analysis workflow
make run-enhanced                    # → logs/enhanced_results.json
make analyze-results                 # Analyzes enhanced_results.json

# Multiple benchmark comparison
make run                            # → logs/quick_results.json
make run-enhanced                   # → logs/enhanced_results.json
make analyze-results FILE=logs/quick_results.json
make analyze-results FILE=logs/enhanced_results.json

# Specialized testing workflow
make run-ensemble                   # → logs/ensemble_results.json
make run-network-test              # → logs/network_results.json
make analyze-ensemble              # Analyzes ensemble_results.json
make analyze-network               # Analyzes network_results.json

# Comprehensive analysis
make run-comprehensive             # → comprehensive_results.json
make analyze-results FILE=comprehensive_results.json
```

---

## Command Line Interface

> **Note:** This project uses `uv` for environment management. All CLI execution should go through `uv run` or `make` commands for consistency.

### Core Benchmarks

```bash
# Quick benchmarks (5 iterations)
uv run agent-benchmark quick                         # Standard task order
uv run agent-benchmark quick --randomize --seed 42   # Randomized with seed
uv run agent-benchmark quick --iterations 10         # Custom iteration count

# Full benchmarks (10+ iterations)
uv run agent-benchmark benchmark --iterations 15     # Standard comprehensive
uv run agent-benchmark benchmark --quick-mode        # Reduced task set
```

### Enhanced Benchmarks

```bash
# Enhanced statistical analysis
uv run agent-benchmark benchmark --enhanced-stats --confidence 0.95

# Tool failure simulation
uv run agent-benchmark benchmark --tool-failures --network unstable

# Complete enhanced benchmark
uv run agent-benchmark benchmark \
    --tool-failures \
    --enhanced-stats \
    --confidence 0.95 \
    --iterations 10 \
    --network unstable \
    --output logs/custom_results.json
```

### Ensemble & Collaboration Testing

```bash
# Full ensemble evaluation
uv run agent-benchmark ensemble --patterns all --combinations mixed

# Specific patterns
uv run agent-benchmark ensemble --patterns pipeline,parallel --iterations 5
uv run agent-benchmark ensemble --patterns consensus --combinations mixed

# Quick ensemble test
uv run agent-benchmark ensemble --patterns parallel --iterations 3
```

### Network Resilience Testing

```bash
# Test different network conditions
uv run agent-benchmark network-test stable --iterations 10      # Baseline
uv run agent-benchmark network-test unstable --iterations 15    # Stress test
uv run agent-benchmark network-test slow --iterations 10        # Slow connections
uv run agent-benchmark network-test degraded --iterations 10    # Packet loss
uv run agent-benchmark network-test peak_hours --iterations 10  # High latency
```

### Analysis & Output

```bash
# Auto-detect and analyze latest results
uv run agent-benchmark analyze --confidence 0.95 --outliers

# Analyze specific results file
uv run agent-benchmark analyze logs/enhanced_results.json --confidence 0.95 --outliers

# Export enhanced statistics
uv run agent-benchmark analyze logs/quick_results.json --export-stats analysis_output.json

# Quiet mode (minimal console output)
uv run agent-benchmark analyze logs/ensemble_results.json --quiet

# Debug mode (detailed error information)
uv run agent-benchmark benchmark --debug --iterations 2
```

---

## Makefile Commands Reference

### Installation & Setup
```bash
make install          # Install core dependencies
make install-dev      # Install dev dependencies and hooks
make version          # Show system + project version info
```

### Basic Benchmarking
```bash
make run                    # Quick 5-iteration → logs/quick_results.json
make run-random            # Randomized task order → logs/quick_random_results.json
make run-enhanced          # Enhanced features → logs/enhanced_results.json
make run-debug             # Debug mode with verbose logging
```

### Specialized Testing
```bash
make run-ensemble          # Multi-agent collaboration → logs/ensemble_results.json
make run-network-test      # Network resilience → logs/network_results.json
make run-comprehensive     # All enhanced features → comprehensive_results.json
make ensemble-quick        # Quick ensemble test → logs/ensemble_results.json
```

### Network Condition Testing
```bash
make run-network-stable    # Baseline → logs/network_results.json
make run-network-slow      # Slow network → logs/network_results.json
make run-network-degraded  # Degraded network → logs/network_results.json
make run-network-peak      # Peak hours → logs/network_results.json
make run-network-unstable  # Variable conditions → logs/network_results.json
```

### Ensemble Pattern Testing
```bash
make ensemble-pipeline     # Pipeline coordination → logs/ensemble_results.json
make ensemble-consensus    # Consensus decision → logs/ensemble_results.json
make ensemble-hierarchical # Hierarchical delegation → logs/ensemble_results.json
```

### Analysis & Validation
```bash
make analyze-results       # Auto-detect and analyze latest results
make analyze-enhanced      # Enhanced results with 99% confidence
make analyze-ensemble      # Ensemble collaboration analysis
make analyze-network       # Network resilience analysis
make show-results         # Show available result files and details
```

### Verbose Variants (For Debugging)
```bash
make run-enhanced-verbose     # Enhanced benchmark with verbose logging
make run-ensemble-verbose     # Ensemble benchmark with verbose logging
make run-network-test-verbose # Network test with verbose logging
```

### Development & QA
```bash
make test-unit             # Run unit tests
make lint                  # Auto-format code (black, isort, ruff)
make lint-check           # Check lint without modifying
make type-check           # Run mypy type checker
make check                # Full quality checks (lint + type + test)
make clean                # Remove logs/, traces/, and cache files
```

### Help & Documentation
```bash
make help                # Show main help message
make help-analysis       # Show detailed analysis command help
make help-enhanced       # Show help for enhanced features
make help-examples       # Show usage examples
```

### Pass Custom Arguments
```bash
make run ARGS="--iterations 10 --quiet --seed 123"
make run-random ARGS="--iterations 15"
make run-enhanced ARGS="--confidence 0.99 --output logs/custom_enhanced.json"
make analyze-results FILE=logs/specific_file.json
```

---

## Results File Management

### File Locations and Naming

All benchmark results are automatically saved with descriptive names:

| Command | Output File | Description |
|---------|-------------|-------------|
| `make run` | `logs/quick_results.json` | Standard 5-iteration benchmark |
| `make run-random` | `logs/quick_random_results.json` | Randomized task order |
| `make run-enhanced` | `logs/enhanced_results.json` | Enhanced features + statistics |
| `make run-ensemble` | `logs/ensemble_results.json` | Multi-agent collaboration |
| `make run-network-test` | `logs/network_results.json` | Network resilience testing |
| `make run-comprehensive` | `comprehensive_results.json` | All enhanced features combined |

### Analysis Workflow

```bash
# Step 1: Run benchmark (generates results file)
make run-enhanced                    # → logs/enhanced_results.json

# Step 2: Analyze results (reads existing file)
make analyze-results                 # Auto-detects enhanced_results.json

# Alternative: Analyze specific file
make analyze-results FILE=logs/enhanced_results.json
```

### Multiple Results Comparison

```bash
# Generate multiple result sets
make run                            # → logs/quick_results.json
make run-enhanced                   # → logs/enhanced_results.json
make run-ensemble                   # → logs/ensemble_results.json

# Compare results from different benchmarks
make analyze-results FILE=logs/quick_results.json
make analyze-results FILE=logs/enhanced_results.json  
make analyze-results FILE=logs/ensemble_results.json
```

### View Available Results

```bash
# See all available result files with timestamps and sizes
make show-results

# Expected output:
# logs/enhanced_results.json     - Enhanced benchmark with tool failures & stats
#    Modified: 2025-01-15 14:30:25 (Size: 15234 bytes)
# logs/ensemble_results.json     - Multi-agent collaboration benchmark
#    Modified: 2025-01-15 13:45:10 (Size: 8976 bytes)
```

### Clean Up Results

```bash
# Remove all logs and cached files
make clean

# Remove specific result files
rm logs/quick_results.json
rm logs/enhanced_results.json
```

---

## Configuration Options

### Agent Types
- **`WrapperAgent`**: Single LLM call pattern (simulates prompt chaining)
- **`MarketingAgent`**: Basic orchestration with limited memory
- **`RealAgent`**: Full autonomous system with planning, memory, and recovery

### Task Complexity Levels

| Complexity | Steps | Description | Success Expectation |
|-----------|-------|-------------|-------------------|
| **Simple** | 1–2 | Atomic operations | 70–90% |
| **Moderate** | 3–5 | Light reasoning chains | 40–70% |
| **Complex** | 6–12 | Multi-stage execution | 20–50% |
| **Enterprise** | 15+ | Full pipeline automation | 10–30% |

### Statistical Options
```bash
--confidence LEVEL         # Confidence level: 0.90, 0.95, 0.99
--enhanced-stats           # Include Cohen's h effect sizes and CI
--suppress-warnings        # Hide statistical warnings (default: True)
```

### Network Conditions
```bash
--network stable           # Normal conditions (baseline)
--network slow             # High latency, normal throughput  
--network unstable         # Variable latency and packet loss
--network degraded         # Consistent packet loss
--network peak_hours       # High latency during peak usage
```

### Ensemble Patterns
```bash
--patterns pipeline        # Sequential task delegation
--patterns parallel        # Concurrent task execution
--patterns hierarchical    # Manager-worker coordination
--patterns consensus       # Democratic decision making
--patterns specialization  # Expert role assignment
--patterns all             # Test all patterns
```

---

## Interpreting Results

### Performance Metrics Explained

#### Core Metrics
- **Success Rate**: Percentage of tasks completed successfully
- **Avg Execution Time**: Mean time per task (includes failures)
- **Context Retention**: Memory utilization and reuse across steps
- **Cost per Success**: Economic efficiency (cost divided by successful tasks)
- **Step Completion**: Percentage of planned steps executed

#### Statistical Analysis
- **Cohen's h**: Effect size for performance differences between agents
- **Confidence Intervals**: Statistical bounds on performance estimates  
- **Performance Gap**: Absolute percentage point difference between agents
- **Resilience Score**: Composite metric for handling adverse conditions

### Example Output Interpretation

```
[REAL_AGENT] EXCELLENT
  Success Rate:        93.3%
  Avg Execution Time:  0.99s
  Context Retention:   93.4%
  Cost per Success:    $0.0314
  Step Completion:     93.8%
  Tasks: 14/15 successful
```

**What this means:**
- Agent completed 14 out of 15 tasks successfully (93.3%)
- Average of 0.99 seconds per task (fast execution)
- Retained 93.4% of context across multi-step tasks (excellent memory)
- Cost $0.0314 per successful task completion (efficient)
- Executed 93.8% of planned steps (high architectural fidelity)

### Performance Comparison

| Agent Type | Success Rate | Context Retention | Step Completion | Interpretation |
|-----------|-------------|------------------|----------------|----------------|
| Wrapper | 15-27% | 11-13% | 20-27% | **Poor**: Basic prompt chaining |
| Marketing | 25-67% | 50-63% | 60% | **Variable**: Inconsistent architecture |
| Real | 75-93% | 91-94% | 94-97% | **Excellent**: Full autonomous system |

---

## Advanced Usage Patterns

### Reproducible Benchmarks
```bash
# Fixed comparison across environments
make run ARGS="--seed 2025 --iterations 10"

# Explore performance variability  
make run-random ARGS="--seed 1337 --iterations 15"
```

### Statistical Power Analysis
```bash
# High confidence statistical analysis
make run-enhanced ARGS="--confidence 0.99 --iterations 20"

# Analyze with enhanced statistics
make analyze-enhanced
```

### Stress Testing Pipeline
```bash
# Comprehensive agent stress test
make run-enhanced              # → logs/enhanced_results.json
make run-network-test          # → logs/network_results.json
make run-ensemble             # → logs/ensemble_results.json

# Analyze each result set
make analyze-enhanced         # Enhanced results with 99% CI
make analyze-network          # Network resilience analysis
make analyze-ensemble         # Ensemble collaboration analysis
```

### Comprehensive Testing
```bash
# Run the most complete benchmark
make run-comprehensive

# Analyze comprehensive results
make analyze-results FILE=comprehensive_results.json
```

---

## Python API Usage

### Basic Agent Execution
```python
from ai_agents_reality_check.agents import WrapperAgent, MarketingAgent, RealAgent
from ai_agents_reality_check.types import TaskComplexity

# Create agent instance
agent = RealAgent()

# Define task
task = {
    "name": "Analyze quarterly sales data", 
    "complexity": TaskComplexity.MODERATE
}

# Execute task
result = await agent.execute_task(task)
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time}s")
print(f"Context retention: {result.context_retention_score:.1%}")
```

### Comprehensive Benchmark
```python
from ai_agents_reality_check import AgentBenchmark

# Create benchmark instance
benchmark = AgentBenchmark(
    quick_mode=True,
    enhanced_features=True
)

# Run comprehensive evaluation
results = await benchmark.run_comprehensive_benchmark(
    iterations=10,
    include_ensemble=True,
    network_conditions=['stable', 'unstable']
)

# Export results
benchmark.export_results("logs/api_results.json", format="json")

# Access results
for agent_type, metrics in results.items():
    print(f"{agent_type}: {metrics['success_rate']:.1%} success")
```

### Custom Agent Implementation
```python
from ai_agents_reality_check.agents.base import BaseAgent
from ai_agents_reality_check.types import TaskResult, AgentType

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "Custom Agent"
        self.agent_type = AgentType.REAL_AGENT
    
    async def execute_task(self, task):
        # Implement custom logic here
        success = await self.custom_execution_logic(task)
        
        return TaskResult(
            task_id=task.get('task_id', self.generate_task_id()),
            agent_type=self.agent_type,
            success=success,
            execution_time=1.0,
            # ... other required fields
        )
```

---

## Troubleshooting

### Common Issues

#### Low Success Rates
```bash
# Check specific failure reasons
make run-debug

# Validate agent-task complexity matching
make run ARGS="--quick-mode --iterations 5"
```

#### Analysis File Not Found
```bash
# Check what results files exist
make show-results

# Generate results first, then analyze
make run                     # → logs/quick_results.json
make analyze-results         # Analyzes quick_results.json

# Specify exact file if auto-detection fails
make analyze-results FILE=logs/quick_results.json
```

#### Slow Execution
```bash
# Use quick mode for faster runs
make run ARGS="--quick-mode --iterations 3"

# Minimize console output
make run ARGS="--quiet --iterations 5"
```

#### Statistical Issues
```bash
# Increase sample size for significance
make run-enhanced ARGS="--iterations 20 --confidence 0.95"

# Check for outliers affecting results
make analyze-results FILE=logs/enhanced_results.json
```

#### Installation Problems
```bash
# Check feature availability
make version

# Clean and reinstall
make clean && make install-dev
```

### Performance Optimization

```bash
# Fastest possible benchmark
make run ARGS="--iterations 3 --quick-mode --quiet"

# Balanced speed/accuracy
make run ARGS="--iterations 5 --quiet"

# Maximum statistical rigor (slower)
make run-comprehensive
```

---

## Benchmarking Best Practices

### For Reliable Comparisons
1. **Use consistent parameters**: Same iterations, seed, and task complexity
2. **Multiple runs**: Run at least 3 separate benchmark sessions
3. **Statistical validation**: Use enhanced stats with sufficient iterations (≥10)
4. **Document conditions**: Note any system load or network conditions
5. **Preserve results**: Keep result files for later comparison and validation

### For Production Decisions
1. **Comprehensive evaluation**: Use `make run-comprehensive` for full analysis
2. **Stress testing**: Include network resilience and tool failure scenarios
3. **Cost analysis**: Focus on cost-per-success metrics, not just success rates
4. **Statistical significance**: Ensure sufficient sample sizes for decision confidence

---

## Output Formats & Integration

### JSON Output
```bash
# Export for external analysis
make run-enhanced ARGS="--output logs/production_results.json"

# Analyze and export enhanced statistics
make analyze-enhanced
```

### Log Analysis
```bash
# Benchmark logs are automatically saved to logs/
ls logs/
# benchmark_YYYYMMDD_HHMMSS.log  - Detailed execution logs
# quick_results.json             - Standard benchmark results
# enhanced_results.json          - Enhanced benchmark results
# ensemble_results.json          - Ensemble collaboration results
# network_results.json           - Network resilience results
```

### Result File Structure
```json
{
  "metadata": {
    "version": "0.1.0",
    "total_results": 45,
    "export_timestamp": 1704067200.0,
    "quick_mode": true,
    "random_seed": 42
  },
  "results": [
    {
      "task_id": "uuid-string",
      "task_name": "Task Name",
      "agent_type": "real_agent",
      "success": true,
      "execution_time": 1.23,
      "context_retention_score": 0.94
    }
  ]
}
```

---

## License & Support

**AI Agents Reality Check** is licensed under the **Apache 2.0** License.
See [`LICENSE`](../LICENSE) for complete terms.

### Getting Help
- **Bug Reports**: [GitHub Issues](https://github.com/Cre4T3Tiv3/ai-agents-reality-check/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/Cre4T3Tiv3/ai-agents-reality-check/discussions)
- **Documentation**: Check README.md and docs/ directory
- **Direct Contact**: [ByteStack Labs](https://bytestacklabs.com)

---

## Author

**AI Agents Reality Check** was built by [Jesse Moses (@Cre4T3Tiv3)](https://github.com/Cre4T3Tiv3) at [ByteStack Labs](https://bytestacklabs.com).

> Could **benchmarking with statistical confidence** be the missing foundation for agentic software?
