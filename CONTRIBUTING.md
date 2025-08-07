# Contributing to AI Agents Reality Check

Thank you for your interest in contributing to AI Agents Reality Check! This project aims to provide rigorous, mathematically sound benchmarking of AI agent architectures.

## Ways to Contribute

### Bug Reports
- Use GitHub Issues with the "bug" label
- Include system information and reproduction steps
- Attach benchmark output logs when relevant

### Feature Requests
- Use GitHub Issues with the "enhancement" label
- Explain the use case and expected behavior
- Consider mathematical rigor and architectural focus

### Research Contributions
- New agent architectures for benchmarking
- Additional statistical analysis methods
- Academic validation studies
- Performance optimization improvements

## Development Setup

```bash
git clone https://github.com/Cre4T3Tiv3/ai-agents-reality-check
cd ai-agents-reality-check
make install-dev
```

## Code Standards

### Python Code Quality
- Use `black` for formatting: `make format`
- Type hints required for all functions
- Docstrings for all public methods
- 90%+ test coverage for new features

### Mathematical Rigor
- Statistical methods must be academically sound
- Include confidence intervals for performance metrics
- Document assumptions and limitations
- Provide references for statistical techniques

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for benchmark pipelines
- Schema validation for data structures
- Performance regression tests

## Pull Request Process

1. **Fork and Branch**: Create feature branch from `main`
2. **Implement**: Follow code standards and testing requirements
3. **Test**: Run `make check` and ensure all tests pass
4. **Document**: Update relevant documentation
5. **Submit**: Create PR with clear description and context

## Review Criteria

- Mathematical accuracy and statistical validity
- Code quality and test coverage
- Documentation completeness
- Backward compatibility
- Performance impact assessment

## Community Guidelines

- Be respectful and constructive
- Focus on technical merit and empirical evidence
- Welcome newcomers and provide helpful feedback
- Maintain the ethos: challenge assumptions with data

## Questions?

- GitHub Discussions for general questions
- GitHub Issues for specific bugs or features
- Email [ByteStack Labs](https://bytestacklabs.com) for research collaborations