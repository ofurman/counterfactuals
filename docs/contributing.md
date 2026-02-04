# Contributing

Guidelines for contributing to the Counterfactuals library.

## Code Style

- Follow **PEP 8** style guidelines
- Use **Ruff** for linting and formatting
- Maximum line length: **100 characters**
- Use **type hints** for all function signatures

## Docstrings

Use **Google-style docstrings**:

```python
def explain(self, X: np.ndarray, y_origin: int, y_target: int) -> ExplanationResult:
    """Generate counterfactual explanations.

    Args:
        X: Input instances to explain. Shape (n_samples, n_features).
        y_origin: Original class label.
        y_target: Target class label.

    Returns:
        ExplanationResult containing counterfactuals and metadata.

    Raises:
        ValueError: If X has wrong shape.
    """
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Write tests for:
- New methods
- Bug fixes
- Edge cases

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting: `ruff check .`
5. Run tests: `pytest`
6. Submit PR with description

## Adding New Methods

1. Create method class inheriting from `BaseCounterfactualMethod`
2. Implement `fit()` and `explain()` methods
3. Add tests in `tests/`
4. Create pipeline in `pipelines/`
5. Add documentation

## Adding New Metrics

1. Create metric class inheriting from `Metric`
2. Use `@register_metric` decorator
3. Implement `required_inputs()` and `__call__()`
4. Add tests

## Development Setup

```bash
# Clone repository
git clone https://github.com/ofurman/counterfactuals.git
cd counterfactuals

# Install with dev dependencies
uv sync

# Install pre-commit hooks
pre-commit install
```
