# Quality Gate

This repo should pass three layers before it is worth showing:

1. `pytest`
2. `ruff check .`
3. `python scripts/quality_gate.py`

The gate checks the actual thesis surface:

- Riemannian drift exists.
- Trajectory drift exists.
- Eval environment exists.
- Real EEG and neural-population fixtures exist.
- Dataset loaders exist.
- Metrics and fixture docs exist.
- Tests are non-token.
- README states the project clearly.
- No placeholder language is left in code/docs.
