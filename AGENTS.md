# AGENTS Guide

This file provides guidelines for automatic agents working with this repository.

## Formatting and linting
- Format Python code with `make format` before committing. This runs `isort` and `black`.
- Verify style and tests with `make run-checks`. This command executes `isort --check`, `black --check`, `ruff check`, and `pytest`.
- All commits should pass `make run-checks`.

## Testing
- Unit tests live under `tests/`. Add or update tests for new features or bug fixes.
- The tests rely on the small dataset in `demo_data/robot_sim.PickNPlace` and run offline using Python 3.10.

## Commit messages
- Keep commit summaries concise and written in the imperative mood.
