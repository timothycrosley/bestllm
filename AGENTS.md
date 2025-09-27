# Repository Guidelines

## Project Structure & Module Organization
- `pyproject.toml` defines metadata, dependencies, and the `uv_build` backend; update it when adding packages or console entry points.
- Runtime code lives under `src/bestllm/`; keep each user-facing feature in its own module and expose entry points through `bestllm.__init__`.
- Store supporting assets (images, prompts, etc.) in `media/` and reference them via relative paths.
- Add future automated tests in a top-level `tests/` package mirroring the `src/bestllm` layout.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package locally and registers the `bestllm` console script.
- `uv pip install --group dev .` (or `python -m pip install ruff ty`) pulls in the linting and type-checking toolchain declared in `pyproject.toml`.
- `python -m bestllm` or `bestllm` runs the current CLI entry point so you can verify local changes quickly.
- `uv build` (fallback: `python -m build`) produces distributable wheels using the configured backend.

## Coding Style & Naming Conventions
- Follow standard PEP 8 conventions with 4-space indentation and descriptive, lowercase module names.
- Favor type hints throughout; keep function interfaces narrow and place shared dataclasses or protocols in dedicated modules.
- Run `ruff check src` (or `uv run ruff check src`) before sending changes; use `ruff format` for quick formatting assistance.
- Maintain clear, action-oriented function names (`select_model`, `load_config`), and reserve `__init__.py` for exports and lightweight wiring.

## Testing Guidelines
- Prefer the built-in `unittest` discovery (`python -m unittest discover`) for quick coverage; structure tests under `tests/feature_name/test_*.py`.
- When edge cases depend on hardware or model availability, guard them with skips and document required fixtures in docstrings.
- Keep high-value smoke checks that exercise the CLI’s selection logic; expand coverage alongside new heuristics or model adapters.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood (“Add CLI selection heuristics”) and keep subject lines under ~72 characters; bundle related changes together.
- Reference linked issues or discussions in the PR description, include reproduction steps, and attach CLI output or screenshots when behavior changes.
- Ensure CI-critical commands (`ruff`, `ty`, tests) pass locally before opening a PR; note any intentionally skipped checks and why.
