AGENTS.md

Project context: Python 3.11, PEP 8, Type hints everywhere

This file is the single source of truth for how automated agents (and human contributors) propose, implement, and verify changes in this repository. Agents must follow the protocol below.


1) Mission & Scope
	- Make small, reviewable changes with clear intent.
	- Prefer plan → change → verify.
	- All code targets Python 3.11; use typing and PEP 8.

2) House Rules (Style & Quality)

2.1 Language & Formatting
	- Python version: 3.11 (use | union types, typing.Self, typing.Literal, typing.TypedDict, dataclasses, enum).
	- Formatting: Ruff (line-length 100). No manual alignment.
	- Lint: Ruff (fix auto-fixable issues). Treat warnings as errors in CI.
	- Docstrings: Google style. Public functions/classes must have docstrings and type hints.
	- Imports: absolute where possible; group stdlib / third-party / local with blank lines.
	- Logging: logging module; no print in library code. Use structured messages.
	- Dependency managagement: use uv whenever possible, inside dockerfiles too.
	- Run code with uv run not python command directly.
