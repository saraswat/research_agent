# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Install: `pip install -r requirements.txt`
- Run: `python usage.py`
- Lint: `flake8 *.py tools/*.py utils/*.py`
- Type check: `mypy --strict *.py tools/*.py utils/*.py`

## Code Style Guidelines
- Imports: Group standard library, third-party, and local imports with a blank line between groups
- Typing: Use type hints from typing module for all function parameters and return values
- Docstrings: Google style docstrings with Args/Returns sections
- Class structure: Docstring under class declaration, followed by `__init__` and methods
- Error handling: Use try/except with specific error types and meaningful error messages
- Variable naming: snake_case for variables/functions, CamelCase for classes
- Line length: Keep lines under 100 characters
- Comments: Explain "why" not "what"; code should be self-explanatory
- Method ordering: Public methods first, followed by private (underscore-prefixed) methods