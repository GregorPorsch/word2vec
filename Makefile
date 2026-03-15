.PHONY: help install test lint format check train clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (runtime + dev)
	uv sync

test: ## Run the full test suite
	uv run pytest tests/ -v

lint: ## Run ruff linter on src/ and tests/
	uv run ruff check src/ tests/

format: ## Auto-format code with ruff
	uv run ruff format src/ tests/

check: lint test ## Run lint + tests (CI-style)

train: ## Train on bundled corpus with default hyperparameters
	uv run word2vec

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/
