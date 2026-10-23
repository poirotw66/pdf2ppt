# pdf2ppt developer shortcuts. Run from the repository root.
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PACKAGE_EXTRAS ?= ocr,server,dev
FRONTEND_DIR := frontend

.PHONY: help install install-core install-dev lint typecheck test test-frontend test-all \
	build-wheel verify-wheel clean docker-lint compose-config

help:
	@echo "Targets:"
	@echo "  install        Editable install with [$(PACKAGE_EXTRAS)]"
	@echo "  install-core   Editable install (core only, no OCR/API)"
	@echo "  install-dev    Alias for install"
	@echo "  lint           ruff check ."
	@echo "  typecheck      mypy on src + bench scripts"
	@echo "  test           pytest -q"
	@echo "  test-frontend  frontend lint + tests"
	@echo "  test-all       lint + typecheck + pytest + frontend"
	@echo "  build-wheel    sdist + wheel via python -m build"
	@echo "  verify-wheel   build wheel, install in clean venv, run CLI on a fixture"
	@echo "  docker-lint    Hadolint if available, else Dockerfile parse checks"
	@echo "  compose-config Validate docker-compose.yml (requires docker compose)"
	@echo "  clean          Remove build/dist/egg-info caches"

install install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[$(PACKAGE_EXTRAS)]"

install-core:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src/pdf2ppt scripts/bench_inpaint.py scripts/bench_slide.py scripts/bench_layout_planner.py

test:
	$(PYTHON) -m pytest -q

test-frontend:
	cd $(FRONTEND_DIR) && npm ci && npm run lint && CI=1 npm test -- --watch=false

test-all: lint typecheck test test-frontend

build-wheel:
	$(PIP) install --upgrade build
	$(PYTHON) -m build

verify-wheel: build-wheel
	@set -e; \
	WHEEL=$$(ls -1 dist/pdf2ppt-*.whl | head -1); \
	echo "Using $$WHEEL"; \
	TMP=$$(mktemp -d); \
	$(PYTHON) -m venv "$$TMP/venv"; \
	"$$TMP/venv/bin/python" -m pip install --upgrade pip; \
	"$$TMP/venv/bin/python" -m pip install "$$WHEEL"; \
	"$$TMP/venv/bin/pdf2ppt" --help >/dev/null; \
	"$$TMP/venv/bin/python" -m pdf2ppt --help >/dev/null; \
	"$$TMP/venv/bin/pdf2ppt" tests/fixtures/solid_background.pdf "$$TMP/out.pptx"; \
	test -s "$$TMP/out.pptx"; \
	echo "verify-wheel OK: $$WHEEL -> $$TMP/out.pptx"; \
	rm -rf "$$TMP"

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -not -path './.git/*' -prune -exec rm -rf {} +

docker-lint:
	@if command -v hadolint >/dev/null 2>&1; then \
		hadolint Dockerfile Dockerfile.frontend; \
	else \
		echo "hadolint not installed; running basic Dockerfile checks"; \
		$(PYTHON) scripts/check_dockerfiles.py; \
	fi

compose-config:
	@if command -v docker >/dev/null 2>&1; then \
		docker compose config >/dev/null && echo "docker compose config OK"; \
	else \
		echo "docker not available; skipped compose-config"; \
		exit 0; \
	fi
