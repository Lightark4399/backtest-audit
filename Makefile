.PHONY: help install test demo lint up down psql schema clean

help:
	@echo "install  install package with dev extras"
	@echo "test     run the test suite"
	@echo "demo     run the offline audit demo (no database or network needed)"
	@echo "lint     ruff check"
	@echo "up       start postgres via docker compose"
	@echo "schema   apply sql/*.sql to the running database"
	@echo "psql     open a psql shell"

install:
	pip install -e ".[dev]"

test:
	python -m pytest

# The demo is deliberately independent of Postgres and of any network call, so
# that a reviewer can see the project's central output with one command.
demo:
	python -m audit.demo --outdir examples/outputs

lint:
	ruff check src tests

up:
	docker compose up -d
	@echo "waiting for postgres to accept connections..."
	@until docker compose exec -T db pg_isready -U audit -q; do sleep 1; done
	@echo "ready"

down:
	docker compose down -v

schema: up
	@for f in sql/001_schema.sql sql/002_pit_views.sql; do \
		echo "applying $$f"; \
		docker compose exec -T db psql -U audit -d audit -v ON_ERROR_STOP=1 -f - < $$f; \
	done

psql:
	docker compose exec db psql -U audit -d audit

clean:
	rm -rf .pytest_cache **/__pycache__ examples/outputs/*.json examples/outputs/*.txt
