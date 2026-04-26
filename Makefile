.PHONY: run test lint docker-up docker-down

run:
	PYTHONPATH=. uv run --env-file .env python parser/parser.py

test:
	PYTHONPATH=. uv run --env-file .env pytest

lint:
	uv run ruff check .

docker-up:
	docker compose up -d

docker-down:
	docker compose down
