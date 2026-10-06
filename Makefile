.PHONY: run calibrate test lint docker-up docker-down

calibrate:
	uv run --env-file .env logpipe calibrate

run:
	uv run --env-file .env logpipe replay

test:
	uv run pytest

lint:
	uv run ruff check .

docker-up:
	docker compose up -d

docker-down:
	docker compose down
