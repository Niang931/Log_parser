.PHONY: setup lint test demo tables docker clean

PYTHON ?= python3
PIP    ?= pip

setup:
	$(PIP) install -e ".[test,lint]"

lint:
	ruff check deepparse tests

test:
	pytest -q

demo:
	./scripts/prepare_paths.sh
	deepparse synth --dataset DemoTiny --mode offline --out artifacts/masks/DemoTiny.json
	deepparse eval --config configs/demo_small.yaml --deterministic
	deepparse table --inputs artifacts/outputs/demo_metrics.csv --out artifacts/outputs/tables/

tables:
	./scripts/regenerate_tables.sh

docker:
	docker build -t deepparse-artifact .

clean:
	rm -rf artifacts/data/DemoTiny artifacts/masks/*.json
	rm -rf artifacts/outputs/*.csv artifacts/outputs/tables artifacts/outputs/logs
