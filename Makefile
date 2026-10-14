# Overridable on the command line, e.g. `make eval-iot LLM=llm:anthropic EPOCHS=10`.
LLM      ?=
EPOCHS   ?= 30
THREADS  ?=
SITE     ?= gemini
BROWSER  ?= camoufox
PROMPT   ?= Reply with OK.
EVAL_OUT ?= artifacts/eval
PYRIGHT  ?= pyright

# .env is optional: load it only when it exists.
RUN      := uv run $(if $(wildcard .env),--env-file .env)
LLM_ARG  := $(if $(LLM),--llm $(LLM))
EVAL     := $(RUN) logpipe eval --out $(EVAL_OUT) $(LLM_ARG)
TRAIN    := --epochs $(EPOCHS) $(if $(THREADS),--threads $(THREADS))
SCRAPER  := $(RUN) python -m deepparse.ai_scraper

.DEFAULT_GOAL := help
.PHONY: help setup calibrate train run replay gen-iot demo \
        data data-loghub data-hdfs data-bgl \
        eval eval-parsing eval-alias eval-iot eval-hdfs eval-bgl eval-report \
        web-setup web-login web-check web-ask \
        test test-fast test-slow test-integration lint format typecheck check \
        docker-up docker-down clean

help: ## list targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## --- setup ------------------------------------------------------------------

setup: ## install deps and create .env from the example
	uv sync --group dev
	@test -f .env || { cp .env.example .env; echo "created .env"; }

## --- pipeline (paths from .env) ---------------------------------------------

calibrate: ## build registry v1 from LOGS_PATH
	$(RUN) logpipe calibrate $(LLM_ARG)

train: ## train LogBERT on per-device windows of LOGS_PATH
	$(RUN) logpipe train-logbert --epochs $(EPOCHS)

run: ## full pipeline: parse, store, enrich, detect
	$(RUN) logpipe run $(LLM_ARG)

replay: ## parse LOGS_PATH with the frozen registry only
	$(RUN) logpipe replay --quiet

gen-iot: ## write the synthetic IoT fleet to artifacts/fleet
	$(RUN) logpipe gen-iot --out artifacts/fleet

demo: gen-iot ## end to end on the synthetic fleet
	$(RUN) logpipe calibrate --logs artifacts/fleet/history $(LLM_ARG)
	$(RUN) logpipe train-logbert --logs artifacts/fleet/history --epochs $(EPOCHS)
	$(RUN) logpipe run --logs artifacts/fleet/test $(LLM_ARG)

## --- datasets (artifacts/datasets) ------------------------------------------

data: ## download every eval dataset (Loghub-2k, HDFS, BGL)
	$(RUN) logpipe fetch-data

data-loghub: ## Loghub-2k (parsing eval)
	$(RUN) logpipe fetch-data loghub2k

data-hdfs: ## HDFS_1 from Zenodo
	$(RUN) logpipe fetch-data HDFS

data-bgl: ## BGL from Zenodo
	$(RUN) logpipe fetch-data BGL

## --- evaluation (results in EVAL_OUT) ---------------------------------------

eval: eval-parsing eval-alias eval-iot eval-hdfs eval-bgl eval-report ## every benchmark, then the report

eval-parsing: data-loghub ## GA/PA on Loghub-2k vs. DeepParse
	$(EVAL) parsing

eval-alias: ## alias decisions on labelled template pairs
	$(EVAL) alias

eval-iot: ## synthetic fleet end to end, with ablations
	$(EVAL) iot $(TRAIN)

eval-hdfs: data-hdfs ## LogBERT on an HDFS subset
	$(EVAL) logbert --dataset HDFS $(TRAIN)

eval-bgl: data-bgl ## LogBERT on a BGL subset
	$(EVAL) logbert --dataset BGL $(TRAIN)

eval-report: ## summarise EVAL_OUT into a report
	$(EVAL) report

## --- web LLM (LLM=llm:web:<site>, no API key) -------------------------------

web-setup: ## download the browser the scraper drives (once)
	$(RUN) $(if $(filter chrome,$(BROWSER)),playwright install chromium,camoufox fetch)

web-login: ## sign in to SITE in a visible window (once per site)
	$(SCRAPER) login $(SITE) --browser $(BROWSER)

web-check: ## check the saved SITE login still works headless
	$(SCRAPER) check $(SITE) --browser $(BROWSER)

web-ask: ## send PROMPT to SITE and print the reply
	$(SCRAPER) ask $(SITE) "$(PROMPT)" --browser $(BROWSER)

## --- quality ----------------------------------------------------------------

test: ## all tests except integration
	$(RUN) pytest -m "not integration"

test-fast: ## skip slow and integration tests
	$(RUN) pytest -m "not slow and not integration"

test-slow: ## only tests that train a model
	$(RUN) pytest -m slow

test-integration: docker-up ## ClickHouse tests
	$(RUN) pytest -m integration

lint: ## ruff
	uv run ruff check logpipe tests

format: ## ruff autofix
	uv run ruff check --fix logpipe tests

typecheck: ## pyright
	$(PYRIGHT)

check: lint typecheck test ## lint, typecheck and test

## --- services ---------------------------------------------------------------

docker-up: ## start ClickHouse and wait until healthy
	docker compose up -d --wait

docker-down: ## stop ClickHouse
	docker compose down

clean: ## remove caches (keeps datasets, models and eval results)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
