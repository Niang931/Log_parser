# Overridable on the command line, e.g. `make eval-iot LLM=llm:anthropic EPOCHS=10`.
LLM      ?=
EPOCHS   ?= 30
THREADS  ?=
SITE     ?= gemini
# camoufox | chromium; defaults to WEB_BROWSER in .env.
BROWSER  ?= $(or $(shell sed -n 's/^WEB_BROWSER=//p' .env 2>/dev/null),camoufox)
PROMPT   ?= Reply with OK.
EVAL_OUT ?= artifacts/eval
HOLDOUT  ?= OpenSSH
MASKS    ?=
PYRIGHT  ?= pyright
DATASET  ?= re3-ob
VIA      ?= direct
LIMIT    ?=
VARIANTS ?=
FRESH    ?=

# .env is optional: load it only when it exists.
RUN      := uv run $(if $(wildcard .env),--env-file .env)
LLM_ARG  := $(if $(LLM),--llm $(LLM))
# DeepParse calls Chromium "chrome"; it runs the `chromium` on PATH.
WEB      := $(if $(filter chromium,$(BROWSER)),chrome,$(BROWSER))
LOGPIPE  := $(RUN) logpipe --browser $(WEB)
EVAL     := $(LOGPIPE) eval --out $(EVAL_OUT) $(LLM_ARG)
TRAIN    := --epochs $(EPOCHS) $(if $(THREADS),--threads $(THREADS))
SCRAPER  := $(RUN) python -m deepparse.ai_scraper
# RCAEval runs in its own tool env: its baselines' deps stay out of ours
RCAEVAL  := uvx --from 'RCAEval[core]' rcaeval-agent
RCA_OUT  := $(EVAL_OUT)/rcaeval
# docs: Material for MkDocs in its own tool env, so it isn't a project dependency
MKDOCS   := NO_MKDOCS_2_WARNING=1 uvx --from mkdocs-material==9.8.1 mkdocs

.DEFAULT_GOAL := help
.PHONY: help setup start calibrate train run run-fast worker replay gen-iot demo bench \
        data data-loghub data-hdfs data-bgl \
        retrain-data retrain-eval \
        eval eval-parsing eval-alias eval-iot eval-hdfs eval-bgl eval-report eval-gate eval-baseline \
        eval-rcaeval \
        web-setup web-login web-check web-ask \
        test test-fast test-slow test-integration lint format typecheck check docs docs-build \
        docker-up docker-down clean

help: ## list targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## --- setup ------------------------------------------------------------------

setup: ## install deps and create .env from the example
	uv sync --group dev
	@test -f .env || { cp .env.example .env; echo "created .env"; }

start: ## interactive: asks what to run (demo or your logs, Kafka or not), sets it up, runs it
	$(RUN) logpipe start

## --- pipeline (paths from .env) ---------------------------------------------

calibrate: ## build registry v1 from LOGS_PATH
	$(LOGPIPE) calibrate $(LLM_ARG)

train: ## train LogBERT on per-device windows of LOGS_PATH
	$(LOGPIPE) train-logbert --epochs $(EPOCHS)

run: ## full pipeline: parse, store, enrich, detect
	$(LOGPIPE) run $(LLM_ARG)

run-fast: ## fast path only: parse, store, publish unknown lines to Kafka
	$(LOGPIPE) run --queue kafka --no-enrich

worker: ## learning worker: consume the Kafka unresolved queue
	$(LOGPIPE) worker --queue kafka $(LLM_ARG)

replay: ## parse LOGS_PATH with the frozen registry only
	$(LOGPIPE) replay --quiet

gen-iot: ## write the synthetic IoT fleet to artifacts/fleet
	$(LOGPIPE) gen-iot --out artifacts/fleet

demo: gen-iot ## end to end on the synthetic fleet
	$(LOGPIPE) calibrate --logs artifacts/fleet/history $(LLM_ARG)
	$(LOGPIPE) train-logbert --logs artifacts/fleet/history --epochs $(EPOCHS)
	$(LOGPIPE) run --logs artifacts/fleet/test $(LLM_ARG)

## --- datasets (artifacts/datasets) ------------------------------------------

data: ## download every eval dataset (Loghub-2k, HDFS, BGL)
	$(LOGPIPE) fetch-data

data-loghub: ## Loghub-2k (parsing eval)
	$(LOGPIPE) fetch-data loghub2k

data-hdfs: ## HDFS_1 from Zenodo
	$(LOGPIPE) fetch-data HDFS

data-bgl: ## BGL from Zenodo
	$(LOGPIPE) fetch-data BGL

## --- evaluation (results in EVAL_OUT) ---------------------------------------

bench: ## lines/s one server keeps up with, its CPU and memory (STORE=clickhouse to include writes)
	$(LOGPIPE) bench --store $(or $(STORE),memory)

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

eval-rcaeval: ## RCA on RCAEval cases, scored by RCAEval (DATASET VIA=direct|mcp LIMIT LLM VARIANTS FRESH=1)
	test -d $(RCA_OUT)/tasks/$(DATASET) || $(RCAEVAL) export --dataset $(DATASET) \
		--out $(RCA_OUT)/tasks --data-root $(RCA_OUT)/data
	WEB_BROWSER=$(WEB) $(RUN) python -m logpipe.eval.rcaeval_bench $(RCA_OUT)/tasks/$(DATASET) --out $(RCA_OUT) \
		--via $(VIA) $(if $(LIMIT),--limit $(LIMIT)) $(if $(filter direct,$(VIA)),$(LLM_ARG)) \
		$(if $(VARIANTS),--variants $(VARIANTS)) $(if $(FRESH),--fresh)
	for p in $(RCA_OUT)/preds.$(DATASET).*.jsonl; do echo "$$p"; \
		$(RCAEVAL) evaluate --tasks $(RCA_OUT)/tasks/$(DATASET) --predictions "$$p" --report-chance; done

eval-report: ## summarise EVAL_OUT into a report
	$(EVAL) report

eval-gate: ## compare EVAL_OUT with eval/baseline.json (the nightly CI check)
	$(EVAL) gate --baseline eval/baseline.json

eval-baseline: ## record EVAL_OUT as the new eval/baseline.json
	$(EVAL) gate --baseline eval/baseline.json --write-baseline

## --- mask-model retrain (run on purpose; fine-tune itself runs on a GPU) ----

retrain-data: ## build the mask-model training set, HOLDOUT=OpenSSH kept out
	cd DeepParse && $(RUN) python -m deepparse.tools.retrain data --holdout $(HOLDOUT) \
		$(if $(wildcard artifacts/datasets/loghub2_full/zips),--loghub2-zips ../artifacts/datasets/loghub2_full/zips)

retrain-eval: ## score adapter mask bundles, e.g. MASKS="v2=artifacts/retrain/masks_v2"
	cd DeepParse && $(RUN) python -m deepparse.tools.retrain eval --holdout $(HOLDOUT) \
		$(foreach m,$(MASKS),--masks $(m)) --out artifacts/retrain

## --- web LLM (LLM=llm:web:<site>, no API key) -------------------------------

web-setup: ## get the browser the scraper drives (once; BROWSER=chromium uses the system one)
ifeq ($(WEB),chrome)
	@command -v chromium >/dev/null || { echo "install chromium and put it on PATH"; exit 1; }
else
	$(RUN) camoufox fetch
endif

web-login: ## sign in to SITE in a visible window (once per site)
	$(SCRAPER) login $(SITE) --browser $(WEB)

web-check: ## check the saved SITE login still works headless
	$(SCRAPER) check $(SITE) --browser $(WEB)

web-ask: ## send PROMPT to SITE and print the reply
	$(SCRAPER) ask $(SITE) "$(PROMPT)" --browser $(WEB)

## --- quality ----------------------------------------------------------------

test: ## all tests except integration
	$(RUN) pytest -m "not integration"

test-fast: ## skip slow and integration tests
	$(RUN) pytest -m "not slow and not integration"

test-slow: ## only tests that train a model
	$(RUN) pytest -m slow

test-integration: docker-up ## ClickHouse and Kafka tests
	$(RUN) pytest -m integration

lint: ## ruff
	uv run ruff check logpipe tests

format: ## ruff autofix
	uv run ruff check --fix logpipe tests

typecheck: ## pyright
	$(PYRIGHT)

check: lint typecheck test ## lint, typecheck and test

docs: ## serve the docs site with live reload on http://127.0.0.1:8000
	$(MKDOCS) serve

docs-build: ## build the docs site into site/ (strict: fails on broken links)
	$(MKDOCS) build --strict

## --- services ---------------------------------------------------------------

docker-up: ## start ClickHouse and Kafka, wait until healthy
	docker compose up -d --wait

docker-down: ## stop the services
	docker compose down

clean: ## remove caches (keeps datasets, models and eval results)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
