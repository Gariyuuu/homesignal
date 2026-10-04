# HomeSignal — every target runs on a laptop CPU with no API keys.
PY := .venv/bin/python
HS := .venv/bin/homesignal

.PHONY: all setup download build backtest ablations tune train explain report export app test lint typecheck check clean

all: setup download build backtest ablations tune train explain export report  ## full pipeline, no manual steps

setup:  ## create the virtualenv and install the package (needs uv or pip)
	@test -d .venv || (command -v uv >/dev/null && uv venv --python 3.11 .venv || python3.11 -m venv .venv)
	@(command -v uv >/dev/null && uv pip install -q -e ".[app,dev]") || $(PY) -m pip install -q -e ".[app,dev]"

download:  ## cached, idempotent downloads of all raw sources (~2.5 GB first time)
	$(HS) download

build:  ## monthly ZIP panel with target + features → data/processed/panel.parquet
	$(HS) build-panel

backtest:  ## baselines, linear and tree models over the expanding-window folds
	$(HS) backtest --tag main

ablations:  ## LightGBM without geography / without ACS demographics
	$(HS) backtest --tag no_geo --models lightgbm --exclude state region division metro_size_bucket
	$(HS) backtest --tag no_acs --models lightgbm --exclude $(shell $(PY) -c "from homesignal.features.demographics import ACS_FEATURES; print(' '.join(ACS_FEATURES))")
	$(HS) backtest --tag no_macro --models lightgbm baseline_trailing_12m --exclude $(shell $(PY) -c "from homesignal.features.macro import MACRO_FEATURES; print(' '.join(MACRO_FEATURES))")

tune:  ## modest Optuna search on recent folds
	$(HS) tune

train:  ## final model on pre-holdout data, single holdout evaluation, artifacts in models/
	$(HS) train-final

explain:  ## SHAP + error analysis + calibration figures
	$(HS) explain

report:  ## reports/evaluation.md from the saved results
	$(HS) report

export:  ## benchmark parquet + schema for the downstream benchmarking project
	$(HS) export

app:  ## run the demo locally
	$(PY) app/prepare_app_data.py && $(PY) app/app.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests app && .venv/bin/ruff format --check src tests app

typecheck:
	.venv/bin/mypy

check: lint typecheck test  ## everything CI runs

clean:
	rm -rf data/interim/* data/processed/*.parquet models/*.joblib models/backtest_predictions reports/results
