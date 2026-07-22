.PHONY: setup lint test audit data data-semcor data-dwug dwug-fetch prototypes contexts cluster evaluate report pilot clean

CONFIG ?= configs/pilot.yaml
RUN    ?= results/oracle_k
PY     ?= python
CORPUS ?= semcor
# Occurrence parquet fed to select_targets; switches with the corpus.
OCC    ?= data/$(if $(filter dwug_en,$(CORPUS)),dwug,semcor)_occurrences.parquet
DWUG_EN_ROOT ?= data/dwug_en

# Keep BLAS single-threaded: POT/numpy/scipy can segfault (exit 139) when the
# system OpenBLAS spawns more threads than it was built for on many-core hosts.
export OPENBLAS_NUM_THREADS ?= 1
export OMP_NUM_THREADS      ?= 1

# --------------------------------------------------------------------------- #
# Environment                                                                  #
# --------------------------------------------------------------------------- #
setup:
	uv venv --python 3.12 .venv
	. .venv/bin/activate && uv pip install -r requirements.txt
	@echo ">>> Now install a CUDA torch build, e.g.:"
	@echo "    . .venv/bin/activate && uv pip install torch torchvision"
	. .venv/bin/activate && $(PY) -c "import nltk; nltk.download('semcor'); nltk.download('wordnet'); nltk.download('omw-1.4')"

lint:
	uv run ruff check src tests

test:
	uv run pytest

# --------------------------------------------------------------------------- #
# Pipeline stages (see run.sh for the full ordered sequence)                   #
# --------------------------------------------------------------------------- #
audit:
	$(PY) -m src.audit --imagenet-root "$$IMAGENET_ROOT" --output box_audit.json

# `make data` extracts the occurrences for $(CORPUS), then selects targets.
data: data-$(CORPUS)
	$(PY) -m src.index_imagenet --root "$$IMAGENET_ROOT" --output data/imagenet_classes.parquet
	$(PY) -m src.select_targets --occurrences $(OCC) \
	  --imagenet-index data/imagenet_classes.parquet --config $(CONFIG) --output data/targets.csv

data-semcor:
	$(PY) -m src.extract_semcor --output data/semcor_occurrences.parquet

# DWUG EN: fetch the dataset first (make dwug-fetch), then extract usages.
data-dwug_en:
	$(PY) -m src.extract_dwug --dwug-root "$(DWUG_EN_ROOT)" --output data/dwug_occurrences.parquet

dwug-fetch:
	@mkdir -p data
	curl -sL "https://zenodo.org/api/records/7387261/files/dwug_en.zip/content" -o data/dwug_en.zip
	cd data && unzip -q -o dwug_en.zip && rm -f dwug_en.zip
	@echo ">>> DWUG EN v3.0.0 extracted to data/dwug_en"

prototypes:
	$(PY) -m src.embed_imagenet --config $(CONFIG) --output cache/imagenet_prototypes.pt

contexts:
	$(PY) -m src.embed_contexts --config $(CONFIG) \
	  --occurrences $(OCC) --targets data/targets.csv \
	  --imagenet-index data/imagenet_classes.parquet \
	  --text-output cache/text_contexts.pt --label-output cache/label_prototypes.pt

cluster:
	$(PY) -m src.cluster --config $(CONFIG) --output $(RUN)

evaluate:
	$(PY) -m src.evaluate --run $(RUN) --output $(RUN)/metrics.csv

report:
	$(PY) -m src.report --run $(RUN) --output $(RUN)/report.md

pilot: audit data prototypes contexts cluster evaluate report

clean:
	rm -rf results/oracle_k results/unknown_k __pycache__ src/__pycache__ src/pilotlib/__pycache__
	find . -name '*.pyc' -delete
