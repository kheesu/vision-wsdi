.PHONY: setup lint test audit data data-semcor data-dwug_en data-semeval2013 data-semeval2010 \
        dwug-fetch semeval2013-fetch semeval2010-fetch prototypes contexts cluster evaluate report pilot clean

CONFIG ?= configs/pilot.yaml
RUN    ?= results/oracle_k
PY     ?= python
CORPUS ?= semcor
# Occurrence parquet fed to select_targets; switches with the corpus.
OCC_semcor      = data/semcor_occurrences.parquet
OCC_dwug_en     = data/dwug_occurrences.parquet
OCC_semeval2013 = data/semeval2013_occurrences.parquet
OCC_semeval2010 = data/semeval2010_occurrences.parquet
OCC    ?= $(OCC_$(CORPUS))
DWUG_EN_ROOT     ?= data/dwug_en
SEMEVAL2013_ROOT ?= data/semeval2013
SEMEVAL2010_ROOT ?= data/semeval2010

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

# SemEval-2013 Task 13 (WSI, graded senses). English; nouns single-sense subset.
data-semeval2013:
	$(PY) -m src.extract_semeval2013 --root "$(SEMEVAL2013_ROOT)" \
	  --output data/semeval2013_occurrences.parquet

semeval2013-fetch:
	@mkdir -p data/semeval2013
	curl -sL "https://zenodo.org/api/records/5638384/files/SemEval-2013-Task-13-test-data.zip/content" -o data/se2013.zip
	cd data && unzip -q -o se2013.zip && rm -f se2013.zip
	cp -r data/SemEval-2013-Task-13-test-data/. data/semeval2013/ && rm -rf data/SemEval-2013-Task-13-test-data
	@echo ">>> SemEval-2013 Task 13 extracted to data/semeval2013"

# SemEval-2010 Task 14 (WSI & Disambiguation). English; nouns.
data-semeval2010:
	$(PY) -m src.extract_semeval2010 --root "$(SEMEVAL2010_ROOT)/test_data" \
	  --gold "$(SEMEVAL2010_ROOT)/evaluation/unsup_eval/keys/all.key" \
	  --output data/semeval2010_occurrences.parquet

semeval2010-fetch:
	@mkdir -p data/semeval2010
	curl -sL "https://zenodo.org/api/records/5638549/files/test_data.tar.gz/content" -o data/se2010_test.tar.gz
	curl -sL "https://zenodo.org/api/records/5638549/files/evaluation.zip/content" -o data/se2010_eval.zip
	cd data/semeval2010 && tar xzf ../se2010_test.tar.gz && unzip -q -o ../se2010_eval.zip
	rm -f data/se2010_test.tar.gz data/se2010_eval.zip
	@echo ">>> SemEval-2010 Task 14 extracted to data/semeval2010"

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
