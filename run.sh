#!/usr/bin/env bash
# Reproduce the full pilot. Run from the vision-lsi/ directory with the venv active.
#   export IMAGENET_ROOT=/path/to/imagenet   # optional; image systems skip if unset
#   bash run.sh
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CONFIG="${CONFIG:-configs/pilot.yaml}"
PY="${PY:-python}"
# Occurrence corpus: semcor (default), dwug_en, semeval2013, or semeval2010.
# Each honours a *_ROOT env var for its extracted dataset (see `make *-fetch`).
CORPUS="${CORPUS:-semcor}"

$PY -m src.audit --imagenet-root "${IMAGENET_ROOT:-}" --output box_audit.json

case "$CORPUS" in
  dwug_en)
    OCC="data/dwug_occurrences.parquet"
    $PY -m src.extract_dwug --dwug-root "${DWUG_EN_ROOT:-data/dwug_en}" --output "$OCC" ;;
  semeval2013)
    OCC="data/semeval2013_occurrences.parquet"
    $PY -m src.extract_semeval2013 --root "${SEMEVAL2013_ROOT:-data/semeval2013}" --output "$OCC" ;;
  semeval2010)
    R="${SEMEVAL2010_ROOT:-data/semeval2010}"
    OCC="data/semeval2010_occurrences.parquet"
    $PY -m src.extract_semeval2010 --root "$R/test_data" \
      --gold "$R/evaluation/unsup_eval/keys/all.key" --output "$OCC" ;;
  *)
    OCC="data/semcor_occurrences.parquet"
    $PY -m src.extract_semcor --output "$OCC" ;;
esac

$PY -m src.index_imagenet --root "${IMAGENET_ROOT:-}" --output data/imagenet_classes.parquet
$PY -m src.select_targets \
  --occurrences "$OCC" \
  --imagenet-index data/imagenet_classes.parquet \
  --config "$CONFIG" --output data/targets.csv

$PY -m src.embed_imagenet --config "$CONFIG" --output cache/imagenet_prototypes.pt
$PY -m src.embed_contexts --config "$CONFIG" \
  --occurrences "$OCC" --targets data/targets.csv \
  --imagenet-index data/imagenet_classes.parquet \
  --text-output cache/text_contexts.pt --label-output cache/label_prototypes.pt

# Controlled representation test (oracle K).
$PY -m src.cluster --config "$CONFIG" --mode oracle --output results/oracle_k
$PY -m src.evaluate --run results/oracle_k --config "$CONFIG"
$PY -m src.report --run results/oracle_k

# Unknown-K experiment.
$PY -m src.cluster --config "$CONFIG" --mode unknown --output results/unknown_k
$PY -m src.evaluate --run results/unknown_k --config "$CONFIG"
$PY -m src.report --run results/unknown_k

echo "Done. See results/oracle_k/report.md and results/unknown_k/report.md"
