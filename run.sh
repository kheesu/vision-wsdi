#!/usr/bin/env bash
# Reproduce the full pilot. Run from the vision-lsi/ directory with the venv active.
#   export IMAGENET_ROOT=/path/to/imagenet   # optional; image systems skip if unset
#   bash run.sh
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CONFIG="${CONFIG:-configs/pilot.yaml}"
PY="${PY:-python}"
# Occurrence corpus: semcor (default) or dwug_en. For dwug_en, set DWUG_EN_ROOT
# to the extracted dataset (see `make dwug-fetch`).
CORPUS="${CORPUS:-semcor}"

$PY -m src.audit --imagenet-root "${IMAGENET_ROOT:-}" --output box_audit.json

if [ "$CORPUS" = "dwug_en" ]; then
  OCC="data/dwug_occurrences.parquet"
  $PY -m src.extract_dwug --dwug-root "${DWUG_EN_ROOT:-data/dwug_en}" --output "$OCC"
else
  OCC="data/semcor_occurrences.parquet"
  $PY -m src.extract_semcor --output "$OCC"
fi

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
