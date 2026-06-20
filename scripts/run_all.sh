#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/vii.yaml}
LIMIT=${LIMIT:-}
SEED=${SEED:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/paper_repro}
PYTHON=${PYTHON:-python}

run_one() {
  local dataset=$1
  local model=$2
  local extra_limit=()
  if [[ -n "${LIMIT}" ]]; then
    extra_limit=(--limit "${LIMIT}")
  fi
  "${PYTHON}" scripts/run_vii_experiment.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_ROOT}/${dataset}_${model}" \
    --seed "${SEED}" \
    "${extra_limit[@]}"
}

# CI smoke test: no commercial backend dispatch, one sample request artifact.
"${PYTHON}" scripts/run_vii_experiment.py \
  --dataset coco_i2v_safetybench \
  --model mock \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/ci_mock" \
  --limit 1 \
  --seed "${SEED}" \
  --dry-run

for dataset in coco_i2v_safetybench conceptrisk; do
  for model in kling veo seedance pixverse; do
    run_one "${dataset}" "${model}"
  done
done
