#!/usr/bin/env bash
set -euo pipefail

# Determinism knobs (best-effort)
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Defaults (override via env or args)
WEIGHTS_A=${WEIGHTS_A:-/workspace/yolo11n.pt}
WEIGHTS_B=${WEIGHTS_B:-/workspace/yolov8n.pt}
CONF_A=${CONF_A:-0.25}
CONF_B=${CONF_B:-0.35}
IMGSZ_A=${IMGSZ_A:-640}
IMGSZ_B=${IMGSZ_B:-416}
FREQ=${FREQ:-20.0}
OVERLAY_CONF=${OVERLAY_CONF:-0.25}
EVERY_N=${EVERY_N:-5}
SYNC_MS=${SYNC_MS:-200}
SAVE_MIN=${SAVE_MIN:-0}

SPLITS=(train val test)
if [[ $# -gt 0 ]]; then
  SPLITS=("$@")
fi

for SPLIT in "${SPLITS[@]}"; do
  echo "=== Running split: ${SPLIT} ==="
  ros2 launch ab_test_bringup dataset_ab.launch.py \
    split:=${SPLIT} \
    frequency:=${FREQ} \
    loop:=false \
    weightsA:=${WEIGHTS_A} confA:=${CONF_A} imgszA:=${IMGSZ_A} \
    weightsB:=${WEIGHTS_B} confB:=${CONF_B} imgszB:=${IMGSZ_B} \
    overlay_conf:=${OVERLAY_CONF} images_every_n:=${EVERY_N} overlay_sync_ms:=${SYNC_MS} save_min_boxes:=${SAVE_MIN}
done

