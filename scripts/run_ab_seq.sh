#!/usr/bin/env bash
set -euo pipefail

# ---------- settings ----------
BAG_PATH="${1:-/workspace/bags/argo_full}"   # 재생할 bag 디렉터리 or .mcap 파일

# A/B detector params (env overrideable)
A_CONF=${A_CONF:-0.25}
A_IMGSZ=${A_IMGSZ:-640}
B_CONF=${B_CONF:-0.35}
B_IMGSZ=${B_IMGSZ:-416}

# Common options
IMG_EVERY_N=${IMG_EVERY_N:-5}
OVERLAY_CONF_MIN=${OVERLAY_CONF_MIN:-0.25}   # overlay_viz.py의 overlay_conf_min
YOLO_EVERY_N=${YOLO_EVERY_N:-1}
# Optional: YOLO_WEIGHTS to override model path
# YOLO_WEIGHTS=${YOLO_WEIGHTS:-}
# ------------------------------

safe_source() { set +u; source "$1"; set -u; }
safe_source /opt/ros/iron/setup.bash
safe_source /workspace/install/setup.bash

cleanup_pids() {
  for pid in "$@"; do
    [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
  done
}

run_case () {
  local NAME="$1"     # A or B
  local DET_CONF="$2"
  local DET_IMGSZ="$3"
  local YOLO_TOPIC="/yolo${NAME}/bounding_boxes"
  local OUT_DIR="/workspace"
  local IMG_DIR="${OUT_DIR}/frames_${NAME}"
  local MP4="${OUT_DIR}/out_${NAME}.mp4"
  local CSV="${OUT_DIR}/out_${NAME}.csv"
  local JSON="${OUT_DIR}/out_${NAME}.json"

  echo ""
  echo "========== CASE ${NAME} (conf=${DET_CONF}, imgsz=${DET_IMGSZ}) =========="
  pkill -f yolo_subscriber_py || true
  pkill -f overlay_viz.py || true

  # Detector (백그라운드)
  if [[ -n "${YOLO_WEIGHTS:-}" ]]; then
    YOLO_WEIGHTS_ARGS=( -p weights:="${YOLO_WEIGHTS}" )
  else
    YOLO_WEIGHTS_ARGS=()
  fi

  ros2 run yolo_subscriber_py yolo_subscriber_py_node --ros-args \
    -p conf:="${DET_CONF}" -p imgsz:="${DET_IMGSZ}" -p every_n:="${YOLO_EVERY_N}" \
    "${YOLO_WEIGHTS_ARGS[@]}" \
    -r /yolo/bounding_boxes:="${YOLO_TOPIC}" \
    >/tmp/yolo_${NAME}.log 2>&1 & DET_PID=$!

  # Overlay (백그라운드) - 입력은 /image_raw, 바운딩박스는 케이스별 토픽
  python3 /workspace/src/overlay_viz.py --ros-args \
    -p overlay_conf_min:="${OVERLAY_CONF_MIN}" \
    -p save_images:=true \
    -p images_dir:="${IMG_DIR}" \
    -p images_every_n:="${IMG_EVERY_N}" \
    -p save_mp4:=true  -p mp4_path:="${MP4}" \
    -p save_boxes_csv:=true  -p csv_path:="${CSV}" \
    -p save_boxes_json:=true -p json_path:="${JSON}" \
    -r /yolo/bounding_boxes:="${YOLO_TOPIC}" \
    >/tmp/overlay_${NAME}.log 2>&1 & OVL_PID=$!

  # give nodes a brief moment to create publishers/subscribers
  sleep 0.5

  # Bag 재생 (한 바퀴)
  echo "[INFO] playing bag: ${BAG_PATH}"
  # Try directory play first; if it fails quickly, fallback to single .mcap with explicit storage
  (
    set -o pipefail
    ros2 bag play "${BAG_PATH}" --clock \
      --read-ahead-queue-size 10 \
      --rate 0.7 \
      --topics /image_raw
  ) >"/tmp/bag_${NAME}.log" 2>&1 & PLY_PID=$!

  # If player exits immediately (e.g., invalid path or no metadata.yaml), fallback to .mcap file
  sleep 0.3
  if ! kill -0 ${PLY_PID} 2>/dev/null; then
    MCAP_FILE=$(find "${BAG_PATH}" -maxdepth 1 -type f -name "*.mcap" | head -n1 || true)
    if [[ -n "${MCAP_FILE}" ]]; then
      echo "[INFO] dir play failed; fallback to file: ${MCAP_FILE}" | tee -a "/tmp/bag_${NAME}.log"
      ros2 bag play -s mcap "${MCAP_FILE}" --clock \
        --read-ahead-queue-size 10 \
        --rate 0.7 \
        --topics /image_raw \
        >>"/tmp/bag_${NAME}.log" 2>&1 & PLY_PID=$!
    else
      echo "[ERROR] No playable bag found at ${BAG_PATH}" | tee -a "/tmp/bag_${NAME}.log"
    fi
  fi

  # 재생이 끝날 때까지 대기
  wait ${PLY_PID} || true
  echo "[INFO] bag finished, stopping nodes (CASE ${NAME})"

  # 백그라운드 종료
  cleanup_pids "${DET_PID}" "${OVL_PID}"
  wait ${DET_PID} 2>/dev/null || true
  wait ${OVL_PID} 2>/dev/null || true

  echo "[DONE] CASE ${NAME} saved: ${MP4}, ${CSV}, ${JSON}, images -> ${IMG_DIR}"
}

# 실제 실행: A → B (env override respected)
run_case "A" "${A_CONF}" "${A_IMGSZ}"
run_case "B" "${B_CONF}" "${B_IMGSZ}"

echo ""
echo "================ ALL DONE ================"
ros2 bag info "${BAG_PATH}" || true
echo "Outputs:"
echo "  /workspace/out_A.mp4  /workspace/out_A.csv  /workspace/out_A.json  /workspace/frames_A/"
echo "  /workspace/out_B.mp4  /workspace/out_B.csv  /workspace/out_B.json  /workspace/frames_B/"

# ---- Summarize A/B automatically ----
echo "[INFO] summarizing A/B results"
export A_JSON="/workspace/out_A.json"
export B_JSON="/workspace/out_B.json"
export A_CSV="/workspace/out_A.csv"
export B_CSV="/workspace/out_B.csv"
export PAIR_CSV="/workspace/ab_pairs.csv"
export SUMMARY_CSV="/workspace/ab_summary.csv"
python3 /workspace/pick_best_frame.py || true
echo "[INFO] summary written -> $PAIR_CSV, $SUMMARY_CSV"
