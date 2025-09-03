#!/usr/bin/env bash
set -euo pipefail

# Single-case runner for YOLO + overlay with bag playback
# Usage examples:
#   scripts/run_single_case.sh --bag /workspace/bags/argo_full/argo_full_0.mcap --name A --conf 0.30 --imgsz 640 --rate 0.5 --read-ahead 1
#   scripts/run_single_case.sh /workspace/bags/argo_full
# Flags:
#   --bag PATH           Bag directory or .mcap file (positional arg also accepted)
#   --name NAME          Label for outputs (default: A)
#   --conf VAL           YOLO confidence (default: 0.25)
#   --imgsz VAL          YOLO imgsz (default: 640)
#   --every-n VAL        YOLO frame skipping (default: 1)
#   --weights PATH       YOLO weights path (optional)
#   --rate VAL           ros2 bag play rate (default: 0.7)
#   --read-ahead N       ros2 bag play read-ahead queue size (default: 1)
#   --start SECONDS      Start offset seconds (optional)
#   --duration SECONDS   Duration seconds (optional)
#   --overlay-conf VAL   overlay_conf_min (default: 0.25)
#   --img-every-n N      Save every N frames (default: 5)

usage() {
  sed -n '1,20p' "$0"
}

# Defaults
BAG_PATH=""
NAME="A"
CONF="0.25"
IMGSZ="640"
YOLO_EVERY_N="1"
YOLO_WEIGHTS=""
RATE="0.7"
READ_AHEAD="1"
START=""
DURATION=""
OVERLAY_CONF_MIN="0.25"
IMG_EVERY_N="5"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag) BAG_PATH="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --every-n) YOLO_EVERY_N="$2"; shift 2;;
    --weights) YOLO_WEIGHTS="$2"; shift 2;;
    --rate) RATE="$2"; shift 2;;
    --read-ahead) READ_AHEAD="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --overlay-conf) OVERLAY_CONF_MIN="$2"; shift 2;;
    --img-every-n) IMG_EVERY_N="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *)
      # positional: bag path
      if [[ -z "$BAG_PATH" ]]; then BAG_PATH="$1"; shift; else echo "Unknown arg: $1"; usage; exit 1; fi;;
  esac
done

if [[ -z "${BAG_PATH}" ]]; then
  echo "[ERROR] --bag PATH (or positional) is required"; usage; exit 1
fi

safe_source() { set +u; source "$1"; set -u; }
safe_source /opt/ros/iron/setup.bash
safe_source /workspace/install/setup.bash

cleanup_pids() { for pid in "$@"; do [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true; done; }

YOLO_TOPIC="/yolo/bounding_boxes"
OUT_DIR="/workspace"
IMG_DIR="${OUT_DIR}/frames_${NAME}"
MP4="${OUT_DIR}/out_${NAME}.mp4"
CSV="${OUT_DIR}/out_${NAME}.csv"
JSON="${OUT_DIR}/out_${NAME}.json"

echo "========== SINGLE CASE ${NAME} (conf=${CONF}, imgsz=${IMGSZ}) =========="
pkill -f yolo_subscriber_py || true
pkill -f overlay_viz.py || true

# Detector
YOLO_WEIGHTS_ARGS=()
if [[ -n "${YOLO_WEIGHTS}" ]]; then YOLO_WEIGHTS_ARGS=( -p weights:="${YOLO_WEIGHTS}" ); fi

ros2 run yolo_subscriber_py yolo_subscriber_py_node --ros-args \
  -p conf:="${CONF}" -p imgsz:="${IMGSZ}" -p every_n:="${YOLO_EVERY_N}" \
  "${YOLO_WEIGHTS_ARGS[@]}" \
  -r /yolo/bounding_boxes:="${YOLO_TOPIC}" \
  >/tmp/yolo_${NAME}.log 2>&1 & DET_PID=$!

# Overlay
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

sleep 0.5

echo "[INFO] playing bag: ${BAG_PATH}"
ARGS=( --clock --topics /image_raw --read-ahead-queue-size "${READ_AHEAD}" --rate "${RATE}" )
if [[ -n "${START}" ]]; then ARGS+=( --start-offset "${START}" ); fi
if [[ -n "${DURATION}" ]]; then ARGS+=( --duration "${DURATION}" ); fi

if [[ -f "${BAG_PATH}" && "${BAG_PATH}" == *.mcap ]]; then
  ros2 bag play -s mcap "${BAG_PATH}" "${ARGS[@]}" >"/tmp/bag_${NAME}.log" 2>&1 & PLY_PID=$!
else
  ros2 bag play "${BAG_PATH}" "${ARGS[@]}" >"/tmp/bag_${NAME}.log" 2>&1 & PLY_PID=$!
fi

wait ${PLY_PID} || true
echo "[INFO] bag finished, stopping nodes (CASE ${NAME})"

cleanup_pids "${DET_PID}" "${OVL_PID}"
wait ${DET_PID} 2>/dev/null || true
wait ${OVL_PID} 2>/dev/null || true

echo "[DONE] CASE ${NAME} saved: ${MP4}, ${CSV}, ${JSON}, images -> ${IMG_DIR}"
