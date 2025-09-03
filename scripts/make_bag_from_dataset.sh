#!/usr/bin/env bash
set -euo pipefail

# ---------- safe source (set -u 보호) ----------
safe_source() { set +u; source "$1"; set -u; }

# ---------- settings ----------
BAG_DIR=${1:-/workspace/bags}        # 저장 폴더
BAG_NAME=${2:-argo_full}             # bag 이름(폴더명)
TOPICS="/image_raw"                  # 녹화할 토픽
# 퍼블리셔: 데이터셋 한 바퀴 후 종료(loop=false). 필요 시 추가 파라미터를 뒤에 붙이세요.
PUB_CMD="python3 /workspace/src/image_pub_test.py --ros-args -p loop:=false"
STORAGE="mcap"                       # 저장 backend
# --------------------------------

echo "[INFO] source ROS"
safe_source /opt/ros/iron/setup.bash
safe_source /workspace/install/setup.bash

mkdir -p "$BAG_DIR"
OUT_DIR="$BAG_DIR/$BAG_NAME"

echo "[INFO] kill leftovers"
pkill -f ros2 || true
pkill -f image_pub_test.py || true

echo "[INFO] start recorder -> $OUT_DIR (storage=$STORAGE)"
# 주의: -o 는 폴더명을 주면 그 폴더 아래에 파일(.mcap)들이 생김
ros2 bag record -s "$STORAGE" -o "$OUT_DIR" $TOPICS --storage-config "" &
REC_PID=$!

cleanup() {
  echo "[INFO] stopping recorder"
  kill $REC_PID 2>/dev/null || true
  wait $REC_PID 2>/dev/null || true
}
trap cleanup EXIT

sleep 1

echo "[INFO] start dataset publisher"
set +e
$PUB_CMD
PUB_RC=$?
set -e
echo "[INFO] publisher exited (rc=$PUB_RC), stopping recorder"

cleanup
trap - EXIT

# ---- 메타데이터 복구 시도(폴더 재생 호환성) ----
if [ ! -f "$OUT_DIR/metadata.yaml" ]; then
  echo "[INFO] metadata.yaml not found. Trying reindex..."
  ros2 bag reindex -s "$STORAGE" "$OUT_DIR" || true
fi

echo "[INFO] bag content"
if ros2 bag info "$OUT_DIR" >/dev/null 2>&1; then
  ros2 bag info "$OUT_DIR"
else
  # 폴더 재생 실패 시 파일 직접 지정(단일 MCAP)
  F=$(find "$OUT_DIR" -maxdepth 1 -type f -name "*.mcap" | head -n1 || true)
  if [ -n "${F:-}" ]; then
    ros2 bag info -s "$STORAGE" "$F" || true
  else
    echo "[WARN] No .mcap file found in $OUT_DIR"
  fi
fi

echo "[DONE] Bag saved at: $OUT_DIR"
echo "[HINT] Play options:"
echo "  ros2 bag play $OUT_DIR --clock              # 폴더 재생(메타데이터 있으면)"
echo "  ros2 bag play -s $STORAGE $OUT_DIR/*.mcap --clock  # 파일 직접 재생"
