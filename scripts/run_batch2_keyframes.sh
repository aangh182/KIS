#!/usr/bin/env bash
# Chuẩn bị keyframe toàn bộ batch2 + đóng gói Kaggle dataset.
# Chạy: bash scripts/run_batch2_keyframes.sh
# Resume được: probe upsert, shots/keyframes bỏ qua video đã xong.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-Dataset/batch2}"
WORKERS="${WORKERS:-8}"
PROFILE="${PROFILE:-hybrid}"
GRID_S="${GRID_S:-3}"
LOG_DIR="${LOG_DIR:-logs}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/pipeline-$STAMP.log"
mkdir -p "$LOG_DIR"
echo "log: $LOG"

exec > >(tee -a "$LOG") 2>&1

export KIS_FFMPEG="${KIS_FFMPEG:-$ROOT/.tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg}"
export KIS_FFPROBE="${KIS_FFPROBE:-$ROOT/.tools/ffmpeg-master-latest-linux64-gpl/bin/ffprobe}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

step() { printf '\n===== %s | %s =====\n' "$(date '+%F %T')" "$*"; }
trap 'printf "\n[%s] exit=%s | log: %s\n" "$(date +%H:%M:%S)" "$?" "$LOG"' EXIT

if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOG_DIR/pipeline.lock"
    flock -n 9 || { echo "đã có instance khác đang chạy ($LOG_DIR/pipeline.lock)"; exit 1; }
fi

step "kiểm tra môi trường"
[[ -x "$KIS_FFPROBE" ]] || { echo "thiếu ffprobe: $KIS_FFPROBE"; exit 1; }
[[ -x "$KIS_FFMPEG" ]] || { echo "thiếu ffmpeg: $KIS_FFMPEG"; exit 1; }
[[ -d "$DATASET" ]] || { echo "thiếu dataset: $DATASET"; exit 1; }
uv run python -c "import av, imagehash, scenedetect" \
    || { echo "thiếu dependency offline: chạy 'uv sync --extra offline'"; exit 1; }

n_video_files=$(find "$DATASET" -type f \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o \
       -iname '*.webm' -o -iname '*.avi' -o -iname '*.m4v' \) | wc -l)
free_gb=$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
echo "dataset: $DATASET | video files: $n_video_files"
echo "workers: $WORKERS | profile: $PROFILE | grid_s: $GRID_S"
echo "free disk: ${free_gb} GB"
(( free_gb >= MIN_FREE_GB )) || { echo "cần >= ${MIN_FREE_GB} GB trống"; exit 1; }

step "1/7 probe"
uv run kis probe --dataset "$DATASET"

step "2/7 shots"
uv run kis shots --workers "$WORKERS"

step "3/7 keyframes"
uv run kis keyframes --profile "$PROFILE" --grid-s "$GRID_S" --workers "$WORKERS"

step "4/7 verify timestamps"
uv run kis debug verify-timestamps --sample 100 --show-failures

step "5/7 coverage gate"
uv run kis eval coverage --report eval/coverage.md

step "6/7 kiểm tra DB"
uv run python - <<'PY'
import sqlite3
import sys

conn = sqlite3.connect("index/meta.db")
videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
with_kf = conn.execute("SELECT COUNT(DISTINCT video_id) FROM keyframes").fetchone()[0]
keyframes = conn.execute("SELECT COUNT(*) FROM keyframes").fetchone()[0]
conn.close()

print(f"videos={videos} videos_with_keyframes={with_kf} keyframes={keyframes}")
if videos == 0 or keyframes == 0:
    sys.exit("DB chưa có dữ liệu keyframe")
if with_kf < videos:
    print(f"WARNING: {videos - with_kf} video không có keyframe", file=sys.stderr)
PY

step "7/7 checkpoint WAL + đóng gói"
uv run python - <<'PY'
import sqlite3
import sys
from pathlib import Path

db = Path("index/meta.db")
conn = sqlite3.connect(db)
busy, log_frames, checkpointed = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
conn.close()

wal = Path(str(db) + "-wal")
wal_bytes = wal.stat().st_size if wal.exists() else 0
print(f"wal_checkpoint busy={busy} log={log_frames} checkpointed={checkpointed} wal_bytes={wal_bytes}")
if busy:
    sys.exit("WAL còn busy: tắt 'kis serve' hoặc process đang ghi DB rồi chạy lại")
if wal_bytes:
    sys.exit("WAL chưa truncate; dừng để tránh copy thiếu dữ liệu")
PY

rm -rf kaggle_upload
mkdir -p kaggle_upload
tar -cf kaggle_upload/batch2_keyframes.tar -C index keyframes
cp index/meta.db kaggle_upload/
du -sh kaggle_upload/batch2_keyframes.tar kaggle_upload/meta.db

step "xong"
cat <<'EOF'

Đóng gói xong. Các bước tiếp theo (chạy tay):
  uv tool install kaggle
  mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
  kaggle datasets init -p kaggle_upload
  # sửa kaggle_upload/dataset-metadata.json: title + id: <username>/<slug>
  kaggle datasets create -p kaggle_upload

Sau đó import notebooks/kaggle_embed.ipynb, attach dataset, bật Internet + GPU T4 x2.
EOF
