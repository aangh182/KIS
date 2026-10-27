# KIS — Self Keyframe Pipeline

Hệ thống tự sinh keyframe cho bài toán Video Retrieval KIS (CPU-only), triển khai theo
[`docs/kis-self-keyframe-retrieval-plan.md`](docs/kis-self-keyframe-retrieval-plan.md).
README này ghi lại **những gì đã làm**, cách chạy, số liệu đo được và các sai lệch so với plan.

---

## 1. Phạm vi đã triển khai

Trọng tâm hiện tại là **M0-A: indexing keyframe**:

```
probe → shots → candidates → extract → dedup → SQLite + JPEG/thumbnail
                                      ↓
                         coverage gate + timestamp verification
```

Chưa triển khai (theo plan, thuộc M0-B/M1/M2):

- Embedding SigLIP-2 + FAISS + `kis search`.
- OCR (RapidOCR), ASR (faster-whisper), FTS5.
- Gemini query structuring, RRF, temporal aggregation, rerank.
- FastAPI UI và DRES submit/mock.
- Eval `queries.jsonl`, recall@20, ablation V0–V3 đầy đủ.

---

## 2. Trạng thái tiến độ

| Hạng mục | Trạng thái |
|---|---|
| Môi trường Python 3.12 + `uv` | ✅ Xong |
| Skeleton `src/kis/` + CLI `kis` | ✅ Xong |
| SQLite schema `videos/shots/keyframes` | ✅ Xong |
| `kis probe` | ✅ 20/20 video dev |
| `kis shots` | ✅ 20/20 video, 6.492 shot |
| `kis keyframes` | 🟡 Mới chạy thử 1 video (370 keyframe, theo policy cũ); **cần chạy lại toàn bộ** sau khi sửa policy |
| Candidate policy V0–V3 | ✅ Đã cài; preview V3 trên 1 video đạt coverage 0.9658 |
| pHash dedup + guard | ✅ Đã cài |
| `kis eval coverage` | ✅ Đã cài, xuất Markdown + gate |
| `kis debug verify-timestamps` | ✅ Đã cài; đã sửa cửa sổ ffprobe, **chưa chạy lại toàn bộ** |
| Unit test | ✅ Đã viết 15 test cho candidates / dedup / coverage (không chạy kiểm tra lại theo yêu cầu) |
| Embed / FAISS / search | ❌ Chưa làm |
| OCR / ASR / FTS | ❌ Chưa làm |
| UI / DRES | ❌ Chưa làm |

---

## 3. Cấu trúc code

```
src/kis/
  __init__.py            entrypoint main()
  cli.py                 argparse CLI: probe|shots|keyframes|eval|debug
  config.py              pydantic-settings (env prefix KIS_)
  db.py                  SQLite connect/migrate/upsert/replace
  verify.py              đối chiếu t_ms với ffprobe
  ingest/
    probe.py             ffprobe → bảng videos
    shots.py             PySceneDetect ContentDetector + ffmpeg fallback
    candidates.py        I-frame pool, grid, policy H§3.3 (V0–V3)
    extract.py           PyAV decode đúng PTS, JPEG/thumbnail, pHash, ghi DB
    dedup.py             pHash 64-bit + Hamming
  eval/
    coverage.py          temporal coverage, max_gap, gate, report Markdown
tests/
  test_candidates.py     policy chọn frame + profile alias
  test_dedup.py          pHash signed 64-bit + Hamming
  test_coverage.py       coverage/max_gap/gate
```

Artifact runtime (đã gitignore):

```
index/
  meta.db                SQLite
  keyframes/<pack>/<video_id>/<kf_n>.jpg      384 px, JPEG q85
  thumbnails/<pack>/<video_id>/<kf_n>.jpg     224 px, JPEG q80
eval/
  coverage.md            báo cáo gate
```

---

## 4. Cài đặt

Yêu cầu: `uv`, `ffmpeg`/`ffprobe` (đã kiểm tra với ffmpeg 8.0.1), Python 3.12.

```bash
uv sync --extra offline
```

`pyproject.toml` pin `>=3.12,<3.13` theo rủi ro wheel ở §11 của plan.
Các nhóm dependency:

| Nhóm | Nội dung |
|---|---|
| core | `numpy`, `pillow`, `pydantic-settings`, `tqdm` |
| `offline` | `av`, `scenedetect`, `imagehash` (đủ cho keyframe) |
| `embed` | `faiss-cpu`, `torch`, `transformers` |
| `ocr` | `rapidocr-onnxruntime` |
| `asr` | `faster-whisper` |

Cài riêng từng nhóm nặng khi cần, tránh kéo torch/CUDA ngay từ đầu.

---

## 5. Cách chạy

```bash
# 0. Probe toàn bộ video
uv run kis probe --dataset Dataset/Video

# 1. Shot detection (máy dev 3 core nên dùng --workers 3)
uv run kis shots --workers 3
#    thêm --force để detect lại, --limit N để chạy thử

# 2. Keyframe extraction
uv run kis keyframes --profile hybrid --grid-s 3 --workers 3
#    --force: xoá và làm lại; --limit N: chạy thử N video

# 3. Kiểm tra timestamp (bắt buộc pass trước khi index full)
uv run kis debug verify-timestamps --sample 100 --show-failures

# 4. Coverage gate
uv run kis eval coverage --report eval/coverage.md

# 5. Unit test
uv run python -m unittest discover -s tests -v
```

Có thể override bằng biến môi trường, ví dụ:

```bash
KIS_INDEX_DIR=index KIS_GRID_S=3 KIS_SAFETY_GRID_S=2 KIS_DEGRADE_ON_LONG_SHOT=0 ...
```

---

## 6. Chính sách chọn keyframe

### 6.1. Candidate pools

- **I-frame pool**: `ffprobe -skip_frame nokey`; fallback `ffmpeg -skip_frame nokey -vf showinfo`.
- **Shot bounds**: PySceneDetect `ContentDetector(threshold=27)`; fallback ffmpeg
  `select='gt(scene,0.30)' + showinfo` (PTS lấy trực tiếp từ demuxer).
- **Grid**: lưới an toàn 1 fps cho video degraded, lưới 2 s toàn cục cho V3.

### 6.2. Profile

| Profile | Mô tả |
|---|---|
| `v0` / `iframe` | chỉ I-frame |
| `v1` / `grid` | lưới đều 1 fps |
| `v2` / `shot` | midpoint mỗi shot |
| `v3` / `hybrid` (mặc định) | I-frame trong shot + midpoint nếu shot không có I-frame + lưới 3 s trong shot dài + lưới an toàn 2 s |

Quy tắc hybrid:

- `MIN_SHOT_S = 0.4 s`: shot ngắn hơn bị gộp vào shot trước (thực hiện ở `shots.py`).
- `MAX_PER_SHOT = 6`: **chỉ giới hạn pick ngữ nghĩa** (I-frame/midpoint); grid không bị cắt,
  tránh lỗ hổng 20–30 s trong shot dài.
- Video chỉ có ≤1 shot → fallback toàn bộ sang lưới 1 fps, `degraded = 1`.

### 6.3. Timestamp

- `t_ms` **chỉ** lấy từ PTS của frame giải mã bằng PyAV (`frame.pts * time_base`).
- Không dùng `frame_idx / fps`.
- `frame_idx` vẫn lưu để tham khảo.
- `verify-timestamps` đối chiếu lại với `ffprobe -show_frames`.

### 6.4. Dedup

- Trong cùng shot: 2 pick cách nhau `< 800 ms` → giữ 1.
- pHash 64-bit (Hamming ≤ 4) trong cửa sổ ±2 shot.
- **Không dedup pHash cho grid/degraded** (shot_id = `NULL`) để bảo toàn coverage.
- **Guard coverage**: không drop pHash nếu khoảng cách tới candidate kế tiếp
  vượt `phash_max_gap_ms = 2000 ms`.
- pHash lưu dạng **signed 64-bit** để vừa `INTEGER` của SQLite; so sánh dùng unsigned.

---

## 7. Schema SQLite

```sql
videos (
  video_id TEXT PK, pack_id TEXT, video_path TEXT, duration_ms INT,
  width INT, height INT, codec TEXT, fps_source REAL, time_base TEXT,
  nb_frames INT, vfr INT, degraded INT DEFAULT 0
)

shots (
  video_id TEXT, shot_id INT, t_start_ms INT, t_end_ms INT, start_frame INT,
  PRIMARY KEY (video_id, shot_id)
)

keyframes (
  keyframe_pk INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id TEXT, kf_n INT NOT NULL, frame_idx INT,
  t_ms INT NOT NULL, pts REAL, shot_id INT,
  image_path TEXT, thumb_path TEXT, source TEXT, phash INTEGER,
  UNIQUE (video_id, kf_n), UNIQUE (video_id, t_ms)
)
```

`image_path` / `thumb_path` lưu tương đối so với `index/`.

---

## 8. Kết quả đo trên 20 video dev

- 20 video `L21_*` / `L22_*`, mỗi video ~20 phút, 1280×720, 25 fps CFR.
- Probe: 20/20.
- Shot detection: 20/20, **6.492 shot**, thời gian ~50–90 s/video trên 1 core.
- Video dev nào cũng có ít nhất một shot >10 s (shot tĩnh/anchor desk).
- Keyframe: mới extract thử `L21_V024` được **370 keyframe** theo policy cũ
  (trước khi sửa `MAX_PER_SHOT`, safety grid và pHash guard) → **số này không còn đại diện**.
- Preview candidate V3 mới (chưa extract) trên `L21_V024`:
  - **731 candidate**
  - **coverage ±1 s = 0.9658**
  - **max gap = 2.760 s**, p99 = 2.760 s
  - `degraded = 0`

Kết luận sơ bộ: với `safety_grid_s = 2`, V3 có thể đạt gate
`max_gap p99 ≤ 3 s` và `coverage ≥ 0.95` mà vẫn giữ keyframe ngữ nghĩa theo shot.

---

## 9. Sai lệch / điều chỉnh so với plan

1. **`requires-python` = 3.12** thay vì 3.14, đúng đối sách rủi ro wheel ở §11.
2. **Tách optional deps** thành `offline` (keyframe), `embed`, `ocr`, `asr`
   thay vì gộp tất cả vào `offline`; tránh kéo torch/onnxruntime khi mới làm keyframe.
3. **`degrade_on_long_shot` mặc định `False`.** Plan §3.3 nói "có shot >10 s → fallback
   toàn bộ video sang 1 fps", nhưng cả 20/20 video dev đều có shot >10 s, nên nếu bật
   thì V3 trùng V1 và mất hoàn toàn đặc tính shot-based. Hybrid đã tự thêm lưới 3 s
   trong shot dài. Có thể bật lại bằng `KIS_DEGRADE_ON_LONG_SHOT=1`.
4. **Thêm lưới an toàn toàn cục `safety_grid_s = 2`.** Lưới 3 s đơn thuần chỉ cho
   coverage ~0.72 ở bán kính ±1 s, không đạt gate 0.95. Đây là đúng tinh thần §11
   ("không qua gate thì giảm grid_s"), nhưng áp dụng như một lớp grid bổ sung.
5. **`MAX_PER_SHOT` chỉ cap pick ngữ nghĩa**, không cap grid; nếu cap cả grid thì
   shot 32 s chỉ còn 6 frame đầu, tạo lỗ hổng >20 s.
6. **pHash dedup có guard khoảng trống** và bỏ qua grid/degraded; nếu không, các shot
   tĩnh dài bị dedup thành 1 frame, coverage tụt còn ~0.55.
7. **`verify-timestamps` dùng cửa sổ ffprobe `+8 s`**, vì `-read_intervals` của ffprobe
   tính độ dài từ keyframe mà nó seek tới, không phải từ mốc yêu cầu; cửa sổ `+1.5 s`
   ban đầu không chứa frame cần kiểm tra.

---

## 10. Việc tiếp theo

1. Chạy lại `kis keyframes` cho cả 20 video sau khi đã sửa policy; xoá 370 keyframe cũ.
2. Chạy `kis debug verify-timestamps --sample 100` trên index mới.
3. Chạy `kis eval coverage` trên cả 20 video và đối chiếu gate.
4. Chạy ablation V0–V3 (cần thêm command `kis eval ablation` hoặc script).
5. Sang M0-B: embedding SigLIP-2 → FAISS, `kis search`, eval queries, rồi M1 (OCR/ASR/UI).

---

## 11. Ghi chú vận hành

- Máy dev hiện tại **3 core**: dùng `--workers 3`; `--workers 8` sẽ oversubscribe và chậm hơn.
- `Dataset/` và `index/` đã nằm trong `.gitignore`.
- SQLite dùng WAL + `busy_timeout=60 s`; mỗi worker mở connection riêng.
- `kis keyframes` mặc định **resume**: video đã có keyframe sẽ bỏ qua; dùng `--force` để làm lại.
