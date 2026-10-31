# KIS — Self Keyframe Pipeline

Hệ thống tự sinh keyframe cho bài toán Video Retrieval KIS (CPU-only), triển khai theo
[`docs/kis-self-keyframe-retrieval-plan.md`](docs/kis-self-keyframe-retrieval-plan.md).
README này ghi lại **những gì đã làm**, cách chạy, số liệu đo được và các sai lệch so với plan.

---

## 1. Phạm vi đã triển khai

Trọng tâm hiện tại là **M0-A/M0-B: keyframe + visual retrieval**:

```
probe → shots → candidates → extract → dedup → SQLite + JPEG/thumbnail
                                      ↓
                         coverage gate + timestamp verification
                                      ↓
              embed (SigLIP-2) → FAISS → search CLI + web UI
```

Chưa triển khai (theo plan, thuộc M1/M2):

- Embed full 20 video đang chạy nền (2/20 xong); index hiện là partial.
- OCR (RapidOCR), ASR (faster-whisper), FTS5.
- Gemini query structuring, RRF, temporal aggregation, rerank.
- DRES submit/mock, timer/EV, pin tray; UI hiện mới có search/grid/detail/video.
- Eval `queries.jsonl` và recall@20. Ablation V0–V3 hiện mới đo temporal coverage,
  chưa đo retrieval quality.

---

## 2. Trạng thái tiến độ

| Hạng mục | Trạng thái |
|---|---|
| Môi trường Python 3.12 + `uv` | ✅ Xong |
| Skeleton `src/kis/` + CLI `kis` | ✅ Xong |
| SQLite schema `videos/shots/keyframes` | ✅ Xong |
| `kis probe` | ✅ 20/20 video test (`Dataset/batch2`) |
| `kis shots` | ✅ 20/20 video, 6.543 shot |
| `kis keyframes` | ✅ 20/20 video, 14.061 keyframe (hybrid/V3) |
| Candidate policy V0–V3 | ✅ Đã cài + ablation trên 20 video |
| pHash dedup + guard | ✅ Đã cài; 296 drop (2.1%) khi chạy thật |
| `kis eval coverage` | ✅ Đã cài; 20 video **PASS** (coverage 0.9672, p99 2.700 ms) |
| `kis eval ablation` | ✅ Mới thêm; so sánh V0–V3 candidate-level |
| `kis debug verify-timestamps` | ✅ 100/100 khớp tuyệt đối |
| Unit test | ✅ 27/27 pass |
| `kis embed` (SigLIP-2) | 🟡 Code xong; đang embed 20 video (**2/20 xong**, 1.428 vector) |
| `kis index build` (FAISS) | 🟡 Partial index 2 video; full index chạy sau khi embed xong |
| `kis search` | ✅ CLI + latency + JSON; chất lượng xem §8.3 |
| `kis serve` (web UI) | ✅ Search/grid/detail/video, đã test API + Range |
| OCR / ASR / FTS | ❌ Chưa làm |
| DRES / timer / pin tray | ❌ Chưa làm |

---

## 3. Cấu trúc code

```
src/kis/
  __init__.py            entrypoint main()
  cli.py                 argparse CLI: probe|shots|keyframes|embed|index|search|serve|eval|debug
  config.py              pydantic-settings (env prefix KIS_)
  db.py                  SQLite connect/migrate/upsert + truy vấn keyframe/embedding
  verify.py              đối chiếu t_ms với ffprobe
  ingest/
    probe.py             ffprobe → bảng videos
    shots.py             PySceneDetect ContentDetector + ffmpeg fallback
    candidates.py        I-frame pool, grid, policy H§3.3 (V0–V3)
    extract.py           PyAV decode đúng PTS, JPEG/thumbnail, pHash, ghi DB
    dedup.py             pHash 64-bit + Hamming
    embed.py             SigLIP-2 image tower → .npy fp16 + sidecar JSON
  index/
    build.py             shard → FAISS IndexIDMap2(FlatIP) + manifest
    store.py             load/validate index, search id → keyframe_pk
  search/
    engine.py            SigLIP-2 text tower (+ dịch VI→EN tuỳ chọn) → FAISS → SQLite
  web/
    app.py               FastAPI: /api/search, /api/health, /api/video (Range)
    static/index.html    UI 1 trang: grid, detail, video, phím tắt
  eval/
    coverage.py          temporal coverage, max_gap, gate, report Markdown
    ablation.py          so sánh candidate V0–V3 (không decode)
tests/
  test_candidates.py     policy chọn frame + profile alias
  test_dedup.py          pHash signed 64-bit + Hamming
  test_coverage.py       coverage/max_gap/gate
  test_embedding.py      shard ghi/validate/checksum
  test_index_store.py    build FAISS + id-map + assert toàn vẹn
  test_search_engine.py  format timestamp/kết quả + nhận diện tiếng Việt
```

Artifact runtime (đã gitignore):

```
index/
  meta.db                SQLite
  keyframes/<pack>/<video_id>/<kf_n>.jpg      384 px, JPEG q85
  thumbnails/<pack>/<video_id>/<kf_n>.jpg     224 px, JPEG q80
eval/
  coverage.md            báo cáo gate
  ablation.md            báo cáo so sánh V0–V3
```

---

## 4. Cài đặt

Yêu cầu: `uv`, `ffmpeg`/`ffprobe`, Python 3.12. Nếu máy không có sẵn ffmpeg/ffprobe,
dùng static build và trỏ `KIS_FFMPEG` / `KIS_FFPROBE` (xem §11).

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
# 0. Probe video (đệ quy; batch hiện tại nằm ở Dataset/batch2)
uv run kis probe --dataset Dataset/batch2

# 1. Shot detection (12 core: dùng --workers 8)
uv run kis shots --workers 8
#    thêm --force để detect lại, --limit N để chạy thử

# 2. Keyframe extraction
uv run kis keyframes --profile hybrid --grid-s 3 --workers 8
#    --force: xoá và làm lại; --limit N: chạy thử N video

# 3. Kiểm tra timestamp (bắt buộc pass trước khi index full)
uv run kis debug verify-timestamps --sample 100 --show-failures

# 4. Coverage gate
uv run kis eval coverage --report eval/coverage.md

# 5. Ablation profile V0–V3 (candidate-level, không decode)
uv run kis eval ablation --report eval/ablation.md

# 6. Embed keyframe bằng SigLIP-2 (resume; --force để làm lại)
uv run kis embed --batch 16
#    --limit N / --video-id ID để chạy thử; --model để đổi model

# 7. Build FAISS + manifest (--allow-partial khi chưa embed đủ video)
uv run kis index build

# 8. Search CLI
uv run kis search "a woman in a white shirt" --k 20
uv run kis search "..." --k 20 --json

# 9. Web UI
uv run kis serve --port 8000
#    mở http://127.0.0.1:8000

# 10. Unit test
uv run python -m unittest discover -s tests -v
```

Cài dependency cho embed/UI:

```bash
uv sync --extra offline --extra embed --extra web
```

Có thể override bằng biến môi trường, ví dụ:

```bash
KIS_INDEX_DIR=index KIS_GRID_S=3 KIS_SAFETY_GRID_S=2 KIS_DEGRADE_ON_LONG_SHOT=0 ...
KIS_EMBED_MODEL=google/siglip2-base-patch16-384 KIS_EMBED_BATCH=16 ...
KIS_EMBED_TEXT_TEMPLATE="This is a photo of {}." KIS_TRANSLATE_QUERY=1 ...
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

## 8. Kết quả đo trên 20 video test

Test slice lấy từ `Dataset/batch2`: 20 video trải đều 10 pack `L13`–`L22`
(2 video/pack, gồm `V001` và `V010`/`V020`), tổng **23.590 s ≈ 6,55 giờ** video.

- Tất cả 1280×720, H.264, CFR, 25 hoặc 30 fps. Không có video VFR.
- Probe: 20/20.
- Shot detection: 20/20 bằng PySceneDetect, **6.543 shot**
  (261–419 shot/video, trung bình 327), shot dài nhất 41,8 s.
  8 worker trên 12 core: **6 phút 47 giây** (≈10,6 core trung bình).
- Keyframe extraction (`--profile hybrid --grid-s 3`):
  - 20/20 video, **14.061 keyframe** giữ lại / 14.357 candidate.
  - 0 gap drop, **296 pHash drop (2,1%)**.
  - 5.864 I-frame được dùng; theo nguồn: `grid` 9.678, `iframe` 3.726, `shot` 657.
  - 567–901 keyframe/video (trung bình 703), không video nào `degraded`.
  - 1 phút 53 giây/video trên 1 core; 19 video còn lại với 8 worker hết
    **5 phút 56 giây** (tổng ~8 phút cho 20 video).
  - Artifact: keyframe JPEG **318 MB** + thumbnail **138 MB** + SQLite **2,9 MB**
    → ≈ **22,8 MB/video**.
- Timestamp: `verify-timestamps --sample 100` → **100/100 khớp tuyệt đối**
  với ffprobe (`diff_ms = 0`).
- Coverage gate (`kis eval coverage`):
  - coverage ±1 s = **0.9672** (từng video 0.9590–0.9729)
  - max gap p50/p95/p99/max = **1900 / 2467 / 2700 / 2767 ms**
  - Gate `coverage ≥ 0.95` và `p99 ≤ 3 s`: **PASS**.

### 8.1. Ablation V0–V3 (candidate-level, cùng 20 video)

Đo trực tiếp trên candidate timestamp (chưa decode), nên số keyframe thực tế sau
dedup thấp hơn một chút. Kết quả V3 khớp với lần extract thật (0.9671 vs 0.9672,
p99 2.700 ms).

| Profile | Chiến lược | Candidate | Coverage ±1 s | p50/p95/p99/max gap (ms) | Gate |
|---|---|---:|---:|---|---|
| V0 | chỉ I-frame | 5.724 | 0.4675 | 4000 / 7000 / 7000 / 7894 | FAIL |
| V1 | lưới 1 fps | 23.596 | 1.0000 | 1000 / 1000 / 1000 / 1000 | PASS |
| V2 | midpoint mỗi shot | 6.530 | 0.5133 | 2617 / 9967 / 14366 / 28660 | FAIL |
| V3 | hybrid (mặc định) | 14.357 | 0.9671 | 1800 / 2467 / 2700 / 2780 | PASS |

V3 đạt gate với **ít hơn 39% candidate so với V1** (14.357 vs 23.596) mà vẫn giữ
keyframe ngữ nghĩa theo shot/I-frame. V0 và V2 fail nặng vì bỏ lỡ vùng giữa shot.

### 8.2. Sweep safety grid (V3)

| `safety_grid_s` | Candidate | Coverage ±1 s | p99 gap (ms) | Gate |
|---:|---:|---:|---:|---|
| 1.5 | 16.141 | 0.9956 | 2180 | PASS |
| **2.0 (mặc định)** | **14.357** | **0.9671** | **2700** | **PASS** |
| 2.5 | 13.338 | 0.8895 | 3100 | FAIL |
| 3.0 | 12.619 | 0.8564 | 3533 | FAIL |
| 4.0 | 11.827 | 0.8198 | 4040 | FAIL |

Đây là bằng chứng đo được cho lựa chọn `safety_grid_s = 2.0`: giảm xuống 1.5 chỉ
tăng coverage thêm 0.028 nhưng tốn thêm 12% keyframe; tăng lên 2.5 là **vỡ gate**
cả coverage lẫn p99. Ở mức đo này, 363 video của `batch2` tương đương khoảng
**255k keyframe** và **8,3 GB** artifact.

### 8.3. Embed / FAISS / search / UI

Đo trên CPU **Xeon E5-2690 (12 luồng, không AVX2)**, model
`google/siglip2-base-patch16-384`:

- Throughput embed thực tế: **~1,4–1,5 ảnh/s** với batch 16. Batch 32/64 chậm hơn;
  bf16 chậm hơn nhiều; dynamic int8 **crash `Illegal instruction`** vì CPU thiếu AVX2.
- 2 video đầu: **1.428 vector / 768-d fp16**; FAISS `IndexIDMap2(FlatIP)` 4,4 MB;
  shard ~1,1 MB/video.
- Search trên partial index (CPU rảnh): encode **~65–75 ms**, FAISS **~2–20 ms**,
  tổng **<100 ms**. Khi embed chạy nền, latency tăng do tranh CPU.
- API UI đã test: `/api/health`, `/api/search`, `/media/*`, `/api/video/...`;
  video trả **HTTP 206 Partial Content** đúng Range nên tua được.
- Chất lượng visual (partial, 2 video): caption ngắn tiếng Anh hoạt động —
  `a black suitcase` → vali, `red and white logo` → graphic HTV9,
  `a woman in a white shirt` → MC áo trắng. Query tiếng Việt thô yếu hơn.
  Đã thêm lớp dịch VI→EN tuỳ chọn (`KIS_TRANSLATE_QUERY=1`, MarianMT,
  +0,6–1 s/query) nhưng **mặc định tắt** vì chưa chứng minh cải thiện top-1.
- Mean-centering embedding không giúp giảm hubness trên mẫu này; giữ cosine thô.
- Full embed 20 video đang chạy nền; khi xong chỉ cần `kis index build`,
  không phải embed lại.

> Chất lượng trên 2 video chỉ là smoke test. Cần `eval/queries.jsonl` + recall@20
> ở bước sau mới kết luận được; các kênh OCR/ASR/RRF dự kiến sẽ bù điểm yếu visual.

---

## 9. Sai lệch / điều chỉnh so với plan

1. **`requires-python` = 3.12** thay vì 3.14, đúng đối sách rủi ro wheel ở §11.
2. **Tách optional deps** thành `offline` (keyframe), `embed`, `ocr`, `asr`
   thay vì gộp tất cả vào `offline`; tránh kéo torch/onnxruntime khi mới làm keyframe.
3. **`degrade_on_long_shot` mặc định `False`.** Plan §3.3 nói "có shot >10 s → fallback
   toàn bộ video sang 1 fps", nhưng cả 20/20 video test đều có shot >10 s, nên nếu bật
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
8. **Thêm `kis eval ablation`** đo candidate V0–V3 mà không decode. Mục đích là chọn
   profile bằng số đo trước khi trả giá CPU cho extraction; kết quả khớp với coverage
   của index thật (0.9671 vs 0.9672). Không thay schema.
9. **Dùng ffmpeg static build** trong `.tools/` và override `KIS_FFMPEG` / `KIS_FFPROBE`
   vì máy test không có `ffmpeg`/`ffprobe` trong PATH. PyAV vẫn là đường decode chính.
10. **`IndexIDMap2(FlatIP)` thay vì "vị trí = keyframe_pk"** của docs §4. `keyframe_pk`
    không liên tục sau `--force`/reindex một phần, nên id tường minh an toàn hơn;
    vẫn là exact search.
11. **`.npy` shard theo video** thay vì theo pack, để resume/rebuild đúng video và
    khớp layout `keyframes/<pack>/<video_id>/`.
12. **Prompt template `This is a photo of {}.`** khi encode text. SigLIP-2 được huấn
    luyện với caption; template chỉ áp dụng lúc query, không đổi image shard.
13. **Lớp dịch VI→EN tuỳ chọn, mặc định tắt** (`KIS_TRANSLATE_QUERY=1`). Đo trên mẫu
    nhỏ thấy dịch chưa cải thiện top-1 rõ rệt và tốn thêm 0,6–1 s/query, nên để opt-in.
14. **torch CPU wheel qua `[tool.uv.index]`** để tránh tải CUDA ~2,5 GB; thêm
    `sentencepiece` cho MarianMT.

---

## 10. Việc tiếp theo

1. ✅ Chạy lại `kis keyframes` cho 20 video sau khi sửa policy — hoàn tất, 14.061 keyframe.
2. ✅ Chạy `kis debug verify-timestamps --sample 100` trên index mới — 100/100 exact.
3. ✅ Chạy `kis eval coverage` trên 20 video — PASS.
4. ✅ Chạy ablation V0–V3 — đã có `kis eval ablation`.
5. ✅ Code embed / FAISS / search CLI / web UI — đã chạy được với partial index.
6. 🟡 Đang embed đủ 20 video (2/20 xong); sau khi xong chạy `kis index build` (strict)
   để có 14.061 vector.
7. Tạo `eval/queries.jsonl` + recall@20/MRR để kết luận chất lượng visual và tune
   template/translation bằng số đo.
8. OCR (RapidOCR) + ASR (faster-whisper) + FTS5, weighted RRF, temporal aggregation.
9. Chạy full `Dataset/batch2` (363 video) rồi mở rộng ra toàn bộ pack; cân nhắc HNSW
   nếu p95 vượt ngân sách.
10. DRES client + mock, timer/EV, pin tray cho M1.

---

## 11. Ghi chú vận hành

- Máy test hiện tại **12 core Xeon E5-2690 (không AVX2)**: dùng `--workers 8` cho
  shot extraction; đặt `OMP_NUM_THREADS=1` / `OPENCV_NUM_THREADS=1` khi chạy pool.
- Embed SigLIP-2 base-384 trên CPU này đạt **~1,4–1,5 ảnh/s**; full 14k keyframe là
  job dài, nên chạy nền và resume (shard đã ghi sẽ được bỏ qua).
- `Dataset/batch2` chứa **363 video, 46 GB**; index test 20 video nằm ở `index/`.
- `.tools/` chứa ffmpeg/ffprobe static; `.tools/`, `Dataset/`, `index/`, `eval/`
  đều nằm trong `.gitignore`.
- Artifact vector: `index/embeddings/<pack>/<video_id>.{npy,json}`, `index/faiss.index`,
  `index/manifest.json`; tất cả nằm dưới `index/` nên đã gitignore.
- SQLite dùng WAL + `busy_timeout=60 s`; mỗi worker mở connection riêng.
- `kis keyframes` và `kis embed` mặc định **resume**: dữ liệu đã có sẽ bỏ qua;
  dùng `--force` để làm lại.
- `kis serve` mặc định bind `127.0.0.1`; UI cần index đã build (partial cũng chạy được).
