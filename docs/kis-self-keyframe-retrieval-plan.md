# Plan: Hệ thống Video Retrieval KIS — tự keyframe, CPU-only

> Bối cảnh: đọc 4 docs hiện có (`docs/Retrieval-System-Plan.md`, `docs/Plan-KIS-Final-Round.md`,
> `docs/Schema-Reference.md`, `docs/KISBranche-Features.md`) và repo hiện trạng:
> chỉ có `Dataset/Video/` (20 video `L21_*`, `L22_*`), chưa có `src/`, chưa có keyframes/OCR/ASR,
> env chỉ có `faiss-cpu`, `torch`, `transformers`, `pillow`, Python pin `3.14`.
>
> 4 quyết định đã chốt với bạn:
> 1. Phạm vi: **full dataset ~800 video / ~292 giờ** (dev trên 20 video trước).
> 2. Hạ tầng: **CPU-only, kể cả offline** — không thuê GPU.
> 3. Keyframing: **hybrid shot-based + lưới đều**.
> 4. Nộp bài DRES: `{"answerSets":[{"answers":[{"mediaItemName": VIDEO_ID, "start": ms, "end": ms}]}]}`
>    → ground truth là **khoảng thời gian**, không phải keyframe id.

---

## 0. Tóm tắt quyết định (TL;DR)

| Hạng mục | Chốt |
|---|---|
| Danh tính keyframe | `keyframe_pk` tự tăng toàn cục (int64) + `(video_id, kf_n)` unique; **bỏ** pack `video_head×10000+kf_n` của risc97 vì 1 fps × video dài sẽ tràn |
| Timestamp | `t_ms` lấy từ **pts của demuxer (PyAV/ffprobe)**; cấm `frame_idx / fps` (VFR) |
| Keyframe | Shot detection (PySceneDetect) + 1 frame/shot + lưới 3 s trong shot dài + I-frame pool + pHash dedup |
| Embedding | SigLIP-2 **base** (768-d) fp32/ONNX-int8 trên CPU; lưu `.npy` fp16 + FAISS `IndexFlatIP` chuẩn hoá L2 |
| OCR | **RapidOCR (ONNX Runtime)** — thay PaddleOCR vì CPU-only; chạy full-res lúc extract, chỉ lưu text phẳng |
| ASR | **faster-whisper int8 + VAD + BatchedInferencePipeline** |
| Text search | SQLite FTS5 kép (có dấu `remove_diacritics 0` + bỏ dấu `remove_diacritics 2`) |
| Fusion | Weighted RRF (k=60) + **temporal aggregation trong cùng video** (đặc thù submit theo interval) |
| Rerank | Gemini top-20 async, cache, timeout 2 s, tắt được |
| UI | FastAPI 1 phiên, keyboard-first, grid + detail + scrub, khay ghim, timer/EV, **nút nộp theo interval** |
| Eval | GT = interval (đúng format DRES) → `Recall@20 theo overlap thời gian`, MRR, p95 latency, **temporal coverage** |
| Môi trường | Hạ `requires-python` xuống **3.12** (xem §11 rủi ro wheel) |

---

## 1. Hệ quả của việc "không có keyframe BTC" — 5 thay đổi so với docs cũ

Docs cũ giả định BTC cấp `Keyframes/` + `map-keyframes/*.csv` + `Clip-features/*.npy`.
Không có 3 thứ đó, kéo theo:

1. **Không có `t_ms` chuẩn** → phải tự sinh bảng thời gian từ pts demuxer. Đây là hạng mục *bắt buộc đúng*, sai là hỏng toàn bộ điểm số (nộp lệch giây).
2. **Không có ground-truth keyframe** → eval không thể đo "có lấy đúng kf_n không". Phải đổi sang **đo theo thời gian**: hit nếu keyframe trả về có `t_ms ∈ [gt.start, gt.end] ± tol`. May mắn là format nộp bài cũng là interval → eval khớp 1:1 với cách chấm.
3. **Recall bị chặn trần bởi mật độ sampling** (bài học ph4mHoa: sai lưới → cap ~20%). Phải có **ablation sampling đo bằng recall**, không chọn bằng cảm tính. → §3.6.
4. **Không có `.npy` của BTC** → tự embed toàn bộ trên CPU, đây là chi phí lớn nhất của dự án → §3.7 + §5.
5. **Không có `mapping` sẵn để đối chiếu** → mọi artifact phải có `manifest.json` + assert tính nhất quán (`FAISS.ntotal == COUNT(keyframes)`) ngay lúc khởi động; index sai âm thầm sẽ phá giờ thi.

---

## 2. Nguyên tắc thiết kế (giữ từ docs, siết thêm cho CPU)

1. **Offline nặng, runtime nhẹ**, nhưng offline bị giới hạn CPU → mọi bước phải **idempotent + resumable + đo throughput thật**.
2. **Một nguồn sự thật**: SQLite (`index/meta.db`) + `index/manifest.json`. Không có Milvus/Qdrant/ES.
3. **Không decode video 2 lần nếu tránh được**: pass 1 shot detection (downscale), pass 2 extract frame theo pts bằng `-ss`/PyAV seek.
4. **Mọi external call có fallback** (Gemini timeout 2 s → dùng nguyên văn query).
5. **Freeze trước giờ thi**, chỉ sửa bug.

---

## 3. Giải pháp keyframing (trọng tâm)

### 3.1. Nguyên tắc

- **"Keyframe" = đơn vị index**, không phải frame nộp bài. Nộp bài là interval → keyframe chỉ cần **đủ dày để top-20 chứa một mốc thời gian nằm trong interval GT**.
- Hai mục tiêu tách biệt cần tối ưu cùng lúc:
  - **Coverage**: mọi khoảng 1–2 s của video đều có 1 keyframe gần đó (giới hạn trần recall).
  - **Chi phí**: số keyframe × (embed + OCR) chạy nổi trên CPU trong vài chục giờ.
- **Shot là đơn vị ngữ nghĩa**: một shot thường dài 2–8 s trong video tin tức; nộp nguyên interval của shot là cách an toàn nhất để "trúng" GT mà không cần biết scoring chi tiết.

### 3.2. Pipeline keyframing (chạy offline, per-video, song song process pool)

```
K1. probe          ffprobe -show_streams -show_format
                   → videos(video_id, width, height, codec, duration_ms,
                            r_frame_rate, avg_frame_rate, time_base,
                            nb_frames, vfr, fps_source)
K2. candidate pool
    K2a  I-frame pool   ffmpeg -skip_frame nokey -vf showinfo → (pts_ms) mỗi I-frame
                        (rất rẻ: không decode P/B; là "lưới gốc" của BTC-keyframe)
    K2b  shot bounds    PySceneDetect ContentDetector, decode downscale → shots(t0_ms,t1_ms)
    K2c  safety grid    ffmpeg -vf fps=1,showinfo → mốc 1 fps (chỉ dùng làm lưới an toàn)
K3. selection policy (xem §3.3) → danh sách mốc thời gian cần extract
K4. extract          decode đúng pts (accurate seek), full-res:
                     ├─ OCR (RapidOCR) → text phẳng  → ocr(video_id, kf_n, text)
                     └─ resize long-side 384 → JPEG q85 → keyframes/<pack>/<video_id>/<kf_n>.jpg
K5. dedup + persist  pHash 64-bit → loại near-duplicate; ghi SQLite + manifest
```

### 3.3. Công thức chọn frame (policy `hybrid`)

Cho mỗi video, sau khi có `shots[]` và `iframes[]`:

```python
GRID_S      = 3.0   # lưới phụ trong shot dài
MAX_PER_SHOT = 6
MIN_SHOT_S  = 0.4   # shot ngắn hơn coi như nhiễu, gộp vào shot trước

for shot in shots:
    dur = shot.t1_ms - shot.t0_ms
    picks = [f for f in iframes if shot.t0_ms <= f <= shot.t1_ms]   # I-frame trong shot
    if not picks:                                    # shot không có I-frame
        picks = [mid(shot)]
    if dur > 4000:                                    # shot dài → thêm lưới 3 s
        picks += [shot.t0_ms + k*GRID_S*1000 for k in range(1, ceil(dur/3000)) if inside]
    picks = dedup_within(picks, min_gap_ms=800)[:MAX_PER_SHOT]
    keep(picks)

# video không detect được shot (n_shots <= 1) hoặc có shot > 10 s:
#   → fall back toàn bộ video sang lưới 1 fps (an toàn), và đánh dấu degraded=1
```

**Vì sao dùng I-frame trong shot thay vì frame giữa shot?**
I-frame gần như trùng khớp với cách BTC chọn keyframe ở các mùa trước (keyframe của encoder), rẻ (đã có sẵn tập), và giữ mật độ ổn định. Frame giữa shot chỉ dùng khi shot không có I-frame.

### 3.4. Timestamp an toàn VFR (bắt buộc, không thương lượng)

- `t_ms` **chỉ** lấy từ: `frame.pts * stream.time_base` (PyAV) hoặc `pts_time` của `showinfo`/`ffprobe -show_frames`.
- Lưu thêm `frame_idx` (đếm tuần tự khi decode) để tham khảo/nộp bài nếu BTC đổi format, nhưng **không bao giờ** dùng để suy ra thời gian.
- `videos.fps_source` chỉ để tham khảo; đánh dấu `vfr = (abs(r_frame_rate - avg_frame_rate) / avg_frame_rate > 0.001)`.
- Test bắt buộc: với 1 video, so `t_ms` của keyframe với `ffprobe -show_frames` tại đúng frame đó — sai số phải = 0.

### 3.5. Dedup near-duplicate

- pHash 64-bit trên ảnh downscale 64×64 (`imagehash`), Hamming ≤ 4 → coi là trùng.
- So sánh trong cửa sổ ±2 shot (không so toàn cục — tránh xoá frame hợp lệ của cảnh lặp lại).
- Loại phụ: nếu 2 pick cách nhau `< 800 ms` → giữ 1 (bỏ frame của slow-pan gần trùng).
- Dedup chạy **trước** embed/OCR để tiết kiệm CPU (đây là lý do đặt ở K3–K5, không phải sau embed).

### 3.6. Ablation sampling — trả lời bằng số, không bằng cảm tính

Chạy 4 profile trên cùng eval set (§8) và cùng 20 video dev:

| Profile | Mô tả | Mục đích |
|---|---|---|
| V0 | I-frame only (`-skip_frame nokey`) | baseline rẻ nhất |
| V1 | lưới đều 1 fps | baseline "an toàn" |
| V2 | 1 frame / shot | baseline ngữ nghĩa |
| **V3 (đề xuất)** | hybrid §3.3 | ứng viên chính |

Đo cho từng profile: `n_keyframes`, `disk_gb`, `build_time_h`, `recall@20 (interval)`, `MRR`, **`temporal_coverage`** (tỉ lệ bin 1 s có ít nhất 1 keyframe trong ±1 s) và **phân bố `max_gap`** (khoảng trống lớn nhất giữa 2 keyframe liên tiếp).

`temporal_coverage` và `max_gap` là chỉ số *không cần GT* — đo được ngay khi chưa có query, và là proxy trực tiếp của trần recall. **Điều kiện chấp nhận V3: `max_gap ≤ 3 s` ở p99 và coverage(±1 s) ≥ 0.95.**

### 3.7. Ngân sách CPU & disk (ước lượng, **phải đo lại ở M0**)

Giả định full dataset: ~800 video / ~292 h / ~1.05 M giây.

| Bước | Khối lượng | Throughput CPU (16 core, ước lượng) | Thời gian |
|---|---|---|---|
| probe | 800 video | tức thời | < 5 phút |
| K2a I-frame pass | 1.05 M s | ~50–100× realtime (skip non-key) | 3–6 h |
| K2b shot detect | 1.05 M s decode downscale | ~8–20× realtime / worker, 8 worker | 6–18 h |
| K3–K4 extract + OCR | ~600 k frame | RapidOCR ~80–150 ms/frame/worker, 8 worker | 8–20 h |
| K4 embed SigLIP-2 base | ~600 k frame | 40–90 img/s (fp32) · 100–200 (ONNX int8) | 2–4 h (int8) → 4–8 h |
| ASR faster-whisper small int8 | ~292 h audio (VAD còn ~65%) | 8–15× realtime, batch pipeline | 12–24 h |
| **Tổng** | | | **~1.5–3 ngày máy CPU, chạy đêm/nhiều máy** |

Disk (target V3, ~600 k keyframe):

| Artifact | Ước lượng |
|---|---|
| keyframes 384px JPEG q85 (~20–28 KB) | 12–17 GB |
| thumbnails 224px (~6–8 KB) | 4–5 GB |
| embeddings 768-d fp16 | ~0.9 GB (fp32 1.8 GB) |
| FAISS `IndexFlatIP` | ~1.8 GB |
| SQLite (keyframes+ocr+transcripts+FTS) | 2–5 GB |
| **Tổng** | **~22–30 GB** |

Giảm tải nếu chậm: (a) `--ocr-shot-reps-only` (chỉ OCR frame đại diện shot), (b) grid 5 s thay 3 s, (c) embed ONNX int8, (d) tạm bỏ ASR ở M0–M1.

---

## 4. Schema & artifact (điều chỉnh so với `docs/Schema-Reference.md`)

```sql
videos      (video_id TEXT PK, pack_id TEXT, video_path TEXT, duration_ms INT,
             width INT, height INT, codec TEXT, fps_source REAL, time_base TEXT,
             nb_frames INT, vfr INT, degraded INT DEFAULT 0)

shots       (video_id TEXT, shot_id INT, t_start_ms INT, t_end_ms INT, start_frame INT,
             PRIMARY KEY (video_id, shot_id))          -- MỚI: cần cho submit interval

keyframes   (keyframe_pk INTEGER PRIMARY KEY AUTOINCREMENT,   -- MỚI: id toàn cục cho FAISS
             video_id TEXT, kf_n INT NOT NULL, frame_idx INT, t_ms INT NOT NULL,
             pts REAL, shot_id INT NULL, image_path TEXT, source TEXT,  -- 'iframe'|'grid'|'shot'
             phash INTEGER,
             UNIQUE (video_id, kf_n), UNIQUE (video_id, t_ms))

ocr         (video_id TEXT, kf_n INT, text TEXT, PRIMARY KEY (video_id, kf_n))
transcripts (video_id TEXT, seg_id INT, t_start_ms INT, t_end_ms INT, text TEXT,
             PRIMARY KEY (video_id, seg_id))
```

**Điều chỉnh so với docs cũ (phải ghi rõ trong plan vì docs nói ngược):**

| Docs cũ | Thay đổi | Lý do |
|---|---|---|
| `vector_id = video_head×10000 + kf_n` | **bỏ**, dùng `keyframe_pk` tự tăng | 1 fps × video > 2.8 h là tràn 10.000; tự keyframe nên mật độ không kiểm soát được |
| `IndexIDMap2` | `IndexFlatIP` + vị trí = `keyframe_pk` | id liên tục 1..N nên vị trí trùng id; đơn giản và nhanh hơn |
| Không có bảng `shots` | **thêm** | submit theo interval cần biên shot; đồng thời dùng cho temporal aggregation |
| `t_ms` từ `pts_time` CSV của BTC | tự sinh từ pts demuxer (§3.4) | không có CSV BTC |
| OCR = PaddleOCR | RapidOCR (ONNX Runtime) | CPU-only, wheel nhẹ, cùng bộ model PP-OCR |
| Assert `ntotal == COUNT(keyframes)` | **giữ nguyên** | chống lệch index/metadata âm thầm |
| `manifest.json` SHA-256 | **giữ nguyên**, mở rộng thêm `sampling_profile`, model revision, code git SHA | tái tạo index deterministic |

`index/manifest.json`:
```json
{"built_at":"...","git_sha":"...","sampling":{"profile":"hybrid","grid_s":3,"max_per_shot":6},
 "nvideos":800,"nkeyframes":612345,
 "models":{"embed":"google/siglip2-base-patch16-384@<rev>","ocr":"rapidocr@<ver>",
           "asr":"faster-whisper-small-int8"},
 "checksums":{"faiss_sha256":"...","meta_db_sha256":"..."}}
```

---

## 5. Offline còn lại: embed / OCR / ASR

**Embed**: SigLIP-2 base, ảnh 384px, `l2_normalize`, batch 32–64, `torch.inference_mode()`, `torch.set_num_threads(core)`. Ghi `.npy` shard theo pack (fp16) → build FAISS `IndexFlatIP` (exact, ~600 k × 768 = 1.8 GB, search 10–30 ms). Nếu sau này > 2 M vector thì mới chuyển `IVF`/HNSW — không làm sớm.

**OCR**: RapidOCR trên ảnh **full-res** (không phải 384px) ngay trong pass extract, lưu text phẳng ghép bằng `\n`. Chạy song song process pool. Chuẩn hoá: NFC, lowercase, gộp khoảng trắng. Ghi 2 cột FTS (`text`, `text_nodia` dùng `unicodedata` bỏ dấu).

**ASR**: faster-whisper `small` int8, `vad_filter=True`, `BatchedInferencePipeline(batch_size=8)`, `language="vi"`. Mỗi segment sinh `(t_start_ms, t_end_ms, text)`. Map transcript → keyframe **theo thời gian** (không theo frame): khi search ASR trả segment, expand ra mọi keyframe có `t_ms ∈ [t_start, t_end]`.

---

## 6. Runtime (CPU, mục tiêu < 2 s p95)

```
query (append dần)
 ├ Gemini structuring (cache key=sha1(query), timeout 2 s, fail → raw)
 │   → {vi_norm, en_for_clip, keywords[], onscreen_text[], time_hint}
 ├ Stream V  : SigLIP-2 text tower (~80–150 ms CPU) → FAISS FlatIP top-200
 ├ Stream O  : FTS5 ocr  (có dấu OR bỏ dấu, phrase query) → top-200 keyframe
 ├ Stream A  : FTS5 transcripts → top-200 segment → expand ra keyframe theo t_ms
 ├ Weighted RRF (k=60, w tune ở M2)
 ├ Temporal aggregation  ← MỚI, đặc thù submit interval
 │   • gom hit theo (video_id) rồi theo cụm t_ms cách nhau ≤ 30 s
 │   • boost cụm có ≥2 hit từ ≥2 stream ("nhiều nguồn xác nhận trong cùng đoạn")
 │   • mỗi cụm gắn interval = shot tương ứng (clamp ≤ 6 s)
 └ (optional) Gemini rerank top-20 — async, cache, tắt được
→ UI: grid + interval đề xuất + nguồn match + điểm
```

Trọng số khởi điểm: `w_V=1.0, w_O=1.0, w_A=0.8`; tune ở M2 bằng ablation trên eval set.

---

## 7. Nộp bài (start/end ms) & UI

**Adapter DRES** (`src/kis/dres.py`), payload đúng format đã cho:

```python
{"answerSets": [{"answers": [{"mediaItemName": video_id,
                              "start": start_ms, "end": end_ms}]}]}
```

**Policy interval** (cấu hình `SUBMIT_POLICY`, tune ở rehearsal):

| Policy | Interval | Khi nào |
|---|---|---|
| `shot` (mặc định) | `[shot.t_start_ms, shot.t_end_ms]` clamp ≤ 6 s | tin cậy trung bình; shot thường = đúng đoạn sự kiện |
| `point` | `[t_ms − 500, t_ms + 500]` | tin cậy cao, frame rất chắc |
| `wide` | `[t_ms − 2000, t_ms + 2000]` | Q&A/Video KIS mô tả mơ hồ |

**UI** (FastAPI + Jinja/HTMX, 1 phiên, keyboard-first):
- Grid 20 kết quả: thumbnail lớn (đọc được banner), badge nguồn (V/O/A), interval đề xuất.
- Detail: frame full-res **extract on-demand từ mp4** (không lưu full-res) + scrub ±2 s (`←/→`), zoom (Q&A đọc số).
- Khay ghim ứng viên; "chỉ search trong khay".
- Panel: countdown `tL`, `ws`, và **điểm nếu nộp đúng ngay bây giờ** `(50−10·ws)+50(1−t/tL)`.
- `Enter` = nộp kèm policy interval; `Shift+Enter` = đổi policy.

---

## 8. Eval harness mới (GT = interval)

`eval/queries.jsonl` — mỗi dòng:
```json
{"qid":"q001","type":"textual|video|qa",
 "reveals":["mô tả phần 1","mô tả phần 2", "..."],
 "gt":{"mediaItemName":"L21_V024","start":734000,"end":739200},
 "tolerance_ms":1500}
```

Cách dựng GT (vì không có keyframe BTC):
1. Chọn ngẫu nhiên một keyframe trong index làm tâm sự kiện.
2. Interval GT = **interval shot chứa keyframe đó** (đây là mức dung sai tự nhiên, khớp cách BTC chấm theo đoạn).
3. Viết mô tả tiếng Việt theo phong cách thể lệ; **viết cách ≥ 1 tuần sau khi xem** để giảm bias.
4. Video KIS: cắt 20 s, người khác mô tả lại từ clip.
5. M0: ≥ 50 query (20 video dev). M2: ≥ 150 query trên full dataset.

Metrics (`src/kis/eval/run_eval.py` → `eval/report.md`):
- **Recall@20 (interval)**: hit nếu tồn tại keyframe trả về có `video_id == gt.mediaItemName` và `t_ms ∈ [start−tol, end+tol]`.
- **MRR@20**, **Recall@1**, **median rank**.
- **Latency p50/p95** (từ query đến render top-20).
- **Temporal coverage / max_gap** (không cần GT, chạy trên index).
- **Bảng ablation**: profile sampling × (recall, n_kf, disk, build_time) và trọng số RRF × (recall).

---

## 9. Cấu trúc code, CLI, deps

```
src/kis/
  __init__.py            main()  → cli.main
  cli.py                 typer/argparse: probe|shots|keyframes|embed|ocr|asr|index|search|eval|serve
  config.py              pydantic-settings; profile sampling; paths
  db.py                  SQLite connect, migrate, FTS5 + triggers, assert_integrity()
  ingest/
    probe.py             ffprobe → videos
    shots.py             PySceneDetect + ffmpeg scene-select fallback
    candidates.py        I-frame pool, grid, policy §3.3
    extract.py           PyAV/ffmpeg accurate seek → OCR + JPEG
    dedup.py             pHash
    embed.py             SigLIP-2 image tower → npy shards
    ocr.py               RapidOCR
    asr.py               faster-whisper
  index/
    build.py             npy → FAISS; manifest; assert ntotal == COUNT
    store.py             FAISS + SQLite lookup
    text.py              FTS5 dual-diacritic search
  search/
    query.py             Gemini structuring (cache/timeout/fallback)
    fusion.py            weighted RRF
    temporal.py          cluster + interval đề xuất
    rerank.py            Gemini rerank (optional)
    pipeline.py          orchestrate 3 stream
  web/                   app.py + templates/ + static/
  dres.py                submit + mock server
  eval/
    make_eval.py, run_eval.py, report.py
index/                   meta.db, embeddings/, faiss.index, manifest.json, keyframes/
eval/                    queries.jsonl, report.md
```

Deps thêm (gộp vào `pyproject.toml`, nhóm optional cho offline):
```toml
requires-python = ">=3.12,<3.13"      # xem rủi ro §11

dependencies = [
  "faiss-cpu>=1.9", "numpy>=1.26", "pillow>=10", "orjson>=3",
  "fastapi>=0.115", "uvicorn>=0.30", "jinja2>=3.1", "pydantic-settings>=2",
  "requests>=2.32", "tqdm>=4.66",
]
[project.optional-dependencies]
offline = ["torch>=2.4", "transformers>=4.49", "av>=13",
           "scenedetect>=0.6", "imagehash>=4.3",
           "rapidocr-onnxruntime>=1.4", "faster-whisper>=1.0"]
```
Cài torch CPU: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` (tránh 2.5 GB CUDA).

---

## 10. Milestones & nghiệm thu

### M0 — Keyframing + index + eval harness (làm trước, không hoãn)
- [ ] `kis probe` + `kis shots` + `kis keyframes` trên 20 video dev → `index/meta.db` đủ `videos/shots/keyframes`, JPEG + thumbnail.
- [ ] `kis embed` SigLIP-2 base → FAISS; `kis search` CLI trả top-20.
- [ ] **Time-correctness test**: `t_ms` khớp `ffprobe -show_frames` (sai số 0) cho 100 keyframe ngẫu nhiên.
- [ ] **Ablation V0–V3** trên 20 video: bảng recall/coverage/max_gap/disk/build_time.
- [ ] `eval/queries.jsonl` ≥ 50 query + `kis eval run` in report.
- **Nghiệm thu**: `max_gap p99 ≤ 3 s` & `coverage ≥ 0.95`; search p95 < 2 s; có số recall@20 baseline V3 ≥ V0/V1/V2.
- *Đo throughput thật để hiệu chỉnh bảng §3.7 trước khi commit full dataset.*

### M1 — Đủ 3 kênh + RRF + UI + DRES mock
- [ ] OCR (RapidOCR) + FTS5 kép; ASR (faster-whisper) + FTS5.
- [ ] Gemini query structuring (cache/timeout 2 s/fallback nguyên văn).
- [ ] Weighted RRF + temporal aggregation.
- [ ] UI grid/detail/scrub/khay ghim/timer/EV + nộp theo `SUBMIT_POLICY`.
- **Nghiệm thu**: recall@20 ≥ 0.70; 10 query liên tiếp không crash; drill 1 Textual KIS < 5 phút.

### M2 — Full dataset + tuning
- [ ] Chạy full ~800 video theo profile V3 (nhiều đêm/máy, resumable), `manifest.json` đầy đủ.
- [ ] Tune trọng số RRF + chọn tập channel bằng ablation có kiểm soát.
- [ ] Gemini rerank top-20 (chỉ giữ nếu recall/precision tăng có ý nghĩa).
- [ ] Rehearsal đúng đồng hồ, mock DRES; tinh chỉnh `SUBMIT_POLICY` và độ rộng interval.
- **Nghiệm thu**: recall@20 ≥ 0.85; Textual KIS trung vị ≤ 2.5 phút; ≤ 1 lần nộp sai trung bình.

### M3 — Cứng hoá + freeze
- [ ] Degraded mode (mất Gemini, thiếu ASR), warm-up index khi khởi động.
- [ ] p95 < 2 s; runbook giờ thi; freeze code.
- *Optional*: OCR boxes cho Q&A, object boost, temporal decomposition query nhiều ý.

---

## 11. Rủi ro & đối sách

| Rủi ro | Mức | Đối sách |
|---|---|---|
| **Python 3.14 thiếu wheel** (`onnxruntime`, `ctranslate2`, `av`, `torch`) | Cao | Hạ `requires-python` xuống **3.12**; nếu bắt buộc giữ 3.14 → tách env `offline` chạy 3.12 qua `uv venv --python 3.12` |
| Shot detection chậm/không chính xác trên tin tức | Cao | Fallback `ffmpeg -vf select='gt(scene,0.3)' + showinfo` (rẻ hơn, cùng interface); lưới 1 fps an toàn cho video degraded |
| Keyframe quá nhiều → vượt ngân sách CPU | Cao | Đo ở M0 rồi chọn `grid_s` (3→5 s) và `MAX_PER_SHOT`; OCR chỉ shot-reps nếu cần |
| **Lệch `t_ms` do VFR** | Cao | Chỉ dùng pts demuxer + test đối chiếu ffprobe (§3.4); cấm `frame_idx/fps` |
| Recall trần thấp do sampling thưa | Cao | `max_gap`/`coverage` là gate ở M0; không qua gate thì giảm `grid_s` trước khi index full |
| OCR/ASR tiếng Việt sai nhiều | Trung | Đo WER/CER trên 1 h mẫu; thay bằng PhoWhisper-small nếu cần; FTS bỏ dấu giảm lỗi dấu |
| Query gõ không dấu | Trung | FTS5 kép 2 cột (`remove_diacritics 0` + `2`), union kết quả |
| Nộp interval sai độ rộng | Trung | Tune ở rehearsal (M2); cho phép đổi policy bằng 1 phím |
| Bias eval (tự viết query, tự chấm) | Trung | Viết query cách ≥ 1 tuần sau khi xem; đo tự động; chéo người viết |
| Gemini lag giờ thi | Trung | Cache + timeout 2 s + fallback raw; mọi kênh local chạy độc lập |
| Index/metadata lệch | Thấp–Trung | `assert ntotal == COUNT(keyframes)` lúc load; manifest SHA-256 |

---

## 12. Verification (cách kiểm chứng khi implement)

```powershell
# 0. Môi trường
uv venv --python 3.12 ; uv sync --extra offline
uv run python -c "import faiss, torch, av, scenedetect, rapidocr_onnxruntime, faster_whisper"

# 1. Probe + shot + keyframe trên 20 video dev
uv run kis probe   --dataset Dataset/Video
uv run kis shots   --workers 8
uv run kis keyframes --profile hybrid --grid-s 3 --workers 8

# 2. Kiểm tra thời gian (bắt buộc pass)
uv run kis debug verify-timestamps --sample 100
#    → so t_ms với `ffprobe -show_frames` của đúng frame; assert diff == 0

# 3. Coverage gate (chưa cần query)
uv run kis eval coverage --report eval/coverage.md
#    → require: max_gap p99 <= 3000 ms, coverage_1s >= 0.95

# 4. Embed + index
uv run kis embed --model google/siglip2-base-patch16-384 --batch 64
uv run kis index build        # assert FAISS.ntotal == COUNT(keyframes)

# 5. Search + latency
uv run kis search "thủ tướng phát biểu tại hội nghị về biến đổi khí hậu" --k 20
uv run kis bench --n 30       # p50/p95

# 6. Ablation sampling (kết quả quyết định profile cuối)
uv run kis eval ablation --profiles v0,v1,v2,v3 --workers 8
#    → bảng trong eval/report.md

# 7. Eval đầy đủ
uv run kis eval run --queries eval/queries.jsonl --report eval/report.md

# 8. UI + DRES mock
uv run kis dres mock --port 8080 &
uv run kis serve --port 8000
#    → nộp thử 1 kết quả, kiểm tra payload đúng format
#      {"answerSets":[{"answers":[{"mediaItemName":"L21_V024","start":734000,"end":739200}]}]}
```

**Definition of Done cho M0**: (a) 20 video có keyframe + embedding + FAISS; (b) `verify-timestamps` pass 100/100;
(c) `max_gap p99 ≤ 3 s`, `coverage ≥ 0.95`; (d) `search` p95 < 2 s; (e) bảng ablation V0–V3 + recall@20 baseline trong `eval/report.md`.

---

## 13. Việc bắt đầu ngay

1. **Chốt môi trường**: hạ `requires-python` → 3.12, thêm optional deps `offline`, tạo skeleton `src/kis/` (§9) để `uv run kis` chạy được.
2. **M0-A (indexing)**: `probe → shots → candidates → extract` trên 3 video đầu, in bảng `n_shots`, `n_keyframes`, `max_gap` để hiệu chỉnh policy §3.3 trước khi chạy 20 video.
3. **M0-B (eval/UI)**: format `eval/queries.jsonl` + `kis search` CLI + FastAPI skeleton + mock DRES.
4. **Chung**: lưu plan này vào `docs/` (vd `docs/Plan-Self-Keyframe.md`) và ghi rõ 5 thay đổi so với `docs/Schema-Reference.md` §5 để tránh implement theo tài liệu cũ.
